"""
Shift Factor (PTDF) Calculator for SPP Transmission Constraints.

This module implements the correct statistical approach to computing Power Transfer
Distribution Factors (PTDFs / shift factors) for ISO transmission constraints.

Mathematical Foundation
-----------------------
For every interval t and every constraint k:

    MCC[node, t] = Σ_k SF[node, k] × ShadowPrice[k, t]

(ShadowPrice is 0 when constraint k is not binding.)  Note the sign: this is
the convention the estimator is fit with, and it fits SPP's published RT data
at R² ≈ 0.999.  SPP publishes binding shadow prices as negative numbers.

Shift factors are a property of the network, not of which constraints happen to
bind together, so ALL intervals are pooled into one regression per node:

    A  (T × K):  shadow price matrix, one row per interval, one column per constraint
    B  (T × N):  observed MCC, one column per node
    X  (K × N):  the unknown shift factors

    A @ X ≈ B   solved with a very lightly ridge-regularised least squares.

An earlier version split intervals by the exact set of binding constraints and
solved each set separately.  On real SPP data that left almost every interval
unsolved (median 4 constraints bind together, so exact sets rarely repeat), so
it was replaced.  Constraints that bind in fewer than ``min_intervals`` intervals
are reported in ``ShiftFactorSolution.unmodeled_constraints`` instead of solved.

Reference
---------
See also: shift_factor_regimes.py for regime diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeSolution:
    """Result of one shift-factor solve (a pooled fit over all supplied intervals)."""
    constraints: FrozenSet[str]
    n_intervals: int
    n_constraints: int
    # shift_factors[node][constraint] = SF value
    shift_factors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Goodness-of-fit: R² of the reconstruction per node
    r2_scores: Dict[str, float] = field(default_factory=dict)
    # Residual std dev of reconstructed vs observed MCCs per node
    residual_std: Dict[str, float] = field(default_factory=dict)
    underdetermined: bool = False
    error: Optional[str] = None
    # Rank / condition number of the shadow price design matrix.  rank < K or a
    # huge condition number means some constraints move together and their
    # individual shift factors are not identifiable (R² will still look great).
    rank: Optional[int] = None
    condition_number: Optional[float] = None


@dataclass
class ShiftFactorSolution:
    """
    Top-level result container for shift factor (PTDF) estimates.

    After calling ``ShiftFactorCalculator.fit()``, call ``.flatten()`` to
    populate the internal lookup, then use:

    - ``.get(node, constraint)`` — single SF lookup
    - ``.to_dataframe(shadow_prices)`` — tidy DataFrame with exposure column
    - ``.compute_mcc(node, shadow_prices)`` — pre-clearing MCC estimate
    - ``.flow_impact(source_node, sink_node, mw, constraint)`` — bilateral what-if
    """
    # regime_key -> RegimeSolution
    regimes: Dict[FrozenSet[str], RegimeSolution] = field(default_factory=dict)
    # Flat lookup: (node, constraint) -> SF — populated after calling .flatten()
    _flat: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # Constraints that were binding but too rarely to estimate (see min_intervals)
    unmodeled_constraints: List[str] = field(default_factory=list)

    def flatten(self) -> None:
        """Populate the flat (node, constraint) lookup from all solved regimes."""
        self._flat.clear()
        weighted_sum: Dict[Tuple[str, str], float] = {}
        weight: Dict[Tuple[str, str], float] = {}
        for regime in self.regimes.values():
            if regime.error or regime.underdetermined:
                continue
            w = float(max(regime.n_intervals, 1))
            for node, sfs in regime.shift_factors.items():
                for constraint, sf in sfs.items():
                    key = (node, constraint)
                    weighted_sum[key] = weighted_sum.get(key, 0.0) + w * sf
                    weight[key] = weight.get(key, 0.0) + w
        # Interval-weighted mean across any solutions that share a (node, constraint)
        for key, total in weighted_sum.items():
            self._flat[key] = total / weight[key]

    def get(self, node: str, constraint: str) -> Optional[float]:
        """Look up a shift factor.  Returns None if not solved."""
        return self._flat.get((node, constraint))

    def to_dataframe(
        self,
        shadow_prices: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Return all solved shift factors as a tidy DataFrame.

        Parameters
        ----------
        shadow_prices : dict, optional
            Mapping of ``{constraint_name: shadow_price}`` for the current
            (or most recent) interval.  When provided, adds two columns:

            - ``shadow_price`` — the live shadow price for that constraint
            - ``sf_exposure``  — GSF × ShadowPrice (the trader's "exposure"
              column: how much $/MWh of congestion cost this SF drives)

        Returns
        -------
        DataFrame with columns: ``node``, ``constraint``, ``shift_factor``
        (plus ``shadow_price``, ``sf_exposure`` when shadow_prices supplied).
        """
        rows = []
        for (node, constraint), sf in self._flat.items():
            row: Dict[str, Any] = {
                "node": node,
                "constraint": constraint,
                "shift_factor": sf,
            }
            if shadow_prices is not None:
                sp = shadow_prices.get(constraint, 0.0)
                row["shadow_price"] = sp
                row["sf_exposure"] = sf * sp  # GSF × SP — the key trader metric
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by absolute exposure descending when available — most impactful first
        if shadow_prices is not None and not df.empty:
            df = df.reindex(
                df["sf_exposure"].abs().sort_values(ascending=False).index
            ).reset_index(drop=True)

        return df

    def compute_mcc(
        self,
        node: str,
        shadow_prices: Dict[str, float],
    ) -> float:
        """
        Estimate the Marginal Cost of Congestion (MCC) for a node given
        a live snapshot of shadow prices.

        Formula (same convention the shift factors are fit with):

            MCC_node = Σ_c (SF_node,c × ShadowPrice_c)

        SPP publishes binding shadow prices as negative numbers, so the sign of
        the published shadow price is already carried in the product.

        This is the pre-clearing MCC prediction — call this every time a
        new set of shadow prices comes in (e.g., every 5-minute RTBM publish)
        to estimate the congestion component of LMP before it officially clears.

        Parameters
        ----------
        node : str
            Settlement location / price node name.
        shadow_prices : dict
            ``{constraint_name: shadow_price}`` for the current interval.
            Only binding constraints (non-zero SPs) need to be included.

        Returns
        -------
        float
            Estimated MCC in $/MWh.  Positive → LMP is above the system
            energy price at this node.  Negative → below it.
        """
        mcc = 0.0
        for constraint, sp in shadow_prices.items():
            sf = self.get(node, constraint)
            if sf is not None:
                mcc += sf * sp
        return mcc

    def flow_impact(
        self,
        source_node: str,
        sink_node: str,
        mw: float,
        constraint: str,
    ) -> Optional[float]:
        """
        Dynamic Flow Impact Calculator — answers the trader's "what-if" question:

            "If I inject an extra {mw} MW at {source_node} and buy {mw} MW
             at {sink_node}, how much does the flow change on {constraint}?"

        Formula:

            ΔFlow_constraint = (SF_source,c - SF_sink,c) × ΔMW

        A positive result means the bilateral injection *increases* flow on the
        constraint (exacerbates congestion).  Negative means it *relieves* it.

        Parameters
        ----------
        source_node : str
            The injection node (where MW is being injected / generated).
        sink_node : str
            The withdrawal node (where MW is consumed / purchased).
        mw : float
            Magnitude of the bilateral injection in MW.  Must be positive.
        constraint : str
            The constraint name to evaluate (canonical or raw, must match
            the names used when the solution was computed).

        Returns
        -------
        float
            ΔFlow in MW on the constraint.  Returns None if shift factors
            for either node are not available.

        Examples
        --------
        ::

            delta = solution.flow_impact(
                source_node="WEST_NODE",
                sink_node="HUB_NORTH",
                mw=50.0,
                constraint="LN HARBINE - BEATRICE",
            )
            if delta and delta > 0:
                print(f"Injection exacerbates congestion by {delta:.1f} MW")
        """
        sf_source = self.get(source_node, constraint)
        sf_sink = self.get(sink_node, constraint)

        if sf_source is None or sf_sink is None:
            missing = []
            if sf_source is None:
                missing.append(f"source='{source_node}'")
            if sf_sink is None:
                missing.append(f"sink='{sink_node}'")
            logger.warning(
                f"flow_impact: no shift factor for {', '.join(missing)} "
                f"on constraint '{constraint}'. "
                "Run ShiftFactorCalculator.fit() with more data."
            )
            return None

        return (sf_source - sf_sink) * mw


# ---------------------------------------------------------------------------
# Core calculator
# ---------------------------------------------------------------------------

class ShiftFactorCalculator:
    """
    Statistical estimator for transmission constraint shift factors (PTDFs).

    Usage
    -----
    ::

        calc = ShiftFactorCalculator(min_intervals=10, tolerance=1e-4)
        solution = calc.fit(constraints_df, lmp_df)
        solution.flatten()

        sf = solution.get("SOUC", "LN HARBINE - BEATRICE")
        df = solution.to_dataframe()

    Parameters
    ----------
    min_intervals : int
        A constraint must be binding in at least this many intervals to get
        shift factors estimated (rarer ones are reported as unmodeled), and at
        least this many intervals must have both shadow prices and LMPs.
    tolerance : float
        Shadow prices with |value| below this threshold are treated as
        non-binding.
    validate_bounds : bool
        If True, shift factors outside [-1, 1] are flagged (logged) but not
        discarded — they may legitimately exceed unity for contingency-based
        flowgates, whose effective factor combines PTDFs and outage distribution factors.
    ridge : float
        Relative strength of the light ridge penalty (0 disables it).
    """

    def __init__(
        self,
        min_intervals: int = 10,
        tolerance: float = 1e-4,
        validate_bounds: bool = True,
        ridge: float = 1e-4,
    ) -> None:
        self.min_intervals = min_intervals
        self.tolerance = tolerance
        self.validate_bounds = validate_bounds
        self.ridge = ridge

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        constraints_df: pd.DataFrame,
        lmp_df: pd.DataFrame,
    ) -> ShiftFactorSolution:
        """
        Estimate shift factors from SPP RT/DA binding constraint + LMP data.

        All intervals are pooled into a single regression per node, with a
        shadow price of 0 for every constraint that is not binding in an
        interval.  Shift factors are a property of the network, not of the
        set of constraints that happen to be binding together, so there is no
        need to (and, on real SPP data, no way to) split intervals by regime.

        Parameters
        ----------
        constraints_df : DataFrame
            Columns: ``interval``, ``constraint_name``, ``shadow_price``
            Each row is one constraint binding in one interval.

        lmp_df : DataFrame
            Columns: ``interval``, ``settlement_location``, ``mcc``
            Each row is the MCC component of LMP for one node in one interval.

        Returns
        -------
        ShiftFactorSolution
            Call ``.flatten()`` on the result then use ``.get()`` or
            ``.to_dataframe()`` to access results.
        """
        # Normalise column names to lower-case BEFORE validation
        constraints_df = constraints_df.copy()
        constraints_df.columns = [c.lower() for c in constraints_df.columns]
        # Also handle SPP CamelCase originals that aren't already lowercased
        _c_rename = {
            "constraint name": "constraint_name",
            "shadow price": "shadow_price",
            "settlement location": "settlement_location",
        }
        constraints_df = constraints_df.rename(columns=_c_rename)
        lmp_df = lmp_df.copy()
        lmp_df.columns = [c.lower() for c in lmp_df.columns]
        lmp_df = lmp_df.rename(columns=_c_rename)

        self._validate_inputs(constraints_df, lmp_df)

        # Standardize interval column to datetime
        for df in (constraints_df, lmp_df):
            if not pd.api.types.is_datetime64_any_dtype(df["interval"]):
                df["interval"] = pd.to_datetime(df["interval"])

        # Filter to binding constraints only
        binding = constraints_df[
            constraints_df["shadow_price"].abs() > self.tolerance
        ]

        solution = ShiftFactorSolution()
        if binding.empty or lmp_df.empty:
            return solution

        # Constraints that bind in too few intervals cannot be estimated.
        bind_counts = binding.groupby("constraint_name")["interval"].nunique()
        modeled = sorted(bind_counts[bind_counts >= self.min_intervals].index)
        solution.unmodeled_constraints = sorted(
            set(bind_counts.index) - set(modeled)
        )
        if not modeled:
            return solution

        # Only intervals where we have both shadow prices and LMPs inform the fit.
        intervals = sorted(set(binding["interval"]) & set(lmp_df["interval"]))
        if len(intervals) < max(self.min_intervals, 1):
            return solution

        sol = self._solve_pooled(binding, lmp_df, intervals, modeled)
        solution.regimes[sol.constraints] = sol
        return solution

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self, constraints_df: pd.DataFrame, lmp_df: pd.DataFrame
    ) -> None:
        required_constraint_cols = {"interval", "constraint_name", "shadow_price"}
        required_lmp_cols = {"interval", "settlement_location", "mcc"}

        cols_c = {c.lower() for c in constraints_df.columns}
        cols_l = {c.lower() for c in lmp_df.columns}

        missing_c = required_constraint_cols - cols_c
        missing_l = required_lmp_cols - cols_l

        if missing_c:
            raise ValueError(
                f"constraints_df is missing required columns: {missing_c}. "
                f"Got: {set(constraints_df.columns)}"
            )
        if missing_l:
            raise ValueError(
                f"lmp_df is missing required columns: {missing_l}. "
                f"Got: {set(lmp_df.columns)}"
            )

    def _solve_pooled(
        self,
        binding_df: pd.DataFrame,
        lmp_df: pd.DataFrame,
        intervals: List,
        constraints: List[str],
    ) -> RegimeSolution:
        """Solve MCC[n, t] = Σ_k x[n, k] · SP[k, t] for every node n at once."""
        K = len(constraints)
        T = len(intervals)
        sol = RegimeSolution(
            constraints=frozenset(constraints),
            n_intervals=T,
            n_constraints=K,
        )

        # A (T × K): shadow prices, 0 where a constraint is not binding.
        A = (
            binding_df[binding_df["constraint_name"].isin(constraints)]
            .pivot_table(
                index="interval",
                columns="constraint_name",
                values="shadow_price",
                aggfunc="first",
            )
            .reindex(index=intervals, columns=constraints)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        # B (T × N): observed MCC per node.
        B_df = (
            lmp_df[lmp_df["interval"].isin(intervals)]
            .pivot_table(
                index="interval",
                columns="settlement_location",
                values="mcc",
                aggfunc="first",
            )
            .reindex(index=intervals)
        )
        nodes = list(B_df.columns)
        B = B_df.to_numpy(dtype=float)

        sv = np.linalg.svd(A, compute_uv=False)
        sol.rank = int(np.sum(sv > sv[0] * 1e-10)) if sv.size and sv[0] > 0 else 0
        sol.condition_number = (
            float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float("inf")
        )
        if sol.rank < K:
            logger.warning(
                f"Pooled design is rank-deficient (rank={sol.rank} < K={K}): "
                "some constraints always move together, so their individual "
                "shift factors are not identifiable."
            )

        logger.info(f"Solving pooled regression: T={T} intervals, K={K} constraints, N={len(nodes)} nodes")

        has_all = ~np.isnan(B).any(axis=0)
        X = np.full((K, len(nodes)), np.nan)

        if has_all.any():
            X[:, has_all] = self._ridge_lstsq(A, B[:, has_all])

        # Nodes with gaps in their MCC history are solved on the rows they have.
        for j in np.flatnonzero(~has_all):
            valid = ~np.isnan(B[:, j])
            if valid.sum() < K:
                continue
            X[:, j] = self._ridge_lstsq(A[valid], B[valid, j][:, None])[:, 0]

        for j, node in enumerate(nodes):
            if np.isnan(X[:, j]).any():
                continue
            x = X[:, j]
            valid = ~np.isnan(B[:, j])
            b = B[valid, j]
            resid = b - A[valid] @ x
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((b - b.mean()) ** 2))
            sol.r2_scores[node] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
            sol.residual_std[node] = float(np.sqrt(ss_res / max(1, len(b) - K)))
            sol.shift_factors[node] = {constraints[k]: float(x[k]) for k in range(K)}

            if self.validate_bounds and np.any(np.abs(x) > 1.0):
                out_of_bounds = [
                    f"{constraints[k]}={x[k]:.4f}" for k in range(K) if abs(x[k]) > 1.0
                ]
                logger.debug(
                    f"Node {node}: shift factor(s) outside [-1, 1]: "
                    f"{', '.join(out_of_bounds)}"
                )

        return sol

    def _ridge_lstsq(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Least squares with a very light ridge penalty on column-scaled A.

        The penalty only matters when columns of A are nearly collinear, where
        it shrinks the unidentifiable direction toward zero instead of letting
        it blow up.  It is negligible for well-conditioned problems.
        """
        K = A.shape[1]
        scale = np.sqrt((A ** 2).mean(axis=0))
        scale[scale == 0] = 1.0
        A_s = A / scale
        if self.ridge > 0:
            lam = np.sqrt(self.ridge * A.shape[0])
            A_aug = np.vstack([A_s, lam * np.eye(K)])
            B_aug = np.vstack([B, np.zeros((K, B.shape[1]))])
        else:
            A_aug, B_aug = A_s, B
        x_s = np.linalg.lstsq(A_aug, B_aug, rcond=None)[0]
        return x_s / scale[:, None]


# ---------------------------------------------------------------------------
# Convenience function (backward-compatible entry point)
# ---------------------------------------------------------------------------

def calculate_shift_factors(
    lmp_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    min_intervals: int = 10,
    tolerance: float = 1e-4,
) -> ShiftFactorSolution:
    """
    Convenience wrapper around ``ShiftFactorCalculator.fit()``.

    Parameters
    ----------
    lmp_df : DataFrame
        Columns: ``interval``, ``settlement_location``, ``mcc``
    constraints_df : DataFrame
        Columns: ``interval``, ``constraint_name``, ``shadow_price``
    min_intervals : int
        A constraint must bind in at least this many intervals to be estimated.
    tolerance : float
        Shadow price absolute threshold for "binding".

    Returns
    -------
    ShiftFactorSolution
        Call ``.flatten()`` then ``.get()`` or ``.to_dataframe()``.
    """
    calc = ShiftFactorCalculator(
        min_intervals=min_intervals,
        tolerance=tolerance,
    )
    solution = calc.fit(constraints_df=constraints_df, lmp_df=lmp_df)
    solution.flatten()
    return solution


# ---------------------------------------------------------------------------
# Validation helpers (public, reusable)
# ---------------------------------------------------------------------------

def validate_shift_factors(
    sf_df: pd.DataFrame,
    lmp_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    tolerance: float = 1.0,
) -> pd.DataFrame:
    """
    Reconstruct MCC from shift factors and compare to observed values.

    Parameters
    ----------
    sf_df : DataFrame
        Output of ``ShiftFactorSolution.to_dataframe()``.
        Columns: ``node``, ``constraint``, ``shift_factor``.
    lmp_df : DataFrame
        Observed LMP data with ``interval``, ``settlement_location``, ``mcc``.
    constraints_df : DataFrame
        Shadow price data with ``interval``, ``constraint_name``, ``shadow_price``.
    tolerance : float
        Max allowed abs error in $/MWh before a reconstruction is flagged.

    Returns
    -------
    DataFrame
        Columns: ``interval``, ``node``, ``observed_mcc``, ``reconstructed_mcc``,
        ``abs_error``, ``flagged``
    """
    lmp_df = lmp_df.copy()
    lmp_df.columns = [c.lower() for c in lmp_df.columns]
    constraints_df = constraints_df.copy()
    constraints_df.columns = [c.lower() for c in constraints_df.columns]
    sf_df = sf_df.copy()
    sf_df.columns = [c.lower() for c in sf_df.columns]

    records = []
    for interval, c_grp in constraints_df.groupby("interval"):
        binding = c_grp[c_grp["shadow_price"].abs() > 1e-6]
        if binding.empty:
            continue

        sp_map = dict(zip(binding["constraint_name"], binding["shadow_price"]))

        interval_lmps = lmp_df[lmp_df["interval"] == interval]
        for _, row in interval_lmps.iterrows():
            node = row["settlement_location"]
            observed = row["mcc"]

            # Sum SF × shadow_price over all binding constraints
            reconstructed = 0.0
            for constraint, sp in sp_map.items():
                sf_row = sf_df[
                    (sf_df["node"] == node) & (sf_df["constraint"] == constraint)
                ]
                if not sf_row.empty:
                    reconstructed += sf_row.iloc[0]["shift_factor"] * sp

            abs_err = abs(observed - reconstructed)
            records.append({
                "interval": interval,
                "node": node,
                "observed_mcc": observed,
                "reconstructed_mcc": reconstructed,
                "abs_error": abs_err,
                "flagged": abs_err > tolerance,
            })

    return pd.DataFrame(records)


def time_split_validation(
    constraints_df: pd.DataFrame,
    lmp_df: pd.DataFrame,
    train_frac: float = 0.7,
    min_intervals: int = 10,
    tolerance: float = 1e-4,
    ridge: float = 1e-4,
) -> Dict[str, Any]:
    """
    Out-of-sample check: fit on the first ``train_frac`` of intervals (in time
    order) and score MCC reconstruction on the remaining intervals.

    Scoring a solution on the data it was fit on (as the in-sample backtest
    does) says almost nothing about whether it will predict tomorrow's MCC.

    Returns a dict with in-sample and out-of-sample R² / MAE / RMSE, the MAE of
    a trivial "predict the mean MCC" baseline, and ``unmodeled_share`` — the
    fraction of out-of-sample shadow-price magnitude carried by constraints
    that had no shift factors (never seen binding in the training window).
    """
    c = constraints_df.copy()
    c.columns = [x.lower() for x in c.columns]
    c = c.rename(columns={"constraint name": "constraint_name", "shadow price": "shadow_price"})
    l = lmp_df.copy()
    l.columns = [x.lower() for x in l.columns]
    l = l.rename(columns={"settlement location": "settlement_location"})
    for df in (c, l):
        if not pd.api.types.is_datetime64_any_dtype(df["interval"]):
            df["interval"] = pd.to_datetime(df["interval"])

    binding = c[c["shadow_price"].abs() > tolerance]
    intervals = sorted(set(binding["interval"]) & set(l["interval"]))
    if len(intervals) < 4:
        return {"status": "insufficient_data", "n_intervals": len(intervals)}
    split = int(len(intervals) * train_frac)
    train_iv, test_iv = intervals[:split], intervals[split:]

    calc = ShiftFactorCalculator(min_intervals=min_intervals, tolerance=tolerance, ridge=ridge)
    sol = calc.fit(
        c[c["interval"].isin(train_iv)],
        l[l["interval"].isin(train_iv)],
    )
    sol.flatten()
    if not sol._flat:
        return {"status": "no_solution", "n_train_intervals": len(train_iv)}

    names = sorted({k[1] for k in sol._flat})
    nodes = sorted({k[0] for k in sol._flat})
    X = np.zeros((len(names), len(nodes)))
    n_idx = {n: i for i, n in enumerate(names)}
    m_idx = {n: j for j, n in enumerate(nodes)}
    for (node, con), sf in sol._flat.items():
        X[n_idx[con], m_idx[node]] = sf

    def _score(ivs):
        A = (
            binding[binding["interval"].isin(ivs)]
            .pivot_table(index="interval", columns="constraint_name", values="shadow_price", aggfunc="first")
            .reindex(index=ivs)
            .fillna(0.0)
        )
        modeled = A.reindex(columns=names).fillna(0.0).to_numpy()
        unmodeled_mag = A.drop(columns=[x for x in names if x in A.columns]).abs().to_numpy().sum()
        total_mag = A.abs().to_numpy().sum()
        B = (
            l[l["interval"].isin(ivs)]
            .pivot_table(index="interval", columns="settlement_location", values="mcc", aggfunc="first")
            .reindex(index=ivs, columns=nodes)
        ).to_numpy(dtype=float)
        P = modeled @ X
        mask = ~np.isnan(B)
        err = (B - P)[mask]
        obs = B[mask]
        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        return {
            "n_intervals": len(ivs),
            "n_observations": int(mask.sum()),
            "r2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 1e-12 else None,
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_baseline_mae": float(np.abs(obs - obs.mean()).mean()),
            "unmodeled_share": float(unmodeled_mag / total_mag) if total_mag else 0.0,
        }

    return {
        "status": "ok",
        "n_constraints_modeled": len(names),
        "unmodeled_constraints_in_train": len(sol.unmodeled_constraints),
        "in_sample": _score(train_iv),
        "out_of_sample": _score(test_iv),
    }



# ---------------------------------------------------------------------------
# Analysis / reporting helpers
# ---------------------------------------------------------------------------

def summarize_solution(solution: ShiftFactorSolution) -> pd.DataFrame:
    """
    Produce a summary table of all solved regimes.

    Returns
    -------
    DataFrame
        Columns: ``constraints``, ``n_intervals``, ``n_constraints``,
        ``n_nodes_solved``, ``mean_r2``, ``solved``
    """
    rows = []
    for regime_key, sol in solution.regimes.items():
        rows.append({
            "constraints": ", ".join(sorted(regime_key)),
            "n_intervals": sol.n_intervals,
            "n_constraints": sol.n_constraints,
            "n_nodes_solved": len(sol.shift_factors),
            "mean_r2": (
                float(np.mean(list(sol.r2_scores.values())))
                if sol.r2_scores
                else None
            ),
            "solved": (not sol.underdetermined and sol.error is None),
        })
    return pd.DataFrame(rows).sort_values("n_intervals", ascending=False)


# ---------------------------------------------------------------------------
# CLI entry point (for quick testing)
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Calculate shift factors from SPP RT/DA data files."
    )
    parser.add_argument("--lmp", required=True, help="Path to LMP CSV file(s)")
    parser.add_argument(
        "--constraints", required=True, help="Path to binding constraints CSV file(s)"
    )
    parser.add_argument(
        "--output",
        default="shift_factors.csv",
        help="Output CSV path (default: shift_factors.csv)",
    )
    parser.add_argument(
        "--min-intervals",
        type=int,
        default=10,
        help="Minimum binding intervals per constraint (default: 10)",
    )
    args = parser.parse_args()

    lmp_df = pd.read_csv(args.lmp)
    constraints_df = pd.read_csv(args.constraints)

    # Normalize column names to expected schema
    # SPP RT LMP columns: Interval, GMTIntervalEnd, Settlement Location, LMP, MLC, MCC, MEC
    lmp_df = lmp_df.rename(
        columns={
            "Settlement Location": "settlement_location",
            "Interval": "interval",
            "MCC": "mcc",
        }
    )

    # SPP RT BC columns: Interval, GMTIntervalEnd, Constraint Name, Shadow Price, ...
    constraints_df = constraints_df.rename(
        columns={
            "Constraint Name": "constraint_name",
            "Interval": "interval",
            "Shadow Price": "shadow_price",
        }
    )

    # Filter to binding only
    if "State" in constraints_df.columns:
        constraints_df = constraints_df[constraints_df["State"] == "BINDING"]

    solution = calculate_shift_factors(
        lmp_df=lmp_df,
        constraints_df=constraints_df,
        min_intervals=args.min_intervals,
    )

    df = solution.to_dataframe()
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} shift factors to {args.output}")

    summary = summarize_solution(solution)
    print("\nRegime Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    _main()