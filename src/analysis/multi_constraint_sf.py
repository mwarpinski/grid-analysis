"""
Shift Factor (PTDF) Calculator for SPP Transmission Constraints.

This module implements the correct statistical approach to computing Power Transfer
Distribution Factors (PTDFs / shift factors) for ISO transmission constraints.

Mathematical Foundation
-----------------------
For any interval t with binding constraints k=1..K:

    MCC[node, t] = Σ_k SF[node, k] × ShadowPrice[k, t]

To solve for SF[node, k] we need many intervals T where the SAME set of K constraints
bind simultaneously (a "constraint regime"). We then build:

    A  (T × K):  each row is the shadow price vector for interval t
    b  (T × 1):  MCC values for a single node across those intervals
    x  (K × 1):  the unknown shift factors SF[node, constraint_1..K]

And solve:  A @ x ≈ b  via ordinary least-squares (numpy.linalg.lstsq).

This requires T ≥ K (more intervals than binding constraints).  Regimes with
insufficient data are flagged rather than solved.

Single-constraint case
----------------------
When K=1, the solution simplifies to the well-known identity:
    SF[node, constraint] = MCC[node] / ShadowPrice[constraint]
averaged across all binding intervals for robustness.

Reference
---------
See also: shift_factor_regimes.py for regime grouping logic.
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
    """Result of solving shift factors for one constraint regime."""
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

    def flatten(self) -> None:
        """Populate the flat (node, constraint) lookup from all solved regimes."""
        self._flat.clear()
        for regime in self.regimes.values():
            if regime.error or regime.underdetermined:
                continue
            for node, sfs in regime.shift_factors.items():
                for constraint, sf in sfs.items():
                    key = (node, constraint)
                    if key not in self._flat:
                        self._flat[key] = sf
                    else:
                        # Average across regimes that share a constraint
                        self._flat[key] = (self._flat[key] + sf) / 2.0

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

        Formula (SPP/MISO convention, note the negative sign):

            MCC_node = -Σ_c (SF_node,c × ShadowPrice_c)

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
            Estimated MCC in $/MWh.  Positive → node is import-constrained
            (LMP > system lambda).  Negative → node is export-constrained.
        """
        mcc = 0.0
        for constraint, sp in shadow_prices.items():
            sf = self.get(node, constraint)
            if sf is not None:
                mcc += sf * sp
        return -mcc  # sign convention: MCC = -Σ(SF × SP)

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
        Minimum number of intervals required in a regime to attempt solving.
        Must be > number of binding constraints for the system to be
        over-determined (preferred) or exactly determined.
    tolerance : float
        Shadow prices with |value| below this threshold are treated as
        non-binding for regime classification.
    validate_bounds : bool
        If True, shift factors outside [-1, 1] are flagged (logged) but not
        discarded — they may legitimately exceed unity in some network configs.
    """

    def __init__(
        self,
        min_intervals: int = 10,
        tolerance: float = 1e-4,
        validate_bounds: bool = True,
    ) -> None:
        self.min_intervals = min_intervals
        self.tolerance = tolerance
        self.validate_bounds = validate_bounds

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
        ].copy()

        solution = ShiftFactorSolution()

        # Group intervals by constraint regime
        regimes = self._build_regimes(binding)
        logger.info(f"Found {len(regimes)} distinct constraint regimes.")

        for regime_key, regime_intervals in regimes.items():
            sol = self._solve_regime(
                regime_key=regime_key,
                regime_intervals=regime_intervals,
                binding_df=binding,
                lmp_df=lmp_df,
            )
            solution.regimes[regime_key] = sol

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

    def _build_regimes(
        self, binding_df: pd.DataFrame
    ) -> Dict[FrozenSet[str], List]:
        """
        Group intervals by their binding constraint set.

        Returns
        -------
        dict mapping frozenset(constraint_names) -> list of interval timestamps
        """
        regimes: Dict[FrozenSet[str], List] = {}

        for interval, grp in binding_df.groupby("interval"):
            key = frozenset(grp["constraint_name"].unique())
            if key not in regimes:
                regimes[key] = []
            regimes[key].append(interval)

        return regimes

    def _solve_regime(
        self,
        regime_key: FrozenSet[str],
        regime_intervals: List,
        binding_df: pd.DataFrame,
        lmp_df: pd.DataFrame,
    ) -> RegimeSolution:
        """Solve shift factors for a single constraint regime."""
        constraints = sorted(regime_key)
        K = len(constraints)
        T = len(regime_intervals)

        sol = RegimeSolution(
            constraints=regime_key,
            n_intervals=T,
            n_constraints=K,
        )

        # Check for sufficient data
        if T < self.min_intervals:
            sol.underdetermined = True
            sol.error = (
                f"Only {T} intervals available; need at least "
                f"{self.min_intervals} (min_intervals). Skipping."
            )
            logger.debug(
                f"Regime {constraints}: {sol.error}"
            )
            return sol

        if T < K:
            sol.underdetermined = True
            sol.error = (
                f"Under-determined: {T} intervals < {K} constraints. "
                "Cannot solve uniquely."
            )
            logger.warning(f"Regime {constraints}: {sol.error}")
            return sol

        # Build shadow price matrix A (T × K)
        A = self._build_shadow_price_matrix(
            binding_df=binding_df,
            intervals=regime_intervals,
            constraints=constraints,
        )

        if A is None:
            sol.error = "Failed to build shadow price matrix — missing data."
            return sol

        # Get all nodes that appear in these intervals
        interval_lmps = lmp_df[lmp_df["interval"].isin(regime_intervals)]
        nodes = interval_lmps["settlement_location"].unique()

        logger.info(
            f"Solving regime [{', '.join(constraints[:3])}{'...' if K > 3 else ''}] "
            f"T={T} intervals, K={K} constraints, N={len(nodes)} nodes"
        )

        # Solve for each node
        for node in nodes:
            node_lmps = interval_lmps[
                interval_lmps["settlement_location"] == node
            ].set_index("interval")

            # Build b vector (T × 1), aligned to regime_intervals order
            b = np.array([
                node_lmps.loc[iv, "mcc"] if iv in node_lmps.index else np.nan
                for iv in regime_intervals
            ], dtype=float)

            # Drop rows where MCC is missing
            valid_mask = ~np.isnan(b)
            if valid_mask.sum() < K:
                logger.debug(
                    f"Node {node}: insufficient MCC data ({valid_mask.sum()} rows). Skipping."
                )
                continue

            A_valid = A[valid_mask]
            b_valid = b[valid_mask]

            if K == 1:
                # Fast path: simple ratio averaged across intervals
                shadow_prices = A_valid[:, 0]
                nonzero = np.abs(shadow_prices) > self.tolerance
                if nonzero.sum() == 0:
                    continue
                sfs = b_valid[nonzero] / shadow_prices[nonzero]
                x = np.array([np.mean(sfs)])
            else:
                # Least-squares solution
                x, residuals, rank, sv = np.linalg.lstsq(
                    A_valid, b_valid, rcond=None
                )
                if rank < K:
                    logger.debug(
                        f"Node {node}: rank-deficient matrix (rank={rank} < K={K}). "
                        "Shift factors may be unreliable."
                    )

            # Compute R² and residual std
            b_hat = A_valid @ x
            ss_res = np.sum((b_valid - b_hat) ** 2)
            ss_tot = np.sum((b_valid - np.mean(b_valid)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
            residual_std = np.sqrt(ss_res / max(1, len(b_valid) - K))

            # Validate shift factor bounds
            if self.validate_bounds and np.any(np.abs(x) > 1.0):
                out_of_bounds = [
                    f"{constraints[i]}={x[i]:.4f}"
                    for i in range(K)
                    if abs(x[i]) > 1.0
                ]
                logger.warning(
                    f"Node {node}: shift factor(s) outside [-1, 1]: "
                    f"{', '.join(out_of_bounds)}"
                )

            sol.shift_factors[node] = {
                constraints[i]: float(x[i]) for i in range(K)
            }
            sol.r2_scores[node] = float(r2)
            sol.residual_std[node] = float(residual_std)

        return sol

    def _build_shadow_price_matrix(
        self,
        binding_df: pd.DataFrame,
        intervals: List,
        constraints: List[str],
    ) -> Optional[np.ndarray]:
        """
        Build the T × K shadow price matrix A.

        Row t corresponds to interval ``intervals[t]``.
        Column k corresponds to constraint ``constraints[k]``.
        Missing shadow prices are filled with 0 (constraint not binding in that interval).
        """
        T = len(intervals)
        K = len(constraints)
        A = np.zeros((T, K), dtype=float)

        # Pivot for fast lookup
        try:
            pivot = (
                binding_df[binding_df["interval"].isin(intervals)]
                .pivot_table(
                    index="interval",
                    columns="constraint_name",
                    values="shadow_price",
                    aggfunc="first",
                )
                .reindex(index=intervals, columns=constraints, fill_value=0.0)
            )
            A = pivot.values.astype(float)
        except Exception as e:
            logger.error(f"Failed to build shadow price matrix: {e}")
            return None

        return A


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
        Minimum intervals per regime to attempt solving.
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
        help="Minimum intervals per regime (default: 10)",
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