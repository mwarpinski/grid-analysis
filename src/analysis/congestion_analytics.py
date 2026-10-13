"""
Congestion analytics built on the pooled shift factors.

Pure functions on DataFrames (no I/O) so each output can be unit-tested against a
synthetic case with a known answer, and called from the API or from scripts.

Conventions
-----------
* SPP labels every interval by its END time, so an hourly bucket is hour-ENDING:
  RT intervals 00:05 .. 01:00 belong to hour-ending 01 (label 01:00), matching the
  Day-Ahead label 01:00.  ``to_hour_ending`` = ``ceil("h")``.
* Shift factors follow the fit convention ``MCC[node] = Σ_c SF[node, c] × SP[c]`` (SPP
  publishes binding shadow prices as negative numbers).  The SF is read as a PTDF
  (MW of flow on constraint c per MW injected at the node, against the system
  reference) *only if* SPP's published shadow price is −μ.  That sign link is
  consistent with the data but is not independently verified, so flow-direction
  outputs (``trip_scenario``) carry that caveat.
* DART spread = RT − DA (positive: RT congestion cost was higher than DA priced).

Outputs
-------
nodal_dart / dart_summary          per-node, per-hour DA vs RT congestion spreads
decompose_dart                     which constraints explain each spread (SF × ΔSP)
constraint_stats                   binding frequency, duration, DA-vs-RT surprises
best_pairs                         node pairs with the largest sensitivity to a constraint
trip_scenario                      flow/stress on every constraint if injection changes
price_impact                       nodal MCC change if given constraints bind at given prices
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

BINDING_TOLERANCE = 1e-4
INTERVAL_MINUTES = 5


# ---------------------------------------------------------------------------
# Time alignment
# ---------------------------------------------------------------------------

def to_hour_ending(interval: pd.Series) -> pd.Series:
    """Map SPP interval-END labels to their hour-ending label (00:05..01:00 -> 01:00)."""
    return pd.to_datetime(interval).dt.ceil("h")


# ---------------------------------------------------------------------------
# Nodal DART
# ---------------------------------------------------------------------------

def rt_hourly_mcc(rt_lmp: pd.DataFrame, min_intervals: int = 10) -> pd.DataFrame:
    """Hourly mean of the 5-minute RT MCC per node; hours with too few intervals are dropped."""
    df = rt_lmp[["interval", "settlement_location", "mcc"]].copy()
    df["hour"] = to_hour_ending(df["interval"])
    g = (
        df.groupby(["hour", "settlement_location"])["mcc"]
        .agg(rt_mcc="mean", n_intervals="count")
        .reset_index()
    )
    return g[g["n_intervals"] >= min_intervals].drop(columns="n_intervals")


def nodal_dart(
    da_lmp: pd.DataFrame,
    rt_lmp: pd.DataFrame,
    min_intervals: int = 10,
) -> pd.DataFrame:
    """
    Per (hour, node): DA MCC, hourly-mean RT MCC and ``spread = rt_mcc − da_mcc``.

    Only hours present on both sides, with at least ``min_intervals`` of 12 RT
    intervals, are returned.
    """
    da = da_lmp.rename(columns={"interval": "hour", "mcc": "da_mcc"})[
        ["hour", "settlement_location", "da_mcc"]
    ].copy()
    da["hour"] = pd.to_datetime(da["hour"])
    out = da.merge(
        rt_hourly_mcc(rt_lmp, min_intervals), on=["hour", "settlement_location"], how="inner"
    )
    out["spread"] = out["rt_mcc"] - out["da_mcc"]
    return out


def dart_summary(dart: pd.DataFrame, min_hours: int = 24) -> pd.DataFrame:
    """
    Per-node distribution of the DART spread.

    ``naive_t_stat`` treats hours as independent.  Consecutive hours are strongly
    correlated, so it overstates significance; use it to rank, not to conclude.
    """
    g = dart.groupby("settlement_location")["spread"]
    s = g.agg(
        n_hours="count",
        mean="mean",
        std="std",
        median="median",
        p05=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
        hit_rate=lambda x: float((x > 0).mean()),
    )
    s["naive_t_stat"] = s["mean"] / (s["std"] / np.sqrt(s["n_hours"]))
    s = s[s["n_hours"] >= min_hours]
    return s.sort_values("naive_t_stat", key=lambda x: x.abs(), ascending=False)


# ---------------------------------------------------------------------------
# Hourly shadow prices (RT & DA) and DART decomposition
# ---------------------------------------------------------------------------

def hourly_shadow_prices_rt(rt_bc: pd.DataFrame, tolerance: float = BINDING_TOLERANCE) -> pd.DataFrame:
    """
    Hour × constraint matrix of the mean RT shadow price over the hour's intervals
    (0 when not binding).  The denominator is the number of RT intervals actually
    present in that hour, so a missing interval does not count as a zero.
    """
    df = rt_bc[["interval", "constraint_name", "shadow_price"]].copy()
    df["interval"] = pd.to_datetime(df["interval"])
    df["hour"] = to_hour_ending(df["interval"])
    n_int = df.groupby("hour")["interval"].nunique()
    priced = df[df["shadow_price"].abs() > tolerance]
    tot = priced.groupby(["hour", "constraint_name"])["shadow_price"].sum().unstack(fill_value=0.0)
    tot = tot.reindex(n_int.index, fill_value=0.0)
    return tot.div(n_int, axis=0)


def hourly_shadow_prices_da(da_bc: pd.DataFrame, tolerance: float = BINDING_TOLERANCE) -> pd.DataFrame:
    """Hour × constraint matrix of DA shadow prices (0 when not binding)."""
    df = da_bc[["interval", "constraint_name", "shadow_price"]].copy()
    df["hour"] = pd.to_datetime(df["interval"])
    priced = df[df["shadow_price"].abs() > tolerance]
    return priced.groupby(["hour", "constraint_name"])["shadow_price"].mean().unstack(fill_value=0.0)


def _sf_matrix(sf_flat: Mapping[Tuple[str, str], float]) -> pd.DataFrame:
    """node × constraint matrix from the flat {(node, constraint): sf} lookup."""
    if not sf_flat:
        return pd.DataFrame()
    s = pd.Series(sf_flat)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["node", "constraint"])
    return s.unstack("constraint")


def decompose_dart(
    dart: pd.DataFrame,
    sf_flat: Mapping[Tuple[str, str], float],
    rt_sp_hourly: pd.DataFrame,
    da_sp_hourly: pd.DataFrame,
) -> Dict[str, object]:
    """
    Explain each node's DART spread by constraint:

        explained[h, n] = Σ_c SF[n, c] × (SP_RT[h, c] − SP_DA[h, c])

    Constraints with no shift factor cannot be attributed and land in the residual,
    so ``unattributed_share`` reports how much of the shadow-price move they carry.

    Returns a dict with
      ``explained``  hour × node DataFrame
      ``actual``     hour × node DataFrame (the DART spread)
      ``summary``    per node: n_hours, r2 (1 − SSE/SST of explained vs actual), mean |spread|
      ``dsp``        hour × constraint DataFrame of RT − DA shadow prices (modeled constraints)
      ``sf``         the node × constraint matrix used
      ``unattributed_share``  Σ|ΔSP| on constraints without shift factors / Σ|ΔSP| overall
    """
    sf = _sf_matrix(sf_flat)
    actual = dart.pivot_table(index="hour", columns="settlement_location", values="spread", aggfunc="first")
    hours = actual.index
    all_cons = sorted(set(rt_sp_hourly.columns) | set(da_sp_hourly.columns))
    dsp_all = (
        rt_sp_hourly.reindex(index=hours, columns=all_cons).fillna(0.0)
        - da_sp_hourly.reindex(index=hours, columns=all_cons).fillna(0.0)
    )
    modeled = [c for c in all_cons if c in sf.columns]
    total_mag = float(dsp_all.abs().to_numpy().sum())
    unattr = float(dsp_all.drop(columns=modeled).abs().to_numpy().sum()) / total_mag if total_mag else 0.0

    dsp = dsp_all[modeled]
    nodes = [n for n in actual.columns if n in sf.index]
    X = sf.loc[nodes, modeled].fillna(0.0)
    explained = pd.DataFrame(dsp.to_numpy() @ X.to_numpy().T, index=hours, columns=nodes)
    act = actual[nodes]

    err = act - explained
    sse = (err ** 2).sum()
    sst = ((act - act.mean()) ** 2).sum()
    summary = pd.DataFrame({
        "n_hours": act.count(),
        "mean_spread": act.mean(),
        "mean_explained": explained.mean(),
        "mean_residual": err.mean(),         # persistent offset the modeled constraints do not explain
        "mean_abs_spread": act.abs().mean(),
        "mean_abs_residual": err.abs().mean(),
        # Centered R2: strongly negative when there is a large unexplained mean offset
        "r2": 1.0 - sse / sst.replace(0, np.nan),
    })
    return {
        "explained": explained, "actual": act, "summary": summary,
        "dsp": dsp, "sf": X, "unattributed_share": unattr,
    }


def node_decomposition(decomp: Dict[str, object], node: str, top: int = 10) -> pd.DataFrame:
    """Per-constraint contribution to one node's DART spread, ranked by total |contribution|."""
    X: pd.DataFrame = decomp["sf"]            # node × constraint
    dsp: pd.DataFrame = decomp["dsp"]         # hour × constraint
    if node not in X.index:
        return pd.DataFrame()
    contrib = dsp.mul(X.loc[node], axis=1)    # hour × constraint
    out = pd.DataFrame({
        "sf": X.loc[node],
        "mean_contribution": contrib.mean(),
        "mean_abs_contribution": contrib.abs().mean(),
        "hours_active": (contrib.abs() > 1e-9).sum(),
    })
    return out.sort_values("mean_abs_contribution", ascending=False).head(top)


def fit_shift_factors_rt_da(
    rt_bc: pd.DataFrame,
    rt_lmp: pd.DataFrame,
    da_bc: pd.DataFrame,
    da_lmp: pd.DataFrame,
    min_intervals: int = 10,
    **kwargs,
):
    """
    Fit shift factors on RT 5-minute data AND DA hourly data together.

    Shift factors are fit from the RT market, so a constraint that only the Day-Ahead market ever
    priced has none, and its part of a DART spread cannot be attributed.  Pooling DA observations
    (one per hour) gives those constraints shift factors too.  DA observations are given a
    1-second offset so they cannot collide with the RT interval stamped on the same minute.

    Assumes the DA and RT network models are close enough to share shift factors; check
    ``RT-fit SFs reproduce DA MCC`` in scripts/sanity_report.py (R2 ~0.7 on the first 14 days).
    """
    from src.analysis.multi_constraint_sf import calculate_shift_factors

    off = pd.Timedelta(seconds=1)
    da_bc_ = da_bc[["interval", "constraint_name", "shadow_price"]].copy()
    da_lmp_ = da_lmp[["interval", "settlement_location", "mcc"]].copy()
    da_bc_["interval"] = pd.to_datetime(da_bc_["interval"]) + off
    da_lmp_["interval"] = pd.to_datetime(da_lmp_["interval"]) + off
    bc = pd.concat([rt_bc[["interval", "constraint_name", "shadow_price"]], da_bc_], ignore_index=True)
    lmp = pd.concat([rt_lmp[["interval", "settlement_location", "mcc"]], da_lmp_], ignore_index=True)
    return calculate_shift_factors(lmp, bc, min_intervals=min_intervals, **kwargs)


def sibling_prior(
    sf_flat: Mapping[Tuple[str, str], float],
    targets: Iterable[str],
    name_key: Mapping[str, str],
    orientation: Mapping[str, int],
    shrink: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    """
    Provisional shift factors for constraints that have none, borrowed from their SIBLINGS: other
    constraints on the same physical element (a different contingency, or the other market's name).

    Same element under another contingency correlates with ~0.75 across nodes (aliases ~0.9), so a
    sibling is a far better guess than nothing for a constraint that has just started binding.  Each
    sibling's vector is oriented to a common direction, averaged, then oriented back to the target's
    own direction and multiplied by ``shrink`` (< 1 pulls the guess toward zero, since the magnitude
    can differ by contingency).  Returns ``{(node, target): sf}`` for targets with at least one
    sibling; the caller decides whether to use it (see scripts/identity_report.py for the measured
    benefit).
    """
    sf = _sf_matrix(sf_flat)
    if sf.empty:
        return {}
    element = {n: k.split("#")[0] for n, k in name_key.items()}
    by_element: Dict[str, List[str]] = {}
    for c in sf.columns:
        if c in element:
            by_element.setdefault(element[c], []).append(c)
    out: Dict[Tuple[str, str], float] = {}
    for t in targets:
        if t in sf.columns or t not in element:
            continue
        sibs = [c for c in by_element.get(element[t], []) if c != t]
        if not sibs:
            continue
        oriented = sf[sibs].mul(pd.Series({c: orientation.get(c, 1) for c in sibs}), axis=1)
        guess = oriented.mean(axis=1) * orientation.get(t, 1) * shrink
        for node, v in guess.dropna().items():
            out[(node, t)] = float(v)
    return out


# ---------------------------------------------------------------------------
# Constraint behaviour
# ---------------------------------------------------------------------------

def _run_lengths(intervals: pd.Series) -> List[int]:
    """Lengths (in intervals) of consecutive runs of 5-minute intervals."""
    iv = np.sort(pd.to_datetime(intervals).unique())
    if len(iv) == 0:
        return []
    gaps = np.diff(iv).astype("timedelta64[m]").astype(int)
    breaks = np.flatnonzero(gaps > INTERVAL_MINUTES)
    edges = np.concatenate([[-1], breaks, [len(iv) - 1]])
    return list(np.diff(edges))


def constraint_stats(
    rt_bc: pd.DataFrame,
    da_bc: Optional[pd.DataFrame] = None,
    tolerance: float = BINDING_TOLERANCE,
    group_by: str = "constraint_name",
) -> pd.DataFrame:
    """
    Behaviour of every constraint that priced at least once in RT.

    Columns: RT/DA binding hours, ``rt_only_hours`` (RT priced, DA did not: the
    surprises), ``da_only_hours``, shadow-price quantiles when binding, mean run
    length of consecutive binding intervals, and ``bind_given_activated`` (share of
    the time the constraint was monitored that it actually priced).

    ``group_by="monitored_facility"`` groups all contingencies on the same physical element.
    Use it for DA-vs-RT surprise counts: DA and RT price different constraint sets (only
    ~53% of priced rows share a constraint name), and the same line is often priced under a
    different contingency in the other market.
    """
    if group_by != "constraint_name":
        rt_bc = rt_bc.assign(constraint_name=rt_bc[group_by].astype(str).str.strip())
        if da_bc is not None and not da_bc.empty:
            da_bc = da_bc.assign(constraint_name=da_bc[group_by].astype(str).str.strip())
    df = rt_bc.copy()
    df["interval"] = pd.to_datetime(df["interval"])
    df["hour"] = to_hour_ending(df["interval"])
    priced = df[df["shadow_price"].abs() > tolerance]

    rows = []
    rt_hours = priced.groupby("constraint_name")["hour"].agg(lambda h: set(h.unique()))
    da_hours: Dict[str, set] = {}
    if da_bc is not None and not da_bc.empty:
        d = da_bc[da_bc["shadow_price"].abs() > tolerance]
        da_hours = d.groupby("constraint_name")["interval"].agg(lambda h: set(pd.to_datetime(h).unique())).to_dict()
    da_window = None
    if da_bc is not None and not da_bc.empty:
        da_window = (pd.to_datetime(da_bc["interval"]).min(), pd.to_datetime(da_bc["interval"]).max())

    monitored = df[~(df["shadow_price"].abs() > tolerance)]
    activated = monitored.groupby("constraint_name")["interval"].nunique()

    for name, g in priced.groupby("constraint_name"):
        sp = g["shadow_price"].abs()
        runs = _run_lengths(g["interval"])
        n_bind = g["interval"].nunique()
        n_act = int(activated.get(name, 0))
        rt_h = rt_hours[name]
        if da_window is not None:
            rt_h_cmp = {h for h in rt_h if da_window[0] <= h <= da_window[1]}   # only hours DA covers
        else:
            rt_h_cmp = rt_h
        da_h = da_hours.get(name, set())
        rows.append({
            group_by if group_by != "constraint_name" else "constraint_name": name,
            "rt_binding_intervals": n_bind,
            "rt_binding_hours": len(rt_h),
            "days_bound": g["interval"].dt.normalize().nunique(),
            "mean_abs_sp": float(sp.mean()),
            "p50_abs_sp": float(sp.quantile(0.5)),
            "p90_abs_sp": float(sp.quantile(0.9)),
            "mean_run_intervals": float(np.mean(runs)) if runs else 0.0,
            "max_run_intervals": int(max(runs)) if runs else 0,
            "bind_given_activated": n_bind / (n_bind + n_act) if (n_bind + n_act) else np.nan,
            "da_binding_hours": len(da_h) if da_window is not None else np.nan,
            "rt_only_hours": len(rt_h_cmp - da_h) if da_window is not None else np.nan,
            "da_only_hours": len(da_h - rt_h_cmp) if da_window is not None else np.nan,
            "first_seen": g["interval"].min(),
            "last_seen": g["interval"].max(),
        })
    return pd.DataFrame(rows).sort_values("rt_binding_intervals", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------

def best_pairs(
    sf_flat: Mapping[Tuple[str, str], float],
    constraint: str,
    top: int = 10,
    sf_alt: Optional[Mapping[Tuple[str, str], float]] = None,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Node pairs with the largest sensitivity to ``constraint``.

    When the constraint binds, ``MCC[a] − MCC[b] = (SF[a] − SF[b]) × SP``.  Because binding
    shadow prices are negative, the node with the LOWEST SF gets the highest MCC.  Each row is
    (``high_node``, ``low_node``): the pair with the widest expected MCC gap per $ of |SP|.

    Nodes with (rounded) identical SFs are collapsed so the list isn't 50 copies of one pair.
    If ``sf_alt`` (a second, independent fit) is given, ``delta_alt`` and ``same_sign`` show
    how stable each pair's spread is across fits.
    """
    sf = _sf_matrix(sf_flat)
    if constraint not in sf.columns:
        return pd.DataFrame()
    x = sf[constraint].dropna()
    uniq = x.groupby(x.round(decimals)).head(1).sort_values()     # one node per distinct SF
    lows, highs = uniq.head(top), uniq.tail(top).iloc[::-1]       # lowest SF -> highest MCC
    n = min(len(lows), len(highs))
    rows = []
    alt = _sf_matrix(sf_alt)[constraint] if sf_alt and constraint in _sf_matrix(sf_alt).columns else None
    for i in range(n):
        hi_node, lo_node = lows.index[i], highs.index[i]
        if highs.iloc[i] - lows.iloc[i] <= 0:      # ran past the middle: no wider pair left
            break
        row = {
            "high_mcc_node": hi_node, "low_mcc_node": lo_node,
            "sf_high_mcc_node": float(lows.iloc[i]), "sf_low_mcc_node": float(highs.iloc[i]),
            "delta_sf": float(highs.iloc[i] - lows.iloc[i]),
        }
        if alt is not None and hi_node in alt.index and lo_node in alt.index:
            d_alt = float(alt[lo_node] - alt[hi_node])
            row["delta_alt"] = d_alt
            row["same_sign"] = bool(np.sign(d_alt) == np.sign(row["delta_sf"]))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def trip_scenario(
    sf_flat: Mapping[Tuple[str, str], float],
    node: str,
    injection_change_mw: float,
    limits: Optional[Mapping[str, float]] = None,
    top: int = 15,
) -> pd.DataFrame:
    """
    Flow change on every modeled constraint if injection at ``node`` changes by
    ``injection_change_mw`` (negative for a generator trip, positive for a load trip).

        Δflow[c] = SF[node, c] × injection_change_mw

    This is only valid for generator/load changes; a LINE outage redistributes flow
    (needs line-outage distribution factors from a network model) and is not modeled.
    The SF is read as a PTDF, which carries the sign caveat in the module docstring.
    ``delta_pct_of_limit`` uses ``limits`` (MW) when given; it shows how much of a
    constraint's limit the change consumes, not the remaining headroom (unknown).
    """
    sf = _sf_matrix(sf_flat)
    if node not in sf.index:
        return pd.DataFrame()
    x = sf.loc[node].dropna()
    out = pd.DataFrame({"sf": x, "delta_flow_mw": x * injection_change_mw})
    if limits is not None:
        out["limit_mw"] = pd.Series({c: limits.get(c, np.nan) for c in out.index})
        out["delta_pct_of_limit"] = 100.0 * out["delta_flow_mw"].abs() / out["limit_mw"].where(out["limit_mw"] > 0)
    key = "delta_pct_of_limit" if limits is not None and out["delta_pct_of_limit"].notna().any() else "delta_flow_mw"
    out = out.reindex(out[key].abs().sort_values(ascending=False).index)
    return out.head(top).rename_axis("constraint").reset_index()


def price_impact(
    sf_flat: Mapping[Tuple[str, str], float],
    shadow_prices: Mapping[str, float],
    nodes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Nodal MCC if the given constraints bind at the given shadow prices
    (``MCC[n] = Σ_c SF[n, c] × SP[c]``).  Use negative shadow prices, as SPP publishes them.
    Constraints without shift factors are reported in ``attrs['unmodeled']``.
    """
    sf = _sf_matrix(sf_flat)
    modeled = {c: sp for c, sp in shadow_prices.items() if c in sf.columns}
    out = pd.DataFrame(index=sf.index if nodes is None else [n for n in nodes if n in sf.index])
    if modeled:
        contrib = sf.loc[out.index, list(modeled)].fillna(0.0) * pd.Series(modeled)
        out["mcc_change"] = contrib.sum(axis=1)
        out["top_constraint"] = contrib.abs().idxmax(axis=1)
    else:
        out["mcc_change"] = 0.0
        out["top_constraint"] = None
    out.attrs["unmodeled"] = sorted(set(shadow_prices) - set(modeled))
    return out.sort_values("mcc_change", key=lambda s: s.abs(), ascending=False).rename_axis("node").reset_index()


def historical_shadow_price(rt_bc: pd.DataFrame, constraints: Iterable[str], quantile: float = 0.5) -> Dict[str, float]:
    """Typical shadow price (a quantile of the signed price while binding) per constraint, for scenario inputs."""
    d = rt_bc[rt_bc["constraint_name"].isin(list(constraints)) & (rt_bc["shadow_price"].abs() > BINDING_TOLERANCE)]
    # Negative prices: the "high" quantile of severity is the LOW quantile of the signed value.
    return d.groupby("constraint_name")["shadow_price"].quantile(1.0 - quantile).to_dict()


def constraint_limits(rt_bc: pd.DataFrame) -> Dict[str, float]:
    """
    Static rating (MW) per constraint: the median ``source_limit_mw`` (falling back to
    ``initial_effective_limit_mw``).

    ``rt_effective_limit_mw`` is deliberately NOT used: it is the optimizer's adjusted limit,
    which is zero or negative on ~15% of priced rows (e.g. -42.6 MW on a 450 MW constraint)
    and would make any "percent of limit" figure meaningless.
    """
    col = next((c for c in ("source_limit_mw", "initial_effective_limit_mw") if c in rt_bc.columns), None)
    if col is None:
        return {}
    d = rt_bc[pd.to_numeric(rt_bc[col], errors="coerce") > 0]
    return d.groupby("constraint_name")[col].median().to_dict()
