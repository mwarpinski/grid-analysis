"""
Shift Factor Validation Router — /api/v1/validation/*

Three endpoints that answer the core question:
"Are our statistically-derived shift factors actually correct?"

Validation strategy
-------------------
1. MCC Back-Prediction Test (primary)
   Ground truth: SPP publishes MCC per node (from LMP files) AND shadow prices
   per constraint (from BC files). We already scrape both every 5 minutes.

   Identity:  MCC_node = Σᵢ ( SF[node,i] × ShadowPrice[i] )

   We reconstruct MCC using our SF estimates and compare to SPP's published
   values. If our SFs are correct the reconstruction error should be small
   (< $0.10/MWh for most nodes).

2. Regime Consistency Check
   Within a stable constraint regime the shadow price / MCC relationship is
   fixed by network topology — SFs should be constant. We measure how much
   our per-interval ratio estimates scatter around the regime mean.

3. Physical Bounds Check
   Shift factors must satisfy |SF| ≤ 1 by network physics (they represent
   the fraction of an injection that appears on a monitored element).
   Values outside this range indicate numerical issues or data quality problems.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.sf_cache import cache
from src.storage.duckdb_store import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/validation", tags=["validation"])


# ---------------------------------------------------------------------------
# Helper: pull and join historical data
# ---------------------------------------------------------------------------

def _load_history(days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull the last `days` days of constraint + LMP history from DuckDB.
    Returns (constraints_df, lmp_df). Either may be empty if not yet available.
    """
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days)

    constraints_df = store.query_constraints(start=start, end=end)
    lmp_df         = store.query_lmp(start=start, end=end)

    # Ensure interval column is datetime
    for df in (constraints_df, lmp_df):
        if not df.empty and "interval" in df.columns:
            df["interval"] = pd.to_datetime(df["interval"])

    return constraints_df, lmp_df


# ---------------------------------------------------------------------------
# 1. MCC Back-Prediction
# ---------------------------------------------------------------------------

@router.get("/mcc-backtest")
async def mcc_backtest(
    days: int = Query(default=7, ge=1, le=90, description="History window in days"),
    min_shadow_price: float = Query(
        default=1.0,
        description="Ignore intervals where all shadow prices are below this ($/MWh). "
                    "Near-zero SPs give noisy SF estimates.",
    ),
    top_n_worst: int = Query(
        default=20, ge=1, le=200,
        description="Number of worst-performing nodes to return in detail.",
    ),
):
    """
    **Primary validation test.**

    Uses SPP's published MCC values as ground truth.  For each interval in the
    history window, reconstructs MCC = Σ(SF × ShadowPrice) using our estimated
    shift factors and computes the error vs. the published MCC.

    Returns aggregate statistics (R², MAE, RMSE) and a per-node breakdown
    sorted by mean absolute error so you can quickly identify problem nodes.

    **Interpretation guide**

    | Metric | Good | Acceptable | Investigate |
    |---|---|---|---|
    | Overall R² | > 0.95 | 0.80–0.95 | < 0.80 |
    | Median node MAE | < $0.10/MWh | $0.10–$0.50 | > $0.50 |
    | % nodes flagged (MAE > $1) | < 5% | 5–15% | > 15% |
    """
    if cache.solution is None:
        raise HTTPException(
            status_code=503,
            detail="SF solution not yet available — wait for first ingest cycle.",
        )

    constraints_df, lmp_df = _load_history(days)

    if constraints_df.empty or lmp_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data available for the past {days} days. "
                   "The store accumulates data as the server runs.",
        )

    # Filter to intervals with meaningful shadow prices
    if "shadow_price" in constraints_df.columns and min_shadow_price > 0:
        loud_intervals = (
            constraints_df[constraints_df["shadow_price"].abs() >= min_shadow_price]
            ["interval"].unique()
        )
        constraints_df = constraints_df[constraints_df["interval"].isin(loud_intervals)]
        lmp_df         = lmp_df[lmp_df["interval"].isin(loud_intervals)]

    if constraints_df.empty:
        return {
            "status": "no_binding_constraints",
            "message": f"No intervals with |shadow_price| ≥ {min_shadow_price} in the past {days} days.",
            "days": days,
        }

    # Build the SF lookup from the current solution
    sf_lookup: dict[tuple[str, str], float] = cache.solution._flat  # (node, constraint) -> SF

    # Reconstruct MCC per (interval, node) and compare to observed
    # Build shadow price pivot: interval -> {constraint: shadow_price}
    sp_pivot = (
        constraints_df[constraints_df["shadow_price"].abs() > 1e-6]
        .groupby(["interval", "constraint_name"])["shadow_price"]
        .first()
        .unstack(fill_value=0.0)
    )

    records = []
    for interval, row in sp_pivot.iterrows():
        sp_map = row[row.abs() > 1e-6].to_dict()  # only binding constraints
        if not sp_map:
            continue

        interval_lmps = lmp_df[lmp_df["interval"] == interval]
        for _, lmp_row in interval_lmps.iterrows():
            node = lmp_row["settlement_location"]
            observed_mcc = lmp_row.get("mcc", np.nan)
            if pd.isna(observed_mcc):
                continue

            # Reconstruct: MCC = Σ(SF × SP)  [note: sign convention below]
            reconstructed = sum(
                sf_lookup.get((node, constraint), 0.0) * sp
                for constraint, sp in sp_map.items()
            )

            records.append({
                "interval":         interval,
                "node":             node,
                "observed_mcc":     float(observed_mcc),
                "reconstructed_mcc": float(reconstructed),
                "abs_error":        abs(float(observed_mcc) - float(reconstructed)),
            })

    if not records:
        return {
            "status": "insufficient_overlap",
            "message": "No intervals had both constraint and LMP data for the same timestamps.",
            "days": days,
            "n_constraint_intervals": len(sp_pivot),
            "n_lmp_rows": len(lmp_df),
        }

    df = pd.DataFrame(records)

    # ── Aggregate metrics ────────────────────────────────────────────────────
    obs  = df["observed_mcc"].values
    pred = df["reconstructed_mcc"].values
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    overall_r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
    overall_mae  = float(df["abs_error"].mean())
    overall_rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    pct_flagged  = float((df["abs_error"] > 1.0).mean() * 100)  # nodes with >$1 error

    # ── Per-node summary ─────────────────────────────────────────────────────
    node_stats = (
        df.groupby("node")
        .agg(
            n_intervals     = ("interval", "nunique"),
            mean_abs_error  = ("abs_error", "mean"),
            max_abs_error   = ("abs_error", "max"),
            mean_observed   = ("observed_mcc", "mean"),
            mean_predicted  = ("reconstructed_mcc", "mean"),
        )
        .reset_index()
        .sort_values("mean_abs_error", ascending=False)
    )
    node_stats["bias"] = node_stats["mean_predicted"] - node_stats["mean_observed"]
    node_stats = node_stats.round(4)

    # ── Per-interval R² ──────────────────────────────────────────────────────
    def _interval_r2(grp):
        o, p = grp["observed_mcc"].values, grp["reconstructed_mcc"].values
        ss_r = np.sum((o - p) ** 2)
        ss_t = np.sum((o - np.mean(o)) ** 2)
        return 1.0 - ss_r / ss_t if ss_t > 1e-12 else 1.0

    interval_r2 = (
        df.groupby("interval")
        .apply(_interval_r2)
        .reset_index(name="r2")
        .sort_values("r2")
    )

    # Quality verdict
    if overall_r2 >= 0.95 and overall_mae < 0.10:
        verdict = "EXCELLENT"
    elif overall_r2 >= 0.80 and overall_mae < 0.50:
        verdict = "ACCEPTABLE"
    else:
        verdict = "INVESTIGATE"

    return {
        "status":          "ok",
        "verdict":         verdict,
        "days":            days,
        "min_shadow_price": min_shadow_price,
        "n_intervals":     int(df["interval"].nunique()),
        "n_nodes":         int(df["node"].nunique()),
        "n_observations":  len(df),
        "aggregate": {
            "r2":                round(overall_r2, 4),
            "mae_per_mwh":       round(overall_mae, 4),
            "rmse_per_mwh":      round(overall_rmse, 4),
            "pct_nodes_flagged": round(pct_flagged, 2),
        },
        "interval_r2": {
            "p10":  round(float(interval_r2["r2"].quantile(0.10)), 4),
            "p25":  round(float(interval_r2["r2"].quantile(0.25)), 4),
            "p50":  round(float(interval_r2["r2"].quantile(0.50)), 4),
            "p75":  round(float(interval_r2["r2"].quantile(0.75)), 4),
            "worst_5": interval_r2.head(5).to_dict(orient="records"),
        },
        "worst_nodes": node_stats.head(top_n_worst).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 2. Regime Consistency
# ---------------------------------------------------------------------------

@router.get("/regime-consistency")
async def regime_consistency():
    """
    **Regime stability check.**

    Within each binding constraint regime (fixed set of binding constraints),
    shift factors should be constant — they are determined by network topology,
    not by load level.  This endpoint measures how much our per-interval ratio
    estimates scatter around the regime mean.

    A regime with high coefficient of variation (CV > 0.3) suggests the
    topology changed mid-regime or the data is noisy.
    """
    if cache.solution is None:
        raise HTTPException(status_code=503, detail="SF solution not yet available.")

    from src.analysis.multi_constraint_sf import summarize_solution

    summary_df = summarize_solution(cache.solution)

    regimes = []
    for _, sol in cache.solution.regimes.items():
        if sol.error or sol.underdetermined or not sol.r2_scores:
            continue

        r2_vals = list(sol.r2_scores.values())
        res_vals = list(sol.residual_std.values())

        regimes.append({
            "constraints":       sorted(sol.constraints),
            "n_constraints":     sol.n_constraints,
            "n_intervals":       sol.n_intervals,
            "n_nodes_solved":    len(sol.shift_factors),
            "r2": {
                "mean":   round(float(np.mean(r2_vals)), 4),
                "p10":    round(float(np.percentile(r2_vals, 10)), 4),
                "min":    round(float(np.min(r2_vals)), 4),
            },
            "residual_std": {
                "mean":   round(float(np.mean(res_vals)), 4),
                "max":    round(float(np.max(res_vals)), 4),
            },
            "verdict": (
                "STABLE"       if np.mean(r2_vals) >= 0.90 else
                "MARGINAL"     if np.mean(r2_vals) >= 0.70 else
                "UNSTABLE"
            ),
        })

    regimes.sort(key=lambda x: x["n_intervals"], reverse=True)

    n_stable   = sum(1 for r in regimes if r["verdict"] == "STABLE")
    n_marginal = sum(1 for r in regimes if r["verdict"] == "MARGINAL")
    n_unstable = sum(1 for r in regimes if r["verdict"] == "UNSTABLE")

    return {
        "n_regimes_solved": len(regimes),
        "summary": {
            "stable":   n_stable,
            "marginal": n_marginal,
            "unstable": n_unstable,
        },
        "regimes": regimes,
    }


# ---------------------------------------------------------------------------
# 3. Physical Bounds Check
# ---------------------------------------------------------------------------

@router.get("/bounds-check")
async def bounds_check():
    """
    **Physical plausibility check.**

    Shift factors represent the fraction of an injected MW that flows across
    a monitored transmission element. By network physics, |SF| ≤ 1.0.

    Values outside this range are not physically meaningful and indicate:
    - Data quality issues (bad shadow prices / MCC values in source files)
    - Numerical instability (rank-deficient regime matrix)
    - Constraint misidentification / name mismatch

    This endpoint flags all such violations sorted by magnitude.
    """
    if cache.solution is None:
        raise HTTPException(status_code=503, detail="SF solution not yet available.")

    violations = []
    within_bounds = 0

    for (node, constraint), sf in cache.solution._flat.items():
        if abs(sf) > 1.0:
            violations.append({
                "node":       node,
                "constraint": constraint,
                "sf_value":   round(float(sf), 6),
                "abs_value":  round(abs(float(sf)), 6),
            })
        else:
            within_bounds += 1

    violations.sort(key=lambda x: x["abs_value"], reverse=True)

    total = len(cache.solution._flat)
    pct_violating = round(len(violations) / total * 100, 2) if total > 0 else 0.0

    # Distribution of all SF values
    all_sf = list(cache.solution._flat.values())
    dist = {
        "min":  round(float(np.min(all_sf)), 4),
        "p5":   round(float(np.percentile(all_sf, 5)), 4),
        "p25":  round(float(np.percentile(all_sf, 25)), 4),
        "p50":  round(float(np.percentile(all_sf, 50)), 4),
        "p75":  round(float(np.percentile(all_sf, 75)), 4),
        "p95":  round(float(np.percentile(all_sf, 95)), 4),
        "max":  round(float(np.max(all_sf)), 4),
    } if all_sf else {}

    verdict = (
        "CLEAN"       if pct_violating == 0 else
        "MINOR"       if pct_violating < 2  else
        "SIGNIFICANT" if pct_violating < 10 else
        "SEVERE"
    )

    return {
        "verdict":           verdict,
        "total_sf_pairs":    total,
        "within_bounds":     within_bounds,
        "violations":        len(violations),
        "pct_violating":     pct_violating,
        "sf_distribution":   dist,
        "violation_details": violations[:50],  # cap response size
    }


# ---------------------------------------------------------------------------
# 4. Summary (all three combined)
# ---------------------------------------------------------------------------

@router.get("/summary")
async def validation_summary(
    days: int = Query(default=7, ge=1, le=90),
):
    """
    **One-stop validation dashboard.**

    Runs all three checks and returns a combined summary.  Use this to quickly
    assess overall SF quality without calling each endpoint individually.
    """
    if cache.solution is None:
        raise HTTPException(status_code=503, detail="SF solution not yet available.")

    # Bounds check (always fast — no DB query)
    total = len(cache.solution._flat)
    n_violations = sum(1 for sf in cache.solution._flat.values() if abs(sf) > 1.0)

    # Regime consistency (no DB query)
    r2_all = []
    for sol in cache.solution.regimes.values():
        if not sol.error and not sol.underdetermined and sol.r2_scores:
            r2_all.extend(sol.r2_scores.values())

    # MCC backtest (DB query — lightweight since just metadata)
    constraints_df, lmp_df = _load_history(days)
    has_history = not constraints_df.empty and not lmp_df.empty

    return {
        "sf_pairs_solved":   total,
        "history_available": has_history,
        "days_of_history":   days,
        "bounds": {
            "violations":    n_violations,
            "pct_clean":     round((total - n_violations) / total * 100, 1) if total else 0,
            "verdict":       "CLEAN" if n_violations == 0 else "HAS_VIOLATIONS",
        },
        "regime_fit": {
            "n_regimes":    len(cache.solution.regimes),
            "mean_r2":      round(float(np.mean(r2_all)), 4) if r2_all else None,
            "p10_r2":       round(float(np.percentile(r2_all, 10)), 4) if r2_all else None,
            "verdict": (
                "STABLE"   if r2_all and np.mean(r2_all) >= 0.90 else
                "MARGINAL" if r2_all and np.mean(r2_all) >= 0.70 else
                "UNSTABLE" if r2_all else
                "NO_DATA"
            ),
        },
        "mcc_backtest": {
            "available":     has_history,
            "note": (
                "Run GET /api/v1/validation/mcc-backtest for full results."
                if has_history else
                f"No historical data yet — backtest requires at least 1 day of stored intervals."
            ),
        },
        "links": {
            "mcc_backtest":        f"/api/v1/validation/mcc-backtest?days={days}",
            "regime_consistency":  "/api/v1/validation/regime-consistency",
            "bounds_check":        "/api/v1/validation/bounds-check",
        },
    }
