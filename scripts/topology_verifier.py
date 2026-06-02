"""
topology_verifier.py — SPP Grid Topology Confidence Scorer.

Runs three independent verification layers against historical RTBM data to
validate the accuracy of our shift factor / PTDF topology model.

Layer A — OLS Residual Test (R² per zone)
  Builds MCC[node,t] = Σ SF[node,c] × ShadowPrice[c,t] via OLS.
  If topology is correct, R² should be > 0.98 per zone.
  R² < 0.95 → alert: possible unmodelled outage or reconfiguration.

Layer B — Mitigated Unit Ground Truth
  Compares our computed shift factors against SPP's published mitigation logs
  (generators flagged as market power mitigated have |SF| ≥ 5% on the constraint).
  Validates directional accuracy of our model.

Layer C — Reference Bus Baseline
  The SPP settlement reference bus must always have MCC ≈ $0.
  Non-zero MCC at the reference bus → matrix alignment error.

Usage:
  python scripts/topology_verifier.py
  python scripts/topology_verifier.py --days 7
  python scripts/topology_verifier.py --zone CSWS
  python scripts/topology_verifier.py --alert-threshold 0.90
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.storage.duckdb_store import DuckDBStore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

TOPOLOGY_DIR = Path(__file__).parent.parent / "src" / "data" / "topology"

# Alert threshold — zones below this R² are flagged untradable
DEFAULT_ALERT_THRESHOLD = 0.95
# Minimum intervals needed to run OLS for a constraint regime
MIN_INTERVALS_OLS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------



def load_window(store: DuckDBStore, days: int, iso: str = "spp") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load LMP + constraint data using all available days with BOTH LMP and BC files,
    up to `days` most-recent days. Skips days missing either file so partial backfills
    still produce useful results.
    """
    end   = date.today()
    start = end - timedelta(days=days - 1)

    lmp_frames, bc_frames = [], []
    days_loaded = 0

    for d in (start + timedelta(days=i) for i in range((end - start).days + 1)):
        lmp_path = store._lmp_dir_for_iso(iso) / f"lmp_{d.strftime('%Y%m%d')}.parquet"
        bc_path  = store._constraints_dir_for_iso(iso) / f"constraints_{d.strftime('%Y%m%d')}.parquet"

        if lmp_path.exists() and bc_path.exists():
            lmp_frames.append(pd.read_parquet(lmp_path))
            bc_frames.append(pd.read_parquet(bc_path))
            days_loaded += 1
        elif lmp_path.exists() and not bc_path.exists():
            logger.debug(f"  {d}: LMP exists but BC missing — skipping")
        elif bc_path.exists() and not lmp_path.exists():
            logger.debug(f"  {d}: BC exists but LMP missing — skipping")

    if not lmp_frames or not bc_frames:
        raise ValueError(
            f"No days with both LMP+BC found in {start} → {end} for {iso.upper()}."
        )

    lmp = pd.concat(lmp_frames, ignore_index=True)
    bc  = pd.concat(bc_frames,  ignore_index=True)

    lmp["interval"] = pd.to_datetime(lmp["interval"], errors="coerce")
    bc["interval"]  = pd.to_datetime(bc["interval"],  errors="coerce")

    # Normalise column names
    lmp = lmp.rename(columns=str.lower)
    lmp = lmp.loc[:, ~lmp.columns.duplicated()]

    # Map raw constraint names to canonical names using the ConstraintRegistry
    # to safeguard against global Flowgate ID shifts/renames
    try:
        from src.analysis.constraint_registry import ConstraintRegistry
        registry = ConstraintRegistry()
        unique_raw = bc["constraint_name"].unique()
        name_map = {}
        for r in unique_raw:
            res = registry.resolve(r)
            name_map[r] = res.canonical_name if res else r
        bc["constraint_name"] = bc["constraint_name"].map(name_map)
        logger.info("  Canonicalized raw constraint names to safeguard against Flowgate ID shifts.")
    except Exception as e:
        logger.warning(f"  Could not canonicalize constraint names: {e}")

    logger.info(
        f"Loaded {len(lmp):,} LMP rows ({lmp['settlement_location'].nunique()} nodes), "
        f"{len(bc):,} BC rows — {days_loaded} days with complete LMP+BC pairs"
    )
    return lmp, bc


def load_topology(iso: str = "spp") -> Optional[pd.DataFrame]:
    """Load the PNode → Zone mapping if available."""
    iso_lower = iso.lower()
    path = TOPOLOGY_DIR / f"pnode_zone_{iso_lower}.parquet" if iso_lower != "spp" else TOPOLOGY_DIR / "pnode_zone.parquet"
    if path.exists():
        return pd.read_parquet(path)
    
    path_default = TOPOLOGY_DIR / "pnode_zone.parquet"
    if iso_lower == "spp" and path_default.exists():
        return pd.read_parquet(path_default)
        
    logger.warning(f"Topology mapping file not found for {iso.upper()} — Zone-level analysis will rely on prefix-based zone inference.")
    return None


# ---------------------------------------------------------------------------
# Layer A — OLS Residual Test
# ---------------------------------------------------------------------------

def layer_a_ols_residual(
    lmp: pd.DataFrame,
    bc: pd.DataFrame,
    topology: Optional[pd.DataFrame],
    alert_threshold: float,
    target_zone: Optional[str] = None,
) -> pd.DataFrame:
    """
    For each node, solve:
        MCC[node, t] = Σ_c SF[node, c] × ShadowPrice[c, t]
    via Bounded L2 Regularized (Ridge) Least Squares and compute R².
    Enforces physical shift factor bounds of [-1.0, 1.0] and prevents weight explosion
    during collinear constraint events.

    Returns a DataFrame with columns:
        node, zone, r2, residual_std, n_intervals, n_constraints, status
    """
    logger.info("Layer A: Bounded Ridge Regression Residual Test")

    # Pivot shadow prices: index=interval, columns=constraint_name
    binding = bc[bc["shadow_price"].abs() > 0.01].copy()
    shadow_pivot = (
        binding.groupby(["interval", "constraint_name"])["shadow_price"]
        .mean()
        .unstack(fill_value=0.0)
    )

    # Pivot MCC: index=interval, columns=settlement_location
    mcc_pivot = (
        lmp.groupby(["interval", "settlement_location"])["mcc"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    # Align intervals
    common_ivs = shadow_pivot.index.intersection(mcc_pivot.index)
    if len(common_ivs) < MIN_INTERVALS_OLS:
        logger.error(f"Only {len(common_ivs)} overlapping intervals — need ≥ {MIN_INTERVALS_OLS}")
        return pd.DataFrame()

    A = shadow_pivot.loc[common_ivs].values          # (T, K)
    constraints = list(shadow_pivot.columns)

    # Zone lookup — use 'area' (readable BA names: CSWS, OKGE, WR, etc.)
    # Fall back to reserve_zone if area is absent
    zone_map: dict[str, str] = {}
    if topology is not None:
        zone_col = "area" if "area" in topology.columns else "reserve_zone"
        zone_map = dict(zip(topology["settlement_location"], topology[zone_col]))

    nodes = list(mcc_pivot.columns)
    
    # Fill in missing nodes dynamically (especially for MISO or where mapping is missing)
    for node in nodes:
        if node not in zone_map:
            zone_map[node] = node.split('.')[0] if '.' in node else "SYSTEM"

    if target_zone:
        nodes = [n for n in nodes if zone_map.get(n) == target_zone]
        logger.info(f"  Filtering to zone {target_zone}: {len(nodes)} nodes")

    from scipy.optimize import lsq_linear

    results = []
    for node in nodes:
        b = mcc_pivot.loc[common_ivs, node].values
        mask = ~np.isnan(b)
        if mask.sum() < MIN_INTERVALS_OLS:
            continue

        A_sub = A[mask]
        b_sub = b[mask]

        try:
            # Multi-collinearity mitigation: Solve analytically as L2-regularized least squares (Ridge).
            # Then physically clip to [-1.0, 1.0].
            alpha = 0.05  # L2 regularization penalty
            K = A_sub.shape[1]
            x = np.linalg.solve(A_sub.T @ A_sub + alpha * np.eye(K), A_sub.T @ b_sub)
            x = np.clip(x, -1.0, 1.0)
            rank = K
            
            b_hat = A_sub @ x
            ss_res = np.sum((b_sub - b_hat) ** 2)
            ss_tot = np.sum((b_sub - b_sub.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
            resid_std = np.sqrt(ss_res / max(1, len(b_sub) - rank))
        except Exception as e:
            logger.debug(f"  Analytical Ridge regression failed for {node}: {e}")
            continue

        zone = zone_map.get(node, "UNKNOWN")
        status = "ok" if r2 >= alert_threshold else "ALERT"
        results.append({
            "node":          node,
            "zone":          zone,
            "r2":            round(r2, 4),
            "residual_std":  round(resid_std, 4),
            "n_intervals":   int(mask.sum()),
            "n_constraints": len(constraints),
            "status":        status,
        })

    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("  No regression results — check data overlap")
        return df

    # Zone-level summary
    zone_summary = (
        df.groupby("zone")[["r2", "residual_std"]]
        .agg({"r2": "mean", "residual_std": "mean"})
        .round(4)
        .reset_index()
    )
    zone_summary["status"] = zone_summary["r2"].apply(
        lambda r: "ok" if r >= alert_threshold else "ALERT"
    )

    logger.info(f"  Tested {len(df)} nodes across {df['zone'].nunique()} zones")
    logger.info(f"  Zone-level R² summary:")
    for _, row in zone_summary.sort_values("r2").iterrows():
        icon = "✓" if row["status"] == "ok" else "✗ ALERT"
        logger.info(f"    {icon}  {row['zone']:<20} R²={row['r2']:.4f}  σ_resid={row['residual_std']:.3f}")

    alerts = zone_summary[zone_summary["status"] == "ALERT"]
    if not alerts.empty:
        logger.warning(
            f"\n  ⚠  {len(alerts)} zone(s) below R²={alert_threshold} threshold:\n"
            + "\n".join(f"    → {r['zone']} (R²={r['r2']:.4f}): "
                        "possible line outage or unmodelled reconfiguration — mark UNTRADABLE"
                        for _, r in alerts.iterrows())
        )

    return df


# ---------------------------------------------------------------------------
# Layer B — Mitigated Unit Ground Truth
# ---------------------------------------------------------------------------

def layer_b_mitigation_check(lmp: pd.DataFrame, bc: pd.DataFrame) -> pd.DataFrame:
    """
    Approximation of the mitigated unit test using market-observable signals.
    Filters historical intervals to only those with EXACTLY ONE active binding constraint
    to prevent multi-constraint blending and ensure high fidelity validation.

    Outputs the top candidates for manual verification against SPP's
    market monitoring mitigation reports.
    """
    logger.info("Layer B: Mitigated Unit Ground Truth (proxy check)")

    # Identify intervals where exactly one constraint is active (shadow price > 0.01) in the footprint
    active_counts = bc[bc["shadow_price"].abs() > 0.01].groupby("interval")["constraint_name"].nunique()
    single_binding_intervals = active_counts[active_counts == 1].index

    # Filter to pure single-constraint intervals and only high-value constraints (> $50)
    binding = bc[
        (bc["interval"].isin(single_binding_intervals)) & 
        (bc["shadow_price"].abs() > 50)
    ].copy()

    if binding.empty:
        logger.info("  No pure single-constraint intervals with high shadow price found — skipping")
        return pd.DataFrame()

    binding["interval"] = pd.to_datetime(binding["interval"], errors="coerce")
    lmp2 = lmp.copy()
    lmp2["interval"] = pd.to_datetime(lmp2["interval"], errors="coerce")

    # Merge MCC onto constraint intervals
    merged = binding.merge(
        lmp2[["interval", "settlement_location", "mcc"]],
        on="interval",
        how="inner",
    )

    # Compute implied shift factor: SF ≈ MCC / ShadowPrice
    merged["implied_sf"] = merged["mcc"] / merged["shadow_price"].replace(0, np.nan)
    merged = merged.dropna(subset=["implied_sf"])

    # Flag nodes with |SF| >= 5% on a constraint (SPP mitigation threshold)
    high_sf = merged[merged["implied_sf"].abs() >= 0.05].copy()
    if high_sf.empty:
        logger.info("  No nodes with |implied SF| ≥ 5% on high-value constraints")
        return pd.DataFrame()

    summary = (
        high_sf.groupby(["constraint_name", "settlement_location"])
        .agg(
            mean_implied_sf=("implied_sf", "mean"),
            n_intervals=("interval", "nunique"),
            mean_shadow_price=("shadow_price", "mean"),
        )
        .reset_index()
        .sort_values("mean_implied_sf", key=abs, ascending=False)
    )

    logger.info(f"  Found {len(summary)} node/constraint pairs with |implied SF| ≥ 5%")
    logger.info("  Top 10 candidates for mitigation verification:")
    for _, row in summary.head(10).iterrows():
        direction = "FLOW→" if row["mean_implied_sf"] > 0 else "FLOW←"
        logger.info(
            f"    {direction} {row['settlement_location']:<32} on {row['constraint_name']:<20} "
            f"SF≈{row['mean_implied_sf']:+.3f}  n={row['n_intervals']}"
        )

    logger.info("  → Compare these SF estimates against SPP Market Monitoring mitigation logs "
                "at https://spp.org/markets-operations/market-monitoring/")
    return summary


# ---------------------------------------------------------------------------
# Layer C — Reference Bus Baseline
# ---------------------------------------------------------------------------

def layer_c_reference_bus(lmp: pd.DataFrame) -> dict:
    """
    The SPP settlement reference bus must always have MCC ≈ $0.
    Evaluates reference bus validity relative to the system average to avoid false positives
    during high volatility events, and enforces a strict 100x separation ratio.
    """
    logger.info("Layer C: Reference Bus Baseline")

    node_stats = (
        lmp.groupby("settlement_location")["mcc"]
        .agg(mean_abs_mcc=lambda x: x.abs().mean(),
             std_mcc="std",
             n_intervals="count")
        .reset_index()
    )

    # Sort nodes by absolute MCC to find candidate and compute statistics for others
    sorted_stats = node_stats.sort_values("mean_abs_mcc")
    ref_bus = sorted_stats.iloc[0]
    
    other_mean_abs_mcc = sorted_stats.iloc[1:]["mean_abs_mcc"].mean()

    logger.info(f"  Identified reference bus candidate: {ref_bus['settlement_location']}")
    logger.info(f"    Mean |MCC|:   ${ref_bus['mean_abs_mcc']:.4f}/MWh")
    logger.info(f"    Std MCC:      ${ref_bus['std_mcc']:.4f}/MWh")
    logger.info(f"    N intervals:  {int(ref_bus['n_intervals'])}")

    # Check known SPP reference buses (SPP uses AECI or a specific EMS bus)
    known_ref_buses = {"AECI", "SPP_REF", "SETTLEMENT_REF", "CSWS_HUB"}
    best_known = node_stats[node_stats["settlement_location"].isin(known_ref_buses)]
    if not best_known.empty:
        logger.info(f"  Known reference bus candidates in data:")
        for _, row in best_known.sort_values("mean_abs_mcc").iterrows():
            status = "✓ OK" if row["mean_abs_mcc"] < 0.01 else "✗ NON-ZERO MCC — check matrix alignment"
            logger.info(f"    {row['settlement_location']:<30} |MCC|={row['mean_abs_mcc']:.4f} {status}")

    # Dynamic threshold: 0.1% of system-wide average absolute MCC (minimum $0.10/MWh)
    system_avg_abs_mcc = lmp["mcc"].abs().mean()
    dynamic_threshold = max(0.10, 0.001 * system_avg_abs_mcc)

    # Relative ratio test: reference bus MCC must be at least 100x lower than the average of other nodes
    ratio = other_mean_abs_mcc / max(1e-6, ref_bus["mean_abs_mcc"])
    ratio_ok = ratio >= 100.0

    # Alert if candidate reference bus fails both checks
    if ref_bus["mean_abs_mcc"] > dynamic_threshold or not ratio_ok:
        logger.warning(
            f"  ✗ ALERT: Reference bus {ref_bus['settlement_location']} failed verification. "
            f"Mean |MCC| = ${ref_bus['mean_abs_mcc']:.4f} > dynamic threshold = ${dynamic_threshold:.4f} "
            f"or separation ratio {ratio:.1f}x < 100x. "
            f"Matrix alignment may be incorrect."
        )
        ref_status = "ALERT"
    else:
        logger.info(f"  ✓ Reference bus MCC verified (mean |MCC| = ${ref_bus['mean_abs_mcc']:.4f}, "
                    f"dynamic threshold = ${dynamic_threshold:.4f}, separation ratio = {ratio:.1f}x) — matrix alignment OK")
        ref_status = "ok"

    return {
        "reference_bus":   ref_bus["settlement_location"],
        "mean_abs_mcc":    round(float(ref_bus["mean_abs_mcc"]), 6),
        "std_mcc":         round(float(ref_bus["std_mcc"]), 6),
        "n_intervals":     int(ref_bus["n_intervals"]),
        "status":          ref_status,
    }


# ---------------------------------------------------------------------------
# Topology Confidence Score
# ---------------------------------------------------------------------------

def compute_confidence_score(layer_a: pd.DataFrame, layer_c: dict) -> dict:
    """
    Aggregate a single Topology Confidence Score (0–1) per zone.
    Combines Layer A R² with Layer C reference bus penalty.
    """
    if layer_a.empty:
        return {}

    ref_penalty = 0.0 if layer_c.get("status") == "ok" else 0.05

    zone_scores = (
        layer_a.groupby("zone")["r2"]
        .mean()
        .apply(lambda r: max(0.0, r - ref_penalty))
        .round(4)
        .to_dict()
    )
    return zone_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=14,
                        help="Number of historical days to analyze (default: 14).")
    parser.add_argument("--iso", type=str, default="spp",
                        help="ISO namespace: spp or miso (default: spp).")
    parser.add_argument("--zone", type=str, default=None,
                        help="Limit Layer A analysis to a specific pricing zone.")
    parser.add_argument("--alert-threshold", type=float, default=DEFAULT_ALERT_THRESHOLD,
                        help=f"R² below which a zone is flagged ALERT (default: {DEFAULT_ALERT_THRESHOLD}).")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full node-level results to this CSV path.")
    args = parser.parse_args()

    store = DuckDBStore()
    iso = args.iso.lower()

    logger.info("=" * 60)
    logger.info(f"{args.iso.upper()} Topology Confidence Verifier")
    logger.info(f"Window: {args.days} days  |  Alert threshold: R² < {args.alert_threshold}")
    logger.info("=" * 60)

    # Load data
    lmp, bc = load_window(store, args.days, iso=iso)
    topology = load_topology(iso=iso)

    # Normalise LMP column names defensively
    for old, new in [("Settlement Location", "settlement_location"), ("MCC", "mcc"),
                     ("LMP", "lmp"), ("MEC", "mec"), ("MLC", "mlc")]:
        if old in lmp.columns and new not in lmp.columns:
            lmp = lmp.rename(columns={old: new})
    lmp.columns = [c.lower() for c in lmp.columns]

    # Run all three layers
    logger.info("")
    layer_a_results = layer_a_ols_residual(lmp, bc, topology, args.alert_threshold, args.zone)

    logger.info("")
    layer_b_results = layer_b_mitigation_check(lmp, bc)

    logger.info("")
    layer_c_result  = layer_c_reference_bus(lmp)

    # Final confidence scores
    logger.info("")
    logger.info("=" * 60)
    logger.info("Topology Confidence Scores by Zone")
    logger.info("=" * 60)
    scores = compute_confidence_score(layer_a_results, layer_c_result)
    if scores:
        for zone, score in sorted(scores.items(), key=lambda x: x[1]):
            bar = "█" * int(score * 20)
            icon = "✓" if score >= args.alert_threshold else "✗ ALERT — mark UNTRADABLE"
            logger.info(f"  {icon}  {zone:<20} {bar:<20} {score:.4f}")
    else:
        logger.warning("  No zone scores computed — check data availability")

    logger.info("")
    logger.info(f"Reference bus: {layer_c_result.get('reference_bus', '?')}  "
                f"|MCC|={layer_c_result.get('mean_abs_mcc', '?'):.4f}  "
                f"status={layer_c_result.get('status', '?').upper()}")

    if args.output and not layer_a_results.empty:
        layer_a_results.to_csv(args.output, index=False)
        logger.info(f"\nNode-level results saved to {args.output}")

    # Exit code: non-zero if any zone is in ALERT
    zone_alerts = False
    if scores:
        zone_alerts = any(score < args.alert_threshold for score in scores.values())
    has_alerts = zone_alerts or layer_c_result.get("status") == "ALERT"

    sys.exit(1 if has_alerts else 0)


if __name__ == "__main__":
    main()
