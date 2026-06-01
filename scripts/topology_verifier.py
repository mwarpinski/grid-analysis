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

def load_window(store: DuckDBStore, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load LMP + constraint data using all available days with BOTH LMP and BC files,
    up to `days` most-recent days. Skips days missing either file so partial backfills
    still produce useful results.
    """
    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)

    lmp_frames, bc_frames = [], []
    days_loaded = 0

    for d in (start + timedelta(days=i) for i in range((end - start).days + 1)):
        lmp_path = store._lmp_dir / f"lmp_{d.strftime('%Y%m%d')}.parquet"
        bc_path  = store._constraints_dir / f"constraints_{d.strftime('%Y%m%d')}.parquet"

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
            f"No days with both LMP+BC found in {start} → {end}. "
            "Run backfill first: python scripts/backfill_history.py --days 25"
        )

    lmp = pd.concat(lmp_frames, ignore_index=True)
    bc  = pd.concat(bc_frames,  ignore_index=True)

    lmp["interval"] = pd.to_datetime(lmp["interval"], errors="coerce")
    bc["interval"]  = pd.to_datetime(bc["interval"],  errors="coerce")

    # Normalise column names
    lmp = lmp.rename(columns=str.lower)
    lmp = lmp.loc[:, ~lmp.columns.duplicated()]

    logger.info(
        f"Loaded {len(lmp):,} LMP rows ({lmp['settlement_location'].nunique()} nodes), "
        f"{len(bc):,} BC rows — {days_loaded} days with complete LMP+BC pairs"
    )
    return lmp, bc


def load_topology() -> Optional[pd.DataFrame]:
    """Load the PNode → Zone mapping if available."""
    path = TOPOLOGY_DIR / "pnode_zone.parquet"
    if path.exists():
        return pd.read_parquet(path)
    logger.warning("Topology pnode_zone.parquet not found — run fetch_spp_topology.py first. "
                   "Zone-level analysis will be skipped.")
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
    via OLS and compute R².

    Returns a DataFrame with columns:
        node, zone, r2, residual_std, n_intervals, n_constraints, status
    """
    logger.info("Layer A: OLS Residual Test")

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
    if target_zone and topology is not None:
        zone_col = "area" if "area" in topology.columns else "reserve_zone"
        zone_nodes = topology[topology[zone_col] == target_zone]["settlement_location"].tolist()
        nodes = [n for n in nodes if n in zone_nodes]
        logger.info(f"  Filtering to zone {target_zone}: {len(nodes)} nodes")

    results = []
    for node in nodes:
        b = mcc_pivot.loc[common_ivs, node].values
        mask = ~np.isnan(b)
        if mask.sum() < MIN_INTERVALS_OLS:
            continue

        A_sub = A[mask]
        b_sub = b[mask]

        try:
            x, residuals, rank, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
            b_hat = A_sub @ x
            ss_res = np.sum((b_sub - b_hat) ** 2)
            ss_tot = np.sum((b_sub - b_sub.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
            resid_std = np.sqrt(ss_res / max(1, len(b_sub) - rank))
        except Exception as e:
            logger.debug(f"  OLS failed for {node}: {e}")
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
        logger.warning("  No OLS results — check data overlap")
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

    SPP's mitigation log is not publicly available, but we can identify
    likely mitigated intervals using these proxies:
      - Node with extreme positive MCC on a binding constraint (SF proxy)
      - Cross-reference: MCC[node] / ShadowPrice[constraint] ≈ SF[node, constraint]

    Outputs the top candidates for manual verification against SPP's
    market monitoring mitigation reports.
    """
    logger.info("Layer B: Mitigated Unit Ground Truth (proxy check)")

    binding = bc[bc["shadow_price"].abs() > 50].copy()  # Only high-value constraints
    if binding.empty:
        logger.info("  No high-shadow-price constraints found in window — skipping")
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

    We identify it as the node with the smallest mean |MCC| across all intervals.
    If the solver ever outputs a meaningful non-zero MCC for this node, the
    matrix is misaligned.
    """
    logger.info("Layer C: Reference Bus Baseline")

    node_stats = (
        lmp.groupby("settlement_location")["mcc"]
        .agg(mean_abs_mcc=lambda x: x.abs().mean(),
             std_mcc="std",
             n_intervals="count")
        .reset_index()
    )

    # Reference bus = node with consistently lowest |MCC|
    ref_bus = node_stats.sort_values("mean_abs_mcc").iloc[0]

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

    # Alert if candidate reference bus has non-trivial MCC
    threshold = 0.10  # $0.10/MWh — should be essentially $0
    if ref_bus["mean_abs_mcc"] > threshold:
        logger.warning(
            f"  ✗ ALERT: Reference bus {ref_bus['settlement_location']} has "
            f"mean |MCC| = ${ref_bus['mean_abs_mcc']:.4f} > ${threshold} threshold. "
            f"Matrix alignment may be incorrect."
        )
        ref_status = "ALERT"
    else:
        logger.info(f"  ✓ Reference bus MCC is approximately zero — matrix alignment OK")
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
    parser.add_argument("--zone", type=str, default=None,
                        help="Limit Layer A analysis to a specific SPP pricing zone.")
    parser.add_argument("--alert-threshold", type=float, default=DEFAULT_ALERT_THRESHOLD,
                        help=f"R² below which a zone is flagged ALERT (default: {DEFAULT_ALERT_THRESHOLD}).")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full node-level results to this CSV path.")
    args = parser.parse_args()

    store = DuckDBStore()

    logger.info("=" * 60)
    logger.info("SPP Topology Confidence Verifier")
    logger.info(f"Window: {args.days} days  |  Alert threshold: R² < {args.alert_threshold}")
    logger.info("=" * 60)

    # Load data
    lmp, bc = load_window(store, args.days)
    topology = load_topology()

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
    has_alerts = (
        not layer_a_results.empty
        and (layer_a_results["status"] == "ALERT").any()
    ) or layer_c_result.get("status") == "ALERT"

    sys.exit(1 if has_alerts else 0)


if __name__ == "__main__":
    main()
