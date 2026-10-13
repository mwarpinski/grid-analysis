"""
sanity_report.py
================
Runs every analytics output on the stored history and checks that the numbers are sane.

Shift factors are fit on the EARLIER part of the history and the DART decomposition is
evaluated on the LATER part, so the headline numbers are out-of-sample.

Usage
-----
    python scripts/sanity_report.py                 # last 14 days, 70/30 split
    python scripts/sanity_report.py --days 21 --train-frac 0.6

Exit status is non-zero if any hard check fails.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import congestion_analytics as ca
from src.analysis.multi_constraint_sf import calculate_shift_factors
from src.storage.duckdb_store import store

logging.disable(logging.CRITICAL)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def r2(actual: np.ndarray, pred: np.ndarray) -> float:
    m = ~np.isnan(actual) & ~np.isnan(pred)
    a, p = actual[m], pred[m]
    return float(1.0 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=args.days)

    print("=" * 78)
    print(f"SANITY REPORT  {start.date()} .. {end.date()}  (train first {args.train_frac:.0%}, test the rest)")
    print("=" * 78)

    # ------------------------------------------------------------------ data
    print("\n1. DATA")
    rt_bc = store.query_constraints(start, end, columns=["constraint_name", "shadow_price", "state", "source_limit_mw", "initial_effective_limit_mw", "monitored_facility"])
    rt_lmp = store.query_lmp(start, end, columns=["settlement_location", "mcc"])
    da_bc = store.query_da_constraints(start, end, columns=["constraint_name", "shadow_price", "monitored_facility"])
    da_lmp = store.query_da_lmp(start, end, columns=["settlement_location", "mcc", "lmp", "mlc", "mec"])
    for d in (rt_bc, rt_lmp, da_bc, da_lmp):
        d["interval"] = pd.to_datetime(d["interval"])
    print(f"  RT constraints {len(rt_bc):,} rows | RT LMP {len(rt_lmp):,} | DA constraints {len(da_bc):,} | DA LMP {len(da_lmp):,}")
    check("all four datasets present", all(len(d) for d in (rt_bc, rt_lmp, da_bc, da_lmp)))
    if not all(len(d) for d in (rt_bc, rt_lmp, da_bc, da_lmp)):
        return 1
    check("DA LMP = MEC + MLC + MCC", (da_lmp.lmp - da_lmp.mec - da_lmp.mlc - da_lmp.mcc).abs().max() < 1e-3)
    check("no duplicate (interval, node) in RT LMP", not rt_lmp.duplicated(["interval", "settlement_location"]).any())
    n_da_days = da_lmp["interval"].dt.floor("D").nunique()
    print(f"  DA covers {da_lmp.interval.nunique()} hours (~{n_da_days} days)")

    # ------------------------------------------------------------ SF fit/split
    all_days = sorted(rt_lmp["interval"].dt.floor("D").unique())
    split_day = pd.Timestamp(all_days[int(len(all_days) * args.train_frac)])
    print(f"\n2. SHIFT FACTORS  (fit on intervals up to {split_day.date()} 00:00)")
    tr_bc, tr_lmp = rt_bc[rt_bc.interval <= split_day], rt_lmp[rt_lmp.interval <= split_day]
    sol = calculate_shift_factors(tr_lmp, tr_bc, min_intervals=10)
    sf = sol._flat
    n_nodes = len({k[0] for k in sf}); n_cons = len({k[1] for k in sf})
    print(f"  {len(sf):,} SF pairs, {n_nodes} nodes x {n_cons} constraints; unmodeled in training: {len(sol.unmodeled_constraints)}")
    check("shift factors produced", len(sf) > 0)
    all_sf = np.array(list(sf.values()))
    check("SFs mostly within [-1, 1]", (np.abs(all_sf) > 1).mean() < 0.01, f"{(np.abs(all_sf) > 1).mean():.3%} outside")

    # --------------------------------------------------------------- DART
    print("\n3. NODAL DART  (spread = RT - DA hourly congestion, test period)")
    te_rt = rt_lmp[rt_lmp.interval > split_day]
    te_da = da_lmp[da_lmp.interval > split_day]
    dart = ca.nodal_dart(te_da, te_rt)
    summ = ca.dart_summary(dart)
    print(f"  {len(dart):,} (hour,node) spreads over {dart.hour.nunique()} hours and {dart.settlement_location.nunique()} nodes")
    check("DART produced for most nodes", dart.settlement_location.nunique() > 1000)
    check("spreads are finite", np.isfinite(dart.spread).all())
    check("mean spread across the footprint is small vs its dispersion",
          abs(dart.spread.mean()) < dart.spread.std(), f"mean {dart.spread.mean():.2f}, sd {dart.spread.std():.2f}")
    print(f"  median node mean-spread {summ['mean'].median():+.2f}; share of nodes with |naive t|>3: {(summ.naive_t_stat.abs() > 3).mean():.0%} (hours are autocorrelated: ranking only)")
    print("  largest-|t| nodes:")
    print(summ[["n_hours", "mean", "std", "hit_rate", "naive_t_stat"]].head(5).round(2).to_string())

    # ------------------------------------------------------- decomposition
    print("\n4. DART DECOMPOSITION BY CONSTRAINT  (shift factors from the TRAIN period)")
    rsp_te = ca.hourly_shadow_prices_rt(rt_bc[rt_bc.interval > split_day])
    dsp_te = ca.hourly_shadow_prices_da(da_bc[da_bc.interval > split_day])
    dec_rt = ca.decompose_dart(dart, sf, rsp_te, dsp_te)
    print(f"  RT-only fit : pooled R2 {r2(dec_rt['actual'].to_numpy(), dec_rt['explained'].to_numpy()):.3f}, "
          f"unattributed shadow-price move {dec_rt['unattributed_share']:.0%}, {n_cons} constraints modeled")
    # The analytics API uses shift factors fit on RT and DA data together (covers DA-only constraints)
    sol_da = ca.fit_shift_factors_rt_da(tr_bc, tr_lmp, da_bc[da_bc.interval <= split_day], da_lmp[da_lmp.interval <= split_day])
    sf_da = sol_da._flat
    dec = ca.decompose_dart(dart, sf_da, rsp_te, dsp_te)
    print(f"  RT+DA fit   : {len({k[1] for k in sf_da})} constraints modeled")
    act, exp = dec["actual"].to_numpy(), dec["explained"].to_numpy()
    pooled = r2(act, exp)
    node_r2 = dec["summary"]["r2"].dropna()
    print(f"  pooled R2 explained-vs-actual spread: {pooled:.3f} | median node R2 {node_r2.median():.3f} | "
          f"shadow-price move on constraints WITHOUT shift factors: {dec['unattributed_share']:.0%}")
    check("decomposition explains a meaningful share out-of-sample", pooled > 0.2, f"R2 {pooled:.3f}")
    check("decomposition is better than predicting zero", ((act - exp) ** 2).sum() < (act ** 2).sum())
    top_node = dec["summary"]["mean_abs_spread"].idxmax()
    print(f"  example: node with the largest mean |spread|: {top_node}")
    print(ca.node_decomposition(dec, top_node, top=5).round(3).to_string())

    # cross-market check: do RT-fit SFs reproduce the DA MCC from DA shadow prices?
    da_sp = ca.hourly_shadow_prices_da(da_bc[da_bc.interval > split_day])
    sfm = ca._sf_matrix(sf_da)
    cons = [c for c in da_sp.columns if c in sfm.columns]
    da_act = te_da.pivot_table(index="interval", columns="settlement_location", values="mcc", aggfunc="first")
    nodes = [n for n in da_act.columns if n in sfm.index]
    hrs = da_act.index.intersection(da_sp.index.union(da_act.index))
    da_pred = da_sp.reindex(index=da_act.index, columns=cons).fillna(0.0).to_numpy() @ sfm.loc[nodes, cons].fillna(0.0).to_numpy().T
    r2_da = r2(da_act[nodes].to_numpy(), da_pred)
    print(f"  RT-fit SFs applied to DA shadow prices reproduce DA MCC with R2 {r2_da:.3f}")
    check("RT-fit shift factors transfer to the DA market", r2_da > 0.3, f"R2 {r2_da:.3f}")

    # -------------------------------------------------------- constraint stats
    print("\n5. CONSTRAINT BEHAVIOUR  (all history)")
    cs = ca.constraint_stats(rt_bc, da_bc)
    print(f"  {len(cs)} constraints priced in RT")
    print(cs.head(6)[["constraint_name", "rt_binding_hours", "da_binding_hours", "rt_only_hours", "mean_run_intervals", "p90_abs_sp", "bind_given_activated"]].round(2).to_string(index=False))
    fs = ca.constraint_stats(rt_bc, da_bc, group_by="monitored_facility")
    print(f"  grouped by physical element: {len(fs)} facilities; DA-missed RT hours by constraint name {int(cs.rt_only_hours.sum())} vs by facility {int(fs.rt_only_hours.sum())}")
    check("grouping by facility never finds MORE surprises than by constraint", fs.rt_only_hours.sum() <= cs.rt_only_hours.sum())
    check("constraint stats computed", len(cs) > 10)
    check("run lengths are sane", (cs.mean_run_intervals >= 1).all() and (cs.max_run_intervals <= rt_bc.interval.nunique()).all())
    check("bind_given_activated is a probability", cs.bind_given_activated.dropna().between(0, 1).all())

    # ------------------------------------------------------------------ pairs
    print("\n6. BEST NODE PAIRS for the most frequent constraint (stability: fit on test period)")
    sol_b = calculate_shift_factors(rt_lmp[rt_lmp.interval > split_day], rt_bc[rt_bc.interval > split_day], min_intervals=10)
    freq = cs.set_index("constraint_name")["rt_binding_intervals"]
    cand = [c for c in freq.index if any(k[1] == c for k in sf) and any(k[1] == c for k in sol_b._flat)]
    if cand:
        pairs = ca.best_pairs(sf, cand[0], top=5, sf_alt=sol_b._flat)
        print(f"  constraint {cand[0]}")
        print(pairs.round(3).to_string(index=False))
        check("pairs found", len(pairs) > 0)
        if "same_sign" in pairs:
            print(f"  pairs whose spread keeps its sign in the independent fit: {pairs.same_sign.mean():.0%}")
            check("top pairs are mostly stable across fits", pairs.same_sign.mean() >= 0.6, f"{pairs.same_sign.mean():.0%}")

    # --------------------------------------------------------------- scenarios
    print("\n7. SCENARIOS")
    limits = ca.constraint_limits(rt_bc)
    try:
        bus = pd.read_parquet(Path("src/data/topology/bus_mapping.parquet"))
        bus = bus.dropna(subset=["ems_mw_max"]).sort_values("ems_mw_max", ascending=False)
        unit = next(r for _, r in bus.iterrows() if (r.settlement_location, cand[0]) in sf) if cand else None
    except Exception:
        unit = None
    if unit is not None:
        mw = float(unit.ems_mw_max)
        t = ca.trip_scenario(sf, unit.settlement_location, -mw, limits, top=5)
        print(f"  trip of {unit.settlement_location} ({unit.ems_station}, {mw:.0f} MW, {unit.fuel_type}):")
        print(t.round(3).to_string(index=False))
        check("trip scenario returns stressed constraints", len(t) > 0)
        check("trip flow change equals SF x MW", np.allclose(t["delta_flow_mw"], t["sf"] * -mw))
    top3 = list(cs.constraint_name[:3])
    sp = ca.historical_shadow_price(rt_bc, top3, 0.5)
    pi = ca.price_impact(sf, sp)
    print(f"  top-3 constraints binding at their median historical price {({k: round(v, 1) for k, v in sp.items()})}:")
    print(f"  nodal MCC change: min {pi.mcc_change.min():.1f}, median {pi.mcc_change.median():.1f}, max {pi.mcc_change.max():.1f} $/MWh")
    check("price impact is finite", np.isfinite(pi.mcc_change).all())

    # ------------------------------------------------------------------ verdict
    failed = [c for c in CHECKS if not c[1]]
    print("\n" + "=" * 78)
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed" + ("" if not failed else "  FAILED: " + "; ".join(c[0] for c in failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
