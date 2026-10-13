"""
identity_report.py
==================
Empirical check of the constraint-identity layer (src/analysis/constraint_identity.py).

The test is physical, not textual: if two constraint names really are the same
element-under-contingency, their shift-factor vectors across the ~1,600 nodes must be
(nearly) identical; unrelated constraints must not be.  Also measures how much merging
aliases shrinks the "unseen constraint" problem, and how well DA and RT line up.

Usage:  python scripts/identity_report.py [--days 14]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import congestion_analytics as ca
from src.analysis import constraint_identity as ci
from src.storage.duckdb_store import store

logging.disable(logging.CRITICAL)


def q(x) -> str:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return "n=0"
    return (f"n={len(x):5d}  median r={np.median(x):+.3f}  p10={np.quantile(x, .1):+.3f}  "
            f"p90={np.quantile(x, .9):+.3f}  share r>0.9={np.mean(x > 0.9):.0%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=args.days)

    cols = ["constraint_name", "shadow_price", "monitored_facility", "contingent_facility"]
    rt_bc = store.query_constraints(start, end, columns=cols + ["nerc_id"])
    da_bc = store.query_da_constraints(start, end, columns=cols)
    rt_lmp = store.query_lmp(start, end, columns=["settlement_location", "mcc"])
    da_lmp = store.query_da_lmp(start, end, columns=["settlement_location", "mcc"])
    for d in (rt_bc, da_bc, rt_lmp, da_lmp):
        d["interval"] = pd.to_datetime(d["interval"])

    print("=" * 78)
    print(f"IDENTITY REPORT  {start.date()} .. {end.date()}")
    print("=" * 78)

    alias = ci.build_alias_table(rt_bc, da_bc)
    print(f"\n1. ALIAS TABLE: {len(alias)} constraint keys (element + contingency) priced in RT and/or DA")
    print(f"   RT only {int((alias.in_rt & ~alias.in_da).sum())} | DA only {int((~alias.in_rt & alias.in_da).sum())} | both {int((alias.in_rt & alias.in_da).sum())}")
    multi = alias[(alias.rt_names.map(len) + alias.da_names.map(len)) > 1]
    print(f"   {len(multi)} keys have more than one name (aliases); the element level has "
          f"{alias.element_key.nunique()} distinct physical elements")
    print(f"   element kinds: {alias.element_kind.value_counts().to_dict()}")

    # element-level DA/RT overlap, by physical element
    el = alias.groupby("element_key").agg(in_rt=("in_rt", "max"), in_da=("in_da", "max"))
    print(f"   physical elements priced in RT {int(el.in_rt.sum())}, in DA {int(el.in_da.sum())}, in both {int((el.in_rt & el.in_da).sum())}")

    # ------------------------------------------------------------- SF validation
    print("\n2. SHIFT-FACTOR TEST OF THE KEYS  (shift factors fit on RT + DA)")
    sol = ca.fit_shift_factors_rt_da(rt_bc, rt_lmp, da_bc, da_lmp, min_intervals=10)
    sol.flatten()
    X = ca._sf_matrix(sol._flat)
    n2k = ci.name_to_key(rt_bc, da_bc)
    orient = ci.name_orientation(rt_bc, da_bc)
    cons = [c for c in X.columns if c in n2k]
    Xc = ci.canonical_sf(X[cons], orient)          # orient by listed direction so aliases are comparable
    V = np.nan_to_num(Xc.to_numpy())
    raw = np.corrcoef(np.nan_to_num(X[cons].to_numpy()).T)
    C = np.corrcoef(V.T)
    idx = {c: i for i, c in enumerate(cons)}
    key = {c: n2k[c] for c in cons}
    elem = {c: key[c].split("#")[0] for c in cons}

    same_key, same_el_diff_cont, diff = [], [], []
    for a, b in itertools.combinations(cons, 2):
        r = C[idx[a], idx[b]]
        if not np.isfinite(r):
            continue
        if key[a] == key[b]:
            same_key.append(r)
        elif elem[a] == elem[b]:
            same_el_diff_cont.append(r)
        else:
            diff.append(r)
    rng = np.random.default_rng(0)
    diff_s = list(rng.choice(diff, size=min(len(diff), 5000), replace=False)) if diff else []
    print(f"   {len(cons)} modeled constraints with an identity key")
    if same_key:
        ridx = [(idx[a], idx[b]) for a, b in itertools.combinations(cons, 2) if key[a] == key[b]]
        print(f"   (before orientation these aliases had median r = {np.median([raw[i, j] for i, j in ridx]):+.2f})")
    print(f"   aliases of the SAME element+contingency : {q(same_key)}")
    print(f"   same element, DIFFERENT contingency      : {q(same_el_diff_cont)}")
    print(f"   unrelated elements                       : {q(diff_s)}")
    sep = (np.median(same_key) - np.median(diff_s)) if same_key and diff_s else float('nan')
    ok1 = bool(same_key) and np.median(same_key) > 0.8
    print(f"   [{'PASS' if ok1 else 'FAIL'}] aliases of one element+contingency share shift factors (median r {np.median(same_key) if same_key else float('nan'):+.2f})")
    ok2 = bool(same_el_diff_cont) and np.median(same_el_diff_cont) < np.median(same_key) if same_key and same_el_diff_cont else False
    print(f"   [{'PASS' if ok2 else 'FAIL'}] the contingency matters: same element under another contingency is less similar than an alias")

    # ----------------------------------------------------------- unseen constraints
    print("\n3. DOES MERGING ALIASES SHRINK THE 'UNSEEN CONSTRAINT' PROBLEM?")
    days = sorted(rt_bc.interval.dt.normalize().unique())
    cut = pd.Timestamp(days[int(len(days) * 0.7)])
    early = rt_bc[(rt_bc.interval <= cut) & (rt_bc.shadow_price.abs() > 1e-4)]
    late = rt_bc[(rt_bc.interval > cut) & (rt_bc.shadow_price.abs() > 1e-4)]
    mag = late.shadow_price.abs().sum()
    new_name = set(late.constraint_name) - set(early.constraint_name)
    early_keys = {n2k.get(n) for n in early.constraint_name}
    early_elems = {n2k[n].split("#")[0] for n in early.constraint_name if n in n2k}
    new_key = {n for n in new_name if n2k.get(n) not in early_keys}
    new_el = {n for n in new_name if n in n2k and n2k[n].split("#")[0] not in early_elems}
    share = lambda S: late[late.constraint_name.isin(S)].shadow_price.abs().sum() / mag
    print(f"   constraints first seen in the last 30% of days, by identity level (share of |shadow price|):")
    print(f"     by name                 : {len(new_name):3d} unseen -> {share(new_name):.0%}")
    print(f"     by element+contingency  : {len(new_key):3d} unseen -> {share(new_key):.0%}")
    print(f"     by physical element only: {len(new_el):3d} unseen -> {share(new_el):.0%}")
    print("   (an unseen ELEMENT has no shift-factor information at all; an unseen constraint on a seen element at least has a neighbour)")
    print("\n" + "=" * 78)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
