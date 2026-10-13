"""
build_geo_crosswalk.py
======================
Builds the node and constraint location tables from PUBLIC data only.

  1. Downloads EIA-860 (public domain) if it is not already in src/data/geo/raw/
  2. Locates SPP generator settlement locations (exact EIA node designation, then a strict
     name/fuel/capacity match)  ->  src/data/geo/node_locations.parquet
  3. Fits shift factors on the stored history and places every constraint seen in RT/DA
     (station anchor -> contingency anchor -> triangulation)
     ->  src/data/geo/constraint_locations.parquet

Coordinates are rounded to 0.01 degrees (~1 km); each constraint carries its method, a quality
label and an uncertainty radius.  See src/analysis/geo_crosswalk.py for how accurate each method is.

Usage:  python scripts/build_geo_crosswalk.py [--days 14]
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import congestion_analytics as ca
from src.analysis import constraint_identity as ci
from src.analysis import geo_crosswalk as gx
from src.storage.duckdb_store import store

logging.disable(logging.CRITICAL)

GEO_DIR = ROOT / "src" / "data" / "geo"
RAW_DIR = GEO_DIR / "raw" / "eia860"
EIA_URL = "https://www.eia.gov/electricity/data/eia860/archive/xls/eia860{year}.zip"
NEEDED = ("2___Plant_Y{year}.xlsx", "3_1_Generator_Y{year}.xlsx")


def ensure_eia860(year: int) -> Path:
    """Download and extract just the two EIA-860 workbooks we need into their own directory."""
    need = [n.format(year=year) for n in NEEDED]
    if all((RAW_DIR / n).exists() for n in need):
        return RAW_DIR
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading EIA-860 {year} ...")
    r = requests.get(EIA_URL.format(year=year), timeout=180)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for n in need:
            (RAW_DIR / n).write_bytes(z.read(n))        # fixed names only: nothing else is extracted
    return RAW_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--eia-year", type=int, default=2024)
    args = ap.parse_args()

    GEO_DIR.mkdir(parents=True, exist_ok=True)
    raw = ensure_eia860(args.eia_year)
    plants = pd.read_excel(raw / f"2___Plant_Y{args.eia_year}.xlsx", header=1)
    gens = pd.read_excel(raw / f"3_1_Generator_Y{args.eia_year}.xlsx", sheet_name=0, header=1)
    bus = pd.read_parquet(ROOT / "src" / "data" / "topology" / "bus_mapping.parquet")

    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=args.days)
    cols = ["constraint_name", "shadow_price", "monitored_facility", "contingent_facility"]
    rt_bc = store.query_constraints(start, end, columns=cols)
    da_bc = store.query_da_constraints(start, end, columns=cols)
    rt_lmp = store.query_lmp(start, end, columns=["settlement_location", "mcc"])
    da_lmp = store.query_da_lmp(start, end, columns=["settlement_location", "mcc"])
    for d in (rt_bc, da_bc, rt_lmp, da_lmp):
        d["interval"] = pd.to_datetime(d["interval"])

    print("Fitting shift factors (RT + DA) ...")
    sol = ca.fit_shift_factors_rt_da(rt_bc, rt_lmp, da_bc, da_lmp, min_intervals=10)
    sol.flatten()
    X = ca._sf_matrix(sol._flat)
    X.index = X.index.astype(str)

    exact = gx.node_locations_from_eia(plants, gens, X.index)
    fuzzy = gx.fuzzy_node_locations(bus, plants, gens, already_located=exact["node"])
    fuzzy = fuzzy[fuzzy["node"].isin(X.index)]
    nodes = pd.concat([exact, fuzzy.drop(columns="score")], ignore_index=True)
    print(f"Located nodes: {len(exact)} exact + {len(fuzzy)} name-matched = {len(nodes)} "
          f"({nodes['plant_code'].nunique()} plants)")
    gx.round_coords(nodes).to_parquet(GEO_DIR / "node_locations.parquet", index=False)

    stations = gx.station_coordinates(bus, nodes)
    tri = gx.triangulate_constraints(X, nodes)
    meta = pd.concat([rt_bc, da_bc])
    meta = meta[meta["shadow_price"].abs() > 1e-4][["constraint_name", "monitored_facility", "contingent_facility"]]
    meta = meta.drop_duplicates("constraint_name")
    placed = gx.place_constraints(meta, stations, tri)
    ident = ci.add_identity_columns(meta)[["constraint_name", "element_key", "constraint_key"]]
    placed = placed.merge(ident, on="constraint_name", how="left")
    placed.to_parquet(GEO_DIR / "constraint_locations.parquet", index=False)

    ok = placed[placed["method"] != "unplaced"]
    print(f"\nConstraints seen: {len(placed)}")
    print(placed.groupby(["method", "quality"]).size().to_string())
    print(f"placed {len(ok)} ({len(ok) / len(placed):.0%}); wrote {GEO_DIR.relative_to(ROOT)}/*.parquet "
          "(served by GET /api/v1/geo/constraints; the data directory is gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
