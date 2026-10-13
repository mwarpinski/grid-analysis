"""
fetch_spp_topology.py — Download and normalize SPP official topology files.

Downloads the following from SPP's public Modeling Reports portal:
  - SL_to_Pnode_to_Zone_with_Area.csv   : PNode → Zone → Area mapping
  - SL_Station_BusNumber_Turbine_and_FuelType.xlsx : PNode → EMS station → PSS/E bus number
  - Hub_Definitions.csv                 : Hub aggregate SL → child PNode weight factors
  - Interface_Definitions.csv           : Interchange (INT) settlement locations -> weighted child PNodes (NOT flowgates)

Saved to: src/data/topology/
Re-run anytime to refresh (SPP updates these continuously).

Usage:
  python scripts/fetch_spp_topology.py
  python scripts/fetch_spp_topology.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOPOLOGY_DIR = Path(__file__).parent.parent / "src" / "data" / "topology"
MANIFEST_PATH = TOPOLOGY_DIR / "topology_manifest.json"

BASE = "https://portal.spp.org/file-browser-api/download/modeling-reports?path=%2F"

TOPOLOGY_FILES = {
    "pnode_zone": {
        "url":  BASE + "SL_to_Pnode_to_Zone_with_Area.csv",
        "file": "SL_to_Pnode_to_Zone_with_Area.csv",
        "desc": "Settlement Location → PNode → Zone → Area mapping",
        "format": "csv",
    },
    "bus_mapping": {
        "url":  BASE + "SL_Station_BusNumber_Turbine_and_FuelType.xlsx",
        "file": "SL_Station_BusNumber_Turbine_and_FuelType.xlsx",
        "desc": "PNode → EMS station name → PSS/E bus number → fuel type",
        "format": "xlsx",
    },
    "hub_definitions": {
        "url":  BASE + "Hub_Definitions.csv",
        "file": "Hub_Definitions.csv",
        "desc": "Hub aggregate SL → child PNode weight factors",
        "format": "csv",
    },
    "interface_definitions": {
        "url":  BASE + "Interface_Definitions.csv",
        "file": "Interface_Definitions.csv",
        "desc": "Interchange (INT) settlement locations and their weighted child PNodes (not flowgate definitions)",
        "format": "csv",
    },
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SPP-Grid-Analysis/1.0 (research)"})


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(key: str, spec: dict, force: bool = False) -> dict:
    """Download a single topology file and return a manifest entry."""
    dest = TOPOLOGY_DIR / spec["file"]
    
    if dest.exists() and not force:
        logger.info(f"  {key}: already present — {spec['file']} (use --force to refresh)")
        return _manifest_entry(key, spec, dest, "cached")

    logger.info(f"  {key}: downloading {spec['url']}")
    try:
        resp = SESSION.get(spec["url"], timeout=60, stream=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        size_kb = len(resp.content) / 1024
        logger.info(f"  {key}: saved {size_kb:.0f} KB → {dest.name}")
        return _manifest_entry(key, spec, dest, "downloaded")
    except Exception as e:
        logger.error(f"  {key}: FAILED — {e}")
        return _manifest_entry(key, spec, dest, "error", str(e))


def _manifest_entry(key, spec, dest, status, error=None):
    return {
        "key":      key,
        "file":     spec["file"],
        "desc":     spec["desc"],
        "status":   status,
        "size_kb":  round(dest.stat().st_size / 1024, 1) if dest.exists() else 0,
        "updated":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error":    error,
    }


# ---------------------------------------------------------------------------
# Normalisation — produce clean Parquet from each raw file
# ---------------------------------------------------------------------------

def normalize_pnode_zone(raw_path: Path) -> pd.DataFrame:
    """
    Parse SL_to_Pnode_to_Zone_with_Area.csv.
    Actual SPP columns: SETLOCNAME, SETLOCTYPE, ASSET_TYPE, PARENTPNODE, CHILDPNODE, ENODE, NODE_AREA, RESZONE
    Output columns: settlement_location, sl_type, asset_type, parent_pnode, child_pnode, area, reserve_zone
    """
    df = pd.read_csv(raw_path, dtype=str).fillna("")
    # Use actual SPP column names
    rename = {
        "SETLOCNAME":   "settlement_location",
        "SETLOCTYPE":   "sl_type",
        "ASSET_TYPE":   "asset_type",
        "PARENTPNODE":  "parent_pnode",
        "CHILDPNODE":   "child_pnode",
        "ENODE":        "enode",
        "NODE_AREA":    "area",
        "RESZONE":      "reserve_zone",
        # Fallback lowercase versions
        "setlocname":   "settlement_location",
        "setloctype":   "sl_type",
        "asset_type":   "asset_type",
        "parentpnode":  "parent_pnode",
        "childpnode":   "child_pnode",
        "node_area":    "area",
        "reszone":      "reserve_zone",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # Convert reserve_zone to string with leading zero padding for consistency
    if "reserve_zone" in df.columns:
        df["reserve_zone"] = df["reserve_zone"].str.strip()
    keep = [c for c in ["settlement_location", "sl_type", "asset_type", "parent_pnode", "child_pnode", "area", "reserve_zone"] if c in df.columns]
    df = df[keep].drop_duplicates()
    n_sl = df["settlement_location"].nunique() if "settlement_location" in df.columns else "?"
    logger.info(f"    pnode_zone: {len(df):,} rows, {n_sl} settlement locations, areas: {sorted(df['area'].unique()) if 'area' in df.columns else '?'}")
    return df


def normalize_bus_mapping(raw_path: Path) -> pd.DataFrame:
    """
    Parse SL_Station_BusNumber_Turbine_and_FuelType.xlsx.
    Output columns: settlement_location, ems_area, ems_station, psse_bus_number, psse_name,
                    ems_kv, ems_unit, ems_mw_max, turbine_type, fuel_type
    """
    df = pd.read_excel(raw_path, dtype=str).fillna("")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {}  # Will inspect actual columns dynamically
    # Lower-case all columns first
    df.columns = [c.strip() for c in df.columns]
    # Map whatever columns exist
    col_lower = {c.lower(): c for c in df.columns}
    for src, dst in [
        ("sl name", "settlement_location"), ("sl_name", "settlement_location"),
        ("settlement location", "settlement_location"),
        ("settlement location name", "settlement_location"),
        ("ems station", "ems_station"), ("ems_station", "ems_station"),
        ("station name", "ems_station"),
        ("bus number", "psse_bus_number"), ("bus_number", "psse_bus_number"),
        ("psse bus number", "psse_bus_number"),
        ("fuel type", "fuel_type"), ("fuel_type", "fuel_type"),
        ("turbine type", "turbine_type"),
        # Actual SPP headers: SETTLEMENTLOCATION, EMS_AREA, EMS_STATION, PSSE_NUM, PSSE_NAME,
        # EMS_KV, EMS_UNIT, EMS_MWMAX, TURBINETYPE, FUELTYPE1, FUELTYPE2, INTERCONNECT
        ("settlementlocation", "settlement_location"),
        ("ems_area", "ems_area"), ("psse_num", "psse_bus_number"), ("psse_name", "psse_name"),
        ("ems_kv", "ems_kv"), ("ems_unit", "ems_unit"), ("ems_mwmax", "ems_mw_max"),
        ("turbinetype", "turbine_type"), ("fueltype1", "fuel_type"),
    ]:
        if src in col_lower:
            rename[col_lower[src]] = dst
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ["settlement_location", "ems_area", "ems_station", "psse_bus_number", "psse_name",
                        "ems_kv", "ems_unit", "ems_mw_max", "turbine_type", "fuel_type"] if c in df.columns]
    df = df[keep].drop_duplicates()
    logger.info(f"    bus_mapping: {len(df):,} rows")
    return df


def normalize_hub_definitions(raw_path: Path) -> pd.DataFrame:
    """
    Parse Hub_Definitions.csv.
    Output columns: hub_name, child_pnode, weight
    The weight factors define how hub LMPs aggregate from child nodes.
    """
    df = pd.read_csv(raw_path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    rename = {
        # Actual SPP columns: Settlement Location Name,Parent PNode,Child PNode,Weighting Factor,Settlement Location Type,BAA
        "Settlement Location Name": "hub_name",
        "Parent PNode":             "parent_pnode",
        "Child PNode":              "child_pnode",
        "Weighting Factor":         "weight",
        "Settlement Location Type": "sl_type",
        "BAA":                      "baa",
        # Fallbacks
        "hub_name":                 "hub_name",
        "hub":                      "hub_name",
        "sl_name":                  "hub_name",
        "child_pnode":              "child_pnode",
        "pnode":                    "child_pnode",
        "weighting_factor":         "weight",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    keep = [c for c in ["hub_name", "parent_pnode", "child_pnode", "weight", "sl_type", "baa"] if c in df.columns]
    df = df[keep].drop_duplicates()
    logger.info(f"    hub_definitions: {len(df):,} rows, {df['hub_name'].nunique() if 'hub_name' in df.columns else '?'} hubs")
    return df


def normalize_interface_definitions(raw_path: Path) -> pd.DataFrame:
    """
    Parse Interface_Definitions.csv.

    NOTE: despite its name this file does NOT define transmission flowgates. It lists SPP's
    interchange ("INT") settlement locations (e.g. ties to neighbouring areas) and the weighted
    child pnodes each one aggregates, in the same layout as Hub_Definitions.csv.
    Output columns: interface_name, parent_pnode, child_pnode, weight, sl_type, baa
    """
    df = pd.read_csv(raw_path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    rename = {
        # Same format as hub_definitions: Settlement Location Name,Parent PNode,Child PNode,Weighting Factor,Settlement Location Type,BAA
        "Settlement Location Name": "interface_name",
        "Parent PNode":             "parent_pnode",
        "Child PNode":              "child_pnode",
        "Weighting Factor":         "weight",
        "Settlement Location Type": "sl_type",
        "BAA":                      "baa",
        # Fallbacks
        "interface_name":   "interface_name",
        "flowgate_name":    "interface_name",
        "monitored_line":   "monitored_line",
        "from_area":        "from_area",
        "to_area":          "to_area",
        "limit_mw":         "limit_mw",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for numeric in ("limit_mw", "weight"):
        if numeric in df.columns:
            df[numeric] = pd.to_numeric(df[numeric], errors="coerce")
    keep = [c for c in ["interface_name", "parent_pnode", "child_pnode", "weight", "sl_type", "baa",
                        "monitored_line", "from_area", "to_area", "limit_mw"] if c in df.columns]
    df = df[keep].drop_duplicates()
    logger.info(f"    interface_definitions: {len(df):,} rows")
    return df


NORMALIZERS = {
    "pnode_zone":            (normalize_pnode_zone,          "pnode_zone.parquet"),
    "bus_mapping":           (normalize_bus_mapping,         "bus_mapping.parquet"),
    "hub_definitions":       (normalize_hub_definitions,     "hub_definitions.parquet"),
    "interface_definitions": (normalize_interface_definitions, "interface_definitions.parquet"),
}


def normalize_all(force: bool = False) -> None:
    """Parse all downloaded raw files and save as Parquet."""
    for key, (fn, parquet_name) in NORMALIZERS.items():
        spec = TOPOLOGY_FILES[key]
        raw_path = TOPOLOGY_DIR / spec["file"]
        parquet_path = TOPOLOGY_DIR / parquet_name

        if not raw_path.exists():
            logger.warning(f"  {key}: raw file missing, skipping normalisation")
            continue

        if parquet_path.exists() and not force:
            mtime_raw = raw_path.stat().st_mtime
            mtime_pq  = parquet_path.stat().st_mtime
            if mtime_pq >= mtime_raw:
                logger.info(f"  {key}: Parquet up-to-date, skipping")
                continue

        try:
            df = fn(raw_path)
            df.to_parquet(parquet_path, index=False)
            logger.info(f"    → saved {parquet_path.name}")
        except Exception as e:
            logger.error(f"  {key}: normalisation failed — {e}")


# ---------------------------------------------------------------------------
# Status / manifest
# ---------------------------------------------------------------------------

def print_status() -> None:
    """Print current state of the topology directory."""
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    logger.info("Topology file status:")
    for key, spec in TOPOLOGY_FILES.items():
        raw  = TOPOLOGY_DIR / spec["file"]
        pq   = TOPOLOGY_DIR / NORMALIZERS[key][1]
        raw_ok = "✓" if raw.exists() else "✗"
        pq_ok  = "✓" if pq.exists() else "✗"
        size_kb = f"{raw.stat().st_size/1024:.0f}KB" if raw.exists() else "—"
        logger.info(f"  {raw_ok} raw  {pq_ok} parquet  {size_kb:>8}  {key}: {spec['desc']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force",  action="store_true", help="Re-download even if files already exist.")
    parser.add_argument("--status", action="store_true", help="Show current file status and exit.")
    args = parser.parse_args()

    TOPOLOGY_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        print_status()
        return

    logger.info("Fetching SPP topology files...")
    manifest = {}
    for key, spec in TOPOLOGY_FILES.items():
        entry = download_file(key, spec, force=args.force)
        manifest[key] = entry

    logger.info("Normalising to Parquet...")
    normalize_all(force=args.force)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(f"Done. Manifest saved to {MANIFEST_PATH}")
    print_status()


if __name__ == "__main__":
    main()
