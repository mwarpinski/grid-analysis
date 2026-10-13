"""
Geo Router — /api/v1/geo/*

Where constraints are, from public data only (see src/analysis/geo_crosswalk.py):

    GET /constraints   every constraint placed by scripts/build_geo_crosswalk.py
    GET /live          the currently binding constraints with a location (new ones are placed on the fly)

Each placement carries ``method`` (monitored_station | contingency_station | triangulation), a
``quality`` label and an uncertainty ``radius_km``.  These are REGIONAL positions, not substation
coordinates, and coordinates are rounded to 0.01 degrees.  Roughly half of all constraints cannot be
placed at all and are simply absent from the results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.analysis import geo_crosswalk as gx
from src.api.sf_cache import cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/geo", tags=["geo"])

_GEO_DIR = Path(__file__).parent.parent.parent / "data" / "geo"
_BUS_PATH = Path(__file__).parent.parent.parent / "data" / "topology" / "bus_mapping.parquet"

_state: dict = {"mtime": None, "table": None, "stations": None, "nodes": None, "extra": {}}


def _load() -> pd.DataFrame:
    """Placement table, reloaded if the build script has rewritten it."""
    path = _GEO_DIR / "constraint_locations.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No constraint locations yet. Run scripts/build_geo_crosswalk.py.")
    mtime = path.stat().st_mtime
    if _state["mtime"] != mtime:
        _state.update(mtime=mtime, table=pd.read_parquet(path), stations=None, nodes=None, extra={})
    return _state["table"]


def _stations() -> Optional[pd.DataFrame]:
    if _state["stations"] is None:
        nodes_path = _GEO_DIR / "node_locations.parquet"
        if not nodes_path.exists() or not _BUS_PATH.exists():
            return None
        nodes = pd.read_parquet(nodes_path)
        _state["nodes"] = nodes
        _state["stations"] = gx.station_coordinates(pd.read_parquet(_BUS_PATH), nodes)
    return _state["stations"]


def _records(df: pd.DataFrame):
    out = df.replace([np.inf, -np.inf], np.nan).astype(object).where(df.notna(), None)
    return out.to_dict(orient="records")


def _place_unknown(rows: pd.DataFrame) -> pd.DataFrame:
    """Place constraints missing from the table (names are reissued daily) using station anchors."""
    stations = _stations()
    if stations is None or rows.empty:
        return pd.DataFrame()
    todo = rows[~rows["constraint_name"].isin(_state["extra"])]
    if not todo.empty:
        placed = gx.place_constraints(todo[["constraint_name", "monitored_facility", "contingent_facility"]], stations, None)
        for r in placed.to_dict(orient="records"):
            _state["extra"][r["constraint_name"]] = r
    got = [_state["extra"][n] for n in rows["constraint_name"] if n in _state["extra"]]
    return pd.DataFrame(got)


@router.get("/constraints")
async def constraint_locations(
    quality: str = Query("good,approximate", description="comma-separated: good, approximate, unplaced"),
    method: Optional[str] = Query(None, description="monitored_station | contingency_station | triangulation"),
    limit: int = Query(2000, ge=1, le=10000),
):
    """Every placed constraint with its method, quality and uncertainty radius."""
    t = _load()
    t = t[t["quality"].isin([q.strip() for q in quality.split(",")])]
    if method:
        t = t[t["method"] == method]
    return {
        "n": int(len(t)),
        "note": "Regional positions (rounded to 0.01 deg) with an uncertainty radius; not substation coordinates.",
        "constraints": _records(t.head(limit)),
    }


@router.get("/live")
async def live_locations(iso: str = Query("spp")):
    """Currently binding constraints with a location. New names are placed on the fly from their facility strings."""
    if iso.lower() != "spp":
        raise HTTPException(status_code=400, detail="Locations are available for SPP only.")
    latest = cache.for_iso("spp").latest_interval
    if latest is None:
        raise HTTPException(status_code=503, detail="No live interval yet.")
    live = pd.DataFrame(latest.constraints)
    if live.empty:
        return {"interval": str(latest.interval), "n": 0, "constraints": []}
    table = _load().set_index("constraint_name")
    known = live[live["constraint_name"].isin(table.index)]
    out = known[["constraint_name", "shadow_price", "state"]].join(
        table[["lat", "lon", "radius_km", "method", "quality"]], on="constraint_name")
    unknown = live[~live["constraint_name"].isin(table.index)]
    if not unknown.empty and {"monitored_facility", "contingent_facility"} <= set(unknown.columns):
        extra = _place_unknown(unknown)
        if not extra.empty:
            add = unknown[["constraint_name", "shadow_price", "state"]].merge(
                extra[["constraint_name", "lat", "lon", "radius_km", "method", "quality"]], on="constraint_name")
            out = pd.concat([out, add], ignore_index=True)
    out = out[out["quality"].isin(["good", "approximate"])]
    return {"interval": str(latest.interval), "n": int(len(out)), "n_binding": int(len(live)),
            "constraints": _records(out.sort_values("shadow_price"))}
