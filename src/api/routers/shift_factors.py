"""
/api/v1/shift-factors — Shift factor lookups and exposure tables.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.models.schemas import (
    ShiftFactorRecord,
    ShiftFactorTableResponse,
    SingleSFResponse,
)
from src.api.sf_cache import cache

router = APIRouter(prefix="/api/v1/shift-factors", tags=["shift-factors"])


# ---------------------------------------------------------------------------
# SPP pricing zone centroids (public market zone geography — no specific
# infrastructure locations).  Positions are approximate zone centroids,
# the same level of precision as any publicly available ISO zone map.
# ---------------------------------------------------------------------------
_SPP_ZONES: Dict[str, dict] = {
    "AECC":  {"lat": 34.8, "lon": -92.3, "name": "Arkansas Electric Cooperative"},
    "AECI":  {"lat": 37.5, "lon": -91.5, "name": "Associated Electric Coop (MO)"},
    "ALTW":  {"lat": 42.1, "lon": -91.5, "name": "Alliant Energy (IA/WI)"},
    "AMRN":  {"lat": 38.5, "lon": -90.5, "name": "Ameren Missouri"},
    "CSWS":  {"lat": 35.5, "lon": -96.5, "name": "AEP/PSO (OK/AR)"},
    "GRDA":  {"lat": 36.3, "lon": -95.0, "name": "Grand River Dam (NE OK)"},
    "KCPL":  {"lat": 39.1, "lon": -94.6, "name": "KC Power & Light"},
    "LES":   {"lat": 40.8, "lon": -96.7, "name": "Lincoln Electric (NE)"},
    "MIDW":  {"lat": 41.6, "lon": -93.6, "name": "MidAmerican Energy (IA)"},
    "NPPD":  {"lat": 41.1, "lon": -97.6, "name": "NE Public Power District"},
    "OKGE":  {"lat": 35.5, "lon": -97.5, "name": "OG&E (Central OK)"},
    "OPPD":  {"lat": 41.3, "lon": -96.0, "name": "Omaha Public Power"},
    "SPRM":  {"lat": 37.1, "lon": -94.5, "name": "Empire District (SW MO)"},
    "SUNC":  {"lat": 38.5, "lon": -100.5, "name": "Sunflower Electric (W KS)"},
    "WAUE":  {"lat": 44.5, "lon": -99.0,  "name": "Western Area Upper Plains East"},
    "WAUW":  {"lat": 47.0, "lon": -106.0, "name": "Western Area Upper Plains West"},
    "WFEC":  {"lat": 35.0, "lon": -99.0,  "name": "W. Farmers Electric (W OK)"},
    "WR":    {"lat": 38.7, "lon": -97.0,  "name": "Evergy Kansas (WR)"},
    "SECI":  {"lat": 33.5, "lon": -93.0,  "name": "SWEPCO (AR/LA/TX)"},
    "CLECO": {"lat": 31.2, "lon": -92.4,  "name": "Cleco (LA)"},
    "BEPM":  {"lat": 42.5, "lon": -97.5,  "name": "Basin Electric (N Plains)"},
    "MRES":  {"lat": 46.8, "lon": -96.8,  "name": "Missouri River Energy (ND)"},
    "NWPS":  {"lat": 44.3, "lon": -96.8,  "name": "NW Public Service (SD)"},
    "WACM":  {"lat": 39.5, "lon": -105.0, "name": "W. Area Colorado-Missouri"},
    "PACE":  {"lat": 42.8, "lon": -108.5, "name": "PacifiCorp East (WY)"},
    "AVA":   {"lat": 47.0, "lon": -117.0, "name": "Avista (WA/ID)"},
    "AEPM":  {"lat": 35.2, "lon": -96.8,  "name": "AEP/PSO Metered (OK)"},
    "INDN":  {"lat": 41.8, "lon": -86.0,  "name": "Indiana Michigan Power"},
}

_ZONE_RE = re.compile(r'^([A-Z]{2,6})[._]')


def _node_to_zone(node: str) -> Optional[str]:
    """Extract SPP pricing zone code from a settlement location name."""
    m = _ZONE_RE.match(node.upper())
    if m:
        prefix = m.group(1)
        if prefix in _SPP_ZONES:
            return prefix
    # Exact match (zone-aggregate nodes like "WR", "KCPL")
    up = node.upper()
    if up in _SPP_ZONES:
        return up
    return None


@router.get("", response_model=SingleSFResponse)
async def get_shift_factor(
    node: str = Query(..., description="Settlement location / price node"),
    constraint: str = Query(..., description="Constraint name (canonical or raw)"),
    shadow_price: Optional[float] = Query(
        None, description="If provided, also returns sf_exposure = SF × SP"
    ),
    iso: str = Query("spp", description="ISO namespace: spp or miso"),
) -> SingleSFResponse:
    """
    Look up the shift factor (PTDF) for a single node / constraint pair.

    Optionally supply a live shadow price to get the **Shift Factor Exposure**
    (GSF × ShadowPrice) — the core trader metric showing how much $/MWh of
    congestion cost this node-constraint relationship drives.
    """
    iso_cache = cache.for_iso(iso)
    if not iso_cache.is_ready() or iso_cache.solution is None:
        raise HTTPException(
            status_code=503,
            detail="Shift factor solver not yet ready — ingest worker still initializing."
        )

    sf = iso_cache.solution.get(node, constraint)
    sp = shadow_price if shadow_price is not None else iso_cache.get_shadow_prices().get(constraint)
    exposure = (sf * sp) if (sf is not None and sp is not None) else None

    return SingleSFResponse(
        node=node,
        constraint=constraint,
        shift_factor=sf,
        shadow_price=sp,
        sf_exposure=exposure,
        solved=sf is not None,
    )


@router.get("/table", response_model=ShiftFactorTableResponse)
async def get_sf_exposure_table(
    constraint: str = Query(..., description="Constraint name to build exposure table for"),
    shadow_price: Optional[float] = Query(
        None,
        description=(
            "Override shadow price for this constraint. "
            "If omitted, uses the most recent observed shadow price."
        )
    ),
    min_abs_exposure: Optional[float] = Query(
        None, ge=0,
        description="Filter out nodes with |sf_exposure| below this threshold"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Max nodes to return"),
    iso: str = Query("spp", description="ISO namespace: spp or miso"),
) -> ShiftFactorTableResponse:
    """
    Returns the full shift factor exposure table for a constraint, sorted by
    absolute exposure descending (most impactful nodes first).

    The **sf_exposure** column (GSF × ShadowPrice) is the key trader metric:
    it shows exactly how much congestion cost contribution each node carries
    at the current shadow price — even a 2% SF matters at a $2,000 shadow price.
    """
    iso_cache = cache.for_iso(iso)
    if not iso_cache.is_ready() or iso_cache.solution is None:
        raise HTTPException(status_code=503, detail="Solver not ready.")

    # Resolve shadow price
    sp = shadow_price
    if sp is None:
        sp = iso_cache.get_shadow_prices().get(constraint)

    # Always pass a dict (even empty) so to_dataframe() always returns
    # named columns (node, constraint, shift_factor). Passing None returns
    # a bare RangeIndex DataFrame with no columns → KeyError on filter.
    sp_map = {constraint: sp} if sp is not None else {}
    df = iso_cache.solution.to_dataframe(shadow_prices=sp_map)

    if df.empty or "constraint" not in df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"No shift factors solved yet — solver needs more intervals. "
                   "Check /api/v1/status for solver readiness."
        )

    # Filter to requested constraint
    df = df[df["constraint"] == constraint]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No shift factors solved for constraint '{constraint}'. "
                   "This constraint may not have appeared in the training window yet."
        )

    # Apply exposure filter
    if min_abs_exposure is not None and "sf_exposure" in df.columns:
        df = df[df["sf_exposure"].abs() >= min_abs_exposure]

    df = df.head(limit)

    records = []
    for _, row in df.iterrows():
        records.append(ShiftFactorRecord(
            node=row["node"],
            constraint=row["constraint"],
            shift_factor=row["shift_factor"],
            shadow_price=row.get("shadow_price"),
            sf_exposure=row.get("sf_exposure"),
        ))

    return ShiftFactorTableResponse(
        constraint=constraint,
        shadow_price=sp,
        n_nodes=len(records),
        records=records,
    )


@router.get("/all-constraints", response_model=List[str])
async def list_solved_constraints(
    iso: str = Query("spp", description="ISO namespace: spp or miso")
) -> List[str]:
    """Returns the list of constraint names for which SFs have been solved."""
    iso_cache = cache.for_iso(iso)
    if not iso_cache.is_ready() or iso_cache.solution is None:
        return []
    constraints = sorted({c for (_, c) in iso_cache.solution._flat})
    return constraints


@router.get("/all-nodes", response_model=List[str])
async def list_solved_nodes(
    iso: str = Query("spp", description="ISO namespace: spp or miso")
) -> List[str]:
    """Returns the list of nodes for which SFs have been solved."""
    iso_cache = cache.for_iso(iso)
    if not iso_cache.is_ready() or iso_cache.solution is None:
        return []
    nodes = sorted({n for (n, _) in iso_cache.solution._flat})
    return nodes


@router.get("/heatmap")
async def get_sf_heatmap(
    constraint: str = Query(..., description="Constraint name to build zone heatmap for"),
    shadow_price: Optional[float] = Query(
        None, description="Override shadow price; uses live value if omitted."
    ),
    iso: str = Query("spp", description="ISO namespace: spp or miso"),
):
    """
    **SF Zone Heatmap** — for map overlay rendering.

    Groups all solved shift factors for a constraint by pricing zone
    and returns zone-level aggregates with lat/lon centroids for Leaflet circle rendering.
    """
    iso_cache = cache.for_iso(iso)
    if iso_cache.solution is None:
        raise HTTPException(status_code=503, detail="SF solution not yet ready.")

    sp = shadow_price or iso_cache.get_shadow_prices().get(constraint)

    # Collect all (node, sf) pairs for this constraint
    pairs = [
        (node, sf)
        for (node, c), sf in iso_cache.solution._flat.items()
        if c == constraint
    ]

    if not pairs:
        raise HTTPException(
            status_code=404,
            detail=f"No shift factors solved for constraint '{constraint}'.",
        )

    # Group by zone
    import numpy as np
    from collections import defaultdict
    zone_sfs: Dict[str, list] = defaultdict(list)
    unmapped = 0

    for node, sf in pairs:
        if iso.lower() == "spp":
            zone = _node_to_zone(node)
            if zone:
                zone_sfs[zone].append(sf)
            else:
                unmapped += 1
        else:
            parts = re.split(r'[._-]', node)
            zone = parts[0].upper() if parts else "UNKNOWN"
            zone_sfs[zone].append(sf)

    zones = []
    for zone_code, sfs in zone_sfs.items():
        arr = np.array(sfs)
        mean_sf = float(np.mean(arr))
        max_abs = float(np.max(np.abs(arr)))
        exposure = mean_sf * sp if sp is not None else None

        if abs(mean_sf) < 0.005:
            direction = "neutral"
        elif mean_sf > 0:
            direction = "exacerbates"
        else:
            direction = "relieves"

        lat, lon, name = 39.0, -89.0, f"MISO Zone {zone_code}"
        if iso.lower() == "spp" and zone_code in _SPP_ZONES:
            lat = _SPP_ZONES[zone_code]["lat"]
            lon = _SPP_ZONES[zone_code]["lon"]
            name = _SPP_ZONES[zone_code]["name"]

        zones.append({
            "zone":        zone_code,
            "name":        name,
            "lat":         lat,
            "lon":         lon,
            "mean_sf":     round(mean_sf, 5),
            "max_abs_sf":  round(max_abs, 5),
            "n_nodes":     len(sfs),
            "direction":   direction,
            "exposure":    round(exposure, 2) if exposure is not None else None,
        })

    # Sort by |mean_sf| descending so the hottest zones come first
    zones.sort(key=lambda z: abs(z["mean_sf"]), reverse=True)

    return {
        "constraint":     constraint,
        "shadow_price":   sp,
        "n_nodes_total":  len(pairs),
        "n_nodes_mapped": len(pairs) - unmapped,
        "n_nodes_unmapped": unmapped,
        "zones":          zones,
    }
