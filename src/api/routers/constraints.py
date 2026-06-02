"""
/api/v1/constraints — Live and historical constraint data.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Query

from src.api.models.schemas import ConstraintSnapshot, LiveConstraintsResponse
from src.api.sf_cache import cache

router = APIRouter(prefix="/api/v1/constraints", tags=["constraints"])


@router.get("/live", response_model=LiveConstraintsResponse)
async def get_live_constraints(
    state: Optional[str] = Query(
        None, description="Filter by state: BINDING, NEAR_BINDING, VIOLATED"
    ),
    min_pct_of_limit: Optional[float] = Query(
        None, ge=0, le=200,
        description="Only return constraints at or above this % of their limit"
    ),
    min_shadow_price: Optional[float] = Query(
        None, ge=0,
        description="Only return constraints with |shadow_price| >= this value"
    ),
    iso: str = Query(
        "spp", description="ISO namespace: spp or miso"
    ),
) -> LiveConstraintsResponse:
    """
    Returns the most recent RTBM binding constraint snapshot.

    Auto-refreshed every ~5 minutes by the ingest worker.
    Use the ``/ws/constraints`` WebSocket endpoint to receive push updates.
    """
    iso_cache = cache.for_iso(iso)
    if not iso_cache.is_ready() or iso_cache.latest_interval is None:
        return LiveConstraintsResponse(
            as_of=datetime.utcnow(),
            interval=None,
            n_binding=0,
            constraints=[],
        )

    interval = iso_cache.latest_interval
    rows = interval.constraints

    # Apply filters
    if state:
        rows = [r for r in rows if r.get("state", "").upper() == state.upper()]
    if min_pct_of_limit is not None:
        rows = [
            r for r in rows
            if r.get("pct_of_limit") is not None
            and r["pct_of_limit"] >= min_pct_of_limit
        ]
    if min_shadow_price is not None:
        rows = [
            r for r in rows
            if abs(r.get("shadow_price", 0)) >= min_shadow_price
        ]

    # Convert to schema
    snapshots = []
    for r in rows:
        try:
            snapshots.append(ConstraintSnapshot(
                interval=r.get("interval", interval.interval),
                constraint_name=r.get("constraint_name", ""),
                canonical_id=r.get("canonical_id"),
                shadow_price=float(r.get("shadow_price", 0)),
                limit_mw=r.get("limit_mw"),
                monitored_mw=r.get("monitored_mw"),
                pct_of_limit=r.get("pct_of_limit"),
                state=r.get("state"),
                contingency_name=r.get("contingency_name"),
                volatility_score=r.get("volatility_score"),
            ))
        except Exception:
            continue

    return LiveConstraintsResponse(
        as_of=iso_cache.last_updated or datetime.utcnow(),
        interval=interval.interval,
        n_binding=len(snapshots),
        constraints=snapshots,
    )


@router.get("/near-limit", response_model=LiveConstraintsResponse)
async def get_near_limit_constraints(
    threshold_pct: float = Query(
        95.0, ge=50, le=100,
        description="Loading % threshold for near-limit alarm (default: 95%)"
    ),
    iso: str = Query(
        "spp", description="ISO namespace: spp or miso"
    ),
) -> LiveConstraintsResponse:
    """
    Returns constraints running at or above the given loading % threshold.

    This is the 'ticking time bomb' list — lines that are about to bind.
    Default threshold: 95% of thermal/stability limit.
    """
    return await get_live_constraints(
        state=None,
        min_pct_of_limit=threshold_pct,
        min_shadow_price=None,
        iso=iso,
    )
