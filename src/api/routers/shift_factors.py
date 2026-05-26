"""
/api/v1/shift-factors — Shift factor lookups and exposure tables.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.models.schemas import (
    ShiftFactorRecord,
    ShiftFactorTableResponse,
    SingleSFResponse,
)
from src.api.sf_cache import cache

router = APIRouter(prefix="/api/v1/shift-factors", tags=["shift-factors"])


@router.get("", response_model=SingleSFResponse)
async def get_shift_factor(
    node: str = Query(..., description="Settlement location / price node"),
    constraint: str = Query(..., description="Constraint name (canonical or raw)"),
    shadow_price: Optional[float] = Query(
        None, description="If provided, also returns sf_exposure = SF × SP"
    ),
) -> SingleSFResponse:
    """
    Look up the shift factor (PTDF) for a single node / constraint pair.

    Optionally supply a live shadow price to get the **Shift Factor Exposure**
    (GSF × ShadowPrice) — the core trader metric showing how much $/MWh of
    congestion cost this node-constraint relationship drives.
    """
    if not cache.is_ready() or cache.solution is None:
        raise HTTPException(
            status_code=503,
            detail="Shift factor solver not yet ready — ingest worker still initializing."
        )

    sf = cache.solution.get(node, constraint)
    sp = shadow_price if shadow_price is not None else cache.get_shadow_prices().get(constraint)
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
) -> ShiftFactorTableResponse:
    """
    Returns the full shift factor exposure table for a constraint, sorted by
    absolute exposure descending (most impactful nodes first).

    The **sf_exposure** column (GSF × ShadowPrice) is the key trader metric:
    it shows exactly how much congestion cost contribution each node carries
    at the current shadow price — even a 2% SF matters at a $2,000 shadow price.
    """
    if not cache.is_ready() or cache.solution is None:
        raise HTTPException(status_code=503, detail="Solver not ready.")

    # Resolve shadow price
    sp = shadow_price
    if sp is None:
        sp = cache.get_shadow_prices().get(constraint)

    # Always pass a dict (even empty) so to_dataframe() always returns
    # named columns (node, constraint, shift_factor). Passing None returns
    # a bare RangeIndex DataFrame with no columns → KeyError on filter.
    sp_map = {constraint: sp} if sp is not None else {}
    df = cache.solution.to_dataframe(shadow_prices=sp_map)

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
async def list_solved_constraints() -> List[str]:
    """Returns the list of constraint names for which SFs have been solved."""
    if not cache.is_ready() or cache.solution is None:
        return []
    constraints = sorted({c for (_, c) in cache.solution._flat})
    return constraints


@router.get("/all-nodes", response_model=List[str])
async def list_solved_nodes() -> List[str]:
    """Returns the list of nodes for which SFs have been solved."""
    if not cache.is_ready() or cache.solution is None:
        return []
    nodes = sorted({n for (n, _) in cache.solution._flat})
    return nodes
