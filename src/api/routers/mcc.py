"""
GET /api/v1/mcc — Pre-clearing MCC estimation.

Computes the Marginal Cost of Congestion (MCC) for any node using the
cached shift factors and the most recent shadow prices:

    MCC_node = -Σ_c (SF_node,c × ShadowPrice_c)

Call this endpoint after each new RTBM interval to get a pre-clearing
estimate of the congestion component of LMP — before SPP officially
publishes the cleared prices.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.models.schemas import MCCContribution, MCCResponse
from src.api.sf_cache import cache

router = APIRouter(prefix="/api/v1/mcc", tags=["mcc"])


@router.get("", response_model=MCCResponse)
async def get_mcc(
    node: str = Query(..., description="Settlement location / price node"),
) -> MCCResponse:
    """
    **Pre-clearing MCC Estimator**

    Returns the estimated Marginal Cost of Congestion for a price node,
    based on cached shift factors × current shadow prices.

    Formula (SPP sign convention):

        MCC_node = -Σ (SF_node,c × ShadowPrice_c)

    A positive MCC means the node is import-constrained (LMP above system
    lambda). Negative MCC means export-constrained (LMP below lambda).

    This is your pre-clearing price signal — call it every 5 minutes as
    soon as SPP publishes new shadow prices.
    """
    if not cache.is_ready() or cache.solution is None:
        raise HTTPException(
            status_code=503,
            detail="Solver not ready — ingest worker still initializing."
        )

    shadow_prices = cache.get_shadow_prices()
    if not shadow_prices:
        raise HTTPException(
            status_code=503,
            detail="No current shadow prices available."
        )

    mcc = cache.solution.compute_mcc(node, shadow_prices)

    # Build per-constraint contribution breakdown
    contributions: List[MCCContribution] = []
    for constraint, sp in shadow_prices.items():
        sf = cache.solution.get(node, constraint)
        contrib = (sf * sp) if sf is not None else 0.0
        contributions.append(MCCContribution(
            constraint=constraint,
            shadow_price=sp,
            shift_factor=sf,
            contribution=contrib,
        ))

    # Sort by |contribution| descending
    contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

    return MCCResponse(
        node=node,
        mcc=round(mcc, 4),
        n_constraints_used=sum(1 for c in contributions if c.shift_factor is not None),
        shadow_prices_used=contributions,
    )


@router.get("/all-nodes", response_model=List[dict])
async def get_mcc_all_nodes(
    min_abs_mcc: float = Query(
        0.0, ge=0,
        description="Filter nodes with |MCC| below this threshold"
    ),
    limit: int = Query(200, ge=1, le=2000),
) -> List[dict]:
    """
    Returns estimated MCC for all nodes that have shift factors solved,
    sorted by |MCC| descending.

    Use this to build a heatmap of nodal price pressure across the grid.
    """
    if not cache.is_ready() or cache.solution is None:
        raise HTTPException(status_code=503, detail="Solver not ready.")

    shadow_prices = cache.get_shadow_prices()
    if not shadow_prices:
        return []

    nodes = sorted({n for (n, _) in cache.solution._flat})
    results = []
    for node in nodes:
        mcc = cache.solution.compute_mcc(node, shadow_prices)
        if abs(mcc) >= min_abs_mcc:
            results.append({"node": node, "mcc": round(mcc, 4)})

    results.sort(key=lambda x: abs(x["mcc"]), reverse=True)
    return results[:limit]
