"""
GET /api/v1/mcc — Pre-clearing MCC estimation.

Computes the Marginal Cost of Congestion (MCC) for any node using the
cached shift factors and the most recent shadow prices:

    MCC_node = Σ_c (SF_node,c × ShadowPrice_c)

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
    iso: str = Query("spp", description="ISO namespace: spp or miso"),
) -> MCCResponse:
    """
    **Pre-clearing MCC Estimator**

    Returns the estimated Marginal Cost of Congestion for a price node,
    based on cached shift factors × current shadow prices.

    Formula (the convention the shift factors are fit with; SPP publishes
    binding shadow prices as negative numbers):

        MCC_node = Σ (SF_node,c × ShadowPrice_c)

    A positive MCC means the node's LMP is above the system energy price;
    negative means below it.

    This is your pre-clearing price signal — call it every 5 minutes as
    soon as SPP publishes new shadow prices.
    """
    iso_cache = cache.for_iso(iso)
    if not iso_cache.is_ready() or iso_cache.solution is None:
        raise HTTPException(
            status_code=503,
            detail="Solver not ready — ingest worker still initializing."
        )

    shadow_prices = iso_cache.get_shadow_prices()
    if not shadow_prices:
        raise HTTPException(
            status_code=503,
            detail="No current shadow prices available."
        )

    mcc = iso_cache.solution.compute_mcc(node, shadow_prices)

    # Build per-constraint contribution breakdown. Constraints with no fitted shift factor may have a
    # provisional one borrowed from a sibling constraint on the same element; it is labelled and kept
    # out of `mcc` (fitted only) so the two can be told apart.
    contributions: List[MCCContribution] = []
    prior_extra = 0.0
    for constraint, sp in shadow_prices.items():
        sf = iso_cache.solution.get(node, constraint)
        source = "fitted" if sf is not None else None
        if sf is None:
            sf = iso_cache.provisional.get((node, constraint))
            if sf is not None:
                source = "sibling_prior"
                prior_extra += sf * sp
        contrib = (sf * sp) if sf is not None else 0.0
        contributions.append(MCCContribution(
            constraint=constraint,
            shadow_price=sp,
            shift_factor=sf,
            contribution=contrib,
            source=source,
        ))

    # Sort by |contribution| descending
    contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

    total_mag = sum(abs(c.shadow_price) for c in contributions)
    fitted_mag = sum(abs(c.shadow_price) for c in contributions if c.source == "fitted")
    covered_mag = sum(abs(c.shadow_price) for c in contributions if c.shift_factor is not None)

    return MCCResponse(
        node=node,
        mcc=round(mcc, 4),
        mcc_with_priors=round(mcc + prior_extra, 4),
        n_constraints_used=sum(1 for c in contributions if c.source == "fitted"),
        shadow_price_coverage=round(fitted_mag / total_mag, 4) if total_mag else None,
        shadow_price_coverage_with_priors=round(covered_mag / total_mag, 4) if total_mag else None,
        shadow_prices_used=contributions,
    )


@router.get("/all-nodes", response_model=List[dict])
async def get_mcc_all_nodes(
    min_abs_mcc: float = Query(
        0.0, ge=0,
        description="Filter nodes with |MCC| below this threshold"
    ),
    limit: int = Query(200, ge=1, le=2000),
    iso: str = Query("spp", description="ISO namespace: spp or miso"),
) -> List[dict]:
    """
    Returns estimated MCC for all nodes that have shift factors solved,
    sorted by |MCC| descending.

    Use this to build a heatmap of nodal price pressure across the grid.
    """
    iso_cache = cache.for_iso(iso)
    if not iso_cache.is_ready() or iso_cache.solution is None:
        raise HTTPException(status_code=503, detail="Solver not ready.")

    shadow_prices = iso_cache.get_shadow_prices()
    if not shadow_prices:
        return []

    nodes = sorted({n for (n, _) in iso_cache.solution._flat})
    results = []
    for node in nodes:
        mcc = iso_cache.solution.compute_mcc(node, shadow_prices)
        if abs(mcc) >= min_abs_mcc:
            results.append({"node": node, "mcc": round(mcc, 4)})

    results.sort(key=lambda x: abs(x["mcc"]), reverse=True)
    return results[:limit]
