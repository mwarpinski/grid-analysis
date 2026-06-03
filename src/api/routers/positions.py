"""
Portfolio Positions — Phase 11.

POST /api/v1/positions          Upload / replace portfolio positions.
GET  /api/v1/positions          List current positions in memory.
GET  /api/v1/portfolio/risk     Aggregate P&L risk across all positions.

The position store is kept as an in-memory list (reset on server restart).
It intentionally mirrors the simple use-case from the requirements:
the trader uploads a CSV of (asset_name, type, node, MW) and gets back
a real-time risk table.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.models.schemas import (
    Position,
    PortfolioRiskResponse,
    PositionRiskRecord,
    PositionsUploadResponse,
)
from src.api.sf_cache import cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["portfolio"])

# ---------------------------------------------------------------------------
# In-memory position store (process-lifetime; cleared on restart)
# ---------------------------------------------------------------------------
_position_store: List[Position] = []

_NEUTRAL_THRESHOLD_MW = 0.01   # flows below 0.01 MW treated as neutral


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/positions", response_model=PositionsUploadResponse)
async def upload_positions(positions: List[Position]) -> PositionsUploadResponse:
    """
    **Upload portfolio positions.**

    Replaces the current in-memory position store with the submitted list.

    Each position:
    - ``asset_name`` — descriptive label (e.g. "WindFarm_01")
    - ``asset_type`` — one of: gen, load, virtual_buy, virtual_sell, ftr, other
    - ``node``       — settlement location name (must match SF-solved nodes)
    - ``mw_signed``  — positive = injection/generation, negative = withdrawal/load

    Example payload::

        [
          {"asset_name": "WindFarm_01", "asset_type": "gen",
           "node": "WEST_NODE", "mw_signed": 120},
          {"asset_name": "Industrial_Load", "asset_type": "load",
           "node": "HUB_NORTH", "mw_signed": -85}
        ]
    """
    global _position_store
    _position_store = positions
    logger.info(f"Positions updated: {len(positions)} entries loaded.")
    return PositionsUploadResponse(accepted=len(positions), positions=positions)


@router.get("/positions", response_model=List[Position])
async def get_positions() -> List[Position]:
    """Returns the current in-memory portfolio position store."""
    return _position_store


@router.delete("/positions")
async def clear_positions() -> dict:
    """Clears all positions from the portfolio store."""
    global _position_store
    count = len(_position_store)
    _position_store = []
    logger.info("Position store cleared.")
    return {"cleared": count}


@router.get("/portfolio/risk", response_model=PortfolioRiskResponse)
async def get_portfolio_risk(
    constraint: str = Query(
        ...,
        description="Constraint name to evaluate portfolio exposure against",
    ),
    shadow_price_override: Optional[float] = Query(
        None, ge=0,
        description="Override shadow price ($/MWh). If omitted, uses live price from ingest cache.",
    ),
    iso: str = Query(
        "spp",
        description="ISO namespace: spp or miso",
    ),
) -> PortfolioRiskResponse:
    """
    **Portfolio P&L Risk Aggregation** — the core REQ-3 deliverable.

    For each position in the store, computes:

    .. code-block::

        net_mw_flow    = GSF_node,constraint × mw_signed
        pnl_risk_rate  = net_mw_flow × shadow_price   ($/hr)

    Returns a sorted table of all positions plus the portfolio total.

    The ``alert_message`` field gives the plain-English summary:

        *Portfolio net: +13.5 MW on LN HARBINE — $13,500/hr at $1,000 shadow price.*
    """
    iso_cache = cache.for_iso(iso)
    if not iso_cache.is_ready() or iso_cache.solution is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Shift factor solver not ready — ingest worker still initializing. "
                "Try again in ~30 seconds after the server warms up."
            ),
        )

    if not _position_store:
        raise HTTPException(
            status_code=400,
            detail=(
                "No positions loaded. POST to /api/v1/positions first."
            ),
        )

    # Resolve shadow price
    sp = shadow_price_override
    if sp is None:
        sp = iso_cache.get_shadow_prices().get(constraint)
    if sp is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No shadow price available for constraint '{constraint}' in {iso.upper()}. "
                "Either the constraint is not currently binding, or override via "
                "?shadow_price_override=<value>."
            ),
        )

    # ── Build per-position risk records ────────────────────────────────────
    records: List[PositionRiskRecord] = []
    total_net_mw = 0.0

    for pos in _position_store:
        gsf = iso_cache.solution.get(pos.node, constraint)
        net_mw = (gsf * pos.mw_signed) if gsf is not None else None
        pnl = (net_mw * sp) if net_mw is not None else None

        if net_mw is None:
            direction = None
        elif abs(net_mw) < _NEUTRAL_THRESHOLD_MW:
            direction = "neutral"
        elif net_mw > 0:
            direction = "exacerbates"
        else:
            direction = "relieves"

        if net_mw is not None:
            total_net_mw += net_mw

        records.append(PositionRiskRecord(
            asset_name=pos.asset_name,
            asset_type=pos.asset_type,
            node=pos.node,
            mw_signed=pos.mw_signed,
            constraint=constraint,
            gsf=round(gsf, 6) if gsf is not None else None,
            net_mw_flow=round(net_mw, 4) if net_mw is not None else None,
            shadow_price=sp,
            pnl_risk_rate=round(pnl, 2) if pnl is not None else None,
            direction=direction,
        ))

    # Sort by |net_mw_flow| descending — biggest contributors first
    records.sort(key=lambda r: abs(r.net_mw_flow or 0), reverse=True)

    total_pnl = total_net_mw * sp
    direction_word = (
        "exacerbates" if total_net_mw > _NEUTRAL_THRESHOLD_MW
        else "relieves" if total_net_mw < -_NEUTRAL_THRESHOLD_MW
        else "neutral impact on"
    )
    alert_msg = (
        f"Portfolio net: {total_net_mw:+.2f} MW on '{constraint}' "
        f"({direction_word} congestion). "
        f"At ${sp:,.0f}/MWh shadow price → ${total_pnl:+,.0f}/hr run rate."
    )

    return PortfolioRiskResponse(
        constraint=constraint,
        shadow_price=sp,
        total_net_mw_flow=round(total_net_mw, 4),
        total_pnl_risk_rate=round(total_pnl, 2),
        alert_message=alert_msg,
        positions=records,
    )
