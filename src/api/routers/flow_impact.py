"""
POST /api/v1/flow-impact — Bilateral injection what-if calculator.

"If I inject an extra 50 MW at Node A and buy 50 MW at Node B,
 how much does the flow change on Binding Constraint X?"

  ΔFlow_constraint = (SF_source,c - SF_sink,c) × ΔMW

Positive ΔFlow → injection exacerbates congestion.
Negative ΔFlow → injection relieves congestion.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.models.schemas import FlowImpactRequest, FlowImpactResponse
from src.api.sf_cache import cache

router = APIRouter(prefix="/api/v1/flow-impact", tags=["flow-impact"])

_NEUTRAL_THRESHOLD_MW = 0.1   # flows < 0.1 MW treated as neutral


@router.post("", response_model=FlowImpactResponse)
async def calculate_flow_impact(body: FlowImpactRequest) -> FlowImpactResponse:
    """
    **Dynamic Flow Impact Calculator** — the trader's bilateral what-if tool.

    Supply a source node (injection), sink node (withdrawal), MW amount,
    and the constraint to evaluate. Returns the change in MW flow on that
    constraint and whether your trade exacerbates or relieves congestion.

    Useful for:
    - Evaluating virtual bids/offers before submission
    - Sizing FTR positions
    - Identifying relief transactions when a constraint is near its limit
    """
    if not cache.is_ready() or cache.solution is None:
        raise HTTPException(
            status_code=503,
            detail="Shift factor solver not ready — ingest worker still initializing."
        )

    delta_flow = cache.solution.flow_impact(
        source_node=body.source_node,
        sink_node=body.sink_node,
        mw=body.mw,
        constraint=body.constraint,
    )

    sf_source = cache.solution.get(body.source_node, body.constraint)
    sf_sink = cache.solution.get(body.sink_node, body.constraint)

    if delta_flow is None:
        missing = []
        if sf_source is None:
            missing.append(f"source node '{body.source_node}'")
        if sf_sink is None:
            missing.append(f"sink node '{body.sink_node}'")
        return FlowImpactResponse(
            source_node=body.source_node,
            sink_node=body.sink_node,
            mw=body.mw,
            constraint=body.constraint,
            delta_flow_mw=None,
            direction=None,
            sf_source=sf_source,
            sf_sink=sf_sink,
            solved=False,
            message=(
                f"No shift factor available for {' and '.join(missing)} "
                f"on constraint '{body.constraint}'. "
                "Expand the training window or check node names."
            ),
        )

    # Determine direction
    if abs(delta_flow) < _NEUTRAL_THRESHOLD_MW:
        direction = "neutral"
    elif delta_flow > 0:
        direction = "exacerbates"
    else:
        direction = "relieves"

    # Get shadow price for context
    sp = cache.get_shadow_prices().get(body.constraint, 0.0)
    pnl_impact = delta_flow * sp  # $/hr impact on congestion cost

    if direction == "exacerbates":
        msg = (
            f"Your {body.mw:.0f} MW injection from {body.source_node} → {body.sink_node} "
            f"adds +{abs(delta_flow):.2f} MW of flow on '{body.constraint}' "
            f"(exacerbates congestion). "
            f"At current shadow price ${sp:,.0f}/MWh → +${abs(pnl_impact):,.0f}/hr congestion cost."
        )
    elif direction == "relieves":
        msg = (
            f"Your {body.mw:.0f} MW injection from {body.source_node} → {body.sink_node} "
            f"removes {abs(delta_flow):.2f} MW of flow on '{body.constraint}' "
            f"(relieves congestion). "
            f"At current shadow price ${sp:,.0f}/MWh → -${abs(pnl_impact):,.0f}/hr congestion cost."
        )
    else:
        msg = (
            f"Your {body.mw:.0f} MW injection has negligible impact "
            f"({delta_flow:.3f} MW) on '{body.constraint}'."
        )

    return FlowImpactResponse(
        source_node=body.source_node,
        sink_node=body.sink_node,
        mw=body.mw,
        constraint=body.constraint,
        delta_flow_mw=round(delta_flow, 4),
        direction=direction,
        sf_source=round(sf_source, 6) if sf_source is not None else None,
        sf_sink=round(sf_sink, 6) if sf_sink is not None else None,
        solved=True,
        message=msg,
    )
