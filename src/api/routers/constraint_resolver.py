"""
/api/v1/resolver — LLM Constraint Name Resolution (Phase 14)

Exposes the LLM constraint resolver results and control endpoints.

Endpoints
---------
GET  /api/v1/resolver/status
    Resolver health: API key configured, queue depth, API call count.

GET  /api/v1/resolver/names
    All resolved constraint names with metadata.
    Query params: ?confidence=HIGH&equipment_type=LINE&limit=200

GET  /api/v1/resolver/names/{constraint}
    Single constraint lookup.

POST /api/v1/resolver/resolve
    Trigger a background resolution pass.
    Body: {"force_refresh": false}
    Returns immediately — resolution runs in background.

GET  /api/v1/resolver/summary
    Aggregate stats: how many resolved by confidence tier, equipment type.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from src.analysis.llm_constraint_resolver import resolver, ResolvedConstraint

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/resolver", tags=["constraint-resolver"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ResolvedConstraintResponse(BaseModel):
    constraint:     str
    from_sub:       Optional[str]
    to_sub:         Optional[str]
    voltage_kv:     Optional[int]
    equipment_type: Optional[str]
    confidence:     str
    reasoning:      Optional[str]
    resolved_at:    Optional[str]
    error:          Optional[str]

    @classmethod
    def from_rc(cls, rc: ResolvedConstraint) -> "ResolvedConstraintResponse":
        return cls(
            constraint     = rc.constraint,
            from_sub       = rc.from_sub,
            to_sub         = rc.to_sub,
            voltage_kv     = rc.voltage_kv,
            equipment_type = rc.equipment_type,
            confidence     = rc.confidence,
            reasoning      = rc.reasoning,
            resolved_at    = rc.resolved_at,
            error          = rc.error,
        )


class ResolveRequest(BaseModel):
    force_refresh: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
async def resolver_status():
    """
    Returns resolver health information:
    - Whether an Anthropic API key is configured
    - How many names are resolved vs. pending
    - API call count and model in use
    """
    return resolver.status()


@router.get("/names", response_model=List[ResolvedConstraintResponse])
async def list_resolved_names(
    confidence: Optional[str] = Query(
        None,
        description="Filter by confidence tier: HIGH, MEDIUM, LOW, UNKNOWN"
    ),
    equipment_type: Optional[str] = Query(
        None,
        description="Filter by equipment type: LINE, TRANSFORMER, INTERFACE, TIE, HUB"
    ),
    limit: int = Query(500, ge=1, le=2000, description="Max records to return"),
):
    """
    Returns all constraint names that have been resolved by the LLM,
    with their decoded physical metadata.

    Only 'named' constraints (pure-alpha SPP codes) are resolved.
    TMP/TEMP/TP contingency monitors are skipped — they embed bus IDs
    that are not meaningful without the full SPP bus database.
    """
    all_resolved = resolver.lookup_all()

    # Apply filters
    if confidence:
        all_resolved = [r for r in all_resolved if r.confidence == confidence.upper()]
    if equipment_type:
        all_resolved = [r for r in all_resolved if r.equipment_type == equipment_type.upper()]

    all_resolved = all_resolved[:limit]
    return [ResolvedConstraintResponse.from_rc(r) for r in all_resolved]


@router.get("/names/{constraint}", response_model=ResolvedConstraintResponse)
async def get_resolved_name(constraint: str):
    """
    Look up a single constraint name in the resolver cache.

    Returns 404 if the constraint has not been resolved yet.
    Use POST /api/v1/resolver/resolve to trigger resolution.
    """
    rc = resolver.lookup(constraint)
    if rc is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Constraint '{constraint}' has not been resolved yet. "
                "POST to /api/v1/resolver/resolve to trigger a resolution pass."
            )
        )
    return ResolvedConstraintResponse.from_rc(rc)


@router.post("/resolve")
async def trigger_resolution(
    body: ResolveRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger a background LLM resolution pass.

    Scans all constraint parquet files for unresolved named constraints
    and queues them for Claude resolution. Returns immediately — resolution
    runs in the background.

    Set `force_refresh=true` to re-resolve names that already have a cached
    result (useful after system prompt improvements).

    Requires ANTHROPIC_API_KEY to be set in the environment or .env file.
    """
    if not resolver.has_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "ANTHROPIC_API_KEY is not configured. "
                "Add it to grid-analysis/.env or set it as an environment variable."
            )
        )

    if resolver._running:
        return {
            "status": "already_running",
            "message": "A resolution pass is already in progress.",
        }

    # Run in background — don't block the response
    async def _run():
        await resolver.resolve_all_pending(force_refresh=body.force_refresh)

    background_tasks.add_task(_run)

    return {
        "status": "started",
        "force_refresh": body.force_refresh,
        "message": (
            "Resolution pass started in background. "
            "Poll GET /api/v1/resolver/status to track progress."
        ),
    }


@router.get("/summary")
async def resolution_summary():
    """
    Aggregate summary of resolved constraints:
    - Breakdown by confidence tier (HIGH / MEDIUM / LOW)
    - Breakdown by equipment type (LINE / TRANSFORMER / etc.)
    - Sample of highest-confidence resolutions
    """
    all_resolved = resolver.lookup_all()
    successful   = [r for r in all_resolved if not r.error]
    errors       = [r for r in all_resolved if r.error]

    # Confidence distribution
    by_confidence: dict[str, int] = {}
    for r in successful:
        by_confidence[r.confidence] = by_confidence.get(r.confidence, 0) + 1

    # Equipment type distribution
    by_type: dict[str, int] = {}
    for r in successful:
        etype = r.equipment_type or "UNKNOWN"
        by_type[etype] = by_type.get(etype, 0) + 1

    # Top HIGH confidence examples
    high_confidence = [
        r for r in successful if r.confidence == "HIGH"
    ][:10]

    return {
        "total_in_cache":       len(all_resolved),
        "successfully_resolved": len(successful),
        "errors":               len(errors),
        "by_confidence":        by_confidence,
        "by_equipment_type":    by_type,
        "resolver_status":      resolver.status(),
        "sample_high_confidence": [
            {
                "constraint":     r.constraint,
                "from_sub":       r.from_sub,
                "to_sub":         r.to_sub,
                "voltage_kv":     r.voltage_kv,
                "equipment_type": r.equipment_type,
                "reasoning":      r.reasoning,
            }
            for r in high_confidence
        ],
        "errors_sample": [
            {"constraint": r.constraint, "error": r.error}
            for r in errors[:5]
        ],
    }
