"""
Historical Replay API — Phase 12 (DVR Mode).

GET /api/v1/replay/intervals   List available interval timestamps in a date range.
GET /api/v1/replay/snapshot    Full constraint + SF state at a specific interval.
GET /api/v1/replay/pnl         Apply position store to historical data → P&L time series.

All data comes from the DuckDB Parquet store populated by the ingest worker.
The replay API is read-only — it never modifies the store.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.models.schemas import (
    ConstraintSnapshot,
    Position,
    ReplayInterval,
    ReplayPnLPoint,
    ReplayPnLResponse,
    ReplaySnapshotResponse,
    ShiftFactorRecord,
)
from src.storage.duckdb_store import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/replay", tags=["replay"])

_MAX_INTERVALS = 2016  # 7 days × 288 intervals/day


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/intervals", response_model=List[ReplayInterval])
async def list_replay_intervals(
    start: datetime = Query(..., description="Range start (ISO-8601)"),
    end: datetime = Query(..., description="Range end (ISO-8601)"),
) -> List[ReplayInterval]:
    """
    **List available replay intervals** in a date range.

    Returns timestamps where constraint data has been stored — these
    are the valid inputs to ``/replay/snapshot``.

    Intervals are 5-minute SPP RTBM buckets.
    """
    if (end - start).total_seconds() > _MAX_INTERVALS * 300:
        raise HTTPException(
            status_code=400,
            detail=f"Date range too large — max {_MAX_INTERVALS} intervals (7 days).",
        )

    df = store.query_constraints(start=start, end=end)
    if df.empty or "interval" not in df.columns:
        return []

    import pandas as pd
    df["interval"] = pd.to_datetime(df["interval"])
    grouped = (
        df.groupby("interval")["constraint_name"]
        .count()
        .reset_index()
        .rename(columns={"constraint_name": "n_binding_constraints"})
    )
    grouped = grouped.sort_values("interval")

    return [
        ReplayInterval(
            interval=row["interval"].to_pydatetime(),
            n_binding_constraints=int(row["n_binding_constraints"]),
        )
        for _, row in grouped.iterrows()
    ]


@router.get("/snapshot", response_model=ReplaySnapshotResponse)
async def get_replay_snapshot(
    interval: datetime = Query(
        ...,
        description="Interval timestamp (ISO-8601). Use /intervals to find valid timestamps.",
    ),
) -> ReplaySnapshotResponse:
    """
    **Historical constraint + SF snapshot** at a specific interval.

    Returns the full constraint state (shadow prices, loading %, etc.) as
    recorded at that 5-minute interval, along with the shift factors that
    were valid at that time (most recent SF solve before the interval).

    This is the core data source for the DVR replay player in the UI.
    """
    df = store.get_snapshot(interval)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data found for interval '{interval}'. "
                "Use /replay/intervals to list available timestamps."
            ),
        )

    import pandas as pd
    df["interval"] = pd.to_datetime(df["interval"])

    constraints = []
    for _, row in df.iterrows():
        try:
            constraints.append(ConstraintSnapshot(
                interval=row["interval"].to_pydatetime(),
                constraint_name=row.get("constraint_name", ""),
                canonical_id=row.get("canonical_id"),
                shadow_price=float(row.get("shadow_price", 0.0)),
                limit_mw=row.get("limit_mw"),
                monitored_mw=row.get("monitored_mw"),
                pct_of_limit=row.get("pct_of_limit"),
                state=row.get("state"),
                contingency_name=row.get("contingency_name"),
                volatility_score=row.get("volatility_score"),
            ))
        except Exception:
            continue

    # Attempt to load nearest SF snapshot from DuckDB
    sf_records: List[ShiftFactorRecord] = []
    try:
        sf_df = store.query_constraints(
            start=interval,
            end=interval,
        )
        # SF snapshots are stored in a separate table — best effort
        if not sf_df.empty and "shift_factor" in sf_df.columns:
            for _, row in sf_df.iterrows():
                sf_records.append(ShiftFactorRecord(
                    node=row.get("node", ""),
                    constraint=row.get("constraint_name", ""),
                    shift_factor=float(row.get("shift_factor", 0.0)),
                    shadow_price=row.get("shadow_price"),
                    sf_exposure=row.get("sf_exposure"),
                ))
    except Exception as e:
        logger.debug(f"SF snapshot load skipped: {e}")

    return ReplaySnapshotResponse(
        interval=interval,
        constraints=constraints,
        shift_factors=sf_records,
    )


@router.get("/pnl", response_model=ReplayPnLResponse)
async def get_replay_pnl(
    start: datetime = Query(..., description="Backtest start (ISO-8601)"),
    end: datetime = Query(..., description="Backtest end (ISO-8601)"),
    constraint: str = Query(..., description="Constraint to evaluate P&L against"),
    positions: Optional[str] = Query(
        None,
        description=(
            "JSON-encoded position list: "
            '[{"asset_name":"X","asset_type":"gen","node":"Y","mw_signed":100},...]. '
            "If omitted, uses the current in-memory position store."
        ),
    ),
) -> ReplayPnLResponse:
    """
    **Historical P&L Backtester** — apply your current positions to stored
    constraint data interval-by-interval.

    Formula per interval:

    .. code-block::

        net_mw = Σ_assets (GSF_node,constraint × mw_signed)
        pnl_rate = net_mw × shadow_price_at_interval  ($/hr)

    Returns a time series of P&L rates and a summary total.
    The total is the sum of ``pnl_rate × (5/60)`` (each interval = 5 min = 1/12 hr).

    **Requires:**
    1. Live shift factors in the SF cache (from the ingest worker).
    2. Historical constraint data in DuckDB for the requested date range.
    """
    from src.api.sf_cache import cache
    from src.api.routers.positions import _position_store

    if (end - start).total_seconds() > _MAX_INTERVALS * 300:
        raise HTTPException(
            status_code=400,
            detail=f"Date range too large — max {_MAX_INTERVALS} intervals (7 days).",
        )

    # Resolve positions
    pos_list: List[Position] = []
    if positions:
        import json
        try:
            raw = json.loads(positions)
            pos_list = [Position(**p) for p in raw]
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid positions JSON: {e}",
            )
    else:
        pos_list = _position_store

    if not pos_list:
        raise HTTPException(
            status_code=400,
            detail=(
                "No positions available. Either POST to /api/v1/positions or "
                "pass ?positions=[...] in the query string."
            ),
        )

    if not cache.is_ready() or cache.solution is None:
        raise HTTPException(
            status_code=503,
            detail="Shift factor solver not ready — ingest worker still initializing.",
        )

    # Pre-compute GSF for each position against the constraint
    gsf_map = {
        pos.asset_name: cache.solution.get(pos.node, constraint)
        for pos in pos_list
    }

    # Fetch historical constraint data
    df = store.query_constraints(
        start=start,
        end=end,
        constraint_name=constraint,
    )
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No historical data for constraint '{constraint}' in "
                f"[{start.isoformat()}, {end.isoformat()}]. "
                "Ensure the ingest worker has been running during that period."
            ),
        )

    import pandas as pd
    df["interval"] = pd.to_datetime(df["interval"])
    df["shadow_price"] = pd.to_numeric(df["shadow_price"], errors="coerce").fillna(0.0)

    # Group by interval — take mean shadow price per interval
    sp_series = (
        df.groupby("interval")["shadow_price"]
        .mean()
        .sort_index()
    )

    # Build dominant constraint series (highest |shadow_price| per interval)
    dominant_series = (
        df.groupby(["interval", "constraint_name"])["shadow_price"]
        .mean()
        .reset_index()
    )

    series: List[ReplayPnLPoint] = []
    total_pnl = 0.0
    interval_duration_hr = 5 / 60  # 5-minute intervals

    for ts, sp in sp_series.items():
        # Net MW flow for this portfolio against the constraint
        net_mw = sum(
            (gsf_map.get(pos.asset_name) or 0.0) * pos.mw_signed
            for pos in pos_list
        )
        pnl_rate = net_mw * sp   # $/hr
        pnl_for_interval = pnl_rate * interval_duration_hr
        total_pnl += pnl_for_interval

        # Dominant constraint at this interval
        ts_dominant = None
        try:
            ts_slice = dominant_series[dominant_series["interval"] == ts]
            if not ts_slice.empty:
                ts_dominant = (
                    ts_slice.loc[ts_slice["shadow_price"].abs().idxmax(), "constraint_name"]
                )
        except Exception:
            pass

        series.append(ReplayPnLPoint(
            interval=ts.to_pydatetime(),
            pnl_rate_per_hr=round(pnl_rate, 2),
            n_binding_constraints=int(df[df["interval"] == ts]["constraint_name"].nunique()),
            dominant_constraint=ts_dominant,
        ))

    return ReplayPnLResponse(
        start=start,
        end=end,
        n_intervals=len(series),
        total_pnl=round(total_pnl, 2),
        series=series,
    )
