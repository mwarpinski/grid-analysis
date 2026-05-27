"""
SPP Transmission Congestion Analysis — FastAPI Server

Startup
-------
::

    # From grid-analysis/ with the venv activated
    uvicorn src.api.server:app --reload --port 8000

    # Or via the CLI helper
    python -m src.api.server

Endpoints
---------
REST
    GET  /api/v1/constraints/live         Live binding constraint snapshot
    GET  /api/v1/constraints/near-limit   Lines at ≥95% of thermal limit
    GET  /api/v1/shift-factors            Single node/constraint SF lookup
    GET  /api/v1/shift-factors/table      Full exposure-sorted SF table
    POST /api/v1/flow-impact              Bilateral injection what-if
    GET  /api/v1/mcc                      Pre-clearing MCC for a node
    GET  /api/v1/mcc/all-nodes            MCC heatmap for all nodes
    POST /api/v1/positions                Upload portfolio positions
    GET  /api/v1/positions                List current positions
    GET  /api/v1/portfolio/risk           Portfolio P&L risk aggregation

    Phase 13 — Price Forecasting
    GET  /api/v1/forecast/{constraint}          Next-interval shadow price + horizon
    GET  /api/v1/forecast/{constraint}/horizon  N-step forecast path
    GET  /api/v1/forecast/all                   All binding constraint forecasts
    GET  /api/v1/forecast/models                Model registry (MAE, R², fallback flag)
    POST /api/v1/forecast/retrain               Trigger background retrain (all)
    POST /api/v1/forecast/retrain/{constraint}  Retrain one model on-demand

WebSocket
    WS   /ws/constraints                  Live constraint push stream
    WS   /ws/portfolio                    Live portfolio P&L stream

Docs
    GET  /docs                            Swagger UI (auto-generated)
    GET  /redoc                           ReDoc UI

Static
    GET  /                                Trading Dashboard UI (Phase 10)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.forecast_cache import forecast_cache
from src.api.ingest_worker import ingest_loop, ingest_once
from src.api.routers import constraints, flow_impact, forecast, mcc, shift_factors, validation, websocket
from src.api.sf_cache import cache

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup:
      1. Run one immediate ingest cycle so the cache is populated
         before the first request arrives.
      2. Start the background ingest loop.

    On shutdown:
      - Cancel the ingest task cleanly.
    """
    logger.info("Server starting up — running initial ingest cycle...")
    try:
        await asyncio.wait_for(ingest_once(), timeout=60.0)
    except asyncio.TimeoutError:
        logger.warning("Initial ingest timed out — server will start with empty cache.")
    except Exception as e:
        logger.warning(f"Initial ingest failed (non-fatal): {e}")

    # Start background polling loop
    task = asyncio.create_task(ingest_loop(), name="ingest_worker")
    logger.info("Ingest worker task started.")

    # Phase 13 — start forecast model training and background retrain loop
    logger.info("Starting forecast cache (Phase 13)...")
    await forecast_cache.startup()

    yield  # Server is running

    logger.info("Server shutting down — cancelling ingest worker and forecast retrain loop...")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await forecast_cache.shutdown()
    logger.info("Ingest worker and forecast retrain loop stopped.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SPP Transmission Congestion Analysis API",
    description=(
        "Real-time shift factor (PTDF) computation, congestion exposure, "
        "bilateral flow-impact calculator, and portfolio P&L risk for "
        "SPP RTBM market participants."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for internal use; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(constraints.router)
app.include_router(shift_factors.router)
app.include_router(flow_impact.router)
app.include_router(mcc.router)
app.include_router(websocket.router)
app.include_router(validation.router)
app.include_router(forecast.router)

# Phase 11 — added when portfolio router is built
try:
    from src.api.routers import positions as positions_router
    app.include_router(positions_router.router)
except ImportError:
    pass

# Phase 12 — added when replay router is built
try:
    from src.api.routers import replay as replay_router
    app.include_router(replay_router.router)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Health / status endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health_check():
    """Basic liveness probe."""
    return {"status": "ok"}


@app.get("/api/v1/status", tags=["meta"])
async def get_status():
    """
    Returns the current state of the ingest worker, SF cache, and on-disk
    storage. Use this to check if the server is ready to serve trading data.
    """
    from src.storage.duckdb_store import store

    return {
        "ready": cache.is_ready(),
        "last_updated": str(cache.last_updated) if cache.last_updated else None,
        "last_solved": str(cache.last_solved) if cache.last_solved else None,
        "current_regime": sorted(list(cache.current_regime)),
        "n_regime_constraints": len(cache.current_regime),
        "n_solved_pairs": len(cache.solution._flat) if cache.solution else 0,
        "history_constraint_rows": len(cache._constraints_history),
        "history_lmp_rows": len(cache._lmp_history),
        "storage": store.storage_stats(),
    }


# ---------------------------------------------------------------------------
# Static UI (Phase 10 — served when ui/ directory exists)
# ---------------------------------------------------------------------------

_UI_DIR = Path(__file__).parent.parent.parent / "ui"
if _UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")
    logger.info(f"Serving Trading Dashboard UI from {_UI_DIR}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
