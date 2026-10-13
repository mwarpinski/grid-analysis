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
import os
from contextlib import asynccontextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env before any imports that need env vars (e.g. ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

def _load_dotenv():
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.forecast_cache import forecast_cache
from src.api.ingest_worker import ingest_loop, ingest_once
from src.api.routers import analytics, geo, graph, constraints, flow_impact, forecast, mcc, shift_factors, validation, websocket
from src.api.routers import constraint_resolver as resolver_router
from src.analysis.llm_constraint_resolver import resolver as llm_resolver
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
    # Seed each ISO's shift-factor cache from stored history so shift factors are
    # available immediately instead of after days of live accumulation.
    async def _seed_cache(iso: str) -> None:
        from src.api.ingest_worker import _refresh_da_solution
        import pandas as pd
        from src.storage.duckdb_store import store
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=7)
        # Read only the columns the solver needs: the full frames are several times larger.
        from src.api.sf_cache import _CONSTRAINT_COLS, _LMP_COLS
        constraints_df = await asyncio.to_thread(
            lambda: store.query_constraints(start, end, iso=iso, columns=_CONSTRAINT_COLS)
        )
        lmp_df = await asyncio.to_thread(
            lambda: store.query_lmp(start, end, iso=iso, columns=_LMP_COLS)
        )
        await asyncio.to_thread(cache.for_iso(iso).seed, constraints_df, lmp_df)
        if iso == "spp":
            await _refresh_da_solution()

    for _iso in ("spp", "miso"):
        try:
            await asyncio.wait_for(_seed_cache(_iso), timeout=120.0)
        except Exception as e:
            logger.warning(f"[{_iso.upper()}] Could not seed shift-factor cache from history: {e}")

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

    # Phase 14 — start LLM constraint resolver (delayed, non-blocking)
    async def _delayed_resolver():
        await asyncio.sleep(60)  # give ingest worker time to settle first
        logger.info("Phase 14: Starting LLM constraint resolver background pass...")
        result = await llm_resolver.resolve_all_pending()
        logger.info(f"Phase 14: LLM resolver complete — {result}")

    if llm_resolver.has_api_key:
        asyncio.create_task(_delayed_resolver(), name="llm_constraint_resolver")
        logger.info("Phase 14: LLM constraint resolver scheduled (starts in 60s).")
    else:
        logger.info(
            "Phase 14: ANTHROPIC_API_KEY not set — constraint resolver inactive. "
            "Add to .env to enable."
        )

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
    allow_credentials=False,   # "*" with credentials is rejected by browsers; no cookies are used
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
app.include_router(analytics.router)
app.include_router(geo.router)
app.include_router(graph.router)
app.include_router(forecast.router)
app.include_router(resolver_router.router)

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

    spp_cache = cache.for_iso("spp")
    miso_cache = cache.for_iso("miso")

    return {
        "ready": spp_cache.is_ready() or miso_cache.is_ready(),
        "spp": {
            "ready": spp_cache.is_ready(),
            "last_updated": str(spp_cache.last_updated) if spp_cache.last_updated else None,
            "last_solved": str(spp_cache.last_solved) if spp_cache.last_solved else None,
            "current_regime": sorted(list(spp_cache.current_regime)),
            "n_regime_constraints": len(spp_cache.current_regime),
            "n_solved_pairs": len(spp_cache.solution._flat) if spp_cache.solution else 0,
            "history_constraint_rows": len(spp_cache._constraints_history),
            "history_lmp_rows": len(spp_cache._lmp_history),
        },
        "miso": {
            "ready": miso_cache.is_ready(),
            "last_updated": str(miso_cache.last_updated) if miso_cache.last_updated else None,
            "last_solved": str(miso_cache.last_solved) if miso_cache.last_solved else None,
            "current_regime": sorted(list(miso_cache.current_regime)),
            "n_regime_constraints": len(miso_cache.current_regime),
            "n_solved_pairs": len(miso_cache.solution._flat) if miso_cache.solution else 0,
            "history_constraint_rows": len(miso_cache._constraints_history),
            "history_lmp_rows": len(miso_cache._lmp_history),
        },
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
