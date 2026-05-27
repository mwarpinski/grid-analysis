"""
Forecast Router — Phase 13.

Endpoints
---------
GET  /api/v1/forecast/all                   Predictions for all currently-binding constraints
GET  /api/v1/forecast/models                Model registry status (fitted models, MAE, R², etc.)
POST /api/v1/forecast/retrain               Trigger an on-demand retrain of all models
POST /api/v1/forecast/retrain/{constraint}  Retrain a single constraint's model
GET  /api/v1/forecast/{constraint}          Next-interval prediction + 6-step horizon
GET  /api/v1/forecast/{constraint}/horizon  N-step-ahead forecast (configurable)

IMPORTANT: Static routes (/all, /models, /retrain) MUST be registered before
the dynamic /{constraint} route to avoid FastAPI matching "all" and "models"
as constraint names.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from src.api.forecast_cache import forecast_cache
from src.api.sf_cache import cache as sf_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])


# ---------------------------------------------------------------------------
# Static routes FIRST — must precede /{constraint} to avoid shadowing
# ---------------------------------------------------------------------------

@router.get("/all", summary="Forecasts for all currently-binding constraints")
async def get_all_forecasts(
    horizon: int = Query(default=6, ge=1, le=24),
):
    """
    Returns next-interval predictions for every constraint in the current
    binding regime. Useful for a dashboard "congestion outlook" panel.

    Constraints with no trained model are omitted.
    """
    if not sf_cache.current_regime:
        return {"constraints": [], "note": "No binding constraints in current regime."}

    results = []
    for constraint in sorted(sf_cache.current_regime):
        fc = forecast_cache.get_forecaster(constraint)
        if fc is None or not fc.is_fitted:
            continue

        sp_now = 0.0
        if sf_cache.latest_interval:
            sp_now = sf_cache.latest_interval.shadow_prices.get(constraint, 0.0)

        current_snapshot = {
            "constraint": constraint,
            "shadow_price": sp_now,
            "interval": sf_cache.latest_interval.interval if sf_cache.latest_interval else None,
        }

        next_price = fc.predict(current_snapshot)
        horizon_steps = fc.predict_horizon(n_steps=horizon, current_snapshot=current_snapshot)
        meta = forecast_cache.get_model_meta(constraint)

        results.append({
            "constraint": constraint,
            "current_shadow_price": sp_now,
            "next_price": round(next_price, 4),
            "is_binding_forecast": abs(next_price) > 0.5,
            "price_direction": (
                "increasing" if next_price > sp_now * 1.05
                else "decreasing" if next_price < sp_now * 0.95
                else "stable"
            ),
            "horizon": horizon_steps,
            "model_fallback": meta.get("fallback", True) if meta else True,
            "mae_cv": meta.get("mae_cv") if meta else None,
        })

    return {
        "n_binding": len(sf_cache.current_regime),
        "n_forecasted": len(results),
        "constraints": results,
    }


@router.get("/models", summary="Model registry — all trained forecasters")
async def get_model_registry():
    """
    Returns training metadata for every constraint forecaster in the registry.
    Includes whether the model is using ML or a persistence fallback, plus
    holdout MAE and R² for quality monitoring.
    """
    models = forecast_cache.list_models()
    return {
        "n_models": len(models),
        "last_retrained": (
            forecast_cache.last_retrained.isoformat()
            if forecast_cache.last_retrained else None
        ),
        "retrain_count": forecast_cache.retrain_count,
        "models": models,
    }


@router.post("/retrain", summary="Retrain all constraint forecasters")
async def retrain_all_models(background_tasks: BackgroundTasks):
    """
    Triggers an asynchronous retrain of all constraint models. Returns
    immediately — the retrain runs in the background.

    Use GET /api/v1/forecast/models to poll for completion.
    """
    background_tasks.add_task(_background_retrain_all)
    return {
        "status": "accepted",
        "message": "Retrain started in background. Poll /api/v1/forecast/models for status.",
    }


@router.post("/retrain/{constraint}", summary="Retrain one constraint's forecaster")
async def retrain_one_model(constraint: str):
    """
    Synchronously retrains the model for a single constraint and returns
    updated fit metadata. Useful for validating a constraint after backfilling data.
    """
    try:
        result = await forecast_cache.retrain_one(constraint)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")

    return {
        "constraint": constraint,
        "fitted": True,
        "fallback": result.fallback,
        "n_training_rows": result.n_training_rows,
        "mae_cv": round(result.mae_cv, 3) if not result.fallback else None,
        "r2_cv": round(result.r2_cv, 3) if not result.fallback else None,
        "trained_at": result.trained_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Dynamic routes LAST — /{constraint} must come after all static paths
# ---------------------------------------------------------------------------

@router.get("/{constraint}", summary="Shadow price forecast for one constraint")
async def get_constraint_forecast(
    constraint: str,
    horizon: int = Query(default=6, ge=1, le=48, description="Intervals to forecast (5-min each)"),
):
    """
    Returns a next-interval point forecast plus a multi-step price path.

    - **next_price**: predicted shadow price for the next 5-min interval ($/MWh)
    - **horizon**: list of step-ahead predictions with confidence intervals
    - **model_meta**: training quality stats (MAE, R², n_training_rows)

    When fewer than 20 training rows are available the model uses a
    **persistence forecast** (last observed shadow price) — still useful as
    a conservative baseline.
    """
    fc = forecast_cache.get_forecaster(constraint)
    if fc is None or not fc.is_fitted:
        # Try retraining on-demand
        try:
            await forecast_cache.retrain_one(constraint)
            fc = forecast_cache.get_forecaster(constraint)
        except Exception as e:
            logger.warning(f"On-demand retrain for {constraint} failed: {e}")

    if fc is None or not fc.is_fitted:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No forecast model for constraint '{constraint}'. "
                "Ensure historical data is available and the backfill script has run."
            ),
        )

    # Build current snapshot from SF cache
    current_snapshot = None
    if sf_cache.latest_interval:
        sp = sf_cache.latest_interval.shadow_prices.get(constraint, 0.0)
        current_snapshot = {
            "constraint": constraint,
            "shadow_price": sp,
            "interval": sf_cache.latest_interval.interval,
        }

    next_price = fc.predict(current_snapshot)
    horizon_steps = fc.predict_horizon(n_steps=horizon, current_snapshot=current_snapshot)
    meta = forecast_cache.get_model_meta(constraint)

    return {
        "constraint": constraint,
        "next_price": round(next_price, 4),
        "is_binding_forecast": abs(next_price) > 0.5,
        "horizon": horizon_steps,
        "model_meta": meta,
        "current_shadow_price": current_snapshot["shadow_price"] if current_snapshot else None,
    }


@router.get("/{constraint}/horizon", summary="Multi-step shadow price forecast")
async def get_constraint_horizon(
    constraint: str,
    steps: int = Query(default=12, ge=1, le=48, description="Number of 5-min steps"),
):
    """
    Returns a detailed multi-step forecast path.
    Each step includes: interval timestamp, predicted shadow price, binding
    probability, and a confidence interval.
    """
    fc = forecast_cache.get_forecaster(constraint)
    if fc is None or not fc.is_fitted:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast model for '{constraint}'. Try POST /api/v1/forecast/retrain/{constraint}.",
        )

    current_snapshot = None
    if sf_cache.latest_interval:
        sp = sf_cache.latest_interval.shadow_prices.get(constraint, 0.0)
        current_snapshot = {
            "constraint": constraint,
            "shadow_price": sp,
            "interval": sf_cache.latest_interval.interval,
        }

    horizon = fc.predict_horizon(n_steps=steps, current_snapshot=current_snapshot)
    return {
        "constraint": constraint,
        "steps": steps,
        "interval_minutes": 5,
        "forecast": horizon,
    }


async def _background_retrain_all() -> None:
    """Wrapped for use as a BackgroundTask."""
    try:
        await forecast_cache.retrain_all()
    except Exception as e:
        logger.error(f"Background retrain failed: {e}", exc_info=True)
