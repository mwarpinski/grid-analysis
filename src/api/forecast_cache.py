"""
Forecast Cache — per-constraint model registry + background retraining.

One ShadowPriceForecaster is maintained per constraint. Retraining is
triggered:
  1. On startup (loads history from Parquet files)
  2. Every RETRAIN_INTERVAL_MINUTES in a background asyncio task
  3. On-demand via POST /api/v1/forecast/retrain

Thread / async safety
---------------------
Retraining is CPU-bound. We run it in an executor (threadpool) so the
asyncio event loop stays responsive during training.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.analysis.shadow_price_forecaster import FitResult, ShadowPriceForecaster
from src.api.sf_cache import cache as sf_cache

logger = logging.getLogger(__name__)

# How often to retrain all models (in minutes)
RETRAIN_INTERVAL_MINUTES: int = 30

# Where Parquet time-series files live
TIMESERIES_DIR = Path(__file__).parent.parent / "data" / "timeseries"


# ---------------------------------------------------------------------------
# Helpers — load history from Parquet
# ---------------------------------------------------------------------------

def _load_constraint_history(
    constraint_name: str,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Load the shadow price history for one constraint from Parquet files.

    Returns a DataFrame with columns: ``interval``, ``shadow_price``,
    optionally ``pct_of_limit``. All intervals are included (zero SP when
    the constraint was not binding — important for training on non-binding hours).
    """
    constraints_dir = TIMESERIES_DIR / "constraints"
    if not constraints_dir.exists():
        return pd.DataFrame(columns=["interval", "shadow_price"])

    cutoff = datetime.now() - timedelta(days=lookback_days)
    dfs: List[pd.DataFrame] = []

    for path in sorted(constraints_dir.glob("constraints_*.parquet")):
        try:
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if "interval" not in df.columns or "constraint_name" not in df.columns:
                continue
            df["interval"] = pd.to_datetime(df["interval"])
            if df["interval"].max() < cutoff:
                continue  # Skip files entirely before the lookback window
            mask = df["constraint_name"] == constraint_name
            if mask.any():
                sub = df[mask][["interval", "shadow_price"]].copy()
                if "pct_of_limit" in df.columns:
                    sub["pct_of_limit"] = df.loc[mask, "pct_of_limit"].values
                dfs.append(sub)
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")

    if not dfs:
        return pd.DataFrame(columns=["interval", "shadow_price"])

    combined = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["interval"])
        .sort_values("interval")
        .reset_index(drop=True)
    )
    combined = combined[combined["interval"] >= cutoff]
    return combined


def _load_all_known_constraints(lookback_days: int = 7) -> List[str]:
    """Return all unique constraint names seen in the Parquet history."""
    constraints_dir = TIMESERIES_DIR / "constraints"
    if not constraints_dir.exists():
        return []

    cutoff = datetime.now() - timedelta(days=lookback_days)
    names: set = set()

    for path in sorted(constraints_dir.glob("constraints_*.parquet")):
        try:
            df = pd.read_parquet(path, columns=["interval", "constraint_name"])
            df["interval"] = pd.to_datetime(df["interval"])
            if df["interval"].max() >= cutoff:
                names.update(df["constraint_name"].dropna().unique())
        except Exception:
            pass

    return sorted(names)


def _train_one(constraint_name: str, lookback_days: int = 7) -> FitResult:
    """Synchronous training function — runs in an executor thread."""
    history = _load_constraint_history(constraint_name, lookback_days=lookback_days)

    # Also pull in live in-memory history from the SF cache
    if not sf_cache._constraints_history.empty:
        live = sf_cache._constraints_history.copy()
        live.columns = [c.lower() for c in live.columns]
        if "constraint_name" in live.columns and "shadow_price" in live.columns:
            live = live[live["constraint_name"] == constraint_name][
                ["interval", "shadow_price"]
            ]
            history = (
                pd.concat([history, live], ignore_index=True)
                .drop_duplicates(subset=["interval"])
                .sort_values("interval")
                .reset_index(drop=True)
            )

    fc = _registry.get(constraint_name)
    if fc is None:
        fc = ShadowPriceForecaster(constraint_name)
        _registry[constraint_name] = fc

    return fc.fit(history)


# ---------------------------------------------------------------------------
# Registry and ForecastCache class
# ---------------------------------------------------------------------------

# Module-level dict: constraint_name → ShadowPriceForecaster
_registry: Dict[str, ShadowPriceForecaster] = {}


class ForecastCache:
    """
    Orchestrates per-constraint model training and prediction.

    Attributes
    ----------
    last_retrained : datetime or None
        Timestamp of the most recent retrain cycle.
    retrain_count : int
        Total number of retrain cycles completed.
    """

    def __init__(self) -> None:
        self.last_retrained: Optional[datetime] = None
        self.retrain_count: int = 0
        self._retrain_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Startup / background loop
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """
        Called once at server startup. Performs an initial training pass
        then starts the background retrain loop.
        """
        logger.info("ForecastCache startup — initial training pass...")
        await self.retrain_all()
        self._retrain_task = asyncio.create_task(
            self._retrain_loop(), name="forecast_retrain_loop"
        )
        logger.info("Forecast retrain background loop started.")

    async def shutdown(self) -> None:
        if self._retrain_task and not self._retrain_task.done():
            self._retrain_task.cancel()
            try:
                await self._retrain_task
            except asyncio.CancelledError:
                pass

    async def _retrain_loop(self) -> None:
        """Background loop: retrain every RETRAIN_INTERVAL_MINUTES minutes."""
        while True:
            await asyncio.sleep(RETRAIN_INTERVAL_MINUTES * 60)
            try:
                await self.retrain_all()
            except Exception as e:
                logger.error(f"Retrain loop error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    async def retrain_all(self) -> Dict[str, FitResult]:
        """
        Retrain models for all known constraints (in a thread pool so the
        event loop stays responsive).

        Returns a dict of {constraint_name: FitResult}.
        """
        loop = asyncio.get_event_loop()

        # Discover all constraints (from Parquet + live cache)
        constraints = _load_all_known_constraints()
        if sf_cache.current_regime:
            constraints = sorted(set(constraints) | set(sf_cache.current_regime))

        if not constraints:
            logger.info("No constraints found for forecasting — skipping.")
            return {}

        logger.info(f"Retraining {len(constraints)} constraint forecasters...")

        results: Dict[str, FitResult] = {}
        for name in constraints:
            try:
                result = await loop.run_in_executor(None, _train_one, name)
                results[name] = result
            except Exception as e:
                logger.error(f"Training failed for {name}: {e}", exc_info=True)

        self.last_retrained = datetime.now()
        self.retrain_count += 1
        n_ml = sum(1 for r in results.values() if not r.fallback)
        n_fb = sum(1 for r in results.values() if r.fallback)
        logger.info(
            f"Retrain complete: {n_ml} ML models, {n_fb} persistence fallbacks."
        )
        return results

    async def retrain_one(self, constraint_name: str) -> FitResult:
        """Retrain a single constraint's model on demand."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _train_one, constraint_name)
        return result

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def get_forecaster(self, constraint_name: str) -> Optional[ShadowPriceForecaster]:
        """Return the forecaster for a constraint, or None if not trained."""
        return _registry.get(constraint_name)

    def predict_next(
        self,
        constraint_name: str,
        current_snapshot: Optional[dict] = None,
    ) -> Optional[float]:
        """
        Predict the next-interval shadow price. Returns None if no model exists.
        """
        fc = _registry.get(constraint_name)
        if fc is None or not fc.is_fitted:
            return None
        return fc.predict(current_snapshot)

    def predict_horizon(
        self,
        constraint_name: str,
        n_steps: int = 6,
        current_snapshot: Optional[dict] = None,
    ) -> Optional[List[dict]]:
        """
        Predict n_steps intervals ahead. Returns None if no model exists.
        """
        fc = _registry.get(constraint_name)
        if fc is None or not fc.is_fitted:
            return None
        return fc.predict_horizon(n_steps=n_steps, current_snapshot=current_snapshot)

    def list_models(self) -> List[dict]:
        """
        Return a summary of all trained models for the /api/v1/forecast/models
        status endpoint.
        """
        rows = []
        for name, fc in sorted(_registry.items()):
            fr = fc.fit_result
            rows.append({
                "constraint": name,
                "fitted": fc.is_fitted,
                "fallback": fr.fallback if fr else True,
                "n_training_rows": fr.n_training_rows if fr else 0,
                "mae_cv": round(fr.mae_cv, 3) if fr and not fr.fallback else None,
                "r2_cv": round(fr.r2_cv, 3) if fr and not fr.fallback else None,
                "trained_at": fr.trained_at.isoformat() if fr else None,
            })
        return rows

    def get_model_meta(self, constraint_name: str) -> Optional[dict]:
        """Return metadata for a single model."""
        fc = _registry.get(constraint_name)
        if fc is None:
            return None
        fr = fc.fit_result
        return {
            "constraint": constraint_name,
            "fitted": fc.is_fitted,
            "fallback": fr.fallback if fr else True,
            "n_training_rows": fr.n_training_rows if fr else 0,
            "mae_cv": round(fr.mae_cv, 3) if fr and not fr.fallback else None,
            "r2_cv": round(fr.r2_cv, 3) if fr and not fr.fallback else None,
            "trained_at": fr.trained_at.isoformat() if fr else None,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
forecast_cache = ForecastCache()
