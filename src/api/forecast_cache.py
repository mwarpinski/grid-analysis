"""
Forecast Cache — per-constraint model registry + background retraining.

One ShadowPriceForecaster is maintained per constraint. Retraining is
triggered:
  1. On startup — loads persisted models from disk (joblib), skipping any
     model that was trained within MODEL_STALE_HOURS hours.  Only truly
     stale or missing models are retrained, so cold-start is nearly instant.
  2. Every RETRAIN_INTERVAL_MINUTES in a background asyncio task.
  3. On-demand via POST /api/v1/forecast/retrain.

Model persistence
-----------------
Each trained model is serialised to:

  src/data/models/forecasters/<constraint_name>.joblib

A registry manifest is kept at:

  src/data/models/forecasters/model_registry.json

The manifest records training metadata (n_rows, MAE, R², trained_at) for
every model so resiliency tracking works even between server restarts.

Thread / async safety
---------------------
Retraining is CPU-bound. We run it in an executor (threadpool) so the
asyncio event loop stays responsive during training.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from src.analysis.shadow_price_forecaster import FitResult, ShadowPriceForecaster
from src.api.sf_cache import cache as sf_cache

logger = logging.getLogger(__name__)

# How often to retrain all models (in minutes)
RETRAIN_INTERVAL_MINUTES: int = 30

# Models trained within this many hours are considered fresh (skip retrain on startup)
MODEL_STALE_HOURS: int = 6

# Where Parquet time-series files live
TIMESERIES_DIR = Path(__file__).parent.parent / "data" / "timeseries"

# Where trained model artefacts are persisted
MODEL_DIR = Path(__file__).parent.parent / "data" / "models" / "forecasters"
MODEL_REGISTRY_PATH = MODEL_DIR / "model_registry.json"


# ---------------------------------------------------------------------------
# Helpers — load history from Parquet
# ---------------------------------------------------------------------------

def _load_constraint_history(
    constraint_name: str,
    lookback_days: int = 21,
) -> pd.DataFrame:
    """
    Load the shadow price history for one constraint from Parquet files.

    Returns a DataFrame with columns: ``interval``, ``shadow_price``,
    optionally ``pct_of_limit``.
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
                continue
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


def _load_all_known_constraints(lookback_days: int = 21) -> List[str]:
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


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

def _model_path(constraint_name: str) -> Path:
    return MODEL_DIR / f"{constraint_name}.joblib"


def _read_registry() -> dict:
    """Load the on-disk model registry manifest (returns {} if missing/corrupt)."""
    if MODEL_REGISTRY_PATH.exists():
        try:
            return json.loads(MODEL_REGISTRY_PATH.read_text())
        except Exception:
            pass
    return {}


def _write_registry(registry: dict) -> None:
    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2, default=str))


def _save_model(fc: ShadowPriceForecaster) -> None:
    """
    Persist a trained ShadowPriceForecaster to disk and update the manifest.
    Saves the full object (model + history + feature cols) via joblib.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_path(fc.constraint_name)
    try:
        joblib.dump(fc, path, compress=3)
    except Exception as e:
        logger.warning(f"Failed to save model for {fc.constraint_name}: {e}")
        return

    # Update registry manifest
    fr = fc.fit_result
    registry = _read_registry()
    registry[fc.constraint_name] = {
        "constraint":      fc.constraint_name,
        "model_file":      path.name,
        "trained_at":      fr.trained_at.isoformat() if fr else None,
        "n_training_rows": fr.n_training_rows if fr else 0,
        "mae_cv":          round(fr.mae_cv, 4) if fr and not fr.fallback else None,
        "r2_cv":           round(fr.r2_cv, 4) if fr and not fr.fallback else None,
        "fallback":        fr.fallback if fr else True,
        "loss_type":       getattr(fr, "loss_type", "squared_error") if fr else "squared_error",
        "use_symlog":      getattr(fr, "use_symlog", False) if fr else False,
        "saved_at":        datetime.now().isoformat(),
    }
    _write_registry(registry)


def _load_model(constraint_name: str) -> Optional[ShadowPriceForecaster]:
    """Load a persisted model from disk. Returns None if not found or corrupt."""
    path = _model_path(constraint_name)
    if not path.exists():
        return None
    try:
        fc = joblib.load(path)
        if not isinstance(fc, ShadowPriceForecaster):
            return None
        return fc
    except Exception as e:
        logger.warning(f"Failed to load persisted model for {constraint_name}: {e}")
        return None


def _model_is_fresh(constraint_name: str, stale_hours: int = MODEL_STALE_HOURS) -> bool:
    """True if the on-disk model was trained within the stale threshold."""
    registry = _read_registry()
    entry = registry.get(constraint_name)
    if not entry or not entry.get("trained_at"):
        return False
    try:
        trained_at = datetime.fromisoformat(entry["trained_at"])
        return (datetime.now() - trained_at) < timedelta(hours=stale_hours)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def _train_one(constraint_name: str, lookback_days: int = 21) -> FitResult:
    """
    Synchronous training function — runs in an executor thread.
    Saves the model to disk after a successful fit.
    """
    history = _load_constraint_history(constraint_name, lookback_days=lookback_days)

    # Merge in live in-memory history from the SF cache
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

    result = fc.fit(history)
    _save_model(fc)          # ← persist immediately after each fit
    return result


# ---------------------------------------------------------------------------
# Registry and ForecastCache class
# ---------------------------------------------------------------------------

# Module-level dict: constraint_name → ShadowPriceForecaster
_registry: Dict[str, ShadowPriceForecaster] = {}


class ForecastCache:
    """
    Orchestrates per-constraint model training, persistence, and prediction.

    Attributes
    ----------
    last_retrained : datetime or None
        Timestamp of the most recent retrain cycle.
    retrain_count : int
        Total number of retrain cycles completed since server start.
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
        Called once at server startup.

        1. Loads all persisted models from disk (fast — no retraining).
        2. Retrains only models that are stale or have no persisted artefact.
        3. Starts the background retrain loop.
        """
        logger.info("ForecastCache startup — loading persisted models from disk...")
        n_loaded = await asyncio.get_event_loop().run_in_executor(
            None, self._load_persisted_models
        )
        logger.info(f"  Loaded {n_loaded} models from disk.")

        # Initialize last_retrained from disk manifest
        try:
            registry = _read_registry()
            times = []
            for entry in registry.values():
                if entry.get("trained_at"):
                    times.append(datetime.fromisoformat(entry["trained_at"]))
            if times:
                self.last_retrained = max(times)
        except Exception as e:
            logger.debug(f"Could not initialize last_retrained from manifest: {e}")

        # Retrain only stale/missing models — in the background, so the server
        # starts serving immediately instead of waiting on every model fit.
        self._startup_train_task = asyncio.create_task(
            self.retrain_all(skip_fresh=True), name="forecast_startup_train"
        )

        self._retrain_task = asyncio.create_task(
            self._retrain_loop(), name="forecast_retrain_loop"
        )
        logger.info("Forecast retrain background loop started.")

    def _load_persisted_models(self) -> int:
        """
        Load all .joblib models from MODEL_DIR into the registry.
        Called once at startup in a thread (joblib.load is blocking I/O).
        Returns the number of models successfully loaded.
        """
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        n = 0
        for path in sorted(MODEL_DIR.glob("*.joblib")):
            constraint_name = path.stem
            fc = _load_model(constraint_name)
            if fc is not None:
                _registry[constraint_name] = fc
                n += 1
        return n

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
                await self.retrain_all(skip_fresh=False)
            except Exception as e:
                logger.error(f"Retrain loop error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    async def retrain_all(
        self, skip_fresh: bool = False, force: bool = False
    ) -> Dict[str, FitResult]:
        """
        Retrain models for all known constraints.

        Parameters
        ----------
        skip_fresh : bool
            If True, skip constraints whose persisted model was trained within
            MODEL_STALE_HOURS (used at startup to avoid redundant work).
        force : bool
            If True, retrain all regardless of freshness (used by the
            POST /retrain endpoint when explicitly requested).

        Returns a dict of {constraint_name: FitResult}.
        """
        loop = asyncio.get_event_loop()

        constraints = _load_all_known_constraints()
        if sf_cache.current_regime:
            constraints = sorted(set(constraints) | set(sf_cache.current_regime))

        if not constraints:
            logger.info("No constraints found for forecasting — skipping.")
            return {}

        # Filter to only stale/missing when skip_fresh=True
        if skip_fresh and not force:
            stale = [c for c in constraints if not _model_is_fresh(c)]
            fresh_count = len(constraints) - len(stale)
            if fresh_count:
                logger.info(
                    f"  Skipping {fresh_count} fresh models (trained < {MODEL_STALE_HOURS}h ago)."
                )
            constraints = stale

        if not constraints:
            logger.info("All models are fresh — no retraining needed at startup.")
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
            f"Retrain complete: {n_ml} ML models, {n_fb} persistence fallbacks. "
            f"All models saved to {MODEL_DIR}."
        )
        return results

    async def retrain_one(self, constraint_name: str) -> FitResult:
        """Retrain a single constraint's model on demand and persist it."""
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
        """Predict the next-interval shadow price. Returns None if no model."""
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
        """Predict n_steps intervals ahead. Returns None if no model."""
        fc = _registry.get(constraint_name)
        if fc is None or not fc.is_fitted:
            return None
        return fc.predict_horizon(n_steps=n_steps, current_snapshot=current_snapshot)

    def list_models(self) -> List[dict]:
        """
        Return a summary of all trained models for the /api/v1/forecast/models
        status endpoint. Merges live registry with on-disk manifest so models
        that are loaded-but-not-yet-retrained still appear.
        """
        registry_manifest = _read_registry()
        all_names = sorted(set(_registry.keys()) | set(registry_manifest.keys()))

        rows = []
        for name in all_names:
            fc = _registry.get(name)
            disk_meta = registry_manifest.get(name, {})

            if fc is not None:
                fr = fc.fit_result
                rows.append({
                    "constraint":      name,
                    "fitted":          fc.is_fitted,
                    "fallback":        fr.fallback if fr else True,
                    "n_training_rows": fr.n_training_rows if fr else 0,
                    "mae_cv":          round(fr.mae_cv, 3) if fr and not fr.fallback else None,
                    "r2_cv":           round(fr.r2_cv, 3) if fr and not fr.fallback else None,
                    "trained_at":      fr.trained_at.isoformat() if fr else None,
                    "persisted":       _model_path(name).exists(),
                    "source":          "live",
                })
            else:
                # Model is on-disk manifest but not yet loaded in this session
                rows.append({
                    "constraint":      name,
                    "fitted":          False,
                    "fallback":        disk_meta.get("fallback", True),
                    "n_training_rows": disk_meta.get("n_training_rows", 0),
                    "mae_cv":          disk_meta.get("mae_cv"),
                    "r2_cv":           disk_meta.get("r2_cv"),
                    "trained_at":      disk_meta.get("trained_at"),
                    "persisted":       _model_path(name).exists(),
                    "source":          "disk_manifest_only",
                })
        return rows

    def registry_summary(self) -> dict:
        """
        High-level summary for the /models endpoint top-level response.
        Adds persistence stats and quality distribution.
        """
        models = self.list_models()
        ml_models  = [m for m in models if not m["fallback"] and m["fitted"]]
        fallbacks  = [m for m in models if m["fallback"]]
        persisted  = [m for m in models if m["persisted"]]

        r2_values = [m["r2_cv"] for m in ml_models if m.get("r2_cv") is not None]

        return {
            "n_models":            len(models),
            "fitted_ml_models":    len(ml_models),
            "persistence_models":  len(fallbacks),
            "persisted_to_disk":   len(persisted),
            "model_dir":           str(MODEL_DIR),
            "last_retrained":      self.last_retrained.isoformat() if self.last_retrained else None,
            "retrain_count":       self.retrain_count,
            "quality": {
                "mean_r2":         round(sum(r2_values) / len(r2_values), 4) if r2_values else None,
                "r2_gte_090":      sum(1 for r in r2_values if r >= 0.90),
                "r2_050_090":      sum(1 for r in r2_values if 0.5 <= r < 0.90),
                "r2_lt_050":       sum(1 for r in r2_values if r < 0.5),
            },
            "models":              models,
        }

    def get_model_meta(self, constraint_name: str) -> Optional[dict]:
        """Return metadata for a single model (merges live + disk manifest)."""
        fc = _registry.get(constraint_name)
        disk_meta = _read_registry().get(constraint_name, {})

        if fc is not None:
            fr = fc.fit_result
            return {
                "constraint":      constraint_name,
                "fitted":          fc.is_fitted,
                "fallback":        fr.fallback if fr else True,
                "n_training_rows": fr.n_training_rows if fr else 0,
                "mae_cv":          round(fr.mae_cv, 3) if fr and not fr.fallback else None,
                "r2_cv":           round(fr.r2_cv, 3) if fr and not fr.fallback else None,
                "trained_at":      fr.trained_at.isoformat() if fr else None,
                "persisted":       _model_path(constraint_name).exists(),
                "disk_trained_at": disk_meta.get("trained_at"),
                "source":          "live",
            }

        if disk_meta:
            return {**disk_meta, "source": "disk_manifest_only", "fitted": False}

        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
forecast_cache = ForecastCache()
