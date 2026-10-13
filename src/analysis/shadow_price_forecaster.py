"""
Shadow Price Forecaster — Phase 13.

Predicts the next-interval shadow price for each binding constraint using a
gradient boosted regressor (sklearn HistGradientBoostingRegressor) with a rich
set of temporal and market-state features.

Architecture
------------
                ┌───────────────────────────────┐
                │   ShadowPriceForecaster        │
                │                               │
                │  fit(history_df)              │
                │  predict(snapshot) → float    │
                │  predict_horizon(n) → list    │
                └───────────────────────────────┘

One Forecaster instance is maintained per constraint by ForecastCache
(src/api/forecast_cache.py).

Feature Engineering
-------------------
Time features (learned by the model but also useful for regime detection):
  - hour_of_day          (0–23)
  - minute_of_hour       (0–55 in steps of 5)
  - day_of_week          (0=Mon … 6=Sun)
  - is_peak_hour         (1 if 07:00–22:00 CT, else 0)
  - is_weekend           (1 if Sat/Sun)

Market-state features (from the constraint's own history):
  - shadow_price_lag1    (t-1 shadow price)
  - shadow_price_lag2    (t-2)
  - shadow_price_lag3    (t-3)
  - shadow_price_lag12   (t-12, ~1 hour back)
  - shadow_price_lag24   (t-24, ~2 hours back)
  - rolling_mean_6       (6-interval rolling mean — ~30 min)
  - rolling_std_6        (6-interval rolling std)
  - rolling_mean_24      (24-interval rolling mean — ~2 hours)
  - binding_streak       (consecutive intervals this constraint has been binding)
  - pct_of_limit         (monitored_mw / limit_mw, if available)

Fallback strategy
-----------------
When fewer than MIN_TRAINING_ROWS training rows are available the model
cannot be reliably fit. In that case `predict()` returns the most recent
observed shadow price (persistence model, a strong baseline for 5-min ahead).

The model is fit incrementally: call `fit()` any time new history arrives.
`predict()` always uses the latest model (or fallback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

# Minimum rows needed before we attempt ML; below this we use persistence.
# 200 rows is ~17 hours of 5-minute data. With the 288-interval lag and 72-interval
# rolling windows, a few dozen rows leaves most features empty.
MIN_TRAINING_ROWS: int = 200

# Interval length in minutes (SPP RTBM is 5-min)
INTERVAL_MINUTES: int = 5


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _naive(ts) -> pd.Timestamp:
    """Timestamp without a timezone, so live snapshots compare with (naive) stored history."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _time_features(ts: pd.Series) -> pd.DataFrame:
    """Extract cyclical and categorical time features from a datetime Series."""
    df = pd.DataFrame(index=ts.index)
    df["hour_of_day"] = ts.dt.hour
    df["minute_of_hour"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek          # 0=Mon, 6=Sun
    # Peak hours: 07:00–22:00 (SPP's on-peak definition is HE07-HE22)
    df["is_peak_hour"] = ((ts.dt.hour >= 7) & (ts.dt.hour < 22)).astype(int)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    # Hour-of-week for capturing day-type × time interactions
    df["hour_of_week"] = ts.dt.dayofweek * 24 + ts.dt.hour
    return df


def build_features(
    history: pd.DataFrame,
    lags: Tuple[int, ...] = (1, 2, 3, 12, 24, 48, 288),
    rolling_windows: Tuple[int, ...] = (6, 24, 72),
) -> pd.DataFrame:
    """
    Build the feature matrix from a shadow-price time series.

    Parameters
    ----------
    history : DataFrame
        Must have columns: ``interval`` (datetime), ``shadow_price`` (float).
        Optionally: ``pct_of_limit`` (float, 0–1).
        Rows must be sorted by ``interval`` ascending.

    Returns
    -------
    DataFrame with one row per interval, containing all features.
    NaN rows (insufficient lags) are **kept** — HistGradientBoosting handles NaN
    natively. We do NOT drop rows so that the row count stays aligned with y.
    """
    df = history.copy().sort_values("interval").reset_index(drop=True)

    sp = df["shadow_price"]

    # Time features
    tf = _time_features(df["interval"])
    feat = pd.concat([df[["interval"]], tf], axis=1)

    # Lagged shadow prices
    for lag in lags:
        feat[f"sp_lag{lag}"] = sp.shift(lag)

    # Rolling statistics
    for w in rolling_windows:
        feat[f"sp_roll_mean_{w}"] = sp.shift(1).rolling(w, min_periods=1).mean()
        feat[f"sp_roll_std_{w}"] = sp.shift(1).rolling(w, min_periods=2).std().fillna(0)
        feat[f"sp_roll_max_{w}"] = sp.shift(1).rolling(w, min_periods=1).max()
        feat[f"sp_roll_bind_freq_{w}"] = (sp.abs() > 0.01).shift(1).rolling(w, min_periods=1).mean()

    # Binding streak (consecutive intervals with |SP| > threshold)
    binding = (sp.abs() > 0.01).astype(int)
    streak = binding.copy()
    for i in range(1, len(binding)):
        if binding.iloc[i]:
            streak.iloc[i] = streak.iloc[i - 1] + 1
    feat["binding_streak"] = streak.shift(1).fillna(0)

    # pct_of_limit if present
    if "pct_of_limit" in df.columns:
        feat["pct_of_limit"] = df["pct_of_limit"].shift(1).fillna(0)

    return feat


# ---------------------------------------------------------------------------
# Model fit result
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Metadata from a training run."""
    constraint: str
    trained_at: datetime
    n_training_rows: int
    mae_cv: float          # MAE on the last-20% holdout. Optimistic: the best of several
                           # configs is picked on this same holdout.
    r2_cv: float           # R² on the same holdout
    fallback: bool         # True if using persistence model
    baseline_mae: float = float("nan")   # Persistence ("same as last interval") MAE on the same holdout
    loss_type: str = "squared_error"
    use_symlog: bool = False
    feature_importances: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class ShadowPriceForecaster:
    """
    Per-constraint shadow price forecaster.

    Usage
    -----
    ::

        fc = ShadowPriceForecaster("TMP386_28084")
        result = fc.fit(history_df)       # train / retrain
        p = fc.predict(snapshot_dict)    # predict next interval
        horizon = fc.predict_horizon(6)  # 6-step-ahead (30 min)

    Parameters
    ----------
    constraint_name : str
        The constraint this forecaster is trained on.
    n_estimators : int
        Number of gradient boosting iterations.
    max_depth : int
        Max tree depth (shallow = fast + less overfit on small data).
    """

    def __init__(
        self,
        constraint_name: str,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ) -> None:
        self.constraint_name = constraint_name
        self._model: Optional[HistGradientBoostingRegressor] = None
        self._fit_result: Optional[FitResult] = None
        self._feature_cols: List[str] = []
        self._last_sp: float = 0.0       # for persistence fallback
        self._history: pd.DataFrame = pd.DataFrame()

        self._model_params = dict(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=5,
            random_state=42,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, history_df: pd.DataFrame) -> FitResult:
        """
        Train (or retrain) on accumulated shadow price history.

        Parameters
        ----------
        history_df : DataFrame
            Columns: ``interval`` (datetime), ``shadow_price`` (float).
            Optionally: ``pct_of_limit``.
            Should span as many intervals as possible — more is better.

        Returns
        -------
        FitResult with training metadata.
        """
        df = history_df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values("interval").reset_index(drop=True)

        # Cache last observed SP for persistence fallback
        if not df.empty:
            self._last_sp = float(df["shadow_price"].iloc[-1])
            self._history = df

        fallback = len(df) < MIN_TRAINING_ROWS

        if fallback:
            logger.info(
                f"[{self.constraint_name}] Only {len(df)} rows — using persistence fallback."
            )
            result = FitResult(
                constraint=self.constraint_name,
                trained_at=datetime.now(),
                n_training_rows=len(df),
                mae_cv=float("nan"),
                r2_cv=float("nan"),
                fallback=True,
            )
            self._fit_result = result
            return result

        # Build features
        feat_df = build_features(df)
        feature_cols = [c for c in feat_df.columns if c != "interval"]
        X = feat_df[feature_cols].values.astype(float)
        y = df["shadow_price"].values.astype(float)

        # Holdout: last 20% for evaluation
        n_test = max(1, int(len(y) * 0.2))
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        # AutoML selection: find configuration with lowest holdout MAE
        candidates = [
            ("squared_error", False),
            ("absolute_error", False),
            ("squared_error", True),
            ("absolute_error", True),
        ]

        # Persistence benchmark on the same holdout: predict "same as the previous interval".
        persistence_pred = np.concatenate([[y_train[-1]], y_test[:-1]])
        baseline_mae = float(mean_absolute_error(y_test, persistence_pred))

        best_mae = float("inf")
        best_r2 = float("-inf")
        best_config = ("absolute_error", True)  # default robust config

        for loss, use_symlog in candidates:
            if use_symlog:
                y_train_fit = np.sign(y_train) * np.log1p(np.abs(y_train))
            else:
                y_train_fit = y_train

            params = self._model_params.copy()
            params["loss"] = loss

            try:
                candidate_model = HistGradientBoostingRegressor(**params)
                candidate_model.fit(X_train, y_train_fit)

                y_pred_fit = candidate_model.predict(X_test)

                if use_symlog:
                    y_pred = np.sign(y_pred_fit) * (np.expm1(np.abs(y_pred_fit)))
                else:
                    y_pred = y_pred_fit

                mae = float(mean_absolute_error(y_test, y_pred))

                # R2 calculation
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 1.0

                # Prioritize lowest MAE, use R2 as tie-breaker
                if mae < best_mae:
                    best_mae = mae
                    best_r2 = r2
                    best_config = (loss, use_symlog)
            except Exception as e:
                logger.warning(
                    f"[{self.constraint_name}] Failed candidate loss={loss}, symlog={use_symlog}: {e}"
                )

        # No candidate could be fit → don't pretend we have a trained model.
        if not np.isfinite(best_mae):
            logger.warning(f"[{self.constraint_name}] All candidate fits failed — using persistence.")
            self._model = None
            self._fit_result = FitResult(
                constraint=self.constraint_name,
                trained_at=datetime.now(),
                n_training_rows=len(df),
                mae_cv=float("nan"),
                r2_cv=float("nan"),
                fallback=True,
                baseline_mae=baseline_mae,
            )
            return self._fit_result

        # A model that cannot beat "same as last interval" is worse than useless.
        if best_mae >= baseline_mae:
            logger.info(
                f"[{self.constraint_name}] ML holdout MAE {best_mae:.3f} does not beat "
                f"persistence {baseline_mae:.3f} — using persistence."
            )
            self._model = None
            self._fit_result = FitResult(
                constraint=self.constraint_name,
                trained_at=datetime.now(),
                n_training_rows=len(df),
                mae_cv=best_mae,
                r2_cv=best_r2,
                fallback=True,
                baseline_mae=baseline_mae,
            )
            return self._fit_result

        # Retrain best configuration on all data (train + holdout)
        best_loss, best_symlog = best_config

        if best_symlog:
            y_fit = np.sign(y) * np.log1p(np.abs(y))
        else:
            y_fit = y

        final_params = self._model_params.copy()
        final_params["loss"] = best_loss

        final_model = HistGradientBoostingRegressor(**final_params)
        final_model.fit(X, y_fit)

        logger.info(
            f"[{self.constraint_name}] Selected best config: loss={best_loss}, use_symlog={best_symlog}. "
            f"Holdout MAE={best_mae:.4f} $/MWh, R²={best_r2:.4f}"
        )

        self._model = final_model
        self._feature_cols = feature_cols
        self._fit_result = FitResult(
            constraint=self.constraint_name,
            trained_at=datetime.now(),
            n_training_rows=len(df),
            mae_cv=best_mae,
            r2_cv=best_r2,
            fallback=False,
            loss_type=best_loss,
            use_symlog=best_symlog,
            baseline_mae=baseline_mae,
        )
        return self._fit_result

    def predict(self, current_snapshot: Optional[dict] = None) -> float:
        """
        Predict the next-interval shadow price.

        Parameters
        ----------
        current_snapshot : dict, optional
            If provided, used to update the last-known state before predicting.
            Expected keys: ``shadow_price``, ``interval`` (datetime).

        Returns
        -------
        float
            Predicted shadow price for the next RTBM interval (5 min ahead).
        """
        if current_snapshot and "shadow_price" in current_snapshot:
            self._last_sp = float(current_snapshot["shadow_price"])

        # Persistence fallback
        if self._model is None or not self._feature_cols:
            return self._last_sp

        # Build a one-row feature vector for the *next* interval
        try:
            next_row = self._build_next_row(current_snapshot)
            X = np.array([[next_row.get(c, np.nan) for c in self._feature_cols]])
            prediction_fit = float(self._model.predict(X)[0])
            if self._fit_result and getattr(self._fit_result, "use_symlog", False):
                prediction = float(np.sign(prediction_fit) * (np.expm1(np.abs(prediction_fit))))
            else:
                prediction = prediction_fit
            return prediction
        except Exception as e:
            logger.warning(f"[{self.constraint_name}] predict() failed: {e}. Using persistence.")
            return self._last_sp

    def predict_horizon(
        self,
        n_steps: int = 6,
        current_snapshot: Optional[dict] = None,
    ) -> List[dict]:
        """
        Generate a multi-step forecast by iteratively feeding predictions back.

        Parameters
        ----------
        n_steps : int
            Number of 5-minute intervals to forecast (default 6 = 30 minutes).
        current_snapshot : dict, optional
            Current market state.

        Returns
        -------
        List of dicts with keys: ``interval`` (datetime), ``shadow_price`` (float),
        ``is_binding`` (bool), ``confidence_interval`` (tuple[float, float]).
        """
        if current_snapshot and "shadow_price" in current_snapshot:
            self._last_sp = float(current_snapshot["shadow_price"])

        if current_snapshot and "interval" in current_snapshot:
            base_time = _naive(current_snapshot["interval"])
        elif not self._history.empty:
            base_time = pd.to_datetime(self._history["interval"].iloc[-1])
        else:
            base_time = pd.Timestamp(datetime.now())

        results = []
        running_sp = self._last_sp

        # History up to and INCLUDING the current interval, so that the first
        # forecast step sees the current shadow price as its lag-1 value.
        cols = [c for c in ("interval", "shadow_price", "pct_of_limit") if c in self._history.columns] or [
            "interval", "shadow_price"
        ]
        running_history = (
            pd.concat(
                [
                    self._history[cols] if not self._history.empty else pd.DataFrame(columns=cols),
                    pd.DataFrame([{"interval": base_time, "shadow_price": self._last_sp}]),
                ],
                ignore_index=True,
            )
            .drop_duplicates(subset=["interval"], keep="last")
            .sort_values("interval")
            .reset_index(drop=True)
        )

        for step in range(1, n_steps + 1):
            interval = base_time + timedelta(minutes=INTERVAL_MINUTES * step)

            if self._model is None or not self._feature_cols:
                predicted = running_sp
            else:
                try:
                    # Placeholder row for the interval being forecast: its lag
                    # features come from earlier rows, its target is unknown.
                    temp_history = pd.concat(
                        [running_history, pd.DataFrame([{"interval": interval, "shadow_price": np.nan}])],
                        ignore_index=True,
                    )
                    feat_df = build_features(temp_history)
                    last_row = feat_df.iloc[-1]
                    X = np.array([[last_row.get(c, np.nan) for c in self._feature_cols]])
                    predicted_fit = float(self._model.predict(X)[0])
                    if self._fit_result and getattr(self._fit_result, "use_symlog", False):
                        predicted = float(np.sign(predicted_fit) * (np.expm1(np.abs(predicted_fit))))
                    else:
                        predicted = predicted_fit
                except Exception as e:
                    logger.debug(f"horizon step {step}: {e}")
                    predicted = running_sp

            # Rough uncertainty band: holdout MAE scaled by sqrt(step). This is a
            # heuristic, NOT a calibrated prediction interval.
            mae = self._fit_result.mae_cv if self._fit_result and not np.isnan(
                self._fit_result.mae_cv
            ) else abs(running_sp) * 0.10  # 10% fallback CI

            ci_half = mae * (step ** 0.5)
            results.append({
                "step": step,
                "interval": interval.isoformat(),
                "shadow_price": round(predicted, 4),
                "is_binding": abs(predicted) > 0.5,
                "confidence_low": round(predicted - ci_half, 4),
                "confidence_high": round(predicted + ci_half, 4),
            })

            running_history = pd.concat(
                [running_history, pd.DataFrame([{"interval": interval, "shadow_price": predicted}])],
                ignore_index=True,
            )
            running_sp = predicted

        return results

    @property
    def is_fitted(self) -> bool:
        """True if a model (or fallback) is ready to produce predictions."""
        return self._fit_result is not None

    @property
    def fit_result(self) -> Optional[FitResult]:
        return self._fit_result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_next_row(self, current_snapshot: Optional[dict]) -> dict:
        """
        Build a feature dict for the next (t+1) interval using rolling history.
        """
        if self._history.empty:
            return {}

        # The "next" interval's features depend on the current state being appended
        now_interval = (
            _naive(current_snapshot["interval"])
            if current_snapshot and "interval" in current_snapshot
            else self._history["interval"].iloc[-1]
        )
        next_interval = now_interval + timedelta(minutes=INTERVAL_MINUTES)

        # Append the current snapshot AND a placeholder row for the interval being
        # predicted.  build_features() lags every feature by one row, so the
        # features of the placeholder row are exactly "what we know at `now`",
        # including the current shadow price as sp_lag1.  (Taking the last
        # feature row of the history instead gives the features for `now`, which
        # never see the current price — the forecast ignored it entirely.)
        extra = pd.DataFrame([
            {"interval": now_interval, "shadow_price": self._last_sp},
            {"interval": next_interval, "shadow_price": np.nan},
        ])
        cols = [c for c in ("interval", "shadow_price", "pct_of_limit") if c in self._history.columns]
        temp_history = (
            pd.concat([self._history[cols], extra], ignore_index=True)
            .drop_duplicates(subset=["interval"], keep="last")
            .sort_values("interval")
        )

        feat_df = build_features(temp_history)
        last_row = feat_df.iloc[-1]
        row_dict = {c: last_row[c] for c in self._feature_cols if c in last_row.index}

        # Override time features for the actual next interval
        tf = _time_features(pd.Series([next_interval]))
        for col, val in tf.iloc[0].items():
            if col in row_dict:
                row_dict[col] = val

        return row_dict
