"""Tests for ShadowPriceForecaster: current-price sensitivity and the persistence guard."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.shadow_price_forecaster import MIN_TRAINING_ROWS, ShadowPriceForecaster


def _sticky_series(n=600, seed=1):
    """Shadow price that stays put ~90% of the time, so 'last value' is a strong forecast."""
    rng = np.random.default_rng(seed)
    sp = np.zeros(n)
    for t in range(1, n):
        if rng.random() < 0.9:
            sp[t] = sp[t - 1]
        else:
            sp[t] = -rng.uniform(20, 300) if rng.random() < 0.5 else 0.0
    iv = pd.date_range("2026-10-01", periods=n, freq="5min")
    return pd.DataFrame({"interval": iv, "shadow_price": sp})


def _autoregressive_series(n=1500, seed=2):
    """Shadow price strongly driven by its previous value plus a daily cycle."""
    rng = np.random.default_rng(seed)
    iv = pd.date_range("2026-10-01", periods=n, freq="5min")
    sp = np.zeros(n)
    for t in range(1, n):
        cycle = -80 * (1 + np.sin(2 * np.pi * (iv[t].hour * 60 + iv[t].minute) / 1440.0))
        sp[t] = 0.7 * sp[t - 1] + 0.3 * cycle + rng.normal(0, 3)
    return pd.DataFrame({"interval": iv, "shadow_price": sp})


def test_falls_back_to_persistence_with_little_data():
    hist = _sticky_series(n=MIN_TRAINING_ROWS - 1)
    fc = ShadowPriceForecaster("X")
    result = fc.fit(hist)
    assert result.fallback
    assert fc.predict({"interval": hist.interval.iloc[-1], "shadow_price": -42.0}) == -42.0


def test_ml_model_is_not_kept_when_it_cannot_beat_persistence():
    """On a sticky series persistence is very hard to beat; the model must defer to it."""
    hist = _sticky_series()
    fc = ShadowPriceForecaster("X")
    result = fc.fit(hist)
    if result.fallback:
        # Deferred to persistence: prediction is exactly the current value.
        assert fc.predict({"interval": hist.interval.iloc[-1], "shadow_price": -123.0}) == -123.0
    else:
        assert result.mae_cv < result.baseline_mae


def test_prediction_depends_on_the_current_shadow_price():
    """Regression test: predict() used to return the same value whatever the current price was."""
    hist = _autoregressive_series()
    train = hist.iloc[:1200]
    fc = ShadowPriceForecaster("X")
    result = fc.fit(train)
    assert not result.fallback, "autoregressive series should be learnable"

    now = train.interval.iloc[-1]
    low = fc.predict({"interval": now, "shadow_price": -300.0})
    high = fc.predict({"interval": now, "shadow_price": 0.0})
    assert low != pytest.approx(high)
    assert low < high  # lower current price -> lower forecast


def test_horizon_first_step_uses_current_price():
    hist = _autoregressive_series()
    train = hist.iloc[:1200]
    fc = ShadowPriceForecaster("X")
    assert not fc.fit(train).fallback
    now = train.interval.iloc[-1]
    a = fc.predict_horizon(3, {"interval": now, "shadow_price": -300.0})
    b = fc.predict_horizon(3, {"interval": now, "shadow_price": 0.0})
    assert a[0]["shadow_price"] != pytest.approx(b[0]["shadow_price"])
    assert len(a) == 3
    assert a[1]["interval"] > a[0]["interval"]


def test_model_beats_persistence_one_step_out_of_sample():
    """The fitted model's real one-step error, not just its self-reported holdout, beats persistence."""
    hist = _autoregressive_series()
    train, test = hist.iloc[:1200].reset_index(drop=True), hist.iloc[1200:].reset_index(drop=True)
    fc = ShadowPriceForecaster("X")
    assert not fc.fit(train).fallback
    model_err, persist_err = [], []
    for i in range(0, 120, 3):
        fc._history = pd.concat([train, test.iloc[: i + 1]], ignore_index=True)
        snap = {"interval": test.interval[i], "shadow_price": test.shadow_price[i]}
        y = test.shadow_price[i + 1]
        model_err.append(abs(fc.predict(snap) - y))
        persist_err.append(abs(test.shadow_price[i] - y))
    assert np.mean(model_err) < np.mean(persist_err)


def test_timezone_aware_snapshot_interval_does_not_break_forecasts():
    """Live snapshots were tagged UTC while stored history is naive; /forecast/{c} returned HTTP 500."""
    hist = _autoregressive_series()
    train = hist.iloc[:1200]
    fc = ShadowPriceForecaster("X")
    assert not fc.fit(train).fallback
    aware_now = train.interval.iloc[-1].tz_localize("UTC")
    snap = {"interval": aware_now, "shadow_price": -50.0}

    assert isinstance(fc.predict(snap), float)
    assert len(fc.predict_horizon(4, snap)) == 4
