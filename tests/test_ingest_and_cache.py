"""Regression tests for the ingest path and the SF cache (no network)."""

import asyncio
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.api import ingest_worker
from src.api.sf_cache import SFCache, MultiISOSFCache
from tests.test_shift_factor_calculator import make_multi_constraint_data


def test_seed_makes_shift_factors_available_immediately():
    constraints_df, lmp_df, true_sfs = make_multi_constraint_data(n_intervals=60, n_nodes=3)
    cache = SFCache()
    assert cache.solution is None

    assert cache.seed(constraints_df, lmp_df) is True

    node = "NODE_00"
    assert cache.solution.get(node, "LN ALPHA - BETA") == pytest.approx(
        true_sfs[node]["LN ALPHA - BETA"], abs=0.02
    )


def test_seed_with_no_history_is_a_noop():
    cache = SFCache()
    assert cache.seed(pd.DataFrame(), pd.DataFrame()) is False
    assert cache.solution is None


def _fake_rt_constraints(interval: pd.Timestamp) -> pd.DataFrame:
    """One BINDING and one BREACHED priced constraint, plus an unpriced ACTIVATED one."""
    return pd.DataFrame({
        "interval": [interval] * 3,
        "constraint_name": ["C_BINDING", "C_BREACHED", "C_IDLE"],
        "state": ["BINDING", "BREACHED", "ACTIVATED"],
        "shadow_price": [-100.0, -250.0, 0.0],
        "limit_mw": [100.0, 100.0, 100.0],
        "monitored_mw": [np.nan] * 3,
        "pct_of_limit": [np.nan] * 3,
    })


def test_ingest_keeps_breached_constraints_in_live_shadow_prices(monkeypatch):
    """
    SPP reports priced constraints as BINDING *or* BREACHED. Live shadow prices used
    to keep only BINDING rows, dropping more than half of the shadow-price magnitude.
    """
    interval = pd.Timestamp("2026-10-10 12:00:00")
    monkeypatch.setattr("src.api.sf_cache.cache", MultiISOSFCache())
    monkeypatch.setattr(ingest_worker, "cache", __import__("src.api.sf_cache", fromlist=["cache"]).cache)
    monkeypatch.setattr(ingest_worker, "_last_da_scraped_date", datetime.now().date())
    monkeypatch.setattr(ingest_worker, "broadcast", lambda msg: asyncio.sleep(0))

    lmp = pd.DataFrame({"interval": [interval], "settlement_location": ["N1"], "mcc": [1.0]})

    with patch("src.scrapers.spp_rt_constraints_scraper.RTConstraintsScraper.fetch_data",
               return_value=_fake_rt_constraints(interval)), \
         patch("src.scrapers.spp_rtlmp_scraper.RTBMLMPScraper.fetch_data", return_value=lmp), \
         patch("src.storage.duckdb_store.DuckDBStore.write_interval"), \
         patch("src.storage.duckdb_store.DuckDBStore.query_constraints", return_value=pd.DataFrame()):
        ok = asyncio.run(ingest_worker.ingest_once("spp"))

    assert ok
    shadow_prices = ingest_worker.cache.for_iso("spp").get_shadow_prices()
    assert shadow_prices == {"C_BINDING": -100.0, "C_BREACHED": -250.0}


def test_market_now_is_central_time():
    """Server may run anywhere; SPP file names are Central time."""
    now = ingest_worker._market_now()
    assert now.tzinfo is None
    central = datetime.now(ingest_worker.MARKET_TZ).replace(tzinfo=None)
    assert abs((now - central).total_seconds()) < 5


def test_daily_da_scrape_runs_and_is_written(monkeypatch):
    """
    The daily Day-Ahead block used `timedelta`, which a later local import turned into
    an unbound local for the whole function, so it raised on every cycle.
    """
    interval = pd.Timestamp("2026-10-10 12:00:00")
    monkeypatch.setattr("src.api.sf_cache.cache", MultiISOSFCache())
    monkeypatch.setattr(ingest_worker, "cache", __import__("src.api.sf_cache", fromlist=["cache"]).cache)
    monkeypatch.setattr(ingest_worker, "_last_da_scraped_date", None)
    monkeypatch.setattr(ingest_worker, "broadcast", lambda msg: asyncio.sleep(0))

    da = pd.DataFrame({
        "interval": [pd.Timestamp("2026-10-10 01:00:00")],
        "constraint_name": ["DA_C"],
        "shadow_price": [-5.0],
    })
    lmp = pd.DataFrame({"interval": [interval], "settlement_location": ["N1"], "mcc": [1.0]})

    with patch("src.scrapers.spp_rt_constraints_scraper.RTConstraintsScraper.fetch_data",
               return_value=_fake_rt_constraints(interval)), \
         patch("src.scrapers.spp_rtlmp_scraper.RTBMLMPScraper.fetch_data", return_value=lmp), \
         patch("src.scrapers.spp_da_constraints_scraper.DAConstraintsScraper.scrape_day", return_value=da), \
         patch("src.storage.duckdb_store.DuckDBStore.write_da_interval") as write_da, \
         patch("src.storage.duckdb_store.DuckDBStore.write_interval"), \
         patch("src.storage.duckdb_store.DuckDBStore.query_constraints", return_value=pd.DataFrame()):
        assert asyncio.run(ingest_worker.ingest_once("spp"))

    assert write_da.call_count == 2          # yesterday and today
    assert ingest_worker._last_da_scraped_date is not None


def test_forecast_training_reads_spp_history_from_multi_iso_cache(monkeypatch, tmp_path):
    """_train_one used to read `_constraints_history` off the multi-ISO wrapper and crash for every model."""
    from src.api import forecast_cache as fcache

    monkeypatch.setattr(fcache, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(fcache, "MODEL_REGISTRY_PATH", tmp_path / "model_registry.json")
    monkeypatch.setattr(fcache, "_registry", {})
    monkeypatch.setattr(fcache, "sf_cache", MultiISOSFCache())

    iv = pd.date_range("2026-10-01", periods=30, freq="5min")
    history = pd.DataFrame({"interval": iv, "shadow_price": -10.0})
    monkeypatch.setattr(fcache, "_load_constraint_history", lambda *a, **k: history)

    result = fcache._train_one("SOME_CONSTRAINT")

    assert result.n_training_rows == 30
    assert result.fallback            # too little data for ML, but training must not crash
