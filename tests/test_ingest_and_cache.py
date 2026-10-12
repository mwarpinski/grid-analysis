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
