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


def test_cache_history_keeps_only_needed_columns():
    """The scraped frames carry many unused columns; holding them all cost ~0.9 GB per 7 days."""
    constraints_df, lmp_df, _ = make_multi_constraint_data(n_intervals=40, n_nodes=2)
    constraints_df["junk_text"] = "x" * 50
    lmp_df["pnode"] = "SOME_PNODE"
    lmp_df["lmp"] = 1.0

    cache = SFCache()
    cache.seed(constraints_df, lmp_df)

    assert set(cache._lmp_history.columns) == {"interval", "settlement_location", "mcc"}
    assert "junk_text" not in cache._constraints_history.columns
    assert cache.solution is not None and len(cache.solution._flat) > 0


def test_ingested_interval_is_not_labelled_utc(monkeypatch):
    interval = pd.Timestamp("2026-10-10 12:00:00")          # SPP label, Central time
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
        asyncio.run(ingest_worker.ingest_once("spp"))

    latest = ingest_worker.cache.for_iso("spp").latest_interval.interval
    assert latest.tzinfo is None
    assert latest == interval


def test_fit_with_da_gives_da_only_constraints_shift_factors():
    """A constraint only the Day-Ahead market priced has no RT shift factors; pooling DA data gives it some."""
    rt_c, rt_l, true = make_multi_constraint_data(n_intervals=60, n_nodes=3, seed=5)
    # DA: hourly, binds "LN DA ONLY" (not in RT) with MCC generated from known shift factors
    hours = pd.date_range("2026-11-02 01:00", periods=40, freq="h")
    rng = np.random.default_rng(1)
    sp = -rng.uniform(20, 120, len(hours))
    da_sf = {"NODE_00": 0.3, "NODE_01": -0.2, "NODE_02": 0.1}
    da_bc = pd.DataFrame({"interval": hours, "constraint_name": "LN DA ONLY", "shadow_price": sp})
    da_lmp = pd.DataFrame([{"interval": h, "settlement_location": n, "mcc": da_sf[n] * s}
                           for h, s in zip(hours, sp) for n in da_sf])
    cache = SFCache()
    cache.seed(rt_c, rt_l)
    assert cache.solution.get("NODE_00", "LN DA ONLY") is None

    assert cache.fit_with_da(da_bc, da_lmp)

    assert cache.solution_with_da.get("NODE_00", "LN DA ONLY") == pytest.approx(0.3, abs=0.02)
    assert cache.solution_with_da.get("NODE_01", "LN DA ONLY") == pytest.approx(-0.2, abs=0.02)
    # the RT constraints keep their shift factors
    assert cache.solution_with_da.get("NODE_00", "LN ALPHA - BETA") == pytest.approx(
        true["NODE_00"]["LN ALPHA - BETA"], abs=0.02)


def test_sibling_prior_borrows_from_the_same_element_and_is_labelled():
    """A newly binding constraint with no shift factors borrows its sibling's (oriented) vector."""
    from src.analysis.multi_constraint_sf import ShiftFactorSolution

    c = SFCache()
    sol = ShiftFactorSolution()
    sol._flat = {("N1", "SIB"): 0.4, ("N2", "SIB"): -0.2}
    c.solution = sol
    # SIB and NEWC are the SAME line under different contingencies; NEWC lists the opposite direction
    frame = pd.DataFrame({
        "constraint_name": ["SIB", "NEWC", "OTHER"],
        "monitored_facility": ["LN A - B", "LN B - A", "LN X - Y"],
        "contingent_facility": ["SPS:P Q:345:1:1", "SPS:R S:345:1:1", "BASE"],
    })
    c._learn_identity(frame)

    assert c.refresh_priors(["NEWC", "OTHER", "SIB"]) == 1        # only NEWC has a sibling; SIB is fitted
    assert c.provisional[("N1", "NEWC")] == pytest.approx(-0.4)   # direction flipped
    assert c.provisional[("N2", "NEWC")] == pytest.approx(0.2)
    assert ("N1", "OTHER") not in c.provisional
    assert all(k[1] != "SIB" for k in c.provisional)


def test_mcc_endpoint_keeps_fitted_and_prior_estimates_apart(monkeypatch):
    from fastapi.testclient import TestClient
    from src.analysis.multi_constraint_sf import ShiftFactorSolution
    from src.api.routers import mcc as mcc_router
    from src.api.server import app

    c = MultiISOSFCache()
    spp = c.for_iso("spp")
    sol = ShiftFactorSolution()
    sol._flat = {("N1", "FITTED"): 0.5}
    spp.solution = sol
    spp.provisional = {("N1", "BORROWED"): 0.1}
    spp.latest_interval = type("L", (), {"shadow_prices": {"FITTED": -100.0, "BORROWED": -100.0, "NONE": -200.0}})()
    spp.last_updated = datetime.now()
    monkeypatch.setattr(mcc_router, "cache", c)

    j = TestClient(app).get("/api/v1/mcc", params={"node": "N1"}).json()

    assert j["mcc"] == pytest.approx(0.5 * -100.0)                       # fitted only
    assert j["mcc_with_priors"] == pytest.approx(0.5 * -100.0 + 0.1 * -100.0)
    assert j["shadow_price_coverage"] == pytest.approx(100 / 400)
    assert j["shadow_price_coverage_with_priors"] == pytest.approx(200 / 400)
    src = {r["constraint"]: r["source"] for r in j["shadow_prices_used"]}
    assert src == {"FITTED": "fitted", "BORROWED": "sibling_prior", "NONE": None}


def test_ingest_populates_canonical_ids_from_the_structured_facility_columns(monkeypatch):
    interval = pd.Timestamp("2026-10-10 12:00:00")
    monkeypatch.setattr("src.api.sf_cache.cache", MultiISOSFCache())
    monkeypatch.setattr(ingest_worker, "cache", __import__("src.api.sf_cache", fromlist=["cache"]).cache)
    monkeypatch.setattr(ingest_worker, "_last_da_scraped_date", datetime.now().date())
    monkeypatch.setattr(ingest_worker, "broadcast", lambda msg: asyncio.sleep(0))
    rt = _fake_rt_constraints(interval)
    rt["monitored_facility"] = ["LN TEKAMHO - SUB1226", "LN FORMAN - FORMNWA", "LN A - B"]
    rt["contingent_facility"] = ["OPPD:SUB3456 NEBRCTY:345:1:11", "BASE", "BASE"]
    lmp = pd.DataFrame({"interval": [interval], "settlement_location": ["N1"], "mcc": [1.0]})

    with patch("src.scrapers.spp_rt_constraints_scraper.RTConstraintsScraper.fetch_data", return_value=rt), \
         patch("src.scrapers.spp_rtlmp_scraper.RTBMLMPScraper.fetch_data", return_value=lmp), \
         patch("src.storage.duckdb_store.DuckDBStore.write_interval"), \
         patch("src.storage.duckdb_store.DuckDBStore.query_constraints", return_value=pd.DataFrame()):
        asyncio.run(ingest_worker.ingest_once("spp"))

    recs = {r["constraint_name"]: r for r in ingest_worker.cache.for_iso("spp").latest_interval.constraints}
    assert recs["C_BINDING"]["canonical_id"] == "LN|SUB1226|TEKAMHO#C|NEBRCTY,SUB3456|345"
    assert recs["C_BINDING"]["element_key"] == "LN|SUB1226|TEKAMHO"
