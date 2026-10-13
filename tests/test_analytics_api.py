"""API tests for /api/v1/analytics/* using synthetic frames (no network, no stored data)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.routers import analytics
from src.api.server import app
from src.api.sf_cache import MultiISOSFCache
from tests.test_congestion_analytics import SF, H1, H2, _da, _rt_5min
from src.analysis import congestion_analytics as ca

client = TestClient(app)


@pytest.fixture(autouse=True)
def _synthetic_world(monkeypatch):
    from src.analysis.multi_constraint_sf import ShiftFactorSolution

    c = MultiISOSFCache()
    sol = ShiftFactorSolution()
    sol._flat = dict(SF)
    c.for_iso("spp").solution = sol
    monkeypatch.setattr(analytics, "cache", c)
    analytics._frames.clear()

    rt_bc, rt_lmp = _rt_5min(hours=2, sp_by_hour={H1: {"C1": -100.0, "C2": -20.0}, H2: {"C1": -50.0}})
    da_bc, da_lmp = _da(hours=2, sp_by_hour={H1: {"C1": -40.0}, H2: {"C1": -80.0, "C2": -10.0}})
    rt_bc["state"] = "BINDING"
    rt_bc["monitored_facility"] = "LN X - Y"
    rt_bc["source_limit_mw"] = 200.0
    da_bc["monitored_facility"] = "LN X - Y"
    dart = ca.nodal_dart(da_lmp, rt_lmp)
    monkeypatch.setattr(analytics, "_load_dart", lambda days: dart)
    monkeypatch.setattr(analytics, "_load_bc", lambda days: (rt_bc, da_bc))
    # the fixture's hours are in 2026-10-10; keep the decomposition's recency filter out of the way
    monkeypatch.setattr(analytics.pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2026-10-10 12:00")))
    yield
    analytics._frames.clear()


def test_dart_nodes():
    r = client.get("/api/v1/analytics/dart/nodes", params={"min_hours": 1})
    assert r.status_code == 200
    j = r.json()
    assert j["n_nodes"] == 3 and j["n_hours"] == 2
    assert {"mean", "std", "hit_rate", "naive_t_stat"} <= set(j["nodes"][0])


def test_dart_decompose_explains_the_spread_exactly():
    r = client.get("/api/v1/analytics/dart/decompose", params={"node": "N1", "days": 3})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["r2"] is None or j["r2"] == pytest.approx(1.0)
    assert j["unattributed_share"] == 0.0
    assert j["in_sample"] is True
    assert j["mean_unexplained_offset"] == pytest.approx(0.0, abs=1e-9)
    assert {c["constraint"] for c in j["constraints"]} == {"C1", "C2"}


def test_dart_decompose_unknown_node_is_404():
    assert client.get("/api/v1/analytics/dart/decompose", params={"node": "NOPE", "days": 3}).status_code == 404


def test_constraint_stats_and_grouping():
    r = client.get("/api/v1/analytics/constraints/stats", params={"group_by": "monitored_facility"})
    assert r.status_code == 200
    assert r.json()["constraints"][0]["monitored_facility"] == "LN X - Y"
    assert client.get("/api/v1/analytics/constraints/stats", params={"group_by": "bogus"}).status_code == 422


def test_pairs():
    r = client.get("/api/v1/analytics/pairs", params={"constraint": "C1", "top": 3})
    assert r.status_code == 200
    first = r.json()["pairs"][0]
    assert first["high_mcc_node"] == "N1" and first["low_mcc_node"] == "N2"       # SF −0.5 vs +0.4
    assert first["delta_sf"] == pytest.approx(0.9)
    assert client.get("/api/v1/analytics/pairs", params={"constraint": "NOPE"}).status_code == 404


def test_injection_change_scenario():
    r = client.post("/api/v1/analytics/scenario/injection-change", json={"node": "N1", "injection_change_mw": -100.0})
    assert r.status_code == 200
    rows = {c["constraint"]: c for c in r.json()["constraints"]}
    assert rows["C1"]["delta_flow_mw"] == pytest.approx(50.0)                    # −0.5 × −100
    assert rows["C1"]["delta_pct_of_limit"] == pytest.approx(25.0)               # 50 MW of a 200 MW rating
    assert client.post("/api/v1/analytics/scenario/injection-change",
                       json={"node": "NOPE", "injection_change_mw": -1}).status_code == 404


def test_constraint_scenario_explicit_and_quantile_and_validation():
    r = client.post("/api/v1/analytics/scenario/constraints", json={"shadow_prices": {"C1": -300.0, "GHOST": -5.0}})
    assert r.status_code == 200
    j = r.json()
    assert j["unmodeled"] == ["GHOST"]
    n1 = next(n for n in j["nodes"] if n["node"] == "N1")
    assert n1["mcc_change"] == pytest.approx(-0.5 * -300.0)

    q = client.post("/api/v1/analytics/scenario/constraints", json={"constraints": ["C1"], "quantile": 1.0})
    assert q.status_code == 200 and q.json()["shadow_prices"]["C1"] == pytest.approx(-100.0)   # most severe RT price
    assert client.post("/api/v1/analytics/scenario/constraints", json={}).status_code == 422


def test_non_spp_is_rejected():
    assert client.get("/api/v1/analytics/pairs", params={"constraint": "C1", "iso": "miso"}).status_code == 400
