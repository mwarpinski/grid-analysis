from datetime import datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.routers import geo
from src.api.server import app
from src.api.sf_cache import MultiISOSFCache

client = TestClient(app)


@pytest.fixture
def geo_dir(tmp_path, monkeypatch):
    table = pd.DataFrame({
        "constraint_name": ["KNOWN_GOOD", "KNOWN_APPROX", "KNOWN_NONE"],
        "lat": [36.5, 37.5, None], "lon": [-97.5, -98.5, None], "radius_km": [40.0, 110.0, None],
        "method": ["monitored_station", "contingency_station", "unplaced"],
        "quality": ["good", "approximate", "unplaced"],
        "element_key": ["LN|A|B", "LN|C|D", "LN|E|F"], "constraint_key": ["k1", "k2", "k3"],
    })
    table.to_parquet(tmp_path / "constraint_locations.parquet")
    pd.DataFrame({"node": ["U1"], "lat": [41.78], "lon": [-96.22], "plant_code": [1], "source": "x"}).to_parquet(tmp_path / "node_locations.parquet")
    bus = tmp_path / "bus_mapping.parquet"
    pd.DataFrame({"settlement_location": ["U1"], "ems_station": ["TEKAMHO"]}).to_parquet(bus)
    monkeypatch.setattr(geo, "_GEO_DIR", tmp_path)
    monkeypatch.setattr(geo, "_BUS_PATH", bus)
    geo._state.update(mtime=None, table=None, stations=None, nodes=None, extra={})
    yield tmp_path
    geo._state.update(mtime=None, table=None, stations=None, nodes=None, extra={})


def test_constraints_filters_by_quality(geo_dir):
    j = client.get("/api/v1/geo/constraints").json()
    assert {c["constraint_name"] for c in j["constraints"]} == {"KNOWN_GOOD", "KNOWN_APPROX"}
    assert "Regional positions" in j["note"]
    only_good = client.get("/api/v1/geo/constraints", params={"quality": "good"}).json()
    assert [c["constraint_name"] for c in only_good["constraints"]] == ["KNOWN_GOOD"]
    assert client.get("/api/v1/geo/constraints", params={"method": "contingency_station"}).json()["n"] == 1


def test_missing_table_is_a_helpful_404(tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "_GEO_DIR", tmp_path)
    geo._state.update(mtime=None, table=None, stations=None, nodes=None, extra={})
    r = client.get("/api/v1/geo/constraints")
    assert r.status_code == 404 and "build_geo_crosswalk" in r.json()["detail"]


def test_live_places_known_and_brand_new_constraints(geo_dir, monkeypatch):
    c = MultiISOSFCache()
    live = [
        {"constraint_name": "KNOWN_GOOD", "shadow_price": -50.0, "state": "BINDING"},
        {"constraint_name": "KNOWN_NONE", "shadow_price": -5.0, "state": "BINDING"},
        # a name the build never saw: placed from its facility string (TEKAMHO is a located station)
        {"constraint_name": "TMP999_1", "shadow_price": -300.0, "state": "BINDING",
         "monitored_facility": "LN TEKAMHO - SUB1226", "contingent_facility": "BASE"},
        {"constraint_name": "TMP998_1", "shadow_price": -10.0, "state": "BINDING",
         "monitored_facility": "LN NOWHERE - NOPLACE", "contingent_facility": "BASE"},
    ]
    c.for_iso("spp").latest_interval = type("L", (), {"constraints": live, "interval": datetime(2026, 10, 12, 12, 0)})()
    monkeypatch.setattr(geo, "cache", c)

    j = client.get("/api/v1/geo/live").json()

    by = {r["constraint_name"]: r for r in j["constraints"]}
    assert set(by) == {"KNOWN_GOOD", "TMP999_1"}                  # unplaced ones are omitted, not guessed
    assert j["n_binding"] == 4
    assert by["TMP999_1"]["method"] == "monitored_station" and by["TMP999_1"]["lat"] == pytest.approx(41.78)
    assert by["KNOWN_GOOD"]["shadow_price"] == -50.0
    assert j["constraints"][0]["constraint_name"] == "TMP999_1"     # most negative shadow price first


def test_live_requires_spp(geo_dir):
    assert client.get("/api/v1/geo/live", params={"iso": "miso"}).status_code == 400
