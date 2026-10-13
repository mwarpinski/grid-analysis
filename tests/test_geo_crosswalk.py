import numpy as np
import pandas as pd
import pytest

from src.analysis import geo_crosswalk as gx


def _loc(rows):
    return pd.DataFrame(rows, columns=["node", "lat", "lon", "plant_code"])


def test_haversine_known_distance():
    # Kansas City to Oklahoma City is about 480 km along the great circle
    d = float(gx.haversine_km(39.10, -94.58, 35.47, -97.52))
    assert 470 < d < 490
    assert float(gx.haversine_km(35.0, -97.0, 35.0, -97.0)) == pytest.approx(0.0)


def test_node_locations_from_eia_matches_the_node_designation_and_filters_bbox():
    plants = pd.DataFrame({"Plant Code": [1, 2, 3], "Latitude": [36.0, 37.0, 10.0], "Longitude": [-97.0, -98.0, -97.0]})
    gens = pd.DataFrame({
        "Plant Code": [1, 2, 3, 1],
        "RTO/ISO LMP Node Designation": ["spp.node.a ", "NODE_B", "NODE_C", None],
        "RTO/ISO Location Designation for Reporting Wholesale Sales Data to FERC": [None, None, None, "NODE_D"],
    })
    out = gx.node_locations_from_eia(plants, gens, ["SPP.NODE.A", "NODE_B", "NODE_C", "NODE_D", "OTHER"])
    assert set(out["node"]) == {"SPP.NODE.A", "NODE_B", "NODE_D"}         # NODE_C is outside the bbox
    assert out.set_index("node").loc["NODE_B", "lat"] == 37.0


def test_triangulation_places_a_constraint_near_its_sensitive_nodes():
    loc = _loc([("A", 36.0, -97.0, 1), ("B", 36.2, -97.2, 2), ("C", 35.9, -96.9, 3), ("D", 36.1, -97.1, 4),
                ("FAR1", 45.0, -100.0, 5), ("FAR2", 30.0, -92.0, 6)])
    X = pd.DataFrame({"C1": [0.5, 0.45, 0.4, 0.48, 0.01, 0.02]}, index=loc["node"])
    t = gx.triangulate_constraints(X, loc, min_nodes=4)
    assert t.loc["C1", "lat"] == pytest.approx(36.05, abs=0.2)
    assert t.loc["C1", "lon"] == pytest.approx(-97.05, abs=0.2)
    assert t.loc["C1", "radius_km"] < 30                                  # evidence is concentrated
    assert t.loc["C1", "n_sites"] == 4


def test_diffuse_evidence_gives_a_large_radius():
    loc = _loc([("A", 45.0, -100.0, 1), ("B", 30.0, -92.0, 2), ("C", 37.0, -105.0, 3), ("D", 33.0, -90.0, 4)])
    X = pd.DataFrame({"C1": [0.4, 0.4, 0.4, 0.4]}, index=loc["node"])
    assert gx.triangulate_constraints(X, loc, min_nodes=4).loc["C1", "radius_km"] > 500


def test_same_plant_nodes_count_once():
    loc = _loc([("U1", 36.0, -97.0, 1), ("U2", 36.0, -97.0, 1), ("U3", 36.0, -97.0, 1), ("B", 37.0, -98.0, 2),
                ("C", 37.1, -98.1, 3), ("D", 37.2, -98.2, 4)])
    X = pd.DataFrame({"C1": [0.5, 0.5, 0.5, 0.4, 0.4, 0.4]}, index=loc["node"])
    t = gx.triangulate_constraints(X, loc, min_nodes=4)
    assert t.loc["C1", "n_sites"] == 4            # 4 distinct plants, not 6 nodes


def test_too_few_sites_means_no_placement():
    loc = _loc([("A", 36.0, -97.0, 1), ("B", 37.0, -98.0, 2)])
    X = pd.DataFrame({"C1": [0.5, 0.4]}, index=loc["node"])
    assert np.isnan(gx.triangulate_constraints(X, loc, min_nodes=4).loc["C1", "lat"])


def test_station_coordinates_average_the_nodes_at_a_station():
    bus = pd.DataFrame({"settlement_location": ["U1", "U2"], "ems_station": ["stn1", "STN1"]})
    loc = pd.DataFrame({"node": ["U1", "U2"], "lat": [36.0, 36.2], "lon": [-97.0, -97.2]})
    st = gx.station_coordinates(bus, loc).set_index("station")
    assert st.loc["STN1", "lat"] == pytest.approx(36.1)


def test_place_constraints_prefers_anchors_then_triangulation_and_rounds():
    stations = pd.DataFrame({"station": ["TEKAMHO", "OSAGE"], "lat": [41.7821, 36.9], "lon": [-96.2234, -96.5]})
    meta = pd.DataFrame({
        "constraint_name": ["MON", "CON", "TRI_GOOD", "TRI_COARSE", "TRI_DIFFUSE", "NONE"],
        "monitored_facility": ["LN TEKAMHO - SUB1226", "LN X - Y", "LN P - Q", "LN P - Q", "LN P - Q", "LN P - Q"],
        "contingent_facility": ["BASE", "OKGE:OSAGE ZZZ:345:1:1", "BASE", "BASE", "BASE", "BASE"],
    })
    tri = pd.DataFrame({"lat": [40.0, 39.0, 38.0], "lon": [-95.0, -96.0, -97.0], "radius_km": [60.0, 180.0, 400.0]},
                       index=["TRI_GOOD", "TRI_COARSE", "TRI_DIFFUSE"])
    out = gx.place_constraints(meta, stations, tri).set_index("constraint_name")

    assert out.loc["MON", "method"] == "monitored_station" and out.loc["MON", "quality"] == "good"
    assert out.loc["MON", "lat"] == 41.78 and out.loc["MON", "lon"] == -96.22          # rounded to 0.01 deg
    assert out.loc["CON", "method"] == "contingency_station" and out.loc["CON", "quality"] == "approximate"
    assert out.loc["TRI_GOOD", "method"] == "triangulation" and out.loc["TRI_GOOD", "quality"] == "good"
    assert out.loc["TRI_COARSE", "quality"] == "approximate"
    assert out.loc["TRI_DIFFUSE", "method"] == "unplaced" and np.isnan(out.loc["TRI_DIFFUSE", "lat"])
    assert out.loc["NONE", "method"] == "unplaced"
