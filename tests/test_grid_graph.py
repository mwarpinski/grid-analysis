import networkx as nx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.analysis import grid_graph as gg
from src.api.routers import graph as graph_router
from src.api.server import app


def _frame(rows):
    df = pd.DataFrame(rows, columns=["monitored_facility", "contingent_facility"])
    df["interval"] = pd.Timestamp("2026-10-10 12:00")
    return df


def test_monitored_and_contingent_elements_become_edges():
    obs = gg.edge_observations(_frame([
        ("LN A - B", "BASE"),
        ("LN C - D", "OPPD:A E:345:1:11"),          # contingency adds the line A-E at 345 kV, circuit 1
    ]))
    G = gg.build_multigraph(obs)
    assert {frozenset((u, v)) for u, v in G.edges()} == {frozenset("AB"), frozenset("CD"), frozenset("AE")}
    assert G.number_of_edges() == 3
    nb = gg.neighbors(G, "a")                         # case-insensitive
    assert set(nb["neighbor"]) == {"B", "E"}
    e = nb.set_index("neighbor").loc["E"]
    assert e["kv"] == [345.0] and e["sources"] == ["contingency"]


def test_direction_does_not_create_a_second_edge():
    G = gg.build_multigraph(gg.edge_observations(_frame([("LN A - B", "BASE"), ("LN B - A", "BASE")])))
    assert G.number_of_edges() == 1


def test_distinct_circuits_become_parallel_edges_in_a_multigraph():
    obs = gg.edge_observations(_frame([("LN X - Y", "OPPD:A B:345:1:1"), ("LN X - Y", "OPPD:A B:345:2:1")]))
    G = gg.build_multigraph(obs)
    assert G.number_of_edges(*"AB") == 2               # two circuits between A and B
    assert gg.summary(G)["station_pairs_with_parallel_lines"] == 1
    assert gg.neighbors(G, "A").set_index("neighbor").loc["B", "n_edges"] == 2


def test_transformers_are_self_loops_and_unusable_contingencies_are_skipped():
    obs = gg.edge_observations(_frame([
        ("XFMR T - T", "NSP:LYON_CO XF TR9_3:345 115 34.5:"),
        ("Multi-Element Constraint", "OKGE:KETCH OMMITCH4 NOWST1 DANFO W"),    # several stations: no single element
        ("LN P - Q", "SPS:HOBBS:COMBINEDCYCLE"),                              # a generating unit, not a line
        ("LN P - Q", "ADD NEW EAST"),
    ]))
    G = gg.build_multigraph(obs)
    assert nx.number_of_selfloops(G) == 2              # T (monitored) and LYON_CO (contingency)
    assert set(G.nodes) == {"T", "LYON_CO", "P", "Q"}
    assert G.number_of_edges() == 3                    # T-T, LYON_CO-LYON_CO, P-Q


def test_summary_counts_components_and_degrees():
    G = gg.build_multigraph(gg.edge_observations(_frame([("LN A - B", "BASE"), ("LN B - C", "BASE"), ("LN X - Y", "BASE")])))
    s = gg.summary(G)
    assert s["stations"] == 5 and s["edges"] == 3 and s["components"] == 2 and s["largest_component"] == 3
    assert s["degree_max"] == 2


def test_external_branch_list_merges_into_the_graph():
    mined = gg.edge_observations(_frame([("LN A - B", "BASE")]))
    ext = gg.external_observations(pd.DataFrame({"a": ["b", "c", "d"], "b": ["c", "d", "d"], "kv": [345, 138, 230]}))
    G = gg.build_multigraph(pd.concat([mined, ext], ignore_index=True))
    assert nx.is_connected(nx.Graph(G))                # the external list bridges the fragments
    assert {d["kind"] for _, _, d in G.edges(data=True)} == {"LN", "XFMR"}          # d-d became a transformer
    assert gg.neighbors(G, "B").set_index("neighbor").loc["C", "sources"] == ["external"]


@pytest.fixture
def client(monkeypatch):
    G = gg.build_multigraph(gg.edge_observations(_frame([("LN A - B", "BASE"), ("LN B - C", "BASE"), ("LN X - Y", "BASE")])))
    monkeypatch.setattr(graph_router, "_graph", lambda days: G)
    return TestClient(app)


def test_graph_api(client):
    assert client.get("/api/v1/graph/summary").json()["components"] == 2
    n = client.get("/api/v1/graph/neighbors", params={"station": "b"}).json()
    assert {x["neighbor"] for x in n["neighbors"]} == {"A", "C"} and n["complete"] is False
    assert client.get("/api/v1/graph/neighbors", params={"station": "ZZZ"}).status_code == 404
    p = client.get("/api/v1/graph/path", params={"a": "A", "b": "C"}).json()
    assert p["connected"] and p["hops"] == 2 and p["path"] == ["A", "B", "C"]
    far = client.get("/api/v1/graph/path", params={"a": "A", "b": "X"}).json()
    assert far["connected"] is False and "does not mean" in far["note"]
    assert client.get("/api/v1/graph/edges", params={"kind": "LN"}).json()["n"] == 3


def test_a_transformer_is_not_a_neighbour_of_its_own_station():
    G = gg.build_multigraph(gg.edge_observations(_frame([("XFMR T - T", "BASE"), ("LN T - U", "BASE")])))
    assert list(gg.neighbors(G, "T")["neighbor"]) == ["U"]
    assert gg.neighbors(gg.build_multigraph(gg.edge_observations(_frame([("XFMR S - S", "BASE")]))), "S").empty
