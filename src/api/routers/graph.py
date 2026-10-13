"""
Graph Router — /api/v1/graph/*

The observed SPP network as a multigraph (stations = vertices, lines/transformers = edges), mined from
the monitored and contingent facilities in SPP's own constraint files.  See src/analysis/grid_graph.py.

IMPORTANT: this is a FRAGMENT of the real network, not the whole of it.  An edge reported here is a real
element named by SPP; a missing edge means "never observed", not "not connected".  Neighbour lists are
lower bounds, and `connected: false` in /path means only "no observed route".
"""

from __future__ import annotations

import time
from typing import Optional

import networkx as nx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.analysis import grid_graph as gg
from src.storage.duckdb_store import store

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])

_TTL_SECONDS = 600
_cache: dict = {}

COMPLETENESS_NOTE = (
    "Observed subgraph only: edges are real elements SPP has monitored or used as a contingency; "
    "absent edges are unknown, not non-existent. Parallel lines are only distinguishable when contingency "
    "strings show distinct circuit numbers."
)


def _graph(days: int) -> nx.MultiGraph:
    hit = _cache.get(days)
    if hit and time.time() - hit[0] < _TTL_SECONDS:
        return hit[1]
    end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days + 1)
    cols = ["monitored_facility", "contingent_facility"]
    rt = store.query_constraints(start, end, columns=cols)
    da = store.query_da_constraints(start, end, columns=cols)
    obs = gg.edge_observations(rt, da)
    if obs.empty:
        raise HTTPException(status_code=404, detail="No constraint history with facility columns in the window.")
    G = gg.build_multigraph(obs)
    _cache[days] = (time.time(), G)
    return G


@router.get("/summary")
async def graph_summary(days: int = Query(365, ge=1, le=3650)):
    """Size and shape of the observed graph (stations, edges, components, degrees)."""
    G = _graph(days)
    return {"days": days, "note": COMPLETENESS_NOTE, **gg.summary(G)}


@router.get("/neighbors")
async def graph_neighbors(station: str = Query(..., description="EMS station name, e.g. CLEVLND7"),
                          days: int = Query(365, ge=1, le=3650)):
    """Stations directly joined to ``station`` by at least one observed edge (a LOWER BOUND)."""
    G = _graph(days)
    st = station.upper().strip()
    if st not in G:
        raise HTTPException(status_code=404, detail=f"Station '{station}' is not in the observed graph.")
    out = gg.neighbors(G, st)
    transformers = sum(1 for u, v, d in G.edges(st, data=True) if u == v)
    return {"station": st, "n_neighbors": int(len(out)), "transformers_at_station": transformers, "complete": False,
            "note": COMPLETENESS_NOTE, "neighbors": out.to_dict(orient="records")}


@router.get("/path")
async def graph_path(a: str = Query(...), b: str = Query(...), days: int = Query(365, ge=1, le=3650)):
    """Shortest observed route between two stations. `connected: false` means no OBSERVED route."""
    G = nx.Graph(_graph(days))
    a, b = a.upper().strip(), b.upper().strip()
    missing = [x for x in (a, b) if x not in G]
    if missing:
        raise HTTPException(status_code=404, detail=f"Not in the observed graph: {', '.join(missing)}")
    try:
        path = nx.shortest_path(G, a, b)
    except nx.NetworkXNoPath:
        return {"a": a, "b": b, "connected": False, "note": "No observed route; this does not mean the stations are disconnected."}
    return {"a": a, "b": b, "connected": True, "hops": len(path) - 1, "path": path, "note": COMPLETENESS_NOTE}


@router.get("/edges")
async def graph_edges(days: int = Query(365, ge=1, le=3650), kind: Optional[str] = Query(None),
                      limit: int = Query(1000, ge=1, le=20000)):
    """The observed edge list (multigraph: parallel circuits appear as separate rows)."""
    G = _graph(days)
    rows = [{"a": u, "b": v, "kind": d["kind"], "label": d.get("label"), "kv": d.get("kv", []),
             "sources": d.get("sources", []), "n_rows": d.get("n_rows")} for u, v, d in G.edges(data=True)]
    if kind:
        rows = [r for r in rows if r["kind"] == kind.upper()]
    return {"n": len(rows), "note": COMPLETENESS_NOTE, "edges": rows[:limit]}
