"""
The observed SPP network as a multigraph.

Vertices are stations (SPP EMS station names); edges are lines and transformers.  No geography is
involved.  The edge list is mined from SPP's own data: every constraint names the element it monitors
(``LN A - B``) and the outage under which it limits (``OPPD:SUB3456 NEBRCTY:345:1:11``, which also gives
the voltage and the CIRCUIT number, so parallel lines between the same two stations become parallel
edges).  Each row of the daily constraint files, binding or not, is an observation of a real element.

What this graph IS
    A subgraph of the real network: every element SPP has monitored or used as a contingency in the
    history we hold.  An edge listed here is a real element; the neighbours of a station are a LOWER
    BOUND (an element that never appeared is simply absent).  Absence of an edge is NOT evidence of
    non-adjacency.

What it is not
    * Complete: only elements that appear in some constraint show up, so congested corridors are dense
      and quiet parts of the system are missing.  Coverage grows with history but never reaches the full
      network (that needs SPP's network model).
    * Bus-level: a station is one vertex, so a transformer between two voltage levels at one station is
      a self-loop (kind ``XFMR``) carrying its voltages, not an edge between two buses.
    * Exact on parallels: circuits are only known from contingency strings; a parallel line that only
      ever appears as a monitored element (which has no circuit number) is not separable.
"""

from __future__ import annotations

from typing import Dict, Optional

import networkx as nx
import pandas as pd

from src.analysis import constraint_identity as ci

_OBS_COLS = ["source", "kind", "a", "b", "kv", "circuit", "transformer", "first_seen", "last_seen", "n_rows"]


def edge_observations(*frames: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    One row per distinct (source, kind, a, b, kv, circuit, transformer) seen in the constraint frames.

    Each frame needs ``interval``, ``monitored_facility`` and ``contingent_facility``.  Pass frames
    UNFILTERED: monitored elements that merely appeared in a flowgate (shadow price 0) are still real
    elements.  ``source`` is ``monitored`` (the limited element) or ``contingency`` (the outaged one).
    """
    rows = []
    for df in frames:
        if df is None or df.empty or "monitored_facility" not in df.columns:
            continue
        d = df[["interval", "monitored_facility"] + (["contingent_facility"] if "contingent_facility" in df.columns else [])]
        d = d.assign(interval=pd.to_datetime(d["interval"]))
        for col, source in (("monitored_facility", "monitored"), ("contingent_facility", "contingency")):
            if col not in d.columns:
                continue
            g = d.groupby(col)["interval"].agg(first_seen="min", last_seen="max", n_rows="size").reset_index()
            for r in g.itertuples(index=False):
                raw = getattr(r, col)
                if source == "monitored":
                    el = ci.parse_monitored(raw)
                    if el.kind in ("MULTI", "UNKNOWN") or len(el.ends) != 2:
                        continue
                    rows.append((source, el.kind, el.ends[0], el.ends[1], None, None, None,
                                 r.first_seen, r.last_seen, r.n_rows))
                else:
                    ct = ci.parse_contingency(raw)
                    if ct.base or any(s.startswith("UNIT:") for s in ct.stations):
                        continue
                    if ct.transformer is not None and len(ct.stations) == 1:
                        st = ct.stations[0]
                        rows.append((source, "XFMR", st, st, ct.kv, None, ct.transformer,
                                     r.first_seen, r.last_seen, r.n_rows))
                    elif ct.transformer is None and len(ct.stations) == 2:
                        rows.append((source, "LN", ct.stations[0], ct.stations[1], ct.kv, ct.circuit, None,
                                     r.first_seen, r.last_seen, r.n_rows))
                    # anything else (multi-element, named, truncated) names no single element: skipped
    if not rows:
        return pd.DataFrame(columns=_OBS_COLS)
    obs = pd.DataFrame(rows, columns=_OBS_COLS)
    return (obs.groupby(["source", "kind", "a", "b", "kv", "circuit", "transformer"], dropna=False)
               .agg(first_seen=("first_seen", "min"), last_seen=("last_seen", "max"), n_rows=("n_rows", "sum"))
               .reset_index())


def external_observations(branches: pd.DataFrame) -> pd.DataFrame:
    """
    Turn an externally supplied branch list into observations that ``build_multigraph`` can merge with
    the mined ones (``pd.concat([edge_observations(...), external_observations(df)])``).

    ``branches`` needs ``a`` and ``b`` (station names, same spelling as SPP's EMS names); optional
    ``kind`` (LN/XFMR/PS, default LN), ``kv``, ``circuit``.  This is the hook for a complete network
    model (e.g. a buyer's licensed bus/branch list): the mined graph is a fragment, a model is the whole.
    Station names are upper-cased and stripped; a == b is a transformer.
    """
    b = branches.copy()
    b["a"] = b["a"].astype(str).str.upper().str.strip()
    b["b"] = b["b"].astype(str).str.upper().str.strip()
    lo = b[["a", "b"]].min(axis=1)
    hi = b[["a", "b"]].max(axis=1)
    now = pd.Timestamp.now()
    out = pd.DataFrame({
        "source": "external",
        "kind": b["kind"].astype(str).str.upper() if "kind" in b else "LN",
        "a": lo, "b": hi,
        "kv": pd.to_numeric(b["kv"], errors="coerce") if "kv" in b else None,
        "circuit": b["circuit"].astype(str) if "circuit" in b else None,
        "transformer": None,
        "first_seen": now, "last_seen": now, "n_rows": 1,
    })
    out.loc[out["a"] == out["b"], "kind"] = out.loc[out["a"] == out["b"], "kind"].where(out["kind"] != "LN", "XFMR")
    return out[_OBS_COLS]


def build_multigraph(obs: pd.DataFrame) -> nx.MultiGraph:
    """
    Merge observations into a multigraph.  Edges between the same two stations of the same ``kind`` are
    one physical element unless contingency strings show several distinct circuits, in which case there
    is one edge per circuit.  Edge attributes: kind, kv (set), circuit, sources (set of
    monitored/contingency), n_rows, first_seen, last_seen.
    """
    G = nx.MultiGraph()
    if obs.empty:
        return G
    o = obs.copy()
    o["pair"] = list(zip(o["a"], o["b"]))
    for (kind, a, b), g in o.groupby(["kind", "a", "b"], sort=False):
        circuits = sorted({c for c in g["circuit"].dropna()})
        xfs = sorted({t for t in g["transformer"].dropna()}) if kind == "XFMR" else []
        # one edge per distinct circuit / transformer; otherwise a single edge for the element
        labels = circuits or xfs or [None]
        for lab in labels:
            sel = g if lab is None else g[(g["circuit"] == lab) | (g["transformer"] == lab)]
            sel_all = g if lab is None or sel.empty else pd.concat([sel, g[g["circuit"].isna() & g["transformer"].isna()]])
            G.add_edge(
                a, b, kind=kind, label=lab,
                kv=sorted({float(k) for k in sel_all["kv"].dropna()}),
                sources=sorted(set(sel_all["source"])),
                n_rows=int(sel_all["n_rows"].sum()),
                first_seen=sel_all["first_seen"].min(), last_seen=sel_all["last_seen"].max(),
            )
    for n in G.nodes:
        kvs = sorted({k for _, _, d in G.edges(n, data=True) for k in d.get("kv", [])})
        G.nodes[n]["kv"] = kvs
    return G


def summary(G: nx.MultiGraph) -> Dict[str, object]:
    """Size and shape of the observed graph."""
    if G.number_of_nodes() == 0:
        return {"stations": 0, "edges": 0}
    simple = nx.Graph(G)                                # collapse parallels for component/degree stats
    comps = sorted((len(c) for c in nx.connected_components(simple)), reverse=True)
    deg = pd.Series(dict(simple.degree()))
    kinds = pd.Series([d["kind"] for _, _, d in G.edges(data=True)]).value_counts().to_dict()
    pairs = pd.Series([tuple(sorted((u, v))) for u, v, d in G.edges(data=True) if d["kind"] == "LN"]).value_counts()
    return {
        "stations": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "edges_by_kind": kinds,
        "self_loops_transformers": int(nx.number_of_selfloops(G)),
        "station_pairs_with_parallel_lines": int((pairs > 1).sum()),
        "components": len(comps),
        "largest_component": comps[0],
        "component_sizes_top5": comps[:5],
        "isolated_pairs_or_tiny_components(<=3)": int(sum(1 for c in comps if c <= 3)),
        "degree_median": float(deg.median()),
        "degree_max": int(deg.max()),
        "share_degree_1": float((deg == 1).mean()),
    }


def neighbors(G: nx.MultiGraph, station: str) -> pd.DataFrame:
    """
    Stations directly joined to ``station`` by at least one observed edge, with the number of parallel
    edges, kinds, voltages and how each edge was observed.  A LOWER BOUND on the true neighbours.
    """
    station = str(station).upper().strip()
    if station not in G:
        return pd.DataFrame()
    rows = {}
    for u, v, d in G.edges(station, data=True):
        if u == v:                     # a transformer at the station itself is not a neighbouring station
            continue
        other = v if u == station else u
        r = rows.setdefault(other, {"neighbor": other, "n_edges": 0, "kinds": set(), "kv": set(), "sources": set()})
        r["n_edges"] += 1
        r["kinds"].add(d["kind"])
        r["kv"].update(d.get("kv", []))
        r["sources"].update(d.get("sources", []))
    if not rows:
        return pd.DataFrame(columns=["neighbor", "n_edges", "kinds", "kv", "sources"])
    out = pd.DataFrame(rows.values())
    for c in ("kinds", "kv", "sources"):
        out[c] = out[c].map(lambda s: sorted(s))
    return out.sort_values(["n_edges", "neighbor"], ascending=[False, True]).reset_index(drop=True)
