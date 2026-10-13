"""
Geographic crosswalk for SPP nodes and constraints, from public data only.

SPP's constraint and substation names are EMS abbreviations that do not appear in public GIS data
(only ~21% of public lines even carry substation names), so names alone cannot place constraints on
a map.  Two independent routes are combined:

1. Nodes -> coordinates.  The EIA-860 generator table lists an "RTO/ISO LMP Node Designation" for
   many generators; where it equals an SPP settlement location the plant's public coordinates give
   that node a position.

2. Constraints -> coordinates by TRIANGULATION.  A constraint's shift factors say which nodes it is
   sensitive to; nodes that are electrically close to a constraint are geographically close to it.
   The constraint is placed at the |shift factor|-weighted centroid of its most sensitive located
   nodes.  This works because shift factors encode geography: leave-one-out, a node's location is
   recovered from its shift-factor neighbours to a median of ~48 km (vs ~350 km for guessing the
   centroid), excluding neighbours at the same plant.  So placements are REGIONAL (tens of km),
   not substation-precise, and every placement carries a ``radius_km`` uncertainty.

Coordinates are rounded to 0.01 degrees (~1 km) on output; that is far finer than the method's
accuracy and avoids publishing precise infrastructure positions.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
COORD_DECIMALS = 2
_NODE_COLS = (
    "RTO/ISO LMP Node Designation",
    "RTO/ISO Location Designation for Reporting Wholesale Sales Data to FERC",
)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (broadcasts over numpy arrays)."""
    p = np.pi / 180.0
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Nodes -> coordinates (EIA-860)
# ---------------------------------------------------------------------------

def node_locations_from_eia(
    plants: pd.DataFrame,
    generators: pd.DataFrame,
    settlement_locations: Iterable[str],
    bbox=(25.0, 50.0, -110.0, -88.0),
) -> pd.DataFrame:
    """
    Locate SPP settlement locations that EIA-860 lists under an RTO/ISO node designation.

    ``plants`` needs ``Plant Code, Latitude, Longitude``; ``generators`` needs ``Plant Code`` and the
    node-designation columns.  Returns ``node, lat, lon, plant_code, source`` (full precision;
    round on output).  ``bbox`` = (lat_min, lat_max, lon_min, lon_max) drops obviously wrong rows.
    """
    pl = plants[["Plant Code", "Latitude", "Longitude"]].copy()
    pl["Latitude"] = pd.to_numeric(pl["Latitude"], errors="coerce")
    pl["Longitude"] = pd.to_numeric(pl["Longitude"], errors="coerce")
    g = generators.merge(pl, on="Plant Code", how="left")
    wanted = {str(s).upper().strip(): str(s) for s in settlement_locations}
    rows = []
    for col in _NODE_COLS:
        if col not in g.columns:
            continue
        t = g.dropna(subset=[col, "Latitude", "Longitude"]).copy()
        t["key"] = t[col].astype(str).str.upper().str.strip()
        t = t[t["key"].isin(wanted)]
        rows.append(pd.DataFrame({
            "node": t["key"].map(wanted), "lat": t["Latitude"], "lon": t["Longitude"],
            "plant_code": t["Plant Code"], "source": "eia860_node_designation",
        }))
    if not rows:
        return pd.DataFrame(columns=["node", "lat", "lon", "plant_code", "source"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates("node")
    lat0, lat1, lon0, lon1 = bbox
    return out[out["lat"].between(lat0, lat1) & out["lon"].between(lon0, lon1)].reset_index(drop=True)


_STOP_WORDS = {
    "WIND", "FARM", "SOLAR", "ENERGY", "CENTER", "STATION", "GENERATING", "PLANT", "POWER", "PROJECT",
    "GEN", "UNIT", "FACILITY", "LLC", "THE", "ELECTRIC", "STORAGE", "BATTERY", "HYDRO", "DAM", "COGEN", "BIOMASS",
}
_FUEL_KEYS = (("WND", "WIND"), ("WIND", "WIND"), ("SUN", "SOLAR"), ("SOLAR", "SOLAR"), ("PV", "SOLAR"),
              ("NG", "GAS"), ("GAS", "GAS"), ("COL", "COAL"), ("COAL", "COAL"), ("NUC", "NUCLEAR"),
              ("WAT", "HYDRO"), ("HYD", "HYDRO"), ("OIL", "OIL"), ("DFO", "OIL"))


def _norm(s) -> str:
    import re
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _fuel_class(f) -> str:
    f = str(f).upper()
    for k, v in _FUEL_KEYS:
        if k in f:
            return v
    return "OTHER"


def fuzzy_node_locations(
    bus_mapping: pd.DataFrame,
    plants: pd.DataFrame,
    generators: pd.DataFrame,
    already_located: Iterable[str] = (),
    min_score: float = 0.8,
    min_margin: float = 0.1,
    bbox=(25.0, 50.0, -110.0, -88.0),
) -> pd.DataFrame:
    """
    Locate generator settlement locations that EIA-860 does not list by node name, by matching SPP's
    own station table (``settlement_location, ems_station, ems_mw_max, fuel_type``) to EIA plants on
    name, fuel and capacity.

    Score = 0.6·name + 0.25·fuel + 0.15·capacity; a match is accepted only if its score is at least
    ``min_score`` AND it beats the runner-up by ``min_margin``.  On held-out nodes whose true plant
    is known, that setting answered ~37% of cases with ~99% precision (looser settings answer more
    but are wrong 20-30% of the time), so the defaults are deliberately strict.
    """
    from rapidfuzz import fuzz
    import re

    skip = {str(x) for x in already_located}
    bus = bus_mapping.copy()
    bus["sl"] = bus["settlement_location"].astype(str)
    bus = bus[~bus["sl"].isin(skip)]
    if bus.empty:
        return pd.DataFrame(columns=["node", "lat", "lon", "plant_code", "source", "score"])
    bus["fuel"] = bus["fuel_type"].map(_fuel_class) if "fuel_type" in bus else "OTHER"
    bus["mw"] = pd.to_numeric(bus.get("ems_mw_max"), errors="coerce")
    bus["sln"] = bus["sl"].map(_norm)
    bus["stn"] = bus["ems_station"].map(_norm) if "ems_station" in bus else ""

    pl = plants[["Plant Code", "Plant Name", "Latitude", "Longitude"]].copy()
    for c in ("Latitude", "Longitude"):
        pl[c] = pd.to_numeric(pl[c], errors="coerce")
    lat0, lat1, lon0, lon1 = bbox
    pl = pl.dropna(subset=["Latitude", "Longitude"])
    pl = pl[pl["Latitude"].between(lat0, lat1) & pl["Longitude"].between(lon0, lon1)]
    g = generators.copy()
    g["mw"] = pd.to_numeric(g["Nameplate Capacity (MW)"], errors="coerce")
    agg = g.groupby("Plant Code").agg(
        mw_max=("mw", "max"),
        fuels=("Energy Source 1", lambda x: {_fuel_class(v) for v in x}),
        techs=("Technology", lambda x: {_fuel_class(v) for v in x}),
    ).reset_index()
    pl = pl.merge(agg, on="Plant Code", how="left")
    pl["tok"] = pl["Plant Name"].map(
        lambda n: [t for t in re.findall(r"[A-Z]{4,}", str(n).upper()) if t not in _STOP_WORDS]
    )
    pl = pl[pl["tok"].map(len) > 0].reset_index(drop=True)

    rows = []
    for _, r in bus.iterrows():
        cands = []
        for _, p in pl.iterrows():
            name = max(max(fuzz.partial_ratio(t, r["sln"]), fuzz.partial_ratio(t, r["stn"])) for t in p["tok"])
            if name < 80:
                continue
            fuels = (p["fuels"] if isinstance(p["fuels"], set) else set()) | (p["techs"] if isinstance(p["techs"], set) else set())
            fuel = 1.0 if r["fuel"] in fuels else (0.5 if r["fuel"] == "OTHER" else 0.0)
            cap = 0.5 if pd.isna(r["mw"]) or pd.isna(p["mw_max"]) else float(
                np.exp(-abs(np.log(max(r["mw"], 0.1) / max(p["mw_max"], 0.1)))))
            cands.append((0.6 * name / 100 + 0.25 * fuel + 0.15 * cap, p))
        if not cands:
            continue
        cands.sort(key=lambda t: t[0], reverse=True)
        score = cands[0][0]
        margin = score - (cands[1][0] if len(cands) > 1 else 0.0)
        if score >= min_score and margin >= min_margin:
            p = cands[0][1]
            rows.append({"node": r["sl"], "lat": p["Latitude"], "lon": p["Longitude"],
                         "plant_code": p["Plant Code"], "source": "eia860_name_match", "score": score})
    return pd.DataFrame(rows, columns=["node", "lat", "lon", "plant_code", "source", "score"])


# ---------------------------------------------------------------------------
# Constraints -> coordinates (triangulation)
# ---------------------------------------------------------------------------

def triangulate_constraints(
    sf_matrix: pd.DataFrame,
    node_locations: pd.DataFrame,
    top_fraction: float = 0.5,
    min_nodes: int = 4,
    power: float = 1.0,
) -> pd.DataFrame:
    """
    Place each constraint at the |SF|-weighted centroid of its most sensitive located nodes.

    Parameters
    ----------
    sf_matrix : node × constraint shift factors.
    node_locations : output of ``node_locations_from_eia`` (``node, lat, lon, plant_code``).
        Nodes at the same plant count once (their shift factors are near-identical, and they would
        otherwise triple-weight one site).
    top_fraction : keep located nodes with |SF| ≥ top_fraction × the constraint's largest |SF|.
    min_nodes : fewer distinct sites than this -> no placement (returned with NaN coordinates).
    power : weights are |SF|^power.

    Returns
    -------
    DataFrame indexed by constraint: ``lat, lon, radius_km`` (weighted RMS distance of the
    contributing sites from the estimate: how spread out the evidence is), ``n_sites``,
    ``max_abs_sf`` (the strongest located node), ``spread_ratio`` (radius ÷ the footprint's typical
    inter-site distance; near 1 means the evidence is everywhere and the placement is not
    informative).
    """
    loc = node_locations.drop_duplicates("plant_code") if "plant_code" in node_locations else node_locations
    loc = loc[loc["node"].isin(sf_matrix.index)]
    if loc.empty:
        return pd.DataFrame(columns=["lat", "lon", "radius_km", "n_sites", "max_abs_sf", "spread_ratio"])
    lat = loc["lat"].to_numpy(float)
    lon = loc["lon"].to_numpy(float)
    S = sf_matrix.loc[loc["node"]].abs().to_numpy(float)          # sites × constraints
    S = np.nan_to_num(S)

    # typical distance between two random located sites: the scale the radius is judged against
    iu = np.triu_indices(len(lat), 1)
    typical = float(np.median(haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])[iu])) if len(lat) > 1 else np.nan

    rows = {}
    for j, c in enumerate(sf_matrix.columns):
        s = S[:, j]
        peak = s.max() if s.size else 0.0
        if peak <= 0:
            rows[c] = (np.nan, np.nan, np.nan, 0, 0.0, np.nan)
            continue
        keep = s >= top_fraction * peak
        if keep.sum() < min_nodes:                     # widen to the strongest min_nodes sites
            keep = np.zeros_like(keep)
            keep[np.argsort(-s)[:min_nodes]] = True
        if keep.sum() < min_nodes:
            rows[c] = (np.nan, np.nan, np.nan, int(keep.sum()), float(peak), np.nan)
            continue
        w = s[keep] ** power
        w = w / w.sum()
        la, lo = lat[keep], lon[keep]
        # centroid in 3D to be safe, then back to lat/lon
        phi, lam = np.radians(la), np.radians(lo)
        x, y, z = np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)
        cx, cy, cz = (w * x).sum(), (w * y).sum(), (w * z).sum()
        clat = float(np.degrees(np.arctan2(cz, np.hypot(cx, cy))))
        clon = float(np.degrees(np.arctan2(cy, cx)))
        d = haversine_km(clat, clon, la, lo)
        radius = float(np.sqrt((w * d ** 2).sum()))
        rows[c] = (clat, clon, radius, int(keep.sum()), float(peak), radius / typical if typical else np.nan)
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["lat", "lon", "radius_km", "n_sites", "max_abs_sf", "spread_ratio"])
    out.index.name = "constraint"
    return out


# ---------------------------------------------------------------------------
# Final placement: anchors first, triangulation as a fallback
# ---------------------------------------------------------------------------

#: nominal uncertainty (km) of each placement method. Anchors: the other end of the element and
#: its length are unknown (EIA-located stations had a median 23 km offset between a constraint's
#: monitored and contingency anchors, p75 110 km). Triangulation uses its own measured radius.
ANCHOR_RADIUS_KM = {"monitored_station": 40.0, "contingency_station": 110.0}
GOOD_RADIUS_KM = 100.0       # triangulation: leave-anchor-out median error ~60 km at or below this
COARSE_RADIUS_KM = 250.0     # above this the evidence is diffuse and error ~ guessing the centroid


def station_coordinates(bus_mapping: pd.DataFrame, node_locations: pd.DataFrame) -> pd.DataFrame:
    """EMS station -> mean coordinates of its located generator nodes (``station, lat, lon``)."""
    b = bus_mapping.copy()
    b["node"] = b["settlement_location"].astype(str)
    j = b.merge(node_locations[["node", "lat", "lon"]], on="node", how="inner")
    j["station"] = j["ems_station"].astype(str).str.upper().str.strip()
    return j.groupby("station").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index()


def place_constraints(
    constraint_meta: pd.DataFrame,
    stations: pd.DataFrame,
    triangulation: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Place constraints on the map, with the method and uncertainty of each placement.

    ``constraint_meta`` has one row per constraint name with ``constraint_name, monitored_facility,
    contingent_facility``.  Order of preference:

    1. ``monitored_station``   a monitored endpoint is a located generator station (needs no shift
                               factors, so it also places constraints that have just started binding)
    2. ``contingency_station`` a station named in the contingency is located (approximate)
    3. ``triangulation``       from shift factors; kept only if ``radius_km`` <= ``COARSE_RADIUS_KM``
    otherwise the constraint is returned ``unplaced`` (NaN coordinates).

    ``quality``: good (monitored station, or triangulation radius <= 100 km), approximate
    (contingency station, or triangulation radius <= 250 km), unplaced.
    """
    from src.analysis import constraint_identity as ci

    xy = stations.set_index("station")
    tri = triangulation if triangulation is not None else pd.DataFrame()
    rows = []
    for _, r in constraint_meta.drop_duplicates("constraint_name").iterrows():
        name = r["constraint_name"]
        mon = [x for x in ci.parse_monitored(r.get("monitored_facility")).ends if x in xy.index]
        con = [x for x in ci.parse_contingency(r.get("contingent_facility")).stations if x in xy.index]
        out = {"constraint_name": name, "lat": np.nan, "lon": np.nan, "radius_km": np.nan,
               "method": "unplaced", "quality": "unplaced"}
        if mon:
            out.update(lat=xy.loc[mon, "lat"].mean(), lon=xy.loc[mon, "lon"].mean(),
                       radius_km=ANCHOR_RADIUS_KM["monitored_station"], method="monitored_station", quality="good")
        elif con:
            out.update(lat=xy.loc[con, "lat"].mean(), lon=xy.loc[con, "lon"].mean(),
                       radius_km=ANCHOR_RADIUS_KM["contingency_station"], method="contingency_station", quality="approximate")
        elif name in tri.index and pd.notna(tri.loc[name, "lat"]) and tri.loc[name, "radius_km"] <= COARSE_RADIUS_KM:
            rad = float(tri.loc[name, "radius_km"])
            out.update(lat=float(tri.loc[name, "lat"]), lon=float(tri.loc[name, "lon"]), radius_km=rad,
                       method="triangulation", quality="good" if rad <= GOOD_RADIUS_KM else "approximate")
        rows.append(out)
    out = round_coords(pd.DataFrame(rows))
    out["radius_km"] = out["radius_km"].round(1)
    return out


def round_coords(df: pd.DataFrame, cols=("lat", "lon")) -> pd.DataFrame:
    """Round coordinates to the module's output precision (~1 km)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].round(COORD_DECIMALS)
    return out
