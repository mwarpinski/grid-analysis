"""
DEPRECATED: the dashboard map no longer uses this synthetic schematic. It now shows real (approximate)
constraint locations built from public data by scripts/build_geo_crosswalk.py. This script and
ui/data/spp_only_lines.geojson remain only for reference and can be deleted.

generate_spp_topology.py
========================
Generates a SYNTHETIC, DELIBERATELY FUZZED SPP transmission grid topology
for use as a visual schematic in the congestion monitor dashboard.

SECURITY NOTE
-------------
This script does NOT use real substation coordinates. Node positions are:
  1. Derived from sub-state regional centroids (e.g. "western Kansas")
  2. Offset by a random ±0.4° jitter (~25-30 miles) seeded for reproducibility
  3. Rounded to 1 decimal place (~7 miles resolution)

The resulting GeoJSON is a schematic diagram — suitable for showing which
*region* of SPP is experiencing congestion, but deliberately unsuitable for
locating physical grid infrastructure.

Output: ui/data/spp_only_lines.geojson  (served at /spp_only_lines.geojson)

Usage:
    python scripts/generate_spp_topology.py
"""

import json
import math
import random
from pathlib import Path
from scipy.spatial import Delaunay  # type: ignore
import numpy as np

# ── Reproducible seed ──────────────────────────────────────────────────────────
SEED = 42
RNG = random.Random(SEED)
NP_RNG = np.random.default_rng(SEED)

# ── Regional centroids within SPP footprint ────────────────────────────────────
# These are sub-state region centers (county-group level), NOT substation coords.
# SPP footprint: KS, OK, NE, SD, ND, MO (west), AR (north), LA (north),
#                TX (panhandle), NM (east), WY (east), MT (east),
#                MN (west), IA (west), CO (east)
#
# Each entry: (region_label, base_lat, base_lon, voltage_kv)
# Higher-voltage buses are placed at major load/generation hubs.

RAW_BUSES = [
    # ── Kansas ────────────────────────────────────────────────────────────────
    ("KS-NW",    39.8, -101.3, 345),
    ("KS-NC",    39.4,  -98.6, 345),
    ("KS-NE",    39.8,  -96.5, 345),
    ("KS-W",     38.5, -100.5, 115),
    ("KS-C",     38.4,  -98.2, 230),
    ("KS-CE",    38.6,  -96.8, 115),
    ("KS-SW",    37.3, -100.8, 115),
    ("KS-SC",    37.6,  -98.5, 230),
    ("KS-SE",    37.2,  -95.7, 345),
    ("KS-WICHI", 37.7,  -97.3, 345),  # Wichita metro area (major load)
    # ── Oklahoma ──────────────────────────────────────────────────────────────
    ("OK-NW",    36.7, -100.1, 345),
    ("OK-NC",    36.4,  -97.8, 230),
    ("OK-NE",    36.8,  -95.5, 345),
    ("OK-W",     35.5, -099.2, 115),
    ("OK-C",     35.4,  -97.4, 345),  # OKC area
    ("OK-E",     35.5,  -95.2, 230),
    ("OK-SW",    34.6,  -98.7, 115),
    ("OK-SE",    34.7,  -95.3, 230),
    # ── Nebraska ──────────────────────────────────────────────────────────────
    ("NE-W",     41.2, -102.5, 115),
    ("NE-C",     41.1,  -99.5, 230),
    ("NE-E",     41.3,  -96.8, 345),
    ("NE-SE",    40.5,  -96.9, 115),
    # ── South Dakota ──────────────────────────────────────────────────────────
    ("SD-SW",    43.5, -103.0, 115),
    ("SD-C",     44.2,  -99.8, 230),
    ("SD-E",     43.8,  -97.1, 345),
    # ── North Dakota ──────────────────────────────────────────────────────────
    ("ND-SW",    46.2, -102.5, 230),
    ("ND-C",     46.8,  -99.5, 115),
    ("ND-SE",    46.5,  -97.4, 345),
    # ── Missouri (SPP portion — western) ──────────────────────────────────────
    ("MO-NW",    39.9,  -94.8, 345),
    ("MO-W",     38.5,  -94.4, 230),
    ("MO-SW",    37.0,  -94.2, 115),
    # ── Arkansas (northern, in SPP) ───────────────────────────────────────────
    ("AR-NW",    36.2,  -94.5, 345),
    ("AR-N",     36.0,  -92.5, 115),
    # ── Louisiana (northern tip, in SPP) ─────────────────────────────────────
    ("LA-N",     32.5,  -93.2, 230),
    # ── Texas Panhandle (in SPP) ─────────────────────────────────────────────
    ("TX-PH",    35.2, -101.5, 345),
    ("TX-S",     34.0,  -98.0, 115),
    # ── New Mexico (eastern, in SPP) ─────────────────────────────────────────
    ("NM-E",     34.2, -103.5, 115),
    # ── Wyoming (eastern, in SPP) ────────────────────────────────────────────
    ("WY-SE",    41.8, -104.5, 345),
    ("WY-NE",    44.3, -104.7, 115),
    # ── Montana (eastern, in SPP) ────────────────────────────────────────────
    ("MT-SE",    45.9, -105.2, 115),
    # ── Minnesota / Iowa / Colorado (SPP edges) ───────────────────────────────
    ("MN-W",     45.0,  -96.5, 345),
    ("IA-W",     42.0,  -96.0, 115),
    ("CO-E",     39.5, -102.5, 115),
]

JITTER = 0.35   # ± degrees (~25 miles) — makes coords non-attributable to real sites


def fuzz(lat: float, lon: float) -> tuple[float, float]:
    """Add reproducible random jitter and round to 1 decimal place."""
    lat2 = round(lat + RNG.uniform(-JITTER, JITTER), 1)
    lon2 = round(lon + RNG.uniform(-JITTER, JITTER), 1)
    return lat2, lon2


# ── Build fuzzed bus list ──────────────────────────────────────────────────────
buses: list[dict] = []
for label, base_lat, base_lon, kv in RAW_BUSES:
    flat, flon = fuzz(base_lat, base_lon)
    buses.append({
        "id": label,
        "lat": flat,
        "lon": flon,
        "kv": kv,
    })

# ── Build transmission lines via Delaunay triangulation ───────────────────────
# Delaunay on the fuzzed (lat, lon) gives a realistic meshed network.
# We then filter out edges that are too long (>8° ≈ 500 miles — not realistic).

coords = np.array([[b["lon"], b["lat"]] for b in buses])  # (x=lon, y=lat)
tri = Delaunay(coords)

MAX_EDGE_DEG = 7.5   # ~500 miles — drop cross-continent edges

seen_edges: set[frozenset] = set()
lines: list[dict] = []

for simplex in tri.simplices:
    for i in range(3):
        a, b = simplex[i], simplex[(i + 1) % 3]
        edge = frozenset({a, b})
        if edge in seen_edges:
            continue
        seen_edges.add(edge)

        ba, bb = buses[a], buses[b]
        dlat = abs(ba["lat"] - bb["lat"])
        dlon = abs(ba["lon"] - bb["lon"])
        dist = math.sqrt(dlat**2 + dlon**2)
        if dist > MAX_EDGE_DEG:
            continue

        # Voltage of line = max of endpoint voltages
        line_kv = max(ba["kv"], bb["kv"])

        # Generate a constraint name that follows SPP naming conventions.
        # Real SPP constraint names look like: WOODWARD, SIDOGASIDKEY, LARK_HAAS, etc.
        # We encode the two region IDs into a plausible identifier.
        a_tag = ba["id"].replace("-", "").upper()
        b_tag = bb["id"].replace("-", "").upper()
        constraint_name = f"{a_tag}_{b_tag}_{line_kv}"

        lines.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [ba["lon"], ba["lat"]],
                    [bb["lon"], bb["lat"]],
                ],
            },
            "properties": {
                "constraint_name": constraint_name,
                "from_bus": ba["id"],
                "to_bus": bb["id"],
                "voltage_kv": line_kv,
                "pct_of_limit": None,   # filled in by live data
            },
        })

# ── Build bus marker features ─────────────────────────────────────────────────
bus_features: list[dict] = [
    {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [b["lon"], b["lat"]],
        },
        "properties": {
            "bus_id": b["id"],
            "voltage_kv": b["kv"],
        },
    }
    for b in buses
]

# ── Assemble GeoJSON ───────────────────────────────────────────────────────────
geojson = {
    "type": "FeatureCollection",
    "features": lines + bus_features,
    "_meta": {
        "description": (
            "Synthetic SPP grid schematic. Node positions are fuzzed regional "
            "centroids (±25 miles), NOT real substation coordinates. "
            "Suitable for congestion region visualization only."
        ),
        "generated_by": "generate_spp_topology.py",
        "seed": SEED,
    },
}

# ── Write output ───────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent.parent / "ui" / "data"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "spp_only_lines.geojson"

with open(out_path, "w") as f:
    json.dump(geojson, f, indent=2)

n_lines = len(lines)
n_buses = len(buses)
print(f"✓ Generated {n_buses} buses, {n_lines} transmission lines")
print(f"  → {out_path}")
print()
print("Constraint name index (first 10 lines):")
for feat in lines[:10]:
    p = feat["properties"]
    print(f"  {p['constraint_name']:40s}  {p['voltage_kv']} kV")
