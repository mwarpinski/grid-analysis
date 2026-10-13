"""
Constraint identity for SPP.

SPP's ``Constraint Name`` is not a stable identifier: RT uses reissued flowgate IDs
(``TMP527_28137``, 58% of the names we saw), DA uses different names for the same physical
limit, and the same id can be recycled.  What IS stable and present in both markets is the
pair of structured columns:

    Monitored Facility   "LN TEKAMHO - SUB1226"    the physical element being limited
    Contingent Facility  "OPPD:SUB3456 NEBRCTY:345:1:11"   the outage that makes it limiting

Identity therefore has two levels:

    element      the monitored line/transformer           -> ``element_key``
    constraint   that element under one contingency        -> ``constraint_key``

The distinction matters physically: the shift factors of a constraint depend on the
contingency (post-contingency flow = own flow + outage distribution × the outaged element's
flow), so two constraints on the same ELEMENT can have very different shift factors, while two
aliases of the same ELEMENT+CONTINGENCY should have the same ones.  See
``tests/test_constraint_identity.py`` and scripts/identity_report.py for the empirical check.

Direction
---------
The ORDER of the endpoints in the monitored-facility string is the monitored flow direction
(``LN SOONER - LIBERAL_`` and ``LN LIBERAL_ - SOONER`` are two different constraints on one line, with
shift factors of opposite sign).  The keys are undirected so that both land in one alias group, and
``orientation`` (+1 if the string lists the endpoints in sorted order, else -1) records the direction.
To compare shift factors of two aliases, multiply each by its orientation: ``canonical_sf``.
The sign of the shadow price does not matter (MCC = SF × SP is invariant); this was checked
empirically: raw alias correlation -0.86, after orientation +0.86.

Limits
------
* Parallel elements (two lines or transformers between the same stations) cannot be told apart by the
  facility string.  Self-ended transformers (``XFMR A - A``) are therefore never merged across names.
* Contingency strings are sometimes truncated by SPP, so equal outages can still get different keys.

All keys are plain strings so they can be stored, joined and compared across markets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

_WS = re.compile(r"\s+")
_KV = re.compile(r"^\d+(?:\.\d+)?$")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Element:
    """The monitored element: ``LN`` / ``XFMR`` / ``PS`` between two stations, or ``MULTI``/``UNKNOWN``."""
    kind: str
    ends: Tuple[str, ...]      # sorted endpoint station names (EMS abbreviations)
    raw: str
    orientation: int = 1       # +1 if the string lists the endpoints in sorted order, -1 if reversed

    @property
    def ambiguous(self) -> bool:
        """True if the facility string cannot identify one element (so it must not be merged by key)."""
        return self.kind in ("MULTI", "UNKNOWN") or (self.kind == "XFMR" and len(self.ends) == 2 and self.ends[0] == self.ends[1])

    @property
    def key(self) -> str:
        return f"{self.kind}|{'|'.join(self.ends)}" if self.ends else f"{self.kind}|{self.raw}"


@dataclass(frozen=True)
class Contingency:
    """The outage under which an element is limiting."""
    base: bool                     # no contingency (base case)
    stations: Tuple[str, ...]      # sorted station tokens named in the contingency
    kv: Optional[float]            # voltage of the outaged element (first voltage listed)
    owners: Tuple[str, ...]
    raw: str

    @property
    def key(self) -> str:
        if self.base:
            return "BASE"
        kv = f"{self.kv:g}" if self.kv is not None else "?"
        return f"C|{','.join(self.stations)}|{kv}" if self.stations else f"C|{self.raw}"


def _clean(s) -> str:
    return _WS.sub(" ", str(s if s is not None and not (isinstance(s, float) and pd.isna(s)) else "")).strip().upper()


def parse_monitored(raw) -> Element:
    """``"LN A - B"`` -> Element("LN", ("A", "B")); unorderable/unknown shapes keep the raw text."""
    s = _clean(raw)
    if " - " in s:
        left, right = s.split(" - ", 1)
        parts = left.split(" ", 1)
        kind, a = (parts[0], parts[1]) if len(parts) == 2 else ("UNKNOWN", parts[0])
        a, b = a.strip(), right.strip()
        if a and b:
            ends = tuple(sorted((a, b)))
            return Element(kind=kind, ends=ends, raw=s, orientation=1 if (a, b) == ends else -1)
    if s.startswith("MULTI"):
        return Element(kind="MULTI", ends=(), raw=s)
    return Element(kind="UNKNOWN", ends=(), raw=s)


def parse_contingency(raw) -> Contingency:
    """
    Parse ``OWNERS:STATION STATION[ XF name]:kV[ kV kV]:circuit:id``.

    SPP's strings are irregular: truncated (``345 13``), multi-element (several stations),
    named (``ADD NEW EAST``), or blank.  Whatever is left over is kept as the raw text so the key
    stays deterministic, and BASE (or blank) means no contingency.
    """
    s = _clean(raw)
    if s in ("", "BASE", "NONE", "NAN"):
        return Contingency(base=True, stations=(), kv=None, owners=(), raw=s)
    parts = s.split(":")
    if len(parts) < 2:                       # a named contingency such as "ADD NEW EAST"
        return Contingency(base=False, stations=(), kv=None, owners=(), raw=s)
    owners = tuple(sorted(parts[0].split()))
    body = parts[1]
    # "A XF TR9_3" is a transformer: keep both stations/name before XF, drop the transformer id
    body = re.sub(r"\bXF\b.*$", "XF", body).strip()
    stations = tuple(sorted(t for t in body.split() if t != "XF"))
    kv = None
    if len(parts) > 2 and parts[2].strip():
        first = parts[2].split()[0]
        if _KV.match(first):
            kv = float(first)
        else:
            # not a voltage: a generating unit at the station (e.g. SPS:HOBBS:COMBINEDCYCLE)
            stations = tuple(sorted(stations + (f"UNIT:{parts[2].strip()}",)))
    return Contingency(base=False, stations=stations, kv=kv, owners=owners, raw=s)


def element_key(monitored_facility) -> str:
    return parse_monitored(monitored_facility).key


def constraint_key(monitored_facility, contingent_facility) -> str:
    """Stable id of an element under one contingency (the thing a shift factor belongs to)."""
    return f"{parse_monitored(monitored_facility).key}#{parse_contingency(contingent_facility).key}"


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def add_identity_columns(
    df: pd.DataFrame,
    monitored_col: str = "monitored_facility",
    contingent_col: str = "contingent_facility",
) -> pd.DataFrame:
    """
    Add ``element_key``, ``element_kind``, ``from_sub``, ``to_sub``, ``contingency_key`` and
    ``constraint_key`` computed from the structured columns (cached per distinct pair, so this is
    cheap on large frames).  Rows without the structured columns are left with null keys.
    """
    out = df.copy()
    if monitored_col not in out.columns:
        for c in ("element_key", "element_kind", "from_sub", "to_sub", "orientation", "contingency_key", "constraint_key"):
            out[c] = None
        return out
    cont = out[contingent_col] if contingent_col in out.columns else pd.Series("", index=out.index)
    pairs = pd.DataFrame({"m": out[monitored_col], "c": cont}).drop_duplicates()
    rows = []
    for m, c in zip(pairs["m"], pairs["c"]):
        el, ct = parse_monitored(m), parse_contingency(c)
        rows.append({
            "m": m, "c": c, "element_key": el.key, "element_kind": el.kind,
            "from_sub": el.ends[0] if len(el.ends) > 0 else None,
            "to_sub": el.ends[1] if len(el.ends) > 1 else None,
            "orientation": el.orientation, "ambiguous": el.ambiguous,
            "contingency_key": ct.key, "constraint_key": f"{el.key}#{ct.key}",
        })
    lookup = pd.DataFrame(rows)
    tmp = pd.DataFrame({"m": out[monitored_col], "c": cont}).merge(lookup, on=["m", "c"], how="left")
    for col in ("element_key", "element_kind", "from_sub", "to_sub", "orientation", "contingency_key", "constraint_key"):
        out[col] = tmp[col].to_numpy()
    # "Multi-Element Constraint", unparseable strings and self-ended transformers do not identify one
    # element, so the structured columns cannot tell two such constraints apart: fall back to the name.
    if "constraint_name" in out.columns:
        anon = tmp["ambiguous"].fillna(False).to_numpy(dtype=bool)
        if anon.any():
            out.loc[anon, "element_key"] = (
                out.loc[anon, "element_key"] + "|" + out.loc[anon, "constraint_name"].astype(str)
            )
            out.loc[anon, "constraint_key"] = out.loc[anon, "element_key"] + "#" + out.loc[anon, "contingency_key"]
    return out


def build_alias_table(
    rt_bc: Optional[pd.DataFrame] = None,
    da_bc: Optional[pd.DataFrame] = None,
    tolerance: float = 1e-4,
) -> pd.DataFrame:
    """
    One row per ``constraint_key`` that priced at least once, listing every alias it appeared under.

    Columns: constraint_key, element_key, element_kind, from_sub, to_sub, contingency_key,
    rt_names, da_names, nerc_ids (when present), in_rt, in_da, first_seen, last_seen.
    This is the DA↔RT crosswalk: a key with both ``in_rt`` and ``in_da`` is the same
    element-under-contingency in both markets, whatever each market called it.
    """
    frames = []
    for market, df in (("rt", rt_bc), ("da", da_bc)):
        if df is None or df.empty:
            continue
        d = df[df["shadow_price"].abs() > tolerance]
        d = add_identity_columns(d)
        d["market"] = market
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    allf = pd.concat(frames, ignore_index=True)
    allf["interval"] = pd.to_datetime(allf["interval"])

    def names(g, market):
        return sorted(set(g.loc[g["market"] == market, "constraint_name"]))

    rows = []
    for key, g in allf.groupby("constraint_key", sort=False):
        head = g.iloc[0]
        rt_n, da_n = names(g, "rt"), names(g, "da")
        rows.append({
            "constraint_key": key, "element_key": head["element_key"], "element_kind": head["element_kind"],
            "from_sub": head["from_sub"], "to_sub": head["to_sub"], "contingency_key": head["contingency_key"],
            "rt_names": rt_n, "da_names": da_n,
            "nerc_ids": sorted(set(g["nerc_id"].dropna().astype(str))) if "nerc_id" in g.columns else [],
            "in_rt": bool(rt_n), "in_da": bool(da_n),
            "first_seen": g["interval"].min(), "last_seen": g["interval"].max(),
        })
    return pd.DataFrame(rows)


def name_orientation(rt_bc: Optional[pd.DataFrame] = None, da_bc: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    """Map every constraint NAME to its orientation (+1 / -1; see the module docstring)."""
    out: Dict[str, int] = {}
    for df in (rt_bc, da_bc):
        if df is None or df.empty:
            continue
        d = add_identity_columns(df[["constraint_name", "monitored_facility", "contingent_facility"]].drop_duplicates())
        for n, o in zip(d["constraint_name"], d["orientation"]):
            out.setdefault(n, 1 if pd.isna(o) else int(o))
    return out


def canonical_sf(sf_matrix: pd.DataFrame, orientation: Dict[str, int]) -> pd.DataFrame:
    """
    Orient a node × constraint shift-factor matrix so aliases are comparable: each column is multiplied
    by its constraint's orientation.  Constraints with unknown orientation are left as they are.
    """
    sign = pd.Series({c: orientation.get(c, 1) for c in sf_matrix.columns}, dtype=float)
    return sf_matrix.mul(sign, axis=1)


def name_to_key(rt_bc: Optional[pd.DataFrame] = None, da_bc: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """Map every constraint NAME seen (either market) to its ``constraint_key`` (first key wins)."""
    out: Dict[str, str] = {}
    for df in (rt_bc, da_bc):
        if df is None or df.empty:
            continue
        d = add_identity_columns(df[["constraint_name", "monitored_facility", "contingent_facility"]].drop_duplicates())
        for n, k in zip(d["constraint_name"], d["constraint_key"]):
            out.setdefault(n, k)
    return out
