"""
Analytics Router — /api/v1/analytics/*

Research outputs built on the pooled shift factors (see src/analysis/congestion_analytics.py
for definitions and conventions):

    GET  /dart/nodes                Per-node distribution of the DA-vs-RT congestion spread
    GET  /dart/decompose            Which constraints explain one node's spread
    GET  /constraints/stats         Binding frequency, duration, DA-vs-RT surprises
    GET  /pairs                     Node pairs most sensitive to a constraint
    POST /scenario/injection-change Flow/stress on every constraint if injection at a node changes
    POST /scenario/constraints      Nodal MCC if given constraints bind at given shadow prices

Shift factors come from the live cache (fit on its rolling window), so decompositions over
recent days are partly in-sample; scripts/sanity_report.py gives the out-of-sample numbers.
SPP only: Day-Ahead history is only collected for SPP.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.analysis import congestion_analytics as ca
from src.api.sf_cache import cache
from src.storage.duckdb_store import store

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

_CACHE_TTL_SECONDS = 300
_frames: Dict[tuple, tuple] = {}     # key -> (loaded_at, value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _records(df: pd.DataFrame) -> List[dict]:
    """JSON-safe records (NaN/inf -> None, Timestamps -> ISO strings)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out = out.replace([np.inf, -np.inf], np.nan).astype(object).where(out.notna(), None)
    return out.to_dict(orient="records")


def _ttl(key: tuple, loader):
    hit = _frames.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1]
    value = loader()
    _frames[key] = (time.time(), value)
    return value


def _check_iso(iso: str) -> None:
    if iso.lower() != "spp":
        raise HTTPException(status_code=400, detail="Analytics are available for SPP only.")


def _sf_flat(iso: str = "spp") -> dict:
    """Shift factors for analytics: the RT+DA fit when available (more constraints), else the RT fit."""
    c = cache.for_iso(iso)
    sol = getattr(c, "solution_with_da", None) or c.solution
    if sol is None or not sol._flat:
        raise HTTPException(status_code=503, detail="Shift factors not available yet — solver has not run.")
    return sol._flat


def _window(days: int):
    end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    return end - pd.Timedelta(days=days + 1), end


def _load_dart(days: int) -> pd.DataFrame:
    """Nodal DART frame. RT prices are loaded a day at a time and reduced to hourly means."""
    def load():
        start, end = _window(days)
        da = store.query_da_lmp(start, end, columns=["settlement_location", "mcc"])
        if da.empty:
            return pd.DataFrame()
        da["interval"] = pd.to_datetime(da["interval"])
        frames = []
        for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
            # SPP labels are interval-END: day D is (D 00:00, D+1 00:00]
            rt = store.query_lmp(day + pd.Timedelta(seconds=1), day + pd.Timedelta(days=1),
                                 columns=["settlement_location", "mcc"])
            if rt.empty:
                continue
            rt["interval"] = pd.to_datetime(rt["interval"])
            frames.append(ca.rt_hourly_mcc(rt))
        if not frames:
            return pd.DataFrame()
        rt_hourly = pd.concat(frames, ignore_index=True).drop_duplicates(["hour", "settlement_location"])
        da = da.rename(columns={"interval": "hour", "mcc": "da_mcc"})
        out = da.merge(rt_hourly, on=["hour", "settlement_location"], how="inner")
        out["spread"] = out["rt_mcc"] - out["da_mcc"]
        return out
    return _ttl(("dart", days), load)


def _load_bc(days: int):
    def load():
        start, end = _window(days)
        rt = store.query_constraints(
            start, end,
            columns=["constraint_name", "shadow_price", "state", "monitored_facility", "source_limit_mw"],
        )
        da = store.query_da_constraints(start, end, columns=["constraint_name", "shadow_price", "monitored_facility"])
        for d in (rt, da):
            if not d.empty:
                d["interval"] = pd.to_datetime(d["interval"])
        return rt, da
    return _ttl(("bc", days), load)


# ---------------------------------------------------------------------------
# DART
# ---------------------------------------------------------------------------

@router.get("/dart/nodes")
async def dart_nodes(
    days: int = Query(14, ge=1, le=90),
    limit: int = Query(50, ge=1, le=2000),
    min_hours: int = Query(24, ge=1),
    iso: str = Query("spp"),
):
    """
    Per-node distribution of the hourly congestion spread, **RT − DA** (positive: RT congestion
    cost was higher than DA priced).  Sorted by |naive t-stat|.  Hours are autocorrelated, so the
    t-stat ranks nodes but overstates significance.
    """
    _check_iso(iso)
    dart = _load_dart(days)
    if dart.empty:
        raise HTTPException(status_code=404, detail="No overlapping DA and RT history. Run the DA backfill.")
    summ = ca.dart_summary(dart, min_hours=min_hours).head(limit).reset_index()
    return {"days": days, "n_hours": int(dart["hour"].nunique()), "n_nodes": int(dart["settlement_location"].nunique()),
            "nodes": _records(summ.round(4))}


@router.get("/dart/decompose")
async def dart_decompose(
    node: str = Query(..., description="Settlement location"),
    days: int = Query(7, ge=1, le=90),
    top: int = Query(10, ge=1, le=100),
    iso: str = Query("spp"),
):
    """
    Which constraints explain a node's DART spread: ``Σ_c SF[node,c] × (SP_RT − SP_DA)``.

    ``r2`` is how much of the spread the decomposition explains; ``unattributed_share`` is the part
    of the shadow-price move carried by constraints with no shift factor.  Shift factors are from
    the live cache (fit on the last 7 days), so a window inside that period is in-sample: see the
    ``in_sample`` flag and treat ``r2`` as an upper bound.
    """
    _check_iso(iso)
    sf = _sf_flat(iso)
    dart = _load_dart(days)
    rt_bc, da_bc = _load_bc(days)
    if dart.empty or rt_bc.empty or da_bc.empty:
        raise HTTPException(status_code=404, detail="Need RT and DA constraints and prices for the window.")
    dart = dart[dart["hour"] > pd.Timestamp.now().normalize() - pd.Timedelta(days=days)]
    dec = ca.decompose_dart(dart, sf, ca.hourly_shadow_prices_rt(rt_bc), ca.hourly_shadow_prices_da(da_bc))
    if node not in dec["sf"].index:
        raise HTTPException(status_code=404, detail=f"No shift factors / DART data for node '{node}'.")
    row = dec["summary"].loc[node]
    return {
        "node": node, "days": days, "n_hours": int(row["n_hours"]),
        # The live shift factors are fit on the most recent 7 days (RT + DA). A decomposition window
        # inside that period is IN-SAMPLE, so r2 is optimistic: scripts/sanity_report.py measures it
        # out-of-sample (about 0.43 pooled across nodes on the first 14 days of data).
        "in_sample": days <= 7,
        "mean_spread": float(row["mean_spread"]),
        "mean_explained": float(row["mean_explained"]),
        "mean_unexplained_offset": float(row["mean_residual"]),
        "mean_abs_spread": float(row["mean_abs_spread"]),
        "r2": None if pd.isna(row["r2"]) else float(row["r2"]),
        "unattributed_share": float(dec["unattributed_share"]),
        "constraints": _records(ca.node_decomposition(dec, node, top=top).round(4).reset_index(names="constraint")),
    }


# ---------------------------------------------------------------------------
# Constraint behaviour & pairs
# ---------------------------------------------------------------------------

@router.get("/constraints/stats")
async def constraint_stats(
    days: int = Query(14, ge=1, le=90),
    limit: int = Query(50, ge=1, le=1000),
    group_by: str = Query("constraint_name", pattern="^(constraint_name|monitored_facility)$"),
    iso: str = Query("spp"),
):
    """
    Behaviour of each constraint that priced in RT: binding hours, run length, shadow-price quantiles,
    and DA-vs-RT surprises (``rt_only_hours``).  Use ``group_by=monitored_facility`` to count a DA
    miss only if DA did not price the same physical element under any contingency.
    """
    _check_iso(iso)
    rt_bc, da_bc = _load_bc(days)
    if rt_bc.empty:
        raise HTTPException(status_code=404, detail="No RT constraint history in the window.")
    cs = ca.constraint_stats(rt_bc, da_bc, group_by=group_by).head(limit)
    return {"days": days, "group_by": group_by, "n": len(cs), "constraints": _records(cs.round(4))}


@router.get("/pairs")
async def pairs(
    constraint: str = Query(...),
    top: int = Query(10, ge=1, le=50),
    iso: str = Query("spp"),
):
    """
    Node pairs with the widest expected congestion gap per $ of |shadow price| on a constraint.
    When it binds, ``MCC[high] − MCC[low] = (SF[high] − SF[low]) × SP``; binding SPs are negative,
    so the lowest-SF node gets the highest MCC.
    """
    _check_iso(iso)
    sf = _sf_flat(iso)
    out = ca.best_pairs(sf, constraint, top=top)
    if out.empty:
        raise HTTPException(status_code=404, detail=f"No shift factors for constraint '{constraint}'.")
    return {"constraint": constraint, "pairs": _records(out.round(5))}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

class InjectionChange(BaseModel):
    node: str
    injection_change_mw: float = Field(..., description="Negative for a generator trip, positive for a load trip")
    top: int = Field(15, ge=1, le=200)
    iso: str = "spp"


@router.post("/scenario/injection-change")
async def scenario_injection_change(body: InjectionChange):
    """
    Flow change on every modeled constraint if injection at ``node`` changes by the given MW
    (``Δflow = SF × ΔMW``), ranked by percent of the constraint's static rating consumed.

    Generator and load changes only.  A LINE outage reroutes flow and needs line-outage factors
    from a network model, which this tool does not have.  The percentage is the change as a share
    of the rating, NOT remaining headroom (line loading is not published).
    """
    _check_iso(body.iso)
    sf = _sf_flat(body.iso)
    rt_bc, _ = _load_bc(14)
    limits = ca.constraint_limits(rt_bc) if not rt_bc.empty else {}
    out = ca.trip_scenario(sf, body.node, body.injection_change_mw, limits, top=body.top)
    if out.empty:
        raise HTTPException(status_code=404, detail=f"No shift factors for node '{body.node}'.")
    return {"node": body.node, "injection_change_mw": body.injection_change_mw, "constraints": _records(out.round(4))}


class ConstraintScenario(BaseModel):
    shadow_prices: Optional[Dict[str, float]] = Field(
        None, description="constraint -> shadow price (negative, as SPP publishes). Overrides `constraints`.")
    constraints: Optional[List[str]] = Field(
        None, description="Constraints to bind at a historical shadow-price quantile")
    quantile: float = Field(0.5, ge=0.0, le=1.0, description="1.0 = the most severe historical price")
    top_nodes: int = Field(25, ge=1, le=2000)
    iso: str = "spp"


@router.post("/scenario/constraints")
async def scenario_constraints(body: ConstraintScenario):
    """
    Nodal MCC change if constraints bind at given shadow prices (``MCC = Σ SF × SP``).  Give explicit
    ``shadow_prices``, or a list of ``constraints`` to bind at a historical quantile of their price.
    Constraints without shift factors are listed under ``unmodeled`` and contribute nothing.
    """
    _check_iso(body.iso)
    sf = _sf_flat(body.iso)
    sp = body.shadow_prices
    if not sp:
        if not body.constraints:
            raise HTTPException(status_code=422, detail="Provide `shadow_prices` or `constraints`.")
        rt_bc, _ = _load_bc(14)
        sp = ca.historical_shadow_price(rt_bc, body.constraints, body.quantile)
        if not sp:
            raise HTTPException(status_code=404, detail="No historical prices for those constraints.")
    out = ca.price_impact(sf, sp)
    return {
        "shadow_prices": sp, "unmodeled": out.attrs.get("unmodeled", []),
        "summary": {"min": float(out["mcc_change"].min()), "median": float(out["mcc_change"].median()),
                    "max": float(out["mcc_change"].max())},
        "nodes": _records(out.head(body.top_nodes).round(4)),
    }
