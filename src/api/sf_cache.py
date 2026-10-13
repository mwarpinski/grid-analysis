"""
Shift Factor Cache — in-memory singleton.

Holds the most recent ShiftFactorSolution and shadow price snapshot so
every API request gets sub-millisecond SF lookups without re-running
the regression.

The cache is updated by the ingest worker (src/api/ingest_worker.py)
after each SPP RTBM poll cycle. The full regression is only re-run when
a new constraint regime is detected (i.e., the set of binding constraints
changes). Within a stable regime, only shadow prices are refreshed.

Thread / async safety
---------------------
FastAPI runs in an asyncio event loop. The ingest worker runs as a
background asyncio Task. All writes to the cache happen from that single
task — no concurrent writes. Reads from route handlers are read-only.
No lock is needed because Python's GIL protects individual attribute
assignments, and we never mutate shared mutable objects in-place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CachedInterval:
    """One RTBM scrape result, normalized and ready to serve."""
    interval: datetime
    constraints: List[dict]              # list of ConstraintSnapshot-compatible dicts
    shadow_prices: Dict[str, float]      # {constraint_name: shadow_price}
    lmp_df: pd.DataFrame                 # settlement_location, mcc
    regime_key: FrozenSet[str]           # frozenset of binding constraint names


# Columns the solver and forecaster actually use. The scraped frames carry many more
# (pnode, lmp, mlc, mec, gmt timestamps, text columns ...); keeping them all made the
# 7-day history ~0.9 GB instead of well under half that.
_CONSTRAINT_COLS = [
    "interval", "constraint_name", "shadow_price", "state", "monitored_facility", "contingent_facility",
]
_LMP_COLS = ["interval", "settlement_location", "mcc"]


@dataclass
class SFCache:
    """
    Singleton in-memory store for the current shift factor solution
    and latest market state.

    Access via the module-level ``cache`` instance.
    """

    # The most recently solved ShiftFactorSolution (or None before first solve)
    solution: Optional[object] = None       # ShiftFactorSolution

    # Latest interval data from the ingest worker
    latest_interval: Optional[CachedInterval] = None

    # Timestamp of the last successful ingest
    last_updated: Optional[datetime] = None

    # Shift factors fit on RT and DA data together (covers constraints only the DA market priced).
    # Used by the analytics endpoints; the live MCC estimate keeps using ``solution``.
    solution_with_da: Optional[object] = None

    # Provisional shift factors for currently-binding constraints that have none, borrowed from sibling
    # constraints on the same physical element ({(node, constraint): sf}). Kept apart from ``solution``
    # so fitted and borrowed values are never confused.
    provisional: Dict[Tuple[str, str], float] = field(default_factory=dict)
    _name_key: Dict[str, str] = field(default_factory=dict)       # constraint name -> constraint_key
    _orientation: Dict[str, int] = field(default_factory=dict)    # constraint name -> +1/-1

    # Timestamp of the last SF solve (only changes when regime changes)
    last_solved: Optional[datetime] = None

    # Most recent regime key (frozenset of binding constraint names)
    current_regime: FrozenSet[str] = field(default_factory=frozenset)

    # Full constraints_df used for the current solve (grows with new intervals)
    _constraints_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    _lmp_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Maximum history retained for regression (rolling window)
    max_history_intervals: int = 2016   # 2016 × 5 min = 7 days

    def update(
        self,
        interval: CachedInterval,
        constraints_df: pd.DataFrame,
        lmp_df: pd.DataFrame,
    ) -> bool:
        """
        Ingest a new interval. Re-runs the SF solver only when the
        binding constraint regime changes.

        Returns True if a new solve was triggered.
        """
        from src.analysis.multi_constraint_sf import calculate_shift_factors

        self.latest_interval = interval
        self.last_updated = datetime.now()
        self._learn_identity(constraints_df)

        # Append to rolling history
        self._constraints_history = self._append_and_trim(
            self._constraints_history, constraints_df, _CONSTRAINT_COLS
        )
        self._lmp_history = self._append_and_trim(
            self._lmp_history, lmp_df, _LMP_COLS
        )

        # Check if the regime changed (new set of binding constraints)
        new_regime = interval.regime_key
        regime_changed = new_regime != self.current_regime

        if regime_changed or self.solution is None:
            logger.info(
                f"Regime changed: {self.current_regime} → {new_regime}. "
                "Re-running shift factor solver."
            )
            self.current_regime = new_regime
            try:
                self.solution = calculate_shift_factors(
                    constraints_df=self._constraints_history,
                    lmp_df=self._lmp_history,
                    min_intervals=10,
                )
                self.last_solved = datetime.now()
                logger.info(
                    f"SF solve complete. "
                    f"History: {len(self._constraints_history)} constraint-rows, "
                    f"{len(self._lmp_history)} LMP rows."
                )
                return True
            except Exception as e:
                logger.error(f"SF solve failed: {e}", exc_info=True)

        return False

    def seed(self, constraints_df: pd.DataFrame, lmp_df: pd.DataFrame) -> bool:
        """
        Pre-load the rolling history (e.g. from the Parquet store at startup) and
        solve once.  Without this the cache starts empty and no shift factors
        exist until ~10 intervals of a constraint have accumulated live.

        Returns True if a solution was produced.
        """
        from src.analysis.multi_constraint_sf import calculate_shift_factors

        if constraints_df.empty or lmp_df.empty:
            return False
        self._learn_identity(constraints_df)
        self._constraints_history = self._append_and_trim(
            pd.DataFrame(), constraints_df, _CONSTRAINT_COLS
        )
        self._lmp_history = self._append_and_trim(pd.DataFrame(), lmp_df, _LMP_COLS)
        try:
            self.solution = calculate_shift_factors(
                constraints_df=self._constraints_history,
                lmp_df=self._lmp_history,
                min_intervals=10,
            )
            self.last_solved = datetime.now()
            logger.info(
                f"Seeded SF cache: {len(self._constraints_history)} constraint rows, "
                f"{len(self._lmp_history)} LMP rows, {len(self.solution._flat)} SF pairs."
            )
            return True
        except Exception as e:
            logger.error(f"SF cache seeding failed: {e}", exc_info=True)
            return False

    def _learn_identity(self, constraints_df: pd.DataFrame) -> None:
        """Remember each constraint name's identity key and direction (from the structured facility columns)."""
        if constraints_df is None or constraints_df.empty or "monitored_facility" not in constraints_df.columns:
            return
        from src.analysis import constraint_identity as ci

        d = constraints_df[[c for c in ("constraint_name", "monitored_facility", "contingent_facility")
                            if c in constraints_df.columns]].drop_duplicates()
        if "contingent_facility" not in d.columns:
            d = d.assign(contingent_facility="")
        d = ci.add_identity_columns(d)
        for n, k, o in zip(d["constraint_name"], d["constraint_key"], d["orientation"]):
            self._name_key.setdefault(n, k)
            self._orientation.setdefault(n, 1 if pd.isna(o) else int(o))

    def refresh_priors(self, names) -> int:
        """
        Borrow shift factors from sibling constraints for those of ``names`` that have none.
        Returns the number of constraints that received a provisional vector.
        """
        from src.analysis.congestion_analytics import sibling_prior

        sol = self.solution
        if sol is None or not getattr(sol, "_flat", None):
            self.provisional = {}
            return 0
        self.provisional = sibling_prior(sol._flat, list(names), self._name_key, self._orientation)
        return len({k[1] for k in self.provisional})

    def fit_with_da(self, da_bc: pd.DataFrame, da_lmp: pd.DataFrame) -> bool:
        """Fit shift factors on the RT history plus Day-Ahead data and keep them as ``solution_with_da``."""
        from src.analysis.congestion_analytics import fit_shift_factors_rt_da

        if (self._constraints_history.empty or self._lmp_history.empty
                or da_bc is None or da_lmp is None or da_bc.empty or da_lmp.empty):
            return False
        try:
            sol = fit_shift_factors_rt_da(
                self._constraints_history, self._lmp_history, da_bc, da_lmp, min_intervals=10
            )
            sol.flatten()
            self.solution_with_da = sol
            logger.info(f"RT+DA shift factors fit: {len(sol._flat)} SF pairs.")
            return True
        except Exception as e:
            logger.error(f"RT+DA shift factor fit failed: {e}", exc_info=True)
            return False

    def get_shadow_prices(self) -> Dict[str, float]:
        """Return the most recent shadow price snapshot."""
        if self.latest_interval is None:
            return {}
        return self.latest_interval.shadow_prices

    def is_ready(self) -> bool:
        """True once at least one successful solve has completed."""
        return self.solution is not None and self.last_updated is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append_and_trim(
        self,
        existing: pd.DataFrame,
        new: pd.DataFrame,
        keep_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Append new rows, keep only the needed columns, and trim to the rolling window."""
        if keep_columns is not None and not new.empty:
            new = new[[c for c in keep_columns if c in new.columns]]
        combined = new.copy() if existing.empty else pd.concat([existing, new], ignore_index=True)

        if "interval" in combined.columns:
            combined = combined.drop_duplicates(
                subset=[c for c in ["interval", "constraint_name", "settlement_location"]
                        if c in combined.columns],
                keep="last",
            )
            # Keep only the most recent max_history_intervals unique intervals
            unique_intervals = sorted(combined["interval"].unique())
            if len(unique_intervals) > self.max_history_intervals:
                cutoff = unique_intervals[-self.max_history_intervals]
                combined = combined[combined["interval"] >= cutoff]

        return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-ISO Cache Registry
# ---------------------------------------------------------------------------
class MultiISOSFCache:
    """
    Wrapper registry that manages individual SFCache instances per ISO.
    Forwards attributes/calls to the 'spp' cache by default for backward compatibility.
    """
    def __init__(self):
        self._caches: Dict[str, SFCache] = {
            "spp": SFCache(),
            "miso": SFCache(),
        }

    def for_iso(self, iso: str) -> SFCache:
        iso = iso.lower()
        if iso not in self._caches:
            self._caches[iso] = SFCache()
        return self._caches[iso]

    @property
    def solution(self):
        return self.for_iso("spp").solution

    @solution.setter
    def solution(self, val):
        self.for_iso("spp").solution = val

    @property
    def latest_interval(self):
        return self.for_iso("spp").latest_interval

    @latest_interval.setter
    def latest_interval(self, val):
        self.for_iso("spp").latest_interval = val

    @property
    def last_updated(self):
        return self.for_iso("spp").last_updated

    @last_updated.setter
    def last_updated(self, val):
        self.for_iso("spp").last_updated = val

    @property
    def last_solved(self):
        return self.for_iso("spp").last_solved

    @last_solved.setter
    def last_solved(self, val):
        self.for_iso("spp").last_solved = val

    @property
    def current_regime(self):
        return self.for_iso("spp").current_regime

    @current_regime.setter
    def current_regime(self, val):
        self.for_iso("spp").current_regime = val

    def update(self, interval: CachedInterval, constraints_df: pd.DataFrame, lmp_df: pd.DataFrame) -> bool:
        return self.for_iso("spp").update(interval, constraints_df, lmp_df)

    def get_shadow_prices(self) -> Dict[str, float]:
        return self.for_iso("spp").get_shadow_prices()

    def seed(self, constraints_df: pd.DataFrame, lmp_df: pd.DataFrame, iso: str = "spp") -> bool:
        return self.for_iso(iso).seed(constraints_df, lmp_df)

    def is_ready(self) -> bool:
        return self.for_iso("spp").is_ready()


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
cache = MultiISOSFCache()
