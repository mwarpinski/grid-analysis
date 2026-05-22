"""
Constraint Regime Analysis for SPP Transmission Data.

A "constraint regime" is the set of transmission constraints that are
simultaneously binding in a group of intervals.  Shift factors can only be
estimated for a regime if we observe it in enough intervals to form an
over-determined system (T > K where T = intervals, K = constraints).

This module handles:
- Identifying and cataloging all regimes in a dataset
- Reporting which regimes have sufficient data for SF estimation
- Providing rich diagnostics about regime frequency, duration, and stability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Regime:
    """Describes one binding constraint regime."""
    key: FrozenSet[str]
    constraints: List[str]
    intervals: List[pd.Timestamp]

    @property
    def n_intervals(self) -> int:
        return len(self.intervals)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    @property
    def is_solvable(self) -> bool:
        """True if we have enough data points to solve the system."""
        return self.n_intervals >= max(self.n_constraints, 2)

    @property
    def overdetermination_ratio(self) -> float:
        """T / K — higher is better (more data relative to unknowns)."""
        if self.n_constraints == 0:
            return float("inf")
        return self.n_intervals / self.n_constraints

    def first_seen(self) -> pd.Timestamp:
        return min(self.intervals)

    def last_seen(self) -> pd.Timestamp:
        return max(self.intervals)


@dataclass
class RegimeReport:
    """Summary statistics for all regimes found in a dataset."""
    total_regimes: int
    solvable_regimes: int
    unsolvable_regimes: int
    total_intervals: int
    covered_intervals: int
    # Sorted list of regimes by n_intervals descending
    regimes: List[Regime] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        if self.total_intervals == 0:
            return 0.0
        return 100.0 * self.covered_intervals / self.total_intervals


# ---------------------------------------------------------------------------
# Core regime builder
# ---------------------------------------------------------------------------

class RegimeBuilder:
    """
    Identifies constraint regimes from SPP binding constraint data.

    Parameters
    ----------
    tolerance : float
        Shadow prices with |value| below this are treated as non-binding.
    min_intervals : int
        Regimes with fewer intervals than this are marked unsolvable.
    """

    def __init__(self, tolerance: float = 1e-4, min_intervals: int = 10) -> None:
        self.tolerance = tolerance
        self.min_intervals = min_intervals

    def build(self, constraints_df: pd.DataFrame) -> RegimeReport:
        """
        Build regime report from binding constraints data.

        Parameters
        ----------
        constraints_df : DataFrame
            Columns: ``interval``, ``constraint_name``, ``shadow_price``
            (case-insensitive; SPP raw column names also accepted)

        Returns
        -------
        RegimeReport
        """
        df = self._normalize(constraints_df)

        # Filter to actually binding constraints
        binding = df[df["shadow_price"].abs() > self.tolerance].copy()

        all_intervals = df["interval"].unique()
        total_intervals = len(all_intervals)

        # Group by interval → get constraint set
        regime_map: Dict[FrozenSet[str], Regime] = {}

        for interval, grp in binding.groupby("interval"):
            key = frozenset(grp["constraint_name"].unique())
            if key not in regime_map:
                regime_map[key] = Regime(
                    key=key,
                    constraints=sorted(key),
                    intervals=[],
                )
            regime_map[key].intervals.append(interval)

        regimes = sorted(
            regime_map.values(),
            key=lambda r: r.n_intervals,
            reverse=True,
        )

        # Apply min_intervals filter for solvability
        solvable = [
            r for r in regimes
            if r.is_solvable and r.n_intervals >= self.min_intervals
        ]
        unsolvable = [r for r in regimes if r not in solvable]

        covered = set()
        for r in solvable:
            covered.update(r.intervals)

        return RegimeReport(
            total_regimes=len(regimes),
            solvable_regimes=len(solvable),
            unsolvable_regimes=len(unsolvable),
            total_intervals=total_intervals,
            covered_intervals=len(covered),
            regimes=regimes,
        )

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and types."""
        df = df.copy()
        # Map common SPP column names
        rename_map = {
            "Constraint Name": "constraint_name",
            "Shadow Price": "shadow_price",
            "Interval": "interval",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df.columns = [c.lower() for c in df.columns]

        if "interval" not in df.columns:
            raise ValueError("constraints_df must have an 'interval' column.")

        if not pd.api.types.is_datetime64_any_dtype(df["interval"]):
            df["interval"] = pd.to_datetime(df["interval"])

        return df


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_regime_report(report: RegimeReport, top_n: int = 20) -> None:
    """Print a human-readable summary of the regime report."""
    print("=" * 72)
    print("CONSTRAINT REGIME REPORT")
    print("=" * 72)
    print(f"Total unique regimes:    {report.total_regimes}")
    print(f"Solvable regimes:        {report.solvable_regimes}")
    print(f"Unsolvable (too few):    {report.unsolvable_regimes}")
    print(f"Total intervals:         {report.total_intervals}")
    print(f"Covered by solvable:     {report.covered_intervals} "
          f"({report.coverage_pct:.1f}%)")
    print()
    print(f"Top {min(top_n, len(report.regimes))} Regimes by Frequency:")
    print("-" * 72)
    print(f"{'N':>6}  {'K':>4}  {'T/K':>6}  {'Solvable':>8}  Constraints")
    print("-" * 72)
    for r in report.regimes[:top_n]:
        solvable_str = "YES" if r.is_solvable else "NO "
        constraint_str = ", ".join(r.constraints[:3])
        if len(r.constraints) > 3:
            constraint_str += f"  (+{len(r.constraints) - 3} more)"
        print(
            f"{r.n_intervals:>6}  {r.n_constraints:>4}  "
            f"{r.overdetermination_ratio:>6.1f}  {solvable_str:>8}  {constraint_str}"
        )
    print("=" * 72)


def regime_report_to_dataframe(report: RegimeReport) -> pd.DataFrame:
    """Convert a RegimeReport to a tidy DataFrame."""
    rows = []
    for r in report.regimes:
        rows.append({
            "constraints": " | ".join(r.constraints),
            "n_constraints": r.n_constraints,
            "n_intervals": r.n_intervals,
            "overdetermination_ratio": r.overdetermination_ratio,
            "solvable": r.is_solvable,
            "first_seen": r.first_seen() if r.intervals else None,
            "last_seen": r.last_seen() if r.intervals else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_regimes(
    constraints_df: pd.DataFrame,
    tolerance: float = 1e-4,
    min_intervals: int = 10,
    verbose: bool = True,
) -> RegimeReport:
    """
    Analyze binding constraint regimes in SPP data.

    Parameters
    ----------
    constraints_df : DataFrame
        Binding constraint data with columns:
        ``interval``, ``constraint_name``, ``shadow_price``
    tolerance : float
        Shadow price threshold for "binding"
    min_intervals : int
        Minimum intervals per regime to mark as solvable
    verbose : bool
        If True, print a summary table to stdout

    Returns
    -------
    RegimeReport
    """
    builder = RegimeBuilder(tolerance=tolerance, min_intervals=min_intervals)
    report = builder.build(constraints_df)
    if verbose:
        print_regime_report(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Analyze constraint regimes in SPP binding constraint data."
    )
    parser.add_argument("--constraints", required=True, help="Path to constraints CSV")
    parser.add_argument(
        "--min-intervals", type=int, default=10,
        help="Minimum intervals per regime to mark as solvable (default: 10)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write regime summary CSV"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.constraints)
    # Filter to binding only if State column present
    if "State" in df.columns:
        df = df[df["State"] == "BINDING"]

    report = build_regimes(df, min_intervals=args.min_intervals, verbose=True)

    if args.output:
        out_df = regime_report_to_dataframe(report)
        out_df.to_csv(args.output, index=False)
        print(f"\nWrote regime summary to {args.output}")


if __name__ == "__main__":
    _main()
