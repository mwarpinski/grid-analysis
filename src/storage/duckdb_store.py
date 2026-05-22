"""
DuckDB Time-Series Store — Phase 12.

Persistent columnar store for constraint intervals, LMP snapshots,
and shift factor history. Enables the DVR-style historical replay API.

Schema
------
Three tables, all partitioned logically by interval date:

  constraint_intervals   — one row per constraint per 5-min interval
  lmp_intervals          — MCC per node per interval
  shift_factor_snapshots — SF matrix snapshots (stored per solve)

Storage layout
--------------
All data is written as Parquet files under src/data/timeseries/ and
queried via DuckDB's native Parquet scanning. This avoids a persistent
DuckDB file lock, making the store safe to query from multiple processes
(e.g., Jupyter + server simultaneously).

Usage
-----
::

    from src.storage.duckdb_store import store

    store.write_interval(constraints_df, lmp_df)

    # Query historical constraints
    df = store.query_constraints(start="2024-11-01", end="2024-11-07")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "src" / "data" / "timeseries"


class DuckDBStore:
    """
    Manages Parquet-backed DuckDB time-series storage for the congestion
    analysis platform.
    """

    def __init__(self, data_dir: Path = _DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self._constraints_dir = self.data_dir / "constraints"
        self._lmp_dir = self.data_dir / "lmp"
        self._sf_dir = self.data_dir / "shift_factors"

        for d in (self._constraints_dir, self._lmp_dir, self._sf_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write_interval(
        self,
        constraints_df: pd.DataFrame,
        lmp_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Persist a single interval's constraint + LMP data to Parquet.
        Called by the ingest worker after each scrape cycle.
        """
        if constraints_df.empty:
            return

        # Determine partition key from interval timestamp
        if "interval" in constraints_df.columns:
            ts = pd.to_datetime(constraints_df["interval"]).max()
        else:
            ts = datetime.utcnow()

        date_str = ts.strftime("%Y%m%d")

        # Write constraints partition (append to today's file)
        self._append_parquet(
            df=constraints_df,
            directory=self._constraints_dir,
            filename=f"constraints_{date_str}.parquet",
        )

        # Write LMP partition
        if lmp_df is not None and not lmp_df.empty:
            self._append_parquet(
                df=lmp_df,
                directory=self._lmp_dir,
                filename=f"lmp_{date_str}.parquet",
            )

    def write_shift_factors(
        self,
        sf_df: pd.DataFrame,
        solved_at: Optional[datetime] = None,
        r2_scores: Optional[dict] = None,
    ) -> None:
        """Persist a shift factor matrix snapshot."""
        if sf_df.empty:
            return

        solved_at = solved_at or datetime.utcnow()
        df = sf_df.copy()
        df["solved_at"] = solved_at

        if r2_scores:
            df["r2_score"] = df["node"].map(r2_scores)

        date_str = solved_at.strftime("%Y%m%d")
        self._append_parquet(
            df=df,
            directory=self._sf_dir,
            filename=f"shift_factors_{date_str}.parquet",
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_constraints(
        self,
        start: str | datetime,
        end: str | datetime,
        constraint_name: Optional[str] = None,
        min_shadow_price: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Query constraint history for a date range.

        Parameters
        ----------
        start, end : str or datetime
            Inclusive date range.
        constraint_name : str, optional
            Filter to one constraint.
        min_shadow_price : float, optional
            Filter by minimum |shadow_price|.

        Returns
        -------
        DataFrame sorted by interval ascending.
        """
        parquet_pattern = str(self._constraints_dir / "constraints_*.parquet")
        return self._query(
            parquet_pattern=parquet_pattern,
            start=start,
            end=end,
            extra_filters=self._build_filters(
                constraint_name=constraint_name,
                min_shadow_price=min_shadow_price,
            ),
        )

    def query_lmp(
        self,
        start: str | datetime,
        end: str | datetime,
        node: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query LMP (MCC component) history."""
        parquet_pattern = str(self._lmp_dir / "lmp_*.parquet")
        extra = f"AND settlement_location = '{node}'" if node else ""
        return self._query(parquet_pattern, start=start, end=end, extra_filters=extra)

    def list_available_intervals(
        self,
        start: str | datetime,
        end: str | datetime,
    ) -> list[datetime]:
        """Return sorted list of available interval timestamps in range."""
        df = self.query_constraints(start=start, end=end)
        if df.empty or "interval" not in df.columns:
            return []
        return sorted(pd.to_datetime(df["interval"]).unique().tolist())

    def get_snapshot(self, interval: str | datetime) -> pd.DataFrame:
        """Return constraint state at a specific interval."""
        ts = pd.Timestamp(interval)
        start = ts - pd.Timedelta(seconds=30)
        end = ts + pd.Timedelta(seconds=30)
        return self.query_constraints(start=start, end=end)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append_parquet(
        self, df: pd.DataFrame, directory: Path, filename: str
    ) -> None:
        """Append rows to a daily Parquet file (read-modify-write)."""
        path = directory / filename
        try:
            if path.exists():
                existing = pd.read_parquet(path)
                # Deduplicate by (interval, constraint_name) or (interval, settlement_location)
                key_cols = [c for c in ["interval", "constraint_name", "settlement_location"]
                            if c in df.columns]
                combined = pd.concat([existing, df], ignore_index=True)
                if key_cols:
                    combined = combined.drop_duplicates(subset=key_cols, keep="last")
                combined.to_parquet(path, index=False)
            else:
                df.to_parquet(path, index=False)
        except Exception as e:
            logger.error(f"Parquet write failed ({path}): {e}")

    def _query(
        self,
        parquet_pattern: str,
        start: str | datetime,
        end: str | datetime,
        extra_filters: str = "",
    ) -> pd.DataFrame:
        """Run a DuckDB query over Parquet files matching the pattern."""
        import glob
        files = glob.glob(parquet_pattern)
        if not files:
            return pd.DataFrame()

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        sql = f"""
            SELECT *
            FROM read_parquet({files!r})
            WHERE CAST(interval AS TIMESTAMP) >= '{start_ts}'
              AND CAST(interval AS TIMESTAMP) <= '{end_ts}'
            {extra_filters}
            ORDER BY interval ASC
        """
        try:
            conn = duckdb.connect(":memory:")
            return conn.execute(sql).df()
        except Exception as e:
            logger.error(f"DuckDB query failed: {e}")
            return pd.DataFrame()

    @staticmethod
    def _build_filters(
        constraint_name: Optional[str] = None,
        min_shadow_price: Optional[float] = None,
    ) -> str:
        parts = []
        if constraint_name:
            safe = constraint_name.replace("'", "''")
            parts.append(f"AND constraint_name = '{safe}'")
        if min_shadow_price is not None:
            parts.append(f"AND ABS(shadow_price) >= {min_shadow_price}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
store = DuckDBStore()
