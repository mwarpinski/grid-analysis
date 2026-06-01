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

Retention
---------
Daily Parquet files older than RETENTION_DAYS (default 90) are automatically
pruned once per calendar day when write_interval() is called. 90 days covers
approximately one quarter of trading data — sufficient for model validation,
backtesting, and regime analysis.

Usage
-----
::

    from src.storage.duckdb_store import store

    store.write_interval(constraints_df, lmp_df)

    # Query historical constraints
    df = store.query_constraints(start="2024-11-01", end="2024-11-07")

    # Check disk usage and retention status
    print(store.storage_stats())
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "src" / "data" / "timeseries"

# Files older than this many days are deleted during the daily pruning pass.
RETENTION_DAYS = 90

# An LMP day is considered "complete" when this fraction of 288 intervals
# has been written.  Allows for a few missing 404s at midnight/boundaries.
LMP_COMPLETE_THRESHOLD = 280


class DuckDBStore:
    """
    Manages Parquet-backed DuckDB time-series storage for the congestion
    analysis platform.
    """

    def __init__(
        self,
        data_dir: Path = _DEFAULT_DATA_DIR,
        retention_days: int = RETENTION_DAYS,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.retention_days = retention_days
        self._constraints_dir = self.data_dir / "constraints"
        self._lmp_dir         = self.data_dir / "lmp"
        self._sf_dir          = self.data_dir / "shift_factors"
        self._manifest_path   = self.data_dir / "manifest.json"

        for d in (self._constraints_dir, self._lmp_dir, self._sf_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Track last pruning run — prune at most once per calendar day
        self._last_pruned: Optional[datetime] = None

        # Bootstrap manifest from existing Parquet files if it doesn't exist
        if not self._manifest_path.exists():
            self._refresh_manifest_from_files()

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

        Also triggers a once-per-day retention pruning pass to delete
        files older than self.retention_days.
        """
        # Allow LMP-only writes (empty constraints_df is fine when lmp_df provided)
        has_constraints = not constraints_df.empty
        has_lmp = lmp_df is not None and not lmp_df.empty

        if not has_constraints and not has_lmp:
            return

        # Determine partition key
        ts = None
        if has_constraints and "interval" in constraints_df.columns:
            ts = pd.to_datetime(constraints_df["interval"]).max()
        elif has_lmp and "interval" in lmp_df.columns:
            ts = pd.to_datetime(lmp_df["interval"]).max()
        if ts is None or pd.isnull(ts):
            ts = datetime.utcnow()

        date_str = ts.strftime("%Y%m%d")
        date_key = ts.strftime("%Y-%m-%d")

        if has_constraints:
            self._append_parquet(
                df=constraints_df,
                directory=self._constraints_dir,
                filename=f"constraints_{date_str}.parquet",
            )
            self._update_manifest(date_key, "bc", constraints_df)

        if has_lmp:
            self._append_parquet(
                df=lmp_df,
                directory=self._lmp_dir,
                filename=f"lmp_{date_str}.parquet",
            )
            self._update_manifest(date_key, "lmp", lmp_df)

        # Once-per-day retention pruning
        self._maybe_prune()

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
    # Retention & Storage Stats
    # ------------------------------------------------------------------

    def prune_old_files(self) -> dict:
        """
        Delete Parquet files whose date suffix is older than retention_days.

        Files are named like ``constraints_YYYYMMDD.parquet`` — the date is
        parsed directly from the filename, so no file I/O is needed to check
        age.  The current day's file is always preserved regardless.

        Returns a summary dict: {directory: {"deleted": N, "kept": M, "freed_mb": F}}
        """
        cutoff = (datetime.utcnow() - timedelta(days=self.retention_days)).date()
        today  = datetime.utcnow().date()
        summary: dict = {}

        for subdir in (self._constraints_dir, self._lmp_dir, self._sf_dir):
            deleted, kept, freed_bytes = 0, 0, 0
            for path in sorted(subdir.glob("*.parquet")):
                # Parse date from filename stem, e.g. "constraints_20240101"
                parts = path.stem.rsplit("_", 1)
                if len(parts) != 2:
                    kept += 1
                    continue
                try:
                    file_date = datetime.strptime(parts[1], "%Y%m%d").date()
                except ValueError:
                    kept += 1
                    continue

                if file_date < cutoff and file_date < today:
                    freed_bytes += path.stat().st_size
                    path.unlink()
                    deleted += 1
                    logger.info(f"Retention pruner: deleted {path.name} ({file_date})")
                else:
                    kept += 1

            summary[subdir.name] = {
                "deleted": deleted,
                "kept": kept,
                "freed_mb": round(freed_bytes / 1_048_576, 2),
            }

        total_freed = sum(v["freed_mb"] for v in summary.values())
        if total_freed > 0:
            logger.info(
                f"Retention pruning complete: freed {total_freed:.1f} MB "
                f"(cutoff={cutoff}, retention={self.retention_days} days)"
            )
        else:
            logger.debug(f"Retention pruner: nothing to delete (cutoff={cutoff}).")

        self._last_pruned = datetime.utcnow()
        return summary

    def storage_stats(self) -> dict:
        """
        Return disk usage and file counts for each data subdirectory.

        Useful for the /api/v1/status endpoint and ad-hoc inspection.
        """
        stats: dict = {}
        total_bytes = 0
        for subdir in (self._constraints_dir, self._lmp_dir, self._sf_dir):
            files = sorted(subdir.glob("*.parquet"))
            size  = sum(f.stat().st_size for f in files)
            total_bytes += size
            dates = []
            for f in files:
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2:
                    try:
                        dates.append(datetime.strptime(parts[1], "%Y%m%d").date())
                    except ValueError:
                        pass
            stats[subdir.name] = {
                "file_count": len(files),
                "size_mb": round(size / 1_048_576, 2),
                "oldest": str(min(dates)) if dates else None,
                "newest": str(max(dates)) if dates else None,
            }
        stats["total_size_mb"] = round(total_bytes / 1_048_576, 2)
        stats["retention_days"] = self.retention_days
        stats["last_pruned"] = str(self._last_pruned) if self._last_pruned else None
        return stats

    def _maybe_prune(self) -> None:
        """Run pruning at most once per calendar day."""
        now = datetime.utcnow()
        if self._last_pruned is None or self._last_pruned.date() < now.date():
            self.prune_old_files()

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
    # Manifest — tracks per-day download status for the backfill
    # ------------------------------------------------------------------

    def read_manifest(self) -> dict:
        """
        Return the full manifest dict.

        Schema::

            {
              "2026-05-07": {
                "bc":  {"status": "complete"|"partial"|"missing",
                        "intervals": 288, "rows": 14297,
                        "stored_at": "2026-05-28T14:00:00"},
                "lmp": {"status": "complete"|"partial"|"missing",
                        "intervals": 288, "rows": 456032,
                        "stored_at": "2026-05-28T15:00:00"}
              },
              ...
            }
        """
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text())
            except Exception:
                pass
        return {}

    def manifest_status(self, date_key: str, data_type: str) -> str:
        """
        Return the status for a specific day/type: 'complete', 'partial', or 'missing'.

        Parameters
        ----------
        date_key  : str  e.g. '2026-05-07'
        data_type : str  'bc' or 'lmp'
        """
        manifest = self.read_manifest()
        entry = manifest.get(date_key, {}).get(data_type, {})
        return entry.get("status", "missing")

    def missing_days(
        self,
        start: str,
        end: str,
        data_type: str = "lmp",
        include_partial: bool = True,
    ) -> list[str]:
        """
        Return a sorted list of date strings that are missing or partial
        for the given data_type between start and end (inclusive).

        Parameters
        ----------
        start, end    : str  e.g. '2026-05-07'
        data_type     : str  'bc' or 'lmp'
        include_partial : bool  If True, partial days are returned as missing.
        """
        from datetime import date, timedelta
        manifest = self.read_manifest()
        start_d = date.fromisoformat(start)
        end_d   = date.fromisoformat(end)
        missing = []
        d = start_d
        while d <= end_d:
            key = d.isoformat()
            status = manifest.get(key, {}).get(data_type, {}).get("status", "missing")
            if status == "missing" or (include_partial and status == "partial"):
                missing.append(key)
            d += timedelta(days=1)
        return missing

    def _update_manifest(
        self, date_key: str, data_type: str, df: pd.DataFrame
    ) -> None:
        """Update the manifest entry for date_key/data_type after a write."""
        manifest = self.read_manifest()
        day_entry = manifest.setdefault(date_key, {})

        intervals = df["interval"].nunique() if "interval" in df.columns else 0
        rows = len(df)

        threshold = LMP_COMPLETE_THRESHOLD if data_type == "lmp" else 280
        existing_intervals = day_entry.get(data_type, {}).get("intervals", 0)
        total_intervals = max(intervals, existing_intervals)

        status = "complete" if total_intervals >= threshold else "partial"
        day_entry[data_type] = {
            "status":    status,
            "intervals": total_intervals,
            "rows":      day_entry.get(data_type, {}).get("rows", 0) + rows,
            "stored_at": datetime.utcnow().isoformat(timespec="seconds"),
        }

        try:
            self._manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
        except Exception as e:
            logger.warning(f"Manifest write failed: {e}")

    def mark_day_exhausted(
        self, date_key: str, data_type: str, fetched: int, attempted: int
    ) -> None:
        """
        Mark a day/data_type as fully attempted, even if below the normal
        interval threshold.  Called after all intervals have been tried —
        if SPP returned 404s for the missing slots those are genuine data
        gaps and retrying will never help.

        Sets status to 'complete' regardless of interval count and records
        fetched/attempted/spp_gaps for auditability.
        """
        manifest = self.read_manifest()
        day_entry = manifest.setdefault(date_key, {})
        existing = day_entry.get(data_type, {})
        day_entry[data_type] = {
            "status":    "complete",
            "intervals": fetched,
            "attempted": attempted,
            "rows":      existing.get("rows", 0),
            "spp_gaps":  attempted - fetched,
            "stored_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        try:
            self._manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
        except Exception as e:
            logger.warning(f"Manifest mark_day_exhausted failed: {e}")

    def _refresh_manifest_from_files(self) -> None:
        """
        Bootstrap the manifest by scanning existing Parquet files.
        Called once on first instantiation when no manifest.json exists.
        """
        manifest: dict = {}
        now_str = datetime.utcnow().isoformat(timespec="seconds")

        for f in sorted(self._constraints_dir.glob("constraints_*.parquet")):
            parts = f.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            try:
                d = datetime.strptime(parts[1], "%Y%m%d").date()
                df = pd.read_parquet(f, columns=["interval"])
                n = df["interval"].nunique()
                key = d.isoformat()
                manifest.setdefault(key, {})["bc"] = {
                    "status":    "complete" if n >= 280 else "partial",
                    "intervals": n,
                    "rows":      len(pd.read_parquet(f)),
                    "stored_at": now_str,
                }
            except Exception:
                pass

        for f in sorted(self._lmp_dir.glob("lmp_*.parquet")):
            parts = f.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            try:
                d = datetime.strptime(parts[1], "%Y%m%d").date()
                df = pd.read_parquet(f, columns=["interval"])
                n = df["interval"].nunique()
                key = d.isoformat()
                manifest.setdefault(key, {})["lmp"] = {
                    "status":    "complete" if n >= LMP_COMPLETE_THRESHOLD else "partial",
                    "intervals": n,
                    "rows":      len(pd.read_parquet(f)),
                    "stored_at": now_str,
                }
            except Exception:
                pass

        try:
            self._manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
            logger.info(
                f"Manifest bootstrapped from existing Parquet files "
                f"({len(manifest)} days tracked)."
            )
        except Exception as e:
            logger.warning(f"Manifest bootstrap failed: {e}")

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
