"""
backfill_history.py
====================
Backfills the last N days of SPP RTBM binding constraint and LMP data
into the DuckDB Parquet store.

SPP keeps per-interval files publicly available on their portal for at
least 60-90 days (sometimes longer). This script iterates over every
5-minute interval in the requested range, fetches the two CSV files
per interval (BC + LMP), and writes them to the Parquet store —
skipping intervals that are already stored.

Design principles
-----------------
- ONE request at a time (concurrency=1) with a 5-second base delay + ±20%
  random jitter. This gives ~1 req/6s = ~10 req/min — well under any
  reasonable rate limit for a public data portal.
- Exponential backoff on HTTP errors: 10s → 20s → 40s → 80s (max 4 retries).
- Checkpoint file (.backfill_checkpoint) tracks the last completed interval
  so the job resumes exactly where it left off if you stop/restart.
- At default settings, 90 days takes approximately 36-48 hours.
  This is intentional — do not rush public data scraping.

Usage
-----
    # Backfill last 90 days (default, very polite)
    python scripts/backfill_history.py

    # Backfill a specific range
    python scripts/backfill_history.py --start 2026-03-01 --end 2026-05-01

    # Dry run — show what would be fetched without downloading
    python scripts/backfill_history.py --dry-run

    # Resume from checkpoint after an interruption
    python scripts/backfill_history.py  # (checkpoint is loaded automatically)

    # Slightly faster if you're comfortable (still very polite)
    python scripts/backfill_history.py --delay 2.0

Rate estimates
--------------
  delay=5.0s (default) : ~1 req/6s → 90 days ≈ 36-48 hours
  delay=2.0s           : ~1 req/3s → 90 days ≈ 18-24 hours
  delay=1.0s           : ~1 req/2s → 90 days ≈ 9-12 hours  (watch closely)

DO NOT set concurrency > 2 or delay < 0.5 for SPP's public portal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.spp_rt_constraints_scraper import RTConstraintsScraper
from src.scrapers.spp_rtlmp_scraper import RTBMLMPScraper
from src.storage.duckdb_store import DuckDBStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interval generation
# ---------------------------------------------------------------------------

def generate_intervals(start: datetime, end: datetime) -> list[datetime]:
    """
    Generate every 5-minute interval timestamp between start and end (inclusive).
    Intervals are rounded down to the nearest 5-minute mark.
    """
    intervals = []
    current = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)
    while current <= end:
        intervals.append(current)
        current += timedelta(minutes=5)
    return intervals


def already_stored(store: DuckDBStore, date: datetime.date) -> bool:
    """
    Returns True if we already have constraint data for this calendar date.
    Checks for existence of the daily Parquet file rather than querying DuckDB
    (much faster for the skip check during backfill).
    """
    parquet_path = store._constraints_dir / f"constraints_{date.strftime('%Y%m%d')}.parquet"
    return parquet_path.exists()


# ---------------------------------------------------------------------------
# Async fetch helpers
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = Path("src/data/.backfill_checkpoint")


def load_checkpoint() -> Optional[str]:
    """Return the ISO timestamp of the last completed interval, or None."""
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            ts = data.get("last_completed")
            if ts:
                logger.info(f"Checkpoint found — resuming after {ts}")
            return ts
        except Exception:
            pass
    return None


def save_checkpoint(interval: datetime) -> None:
    """Persist the last successfully completed interval timestamp."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps({
        "last_completed": interval.isoformat(),
        "saved_at": datetime.now(UTC).isoformat(),
    }))


def clear_checkpoint() -> None:
    """Remove checkpoint after a successful full run."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


async def fetch_interval(
    bc_scraper: RTConstraintsScraper,
    lmp_scraper: RTBMLMPScraper,
    interval: datetime,
    semaphore: asyncio.Semaphore,
    delay: float,
    max_retries: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch BC + LMP for a single interval under the shared semaphore.

    - Adds ±20% random jitter to the delay so requests don't look robotic.
    - Retries up to max_retries times with exponential backoff on errors.
    - Returns (constraints_df, lmp_df) — either may be empty on 404 (normal).
    """
    async with semaphore:
        # Jittered delay: base ± 20%
        jitter = delay * random.uniform(0.8, 1.2)
        await asyncio.sleep(jitter)

        backoff = 10.0  # seconds, doubles on each retry
        for attempt in range(max_retries + 1):
            try:
                bc_df  = await asyncio.to_thread(bc_scraper.fetch_data,  interval, "interval")
                lmp_df = await asyncio.to_thread(lmp_scraper.fetch_data, interval, "interval")
                return bc_df, lmp_df
            except Exception as e:
                if attempt == max_retries:
                    logger.warning(f"  {interval}: gave up after {max_retries} retries — {e}")
                    return pd.DataFrame(), pd.DataFrame()
                wait = backoff * (2 ** attempt)
                logger.warning(
                    f"  {interval}: fetch error (attempt {attempt+1}/{max_retries}) — "
                    f"{e}. Backing off {wait:.0f}s ..."
                )
                await asyncio.sleep(wait)

    return pd.DataFrame(), pd.DataFrame()


# ---------------------------------------------------------------------------
# Main backfill loop
# ---------------------------------------------------------------------------

async def backfill(
    start: datetime,
    end: datetime,
    store: DuckDBStore,
    concurrency: int = 1,
    delay: float = 5.0,
    dry_run: bool = False,
    skip_existing_days: bool = True,
) -> None:
    """
    Async backfill loop — fetches and stores all intervals in [start, end].

    Parameters
    ----------
    start, end      : datetime range to backfill (inclusive)
    store           : DuckDBStore instance to write to
    concurrency     : max simultaneous HTTP requests to SPP portal
    delay           : seconds to sleep between requests (per-slot)
    dry_run         : if True, only log what would be fetched — no downloads
    skip_existing_days : if True, skip calendar days that already have a Parquet file
    """
    intervals = generate_intervals(start, end)
    total = len(intervals)
    est_hours = (total * (delay * 1.2 + 2)) / 3600  # rough estimate
    logger.info(f"Backfill range: {start.date()} → {end.date()}")
    logger.info(f"Total intervals: {total:,}  ({total * 2:,} HTTP requests)")
    logger.info(f"Concurrency: {concurrency}  Delay: {delay}s ±20% jitter")
    logger.info(f"Estimated time: {est_hours:.1f} hours (at default settings)")

    if dry_run:
        logger.info("DRY RUN — no data will be downloaded.")
        for iv in intervals[:5]:
            logger.info(f"  Would fetch: {iv}")
        if total > 5:
            logger.info(f"  ... and {total - 5:,} more.")
        logger.info(f"Checkpoint file: {CHECKPOINT_FILE}")
        return

    # Resume from checkpoint if available
    checkpoint_ts = load_checkpoint()
    if checkpoint_ts:
        resume_after = datetime.fromisoformat(checkpoint_ts)
        intervals = [iv for iv in intervals if iv > resume_after]
        logger.info(f"Resuming: {len(intervals):,} intervals remaining after checkpoint.")

    # Skip days that already have full Parquet files
    if skip_existing_days:
        skipped_days: set[datetime.date] = set()
        filtered = []
        for iv in intervals:
            day = iv.date()
            if day in skipped_days:
                continue
            if already_stored(store, day):
                skipped_days.add(day)
                logger.info(f"Skipping {day} — Parquet already exists.")
                continue
            filtered.append(iv)
        intervals = filtered
        if skipped_days:
            logger.info(f"Skipped {len(skipped_days)} existing day(s). "
                        f"Remaining: {len(intervals):,} intervals.")

    if not intervals:
        logger.info("Nothing to backfill — all intervals already stored.")
        return

    # Scrapers (stateless — safe to reuse across concurrent tasks)
    bc_scraper  = RTConstraintsScraper()
    lmp_scraper = RTBMLMPScraper()
    semaphore   = asyncio.Semaphore(concurrency)

    # Column normalization (mirrors ingest_worker.py)
    col_rename = {
        "Constraint Name":   "constraint_name",
        "Shadow Price":      "shadow_price",
        "Interval":          "interval",
        "Settlement Location": "settlement_location",
        "MCC":               "mcc",
    }

    # Progress tracking
    done = 0
    ok_bc = 0
    ok_lmp = 0
    miss_bc = 0
    miss_lmp = 0

    # Process in day-sized batches so we can write a Parquet per day
    # and show clean progress
    from itertools import groupby
    from datetime import date as date_type

    def day_key(iv: datetime) -> date_type:
        return iv.date()

    for day, day_intervals in groupby(intervals, key=day_key):
        day_intervals_list = list(day_intervals)
        logger.info(f"Fetching {day}  ({len(day_intervals_list)} intervals) ...")

        tasks = [
            fetch_interval(bc_scraper, lmp_scraper, iv, semaphore, delay)
            for iv in day_intervals_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        day_bc_frames  = []
        day_lmp_frames = []

        for iv, result in zip(day_intervals_list, results):
            if isinstance(result, Exception):
                logger.debug(f"  {iv}: error — {result}")
                miss_bc += 1
                miss_lmp += 1
                continue

            bc_df, lmp_df = result

            if not bc_df.empty:
                bc_df = bc_df.rename(
                    columns={k: v for k, v in col_rename.items() if k in bc_df.columns}
                )
                if "interval" not in bc_df.columns:
                    bc_df["interval"] = iv
                day_bc_frames.append(bc_df)
                ok_bc += 1
            else:
                miss_bc += 1

            if not lmp_df.empty:
                lmp_df = lmp_df.rename(
                    columns={k: v for k, v in col_rename.items() if k in lmp_df.columns}
                )
                if "interval" not in lmp_df.columns:
                    lmp_df["interval"] = iv
                day_lmp_frames.append(lmp_df)
                ok_lmp += 1
            else:
                miss_lmp += 1

            # Save checkpoint after every interval
            save_checkpoint(iv)

        done += len(day_intervals_list)

        # Write the day's data to Parquet
        if day_bc_frames:
            combined_bc = pd.concat(day_bc_frames, ignore_index=True)
            store.write_interval(combined_bc)
            logger.info(f"  → Stored {len(day_bc_frames)} BC intervals "
                        f"({len(combined_bc):,} rows)")

        if day_lmp_frames:
            combined_lmp = pd.concat(day_lmp_frames, ignore_index=True)
            # write_interval expects constraints_df as first arg; pass lmp separately
            # Use a dummy empty DF for constraints since we already wrote it above
            empty = pd.DataFrame(columns=["interval", "constraint_name"])
            store.write_interval(empty, lmp_df=combined_lmp)
            logger.info(f"  → Stored {len(day_lmp_frames)} LMP intervals "
                        f"({len(combined_lmp):,} rows)")

        pct = done / len(intervals) * 100
        remaining = len(intervals) - done
        est_remaining_h = (remaining * (delay * 1.2 + 2)) / 3600
        logger.info(
            f"Progress: {done:,}/{len(intervals):,} ({pct:.1f}%)  "
            f"BC: {ok_bc} ok / {miss_bc} miss  "
            f"LMP: {ok_lmp} ok / {miss_lmp} miss  "
            f"ETA: ~{est_remaining_h:.1f}h remaining"
        )

    # Final summary
    clear_checkpoint()
    logger.info("=" * 60)
    logger.info("Backfill complete.")
    logger.info(f"  BC intervals fetched:  {ok_bc:,} / {ok_bc + miss_bc:,}")
    logger.info(f"  LMP intervals fetched: {ok_lmp:,} / {ok_lmp + miss_lmp:,}")
    logger.info(f"  Miss rate (BC):  {miss_bc / max(1, ok_bc + miss_bc) * 100:.1f}%")
    logger.info(f"  Miss rate (LMP): {miss_lmp / max(1, ok_lmp + miss_lmp) * 100:.1f}%")
    logger.info("")
    logger.info("Storage stats after backfill:")
    stats = store.storage_stats()
    for name, info in stats.items():
        if isinstance(info, dict):
            logger.info(f"  {name}: {info['file_count']} files, {info['size_mb']} MB "
                        f"({info['oldest']} → {info['newest']})")
    logger.info(f"  Total: {stats['total_size_mb']} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill SPP RTBM history into the DuckDB Parquet store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of days to backfill (default: 90). Ignored if --start/--end given.",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date (YYYY-MM-DD). Overrides --days.",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD, inclusive). Defaults to yesterday.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Max simultaneous HTTP requests to SPP portal (default: 1). "
             "DO NOT exceed 2 for SPP's public portal.",
    )
    parser.add_argument(
        "--delay", type=float, default=5.0,
        help="Base seconds between requests, ±20%% jitter applied (default: 5.0). "
             "At 5s: 90 days ≈ 36-48 hours. At 2s: ≈ 18-24 hours.",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-fetch days that already have Parquet files (forces re-download).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without downloading anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve date range
    now = datetime.now(UTC).replace(tzinfo=None)
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=55)
    else:
        end = (now - timedelta(days=1)).replace(hour=23, minute=55, second=0, microsecond=0)

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(hour=0, minute=0)
    else:
        start = (end - timedelta(days=args.days - 1)).replace(hour=0, minute=0)

    logger.info(f"SPP RTBM Historical Backfill")
    logger.info(f"Range: {start.date()} → {end.date()} ({(end - start).days + 1} days)")

    store = DuckDBStore()

    asyncio.run(backfill(
        start=start,
        end=end,
        store=store,
        concurrency=args.concurrency,
        delay=args.delay,
        dry_run=args.dry_run,
        skip_existing_days=not args.no_skip_existing,
    ))


if __name__ == "__main__":
    main()
