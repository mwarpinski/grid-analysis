"""
backfill_history.py
====================
Backfills the last N days of SPP RTBM binding constraint and LMP data
into the Parquet store.

Data availability on SPP portal
---------------------------------
- **BC (binding constraints)**: SPP only publishes live per-interval BC files
  for the *current day*. For historical dates, use the daily aggregate file:
    RTBM-DAILY-BC-YYYYMMDD.csv  →  1 file per day, all 288 intervals combined
  These are available under By_Day/ and are retained for ~90 days.

- **LMP**: SPP publishes per-interval LMP files for historical dates:
    RTBM-LMP-SL-YYYYMMDDhhmm.csv  →  288 files per day
  Note: intervals start at 00:05, not 00:00 (SPP convention).

Strategy
--------
  1. Fetch 1 daily BC file per day  (fast — ~1 request/day)
  2. Fetch per-interval LMP files   (288 requests/day — rate limited)

Design principles
-----------------
- ONE request at a time with a 2-second base delay + ±20% random jitter.
- Exponential backoff on HTTP errors.
- Checkpoint file tracks the last completed day so the job resumes after restart.
- Skips days that already have Parquet files.

Usage
-----
    # Backfill last 21 days (3 full weeks — good day-of-week seasonality)
    python scripts/backfill_history.py --days 21

    # Backfill a specific range
    python scripts/backfill_history.py --start 2026-05-01 --end 2026-05-25

    # Dry run
    python scripts/backfill_history.py --dry-run

Rate estimates (LMP-dominated)
---------------------------------
  21 days × 288 intervals × ~2.5s/request ≈ ~4.2 hours
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone, date as date_type
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.spp_rt_constraints_scraper import RTConstraintsScraper
from src.scrapers.spp_rtlmp_scraper import RTBMLMPScraper
from src.storage.duckdb_store import DuckDBStore

UTC = timezone.utc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = Path("src/data/.backfill_checkpoint")


def generate_days(start: date_type, end: date_type) -> List[date_type]:
    """Return every calendar date in [start, end]."""
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def generate_lmp_intervals(day: date_type) -> List[datetime]:
    """
    Return all 288 RTBM interval timestamps for a calendar day.
    SPP intervals run 00:05 → 23:55 (5-min steps, no 00:00 file).
    """
    intervals = []
    base = datetime(day.year, day.month, day.day, 0, 5)
    for i in range(288):
        intervals.append(base + timedelta(minutes=5 * i))
    return intervals


def already_stored(store: DuckDBStore, day: date_type) -> bool:
    """True if a daily Parquet file already exists for this date."""
    parquet_path = store._constraints_dir / f"constraints_{day.strftime('%Y%m%d')}.parquet"
    return parquet_path.exists()


def load_checkpoint() -> Optional[str]:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            ts = data.get("last_completed_day")
            if ts:
                logger.info(f"Checkpoint found — resuming after {ts}")
            return ts
        except Exception:
            pass
    return None


def save_checkpoint(day: date_type) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps({
        "last_completed_day": day.isoformat(),
        "saved_at": datetime.now(UTC).isoformat(),
    }))


def clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

async def fetch_daily_bc(
    scraper: RTConstraintsScraper,
    day: date_type,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch the daily aggregate BC file for one calendar day.
    Returns empty DataFrame on failure (404 or repeated errors).
    """
    day_dt = datetime(day.year, day.month, day.day)
    backoff = 5.0
    for attempt in range(max_retries + 1):
        try:
            df = await asyncio.to_thread(scraper.fetch_data, day_dt, "day")
            if not df.empty:
                logger.info(f"  BC {day}: {len(df)} rows fetched")
            return df
        except Exception as e:
            if "404" in str(e):
                logger.warning(f"  BC {day}: 404 — daily file not available (skipping BC)")
                return pd.DataFrame()
            if attempt == max_retries:
                logger.warning(f"  BC {day}: gave up after {max_retries} retries — {e}")
                return pd.DataFrame()
            wait = backoff * (2 ** attempt) * random.uniform(0.8, 1.2)
            logger.warning(f"  BC {day}: error (attempt {attempt+1}) — {e}. Retrying in {wait:.0f}s")
            await asyncio.sleep(wait)
    return pd.DataFrame()


async def fetch_lmp_interval(
    scraper: RTBMLMPScraper,
    interval: datetime,
    delay: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2,
) -> pd.DataFrame:
    """Fetch one 5-minute LMP interval file with rate limiting."""
    async with semaphore:
        await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        backoff = 5.0
        for attempt in range(max_retries + 1):
            try:
                df = await asyncio.to_thread(scraper.fetch_data, interval, "interval")
                return df
            except Exception as e:
                if "404" in str(e):
                    return pd.DataFrame()   # interval simply doesn't exist
                if attempt == max_retries:
                    logger.debug(f"  LMP {interval}: gave up — {e}")
                    return pd.DataFrame()
                await asyncio.sleep(backoff * (2 ** attempt))
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main backfill loop
# ---------------------------------------------------------------------------

async def backfill(
    start: date_type,
    end: date_type,
    store: DuckDBStore,
    delay: float = 2.0,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> None:
    all_days = generate_days(start, end)
    logger.info(f"Backfill range: {start} → {end}  ({len(all_days)} days)")

    if dry_run:
        logger.info("DRY RUN — no data will be downloaded.")
        for d in all_days[:5]:
            logger.info(f"  Would fetch BC(1 req) + LMP(288 req) for {d}")
        if len(all_days) > 5:
            logger.info(f"  ... and {len(all_days) - 5} more days")
        est = len(all_days) * 288 * (delay * 1.1) / 3600
        logger.info(f"  Estimated time: {est:.1f} hours")
        return

    # Resume from checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        resume_after = date_type.fromisoformat(checkpoint)
        all_days = [d for d in all_days if d > resume_after]
        logger.info(f"Resuming: {len(all_days)} days remaining")

    # Skip already-stored days
    if skip_existing:
        skipped = []
        filtered = []
        for d in all_days:
            if already_stored(store, d):
                skipped.append(d)
            else:
                filtered.append(d)
        if skipped:
            logger.info(f"Skipping {len(skipped)} day(s) already in Parquet store")
        all_days = filtered

    if not all_days:
        logger.info("Nothing to backfill — all days already stored.")
        return

    est = len(all_days) * 288 * (delay * 1.1) / 3600
    logger.info(f"{len(all_days)} days to fetch. Estimated time: {est:.1f} hours")

    bc_scraper  = RTConstraintsScraper()
    lmp_scraper = RTBMLMPScraper()
    semaphore   = asyncio.Semaphore(1)   # 1 concurrent request — polite

    col_rename = {
        "Constraint Name":     "constraint_name",
        "Shadow Price":        "shadow_price",
        "Interval":            "interval",
        "Settlement Location": "settlement_location",
        "MCC":                 "mcc",
    }

    total_days = len(all_days)
    for idx, day in enumerate(all_days, 1):
        logger.info(f"[{idx}/{total_days}] Fetching {day} ...")

        # ── 1. Daily BC aggregate (1 request) ───────────────────────────
        bc_df = await fetch_daily_bc(bc_scraper, day)
        if not bc_df.empty:
            bc_df = bc_df.rename(columns={
                k: v for k, v in col_rename.items() if k in bc_df.columns
            })
            bc_df.columns = [c.lower() for c in bc_df.columns]
            if "interval" in bc_df.columns:
                bc_df["interval"] = pd.to_datetime(bc_df["interval"], errors="coerce")
            store.write_interval(bc_df)
            logger.info(f"  → BC stored: {len(bc_df)} rows")
        else:
            logger.warning(f"  → BC: no data for {day}")

        # ── 2. Per-interval LMP (288 requests) ──────────────────────────
        intervals = generate_lmp_intervals(day)
        tasks = [
            fetch_lmp_interval(lmp_scraper, iv, delay, semaphore)
            for iv in intervals
        ]
        results = await asyncio.gather(*tasks)

        lmp_frames = []
        ok = miss = 0
        for iv, df in zip(intervals, results):
            if df is not None and not df.empty:
                df = df.rename(columns={
                    k: v for k, v in col_rename.items() if k in df.columns
                })
                df.columns = [c.lower() for c in df.columns]
                if "interval" not in df.columns:
                    df["interval"] = iv
                else:
                    df["interval"] = pd.to_datetime(df["interval"], errors="coerce")
                lmp_frames.append(df)
                ok += 1
            else:
                miss += 1

        if lmp_frames:
            combined_lmp = pd.concat(lmp_frames, ignore_index=True)
            empty_bc = pd.DataFrame(columns=["interval", "constraint_name"])
            store.write_interval(empty_bc, lmp_df=combined_lmp)
            logger.info(f"  → LMP stored: {ok} intervals ({len(combined_lmp):,} rows), {miss} missed")
        else:
            logger.warning(f"  → LMP: no data for {day}")

        save_checkpoint(day)
        pct = idx / total_days * 100
        remaining_h = (total_days - idx) * 288 * (delay * 1.1) / 3600
        logger.info(f"  Progress: {idx}/{total_days} ({pct:.0f}%)  ETA ~{remaining_h:.1f}h")

    clear_checkpoint()
    logger.info("=" * 60)
    logger.info("Backfill complete.")
    stats = store.storage_stats()
    for name, info in stats.items():
        if isinstance(info, dict):
            logger.info(
                f"  {name}: {info['file_count']} files, {info['size_mb']} MB "
                f"({info.get('oldest','?')} → {info.get('newest','?')})"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill SPP RTBM history (daily BC + per-interval LMP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--days", type=int, default=21,
        help="Number of days to backfill ending yesterday (default: 21).",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end",   type=str, default=None, help="End date YYYY-MM-DD (inclusive, default: yesterday).")
    parser.add_argument("--delay", type=float, default=2.0, help="Base seconds between LMP requests (default: 2.0).")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-fetch days already in Parquet.")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(UTC).date()

    end  = date_type.fromisoformat(args.end)   if args.end   else now - timedelta(days=1)
    start = date_type.fromisoformat(args.start) if args.start else end - timedelta(days=args.days - 1)

    logger.info(f"SPP RTBM Historical Backfill")
    logger.info(f"Range: {start} → {end}  ({(end - start).days + 1} days)")
    logger.info(f"Strategy: 1 daily BC file/day + 288 LMP interval files/day")

    store = DuckDBStore()
    asyncio.run(backfill(
        start=start,
        end=end,
        store=store,
        delay=args.delay,
        dry_run=args.dry_run,
        skip_existing=not args.no_skip_existing,
    ))


if __name__ == "__main__":
    main()
