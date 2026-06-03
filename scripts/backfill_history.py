"""
backfill_history.py
====================
Backfills the last N days of SPP RTBM or MISO RT market binding constraint and LMP data
into the Parquet store.

Usage
-----
    # Backfill SPP last 21 days
    python scripts/backfill_history.py --iso spp --days 21

    # Backfill MISO last 45 days
    python scripts/backfill_history.py --iso miso --days 45

    # Dry run
    python scripts/backfill_history.py --iso miso --days 45 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone, date as date_type
from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.spp_rt_constraints_scraper import RTConstraintsScraper
from src.scrapers.spp_rtlmp_scraper import RTBMLMPScraper
from src.storage.duckdb_store import store as default_store, DuckDBStore

UTC = timezone.utc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_checkpoint_file(iso: str) -> Path:
    return Path(f"src/data/.backfill_checkpoint_{iso.lower()}")


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


def bc_already_stored(store: DuckDBStore, day: date_type, iso: str = "spp") -> bool:
    """True if the BC Parquet file already exists for this date."""
    return (store._constraints_dir_for_iso(iso) / f"constraints_{day.strftime('%Y%m%d')}.parquet").exists()


def lmp_already_stored(store: DuckDBStore, day: date_type, iso: str = "spp") -> bool:
    """
    True if the LMP parquet exists AND has enough expected
    intervals for the day.
    """
    path = store._lmp_dir_for_iso(iso) / f"lmp_{day.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["interval"])
        thresh = 24 if iso.lower() == "miso" else 280
        return df["interval"].nunique() >= thresh
    except Exception:
        return False


def lmp_stored_intervals(store: DuckDBStore, day: date_type, iso: str = "spp") -> set:
    """
    Return the set of interval timestamps already written to the LMP parquet
    for this day, floored to the minute.
    """
    path = store._lmp_dir_for_iso(iso) / f"lmp_{day.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path, columns=["interval"])
        return set(pd.to_datetime(df["interval"]).dt.floor("min").unique())
    except Exception:
        return set()


def load_checkpoint(iso: str) -> Optional[str]:
    checkpoint_file = get_checkpoint_file(iso)
    if checkpoint_file.exists():
        try:
            data = json.loads(checkpoint_file.read_text())
            ts = data.get("last_completed_day")
            if ts:
                logger.info(f"Checkpoint found for {iso.upper()} — resuming after {ts}")
            return ts
        except Exception:
            pass
    return None


def save_checkpoint(day: date_type, iso: str) -> None:
    checkpoint_file = get_checkpoint_file(iso)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file.write_text(json.dumps({
        "last_completed_day": day.isoformat(),
        "saved_at": datetime.now(UTC).isoformat(),
    }))


def clear_checkpoint(iso: str) -> None:
    checkpoint_file = get_checkpoint_file(iso)
    if checkpoint_file.exists():
        checkpoint_file.unlink()


# ---------------------------------------------------------------------------
# MISO-Specific Parsing Helpers
# ---------------------------------------------------------------------------

def parse_miso_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw MISO yearly binding constraints CSV."""
    df = df[df['Market Date'].astype(str).str.contains('/', na=False)].copy()
    
    # Parse shadow price (MISO puts negative sign or parentheses, we want absolute)
    prices = df['Preliminary Shadow Price'].astype(str).str.replace('(', '', regex=False)
    prices = prices.str.replace(')', '', regex=False).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    df['shadow_price'] = pd.to_numeric(prices, errors='coerce').abs()
    
    # Parse interval (localized to America/New_York and converted to UTC timezone-naive)
    df['interval'] = pd.to_datetime(df['Market Date'] + ' ' + df['Hour of Occurrence'])
    try:
        df['interval'] = df['interval'].dt.tz_localize("America/New_York", ambiguous='NaT', nonexistent='NaT').dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
        
    df = df.rename(columns={
        'Constraint Name': 'constraint_name',
        'Contingency Description': 'contingency_name'
    })
    
    df['state'] = 'BINDING'
    df['limit_mw'] = None
    df['monitored_mw'] = None
    df['pct_of_limit'] = 100.0
    
    cols = ['constraint_name', 'interval', 'shadow_price', 'state', 'limit_mw', 'monitored_mw', 'pct_of_limit', 'contingency_name']
    return df[[c for c in cols if c in df.columns]]


def parse_miso_lmp(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Parse raw MISO daily LMP CSV with HE format."""
    df = df.rename(columns={'Node': 'settlement_location', 'Value': 'value'})
    df = df[df['value'].isin(['LMP', 'MCC', 'MLC'])]
    
    he_cols = [f'HE {i}' for i in range(1, 25)]
    melted = df.melt(
        id_vars=['settlement_location', 'value'],
        value_vars=[c for c in he_cols if c in df.columns],
        var_name='he',
        value_name='price'
    )
    
    melted['hour'] = melted['he'].str.replace('HE ', '', regex=False).astype(int)
    melted['price'] = pd.to_numeric(melted['price'], errors='coerce')
    
    pivoted = melted.pivot(
        index=['settlement_location', 'hour'],
        columns='value',
        values='price'
    ).reset_index()
    
    for col in ['LMP', 'MCC', 'MLC']:
        if col not in pivoted.columns:
            pivoted[col] = None
            
    pivoted = pivoted.rename(columns={
        'LMP': 'lmp',
        'MCC': 'mcc',
        'MLC': 'mlc'
    })
    
    base_date = pd.to_datetime(date_str)
    pivoted['interval'] = base_date + pd.to_timedelta(pivoted['hour'] - 1, unit='h')
    
    try:
        pivoted['interval'] = pivoted['interval'].dt.tz_localize("America/New_York", ambiguous='NaT', nonexistent='NaT').dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
        
    pivoted = pivoted.dropna(subset=['interval'])
    return pivoted[['interval', 'settlement_location', 'lmp', 'mcc', 'mlc']]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

async def fetch_daily_bc(
    scraper: RTConstraintsScraper,
    day: date_type,
    max_retries: int = 3,
) -> pd.DataFrame:
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
    async with semaphore:
        await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        backoff = 5.0
        for attempt in range(max_retries + 1):
            try:
                df = await asyncio.to_thread(scraper.fetch_data, interval, "interval")
                return df
            except Exception as e:
                if "404" in str(e):
                    return pd.DataFrame()
                if attempt == max_retries:
                    logger.debug(f"  LMP {interval}: gave up — {e}")
                    return pd.DataFrame()
                await asyncio.sleep(backoff * (2 ** attempt))
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# MISO Year Constraints Loader
# ---------------------------------------------------------------------------

async def fetch_miso_year_constraints(year: int) -> pd.DataFrame:
    url = f"https://docs.misoenergy.org/marketreports/{year}_rt_bc_HIST.csv"
    logger.info(f"Downloading MISO yearly constraints for {year} from {url} ...")
    try:
        def _get():
            res = requests.get(url, timeout=45)
            res.raise_for_status()
            return res.text
        text = await asyncio.to_thread(_get)
        df = pd.read_csv(StringIO(text), skiprows=2)
        parsed = parse_miso_constraints(df)
        logger.info(f"Successfully loaded and parsed {len(parsed)} MISO constraint records for {year}")
        return parsed
    except Exception as e:
        logger.error(f"Failed to fetch MISO constraints for {year}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main backfill loop
# ---------------------------------------------------------------------------

async def backfill_miso(
    start: date_type,
    end: date_type,
    store: DuckDBStore,
    delay: float = 1.0,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    all_days = generate_days(start, end)
    logger.info(f"MISO Backfill range: {start} → {end}  ({len(all_days)} days)")

    if dry_run:
        manifest = store.read_manifest(iso="miso")
        logger.info("DRY RUN — manifest status per day:")
        for d in all_days:
            key = d.isoformat()
            bc  = manifest.get(key, {}).get("bc",  {}).get("status", "missing")
            lmp = manifest.get(key, {}).get("lmp", {}).get("status", "missing")
            flag = "✓" if bc == "complete" and lmp == "complete" else "✗"
            logger.info(f"  {flag} {key}  BC={bc:<8} LMP={lmp}")
        return

    # Filter days to process using the manifest
    if force:
        days_to_process = all_days
        logger.info("--force: re-fetching all days regardless of manifest")
    else:
        days_to_process = [
            d for d in all_days
            if store.manifest_status(d.isoformat(), "bc", iso="miso")  != "complete"
            or store.manifest_status(d.isoformat(), "lmp", iso="miso") != "complete"
        ]
        skipped = len(all_days) - len(days_to_process)
        if skipped:
            logger.info(f"Manifest: skipping {skipped} day(s) already complete")

    if not days_to_process:
        logger.info("Nothing to backfill — all days complete per manifest.")
        return

    # Load constraints by year
    years_needed = sorted(list(set(d.year for d in days_to_process)))
    miso_year_constraints = {}
    for yr in years_needed:
        miso_year_constraints[yr] = await fetch_miso_year_constraints(yr)

    total_days = len(days_to_process)
    checkpoint = load_checkpoint("miso")
    if checkpoint and force:
        clear_checkpoint("miso")

    for idx, day in enumerate(days_to_process, 1):
        day_str = day.isoformat()
        logger.info(f"[{idx}/{total_days}] Fetching MISO data for {day} ...")

        # ── 1. Constraints ──
        if store.manifest_status(day_str, "bc", iso="miso") == "complete" and not force:
            logger.info(f"  BC {day}: already complete, skipping")
        else:
            yr_df = miso_year_constraints.get(day.year)
            if yr_df is not None and not yr_df.empty:
                try:
                    est_dates = pd.to_datetime(yr_df['interval']).dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.date
                    day_bc_df = yr_df[est_dates == day].copy()
                except Exception:
                    day_bc_df = yr_df[pd.to_datetime(yr_df['interval']).dt.date == day].copy()
                
                if not day_bc_df.empty:
                    # Group by UTC date and write each group separately
                    groups = day_bc_df.groupby(pd.to_datetime(day_bc_df["interval"]).dt.date)
                    for utc_date, group_df in groups:
                        store.write_interval(group_df, iso="miso")
                    logger.info(f"  → BC stored: {len(day_bc_df)} rows")
                else:
                    store.mark_day_exhausted(day_str, "bc", 0, 0, iso="miso")
                    logger.info(f"  → BC: no binding constraints (marked complete)")
            else:
                store.mark_day_exhausted(day_str, "bc", 0, 0, iso="miso")
                logger.info(f"  → BC: no yearly data loaded (marked complete)")

        # ── 2. LMPs ──
        if store.manifest_status(day_str, "lmp", iso="miso") == "complete" and not force:
            logger.info(f"  LMP {day}: already complete, skipping")
        else:
            date_key_yyyymmdd = day.strftime('%Y%m%d')
            url = f"https://docs.misoenergy.org/marketreports/{date_key_yyyymmdd}_rt_lmp_final.csv"
            logger.info(f"  LMP {day}: fetching daily file from {url} ...")
            
            await asyncio.sleep(delay)
            
            try:
                def _get():
                    res = requests.get(url, timeout=20)
                    if res.status_code == 404:
                        return None
                    res.raise_for_status()
                    return res.text
                    
                csv_text = await asyncio.to_thread(_get)
                if csv_text:
                    raw_df = pd.read_csv(StringIO(csv_text), skiprows=4)
                    day_lmp_df = parse_miso_lmp(raw_df, day_str)
                    if not day_lmp_df.empty:
                        # Group by UTC date and write each group separately
                        groups = day_lmp_df.groupby(pd.to_datetime(day_lmp_df["interval"]).dt.date)
                        for utc_date, group_df in groups:
                            store.write_interval(pd.DataFrame(columns=["interval", "constraint_name"]), lmp_df=group_df, iso="miso")
                        logger.info(f"  → LMP stored: {day_lmp_df['interval'].nunique()} intervals, {len(day_lmp_df)} rows")
                    else:
                        logger.warning(f"  → LMP: parsed data is empty for {day}")
                else:
                    logger.warning(f"  → LMP: 404 - report not published yet for {day} (will retry later)")
            except Exception as e:
                logger.error(f"  → LMP error fetching {day}: {e}")

        save_checkpoint(day, "miso")
        pct = idx / total_days * 100
        logger.info(f"  Progress: {idx}/{total_days} ({pct:.0f}%)")

    clear_checkpoint("miso")
    logger.info("=" * 60)
    logger.info("MISO Backfill complete.")


async def backfill_spp(
    start: date_type,
    end: date_type,
    store: DuckDBStore,
    delay: float = 2.0,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    all_days = generate_days(start, end)
    logger.info(f"SPP Backfill range: {start} → {end}  ({len(all_days)} days)")

    if dry_run:
        manifest = store.read_manifest(iso="spp")
        logger.info("DRY RUN — manifest status per day:")
        for d in all_days:
            key = d.isoformat()
            bc  = manifest.get(key, {}).get("bc",  {}).get("status", "missing")
            lmp = manifest.get(key, {}).get("lmp", {}).get("status", "missing")
            flag = "✓" if bc == "complete" and lmp == "complete" else "✗"
            logger.info(f"  {flag} {key}  BC={bc:<8} LMP={lmp}")
        to_fetch = [d for d in all_days
                    if store.manifest_status(d.isoformat(), "bc", iso="spp")  != "complete"
                    or store.manifest_status(d.isoformat(), "lmp", iso="spp") != "complete"]
        est = len(to_fetch) * 288 * (delay * 1.1) / 3600
        logger.info(f"  {len(to_fetch)} days would be fetched. Estimated time: {est:.1f} hours")
        return

    checkpoint = load_checkpoint("spp")
    if checkpoint and force:
        clear_checkpoint("spp")

    if force:
        days_to_process = all_days
        logger.info("--force: re-fetching all days regardless of manifest")
    else:
        days_to_process = [
            d for d in all_days
            if store.manifest_status(d.isoformat(), "bc", iso="spp")  != "complete"
            or store.manifest_status(d.isoformat(), "lmp", iso="spp") != "complete"
        ]
        skipped = len(all_days) - len(days_to_process)
        if skipped:
            logger.info(f"Manifest: skipping {skipped} day(s) already complete")

    if not days_to_process:
        logger.info("Nothing to backfill — all days complete per manifest.")
        return

    est = len(days_to_process) * 288 * (delay * 1.1) / 3600
    logger.info(f"{len(days_to_process)} days to fetch. Estimated time: {est:.1f} hours")

    bc_scraper  = RTConstraintsScraper()
    lmp_scraper = RTBMLMPScraper()
    semaphore   = asyncio.Semaphore(1)

    col_rename = {
        "Constraint Name":     "constraint_name",
        "Shadow Price":        "shadow_price",
        "Interval":            "interval",
        "Settlement Location": "settlement_location",
        "MCC":                 "mcc",
    }

    BATCH_SIZE = 12
    total_days = len(days_to_process)
    
    for idx, day in enumerate(days_to_process, 1):
        logger.info(f"[{idx}/{total_days}] Fetching SPP data for {day} ...")

        # ── 1. Daily BC aggregate ──
        if bc_already_stored(store, day, iso="spp") and not force:
            logger.info(f"  BC {day}: already stored, skipping")
        else:
            bc_df = await fetch_daily_bc(bc_scraper, day)
            if not bc_df.empty:
                bc_df = bc_df.rename(columns={
                    k: v for k, v in col_rename.items() if k in bc_df.columns
                })
                bc_df.columns = [c.lower() for c in bc_df.columns]
                if "interval" in bc_df.columns:
                    bc_df["interval"] = pd.to_datetime(bc_df["interval"], errors="coerce")
                store.write_interval(bc_df, iso="spp")
                logger.info(f"  → BC stored: {len(bc_df)} rows")
            else:
                logger.warning(f"  → BC: no data for {day}")

        # ── 2. Per-interval LMP ──
        if lmp_already_stored(store, day, iso="spp") and not force:
            logger.info(f"  LMP {day}: already stored (280+ intervals), skipping")
        else:
            intervals = generate_lmp_intervals(day)
            already_fetched = lmp_stored_intervals(store, day, iso="spp")
            if already_fetched:
                intervals_to_fetch = [iv for iv in intervals if iv not in already_fetched]
                logger.info(f"  LMP {day}: {len(already_fetched)} intervals on disk, "
                            f"fetching {len(intervals_to_fetch)} remaining")
            else:
                intervals_to_fetch = intervals
            ok   = len(already_fetched)
            miss = 0
            empty_bc = pd.DataFrame(columns=["interval", "constraint_name"])

            for batch_start in range(0, len(intervals_to_fetch), BATCH_SIZE):
                batch = intervals_to_fetch[batch_start:batch_start + BATCH_SIZE]
                tasks = [
                    fetch_lmp_interval(lmp_scraper, iv, delay, semaphore)
                    for iv in batch
                ]
                results = await asyncio.gather(*tasks)

                batch_frames = []
                for iv, df in zip(batch, results):
                    if df is not None and not df.empty:
                        df = df.rename(columns={
                            k: v for k, v in col_rename.items() if k in df.columns
                        })
                        df.columns = [c.lower() for c in df.columns]
                        if "interval" not in df.columns:
                            df["interval"] = iv
                        else:
                            df["interval"] = pd.to_datetime(df["interval"], errors="coerce")
                        batch_frames.append(df)
                        ok += 1
                    else:
                        miss += 1

                if batch_frames:
                    combined = pd.concat(batch_frames, ignore_index=True)
                    store.write_interval(empty_bc, lmp_df=combined, iso="spp")

            fetched   = ok
            attempted = ok + miss
            store.mark_day_exhausted(day.isoformat(), "lmp", fetched, attempted, iso="spp")
            if ok > 0:
                gaps = miss - len(already_fetched) if already_fetched else miss
                msg  = f"  → LMP stored: {ok} intervals"
                if gaps > 0:
                    msg += f"  ({gaps} genuine SPP gaps — marked complete)"
                logger.info(msg)
            else:
                logger.warning(f"  → LMP: no data for {day} — all 404s, marked complete")

        save_checkpoint(day, "spp")
        pct = idx / total_days * 100
        remaining_h = (total_days - idx) * 288 * (delay * 1.1) / 3600
        logger.info(f"  Progress: {idx}/{total_days} ({pct:.0f}%)  ETA ~{remaining_h:.1f}h")

    clear_checkpoint("spp")
    logger.info("=" * 60)
    logger.info("SPP Backfill complete.")


async def backfill_spp_da(
    start: date_type,
    end: date_type,
    store: DuckDBStore,
    delay: float = 2.0,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    from src.scrapers.spp_da_constraints_scraper import DAConstraintsScraper

    all_days = generate_days(start, end)
    logger.info(f"SPP Day-Ahead Backfill range: {start} → {end}  ({len(all_days)} days)")

    if dry_run:
        manifest = store.read_manifest(iso="spp")
        logger.info("DRY RUN — manifest status per day:")
        for d in all_days:
            key = d.isoformat()
            status = manifest.get(key, {}).get("da_bc", {}).get("status", "missing")
            flag = "✓" if status == "complete" else "✗"
            logger.info(f"  {flag} {key}  DA_BC={status}")
        return

    if force:
        days_to_process = all_days
    else:
        days_to_process = [
            d for d in all_days
            if store.manifest_status(d.isoformat(), "da_bc", iso="spp") != "complete"
        ]

    if not days_to_process:
        logger.info("Nothing to backfill — all days complete per manifest.")
        return

    da_scraper = DAConstraintsScraper()
    total_days = len(days_to_process)

    for idx, day in enumerate(days_to_process, 1):
        day_str = day.isoformat()
        logger.info(f"[{idx}/{total_days}] Fetching SPP DA Constraints for {day} ...")

        if idx > 1:
            await asyncio.sleep(delay)

        try:
            target_dt = datetime(day.year, day.month, day.day)
            df = await asyncio.to_thread(da_scraper.scrape_day, target_dt)
            if not df.empty:
                df = df.rename(columns={
                    "interval": "interval",
                    "constraint_name": "constraint_name",
                    "shadow_price": "shadow_price",
                })
                if "interval" in df.columns:
                    df["interval"] = pd.to_datetime(df["interval"])
                
                store.write_da_interval(df, iso="spp")
                logger.info(f"  → DA_BC stored: {len(df)} rows")
            else:
                store.mark_day_exhausted(day_str, "da_bc", 0, 0, iso="spp")
                logger.info(f"  → DA_BC: no binding constraints (marked complete)")
        except Exception as e:
            logger.error(f"  → DA_BC error fetching {day}: {e}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill SPP or MISO market history (daily BC + LMP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iso", type=str, default="spp", choices=["spp", "miso"],
        help="ISO footprint to backfill (default: spp).",
    )
    parser.add_argument(
        "--days", type=int, default=21,
        help="Number of days to backfill ending yesterday (default: 21).",
    )
    parser.add_argument(
        "--market", type=str, default="rtbm", choices=["rtbm", "dam"],
        help="Market to backfill: rtbm or dam (default: rtbm).",
    )
    parser.add_argument("--start",  type=str,   default=None, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end",    type=str,   default=None, help="End date YYYY-MM-DD (inclusive, default: yesterday).")
    parser.add_argument("--delay",  type=float, default=None,  help="Base seconds between requests (default: 2.0 for SPP, 1.0 for MISO).")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch all days regardless of manifest (use for revisions/corrections).",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print manifest status for the date range and exit (no downloads).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iso = args.iso.lower()
    now = datetime.now(UTC).date()

    end  = date_type.fromisoformat(args.end)   if args.end   else now - timedelta(days=1)
    start = date_type.fromisoformat(args.start) if args.start else end - timedelta(days=args.days - 1)

    logger.info(f"{iso.upper()} Market Historical Backfill")
    logger.info(f"Range: {start} \u2192 {end}  ({(end - start).days + 1} days)")

    store = DuckDBStore()

    if args.status:
        # Print manifest and exit — no downloads
        manifest = store.read_manifest(iso=iso)
        if args.market == "dam":
            logger.info(f"Manifest status for {iso.upper()} DAY-AHEAD:")
            d = start
            while d <= end:
                key = d.isoformat()
                da_bc = manifest.get(key, {}).get("da_bc", {}).get("status", "missing")
                da_bc_iv = manifest.get(key, {}).get("da_bc", {}).get("intervals", 0)
                flag = "✓" if da_bc == "complete" else "✗"
                logger.info(f"  {flag} {key}  DA_BC={da_bc:<8} ({da_bc_iv:>3} iv)")
                d += timedelta(days=1)
        else:
            logger.info(f"Manifest status for {iso.upper()} REAL-TIME:")
            d = start
            while d <= end:
                key = d.isoformat()
                bc  = manifest.get(key, {}).get("bc",  {}).get("status", "missing")
                lmp = manifest.get(key, {}).get("lmp", {}).get("status", "missing")
                bc_iv  = manifest.get(key, {}).get("bc",  {}).get("intervals", 0)
                lmp_iv = manifest.get(key, {}).get("lmp", {}).get("intervals", 0)
                flag = "✓" if bc == "complete" and lmp == "complete" else "✗"
                logger.info(f"  {flag} {key}  BC={bc:<8} ({bc_iv:>3} iv)  LMP={lmp:<8} ({lmp_iv:>3} iv)")
                d += timedelta(days=1)
        return

    delay = args.delay if args.delay is not None else (1.0 if iso == "miso" else 2.0)

    if args.market == "dam":
        if iso == "miso":
            logger.error("Day-Ahead backfill is only supported for SPP currently.")
            sys.exit(1)
        asyncio.run(backfill_spp_da(
            start=start,
            end=end,
            store=store,
            delay=delay,
            dry_run=args.dry_run,
            force=args.force,
        ))
    else:
        if iso == "miso":
            asyncio.run(backfill_miso(
                start=start,
                end=end,
                store=store,
                delay=delay,
                dry_run=args.dry_run,
                force=args.force,
            ))
        else:
            asyncio.run(backfill_spp(
                start=start,
                end=end,
                store=store,
                delay=delay,
                dry_run=args.dry_run,
                force=args.force,
            ))


if __name__ == "__main__":
    main()
