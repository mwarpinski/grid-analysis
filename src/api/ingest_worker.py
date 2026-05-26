"""
Ingest Worker — background asyncio task.

Polls SPP RTBM every ~290 seconds (just under the 5-minute dispatch interval),
normalizes constraints, updates the SF cache, writes to DuckDB, and broadcasts
new state to all connected WebSocket clients.

Lifecycle
---------
The worker is started as an asyncio background Task in the FastAPI lifespan
context manager (server.py). It runs for the lifetime of the server process.

SPP RTBM timing
---------------
SPP publishes new RTBM results roughly every 5 minutes.  We poll at 290s
(not 300s) to ensure we always catch the new file before the next cycle,
accounting for SPP's publication latency (typically 30–90s after interval end).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Set

import pandas as pd

from src.api.sf_cache import CachedInterval, cache

logger = logging.getLogger(__name__)

# Seconds between SPP portal polls
SPP_POLL_INTERVAL_SECONDS = 290

# WebSocket connection registry — router/websocket.py populates this
_ws_connections: Set = set()


def register_ws(ws) -> None:
    """Register a WebSocket connection for broadcast."""
    _ws_connections.add(ws)


def deregister_ws(ws) -> None:
    """Remove a closed WebSocket connection."""
    _ws_connections.discard(ws)


async def broadcast(message: dict) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    import json
    if not _ws_connections:
        return
    payload = json.dumps(message, default=str)
    dead = set()
    for ws in list(_ws_connections):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _ws_connections.discard(ws)


async def _scrape_rt_constraints(bc_scraper, now: datetime) -> "pd.DataFrame":
    """
    Fetch the most recent RTBM binding constraint data.

    Strategy: try the latest per-interval file, then walk back up to 6
    intervals (30 min) to account for SPP publication lag (~30-90 s).
    The daily rollup file only exists after midnight, so we always use
    the per-interval endpoint during the live trading day.
    """
    from datetime import timedelta

    rounded = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)

    for offset in range(7):   # 0 … 30 minutes back
        target = rounded - timedelta(minutes=offset * 5)
        df = await asyncio.to_thread(bc_scraper.fetch_data, target, "interval")
        if not df.empty:
            logger.info(
                f"RT constraints fetched for interval {target}"
                + (f" (walked back {offset*5} min)" if offset else "")
            )
            return df

    logger.warning("Per-interval RT constraint files unavailable for the past 30 min.")
    return pd.DataFrame()


async def ingest_once() -> bool:
    """
    Run a single scrape → normalize → cache → broadcast cycle.

    Returns True if data was successfully ingested, False on error.
    """
    from src.scrapers.spp_rt_constraints_scraper import RTConstraintsScraper
    from src.scrapers.spp_rtlmp_scraper import RTBMLMPScraper
    from src.analysis.constraint_normalizer import normalize_constraints_df

    logger.info("Ingest cycle starting...")

    now = datetime.now()

    # ── Step 1: Scrape latest RTBM constraint data (per-interval) ───────
    try:
        bc_scraper = RTConstraintsScraper()
        constraints_df = await _scrape_rt_constraints(bc_scraper, now)
    except Exception as e:
        logger.warning(f"RT constraint scrape failed: {e}")
        constraints_df = pd.DataFrame()

    # ── Step 2: Scrape latest RTBM LMP (per-interval) ────────────────────
    try:
        lmp_scraper = RTBMLMPScraper()
        lmp_df = await asyncio.to_thread(lmp_scraper.fetch_latest_lmps)
    except Exception as e:
        logger.warning(f"RT LMP scrape failed: {e}")
        lmp_df = pd.DataFrame()

    if constraints_df.empty:
        logger.warning("No constraint data available — skipping cycle.")
        return False

    # ── Step 2: Normalize column names ──────────────────────────────────
    col_rename = {
        "Constraint Name": "constraint_name",
        "Shadow Price": "shadow_price",
        "Interval": "interval",
        "Settlement Location": "settlement_location",
        "MCC": "mcc",
    }
    constraints_df = constraints_df.rename(
        columns={k: v for k, v in col_rename.items() if k in constraints_df.columns}
    )
    if not lmp_df.empty:
        lmp_df = lmp_df.rename(
            columns={k: v for k, v in col_rename.items() if k in lmp_df.columns}
        )

    # ── Step 3: Constraint name normalization ────────────────────────────
    if "constraint_name" in constraints_df.columns:
        constraints_df = normalize_constraints_df(constraints_df, name_col="constraint_name")

    # ── Step 4: Build interval snapshot ─────────────────────────────────
    # Get the most recent interval
    binding = constraints_df.copy()
    if "state" in binding.columns:
        binding = binding[binding["state"].str.upper() == "BINDING"]
    if "shadow_price" in binding.columns:
        binding = binding[binding["shadow_price"].abs() > 1e-4]

    if binding.empty:
        logger.info("No binding constraints in current snapshot.")
        regime_key = frozenset()
        shadow_prices = {}
        latest_interval_ts = datetime.now(timezone.utc)
    else:
        latest_interval_ts = (
            pd.to_datetime(binding["interval"]).max()
            if "interval" in binding.columns
            else datetime.now(timezone.utc)
        )
        binding_names = binding["constraint_name"].unique().tolist()
        regime_key = frozenset(binding_names)
        shadow_prices = (
            binding.groupby("constraint_name")["shadow_price"].mean().to_dict()
        )

    interval_snapshot = CachedInterval(
        interval=latest_interval_ts,
        constraints=binding.to_dict("records"),
        shadow_prices=shadow_prices,
        lmp_df=lmp_df,
        regime_key=regime_key,
    )

    # ── Step 5: Update SF cache (re-solves if regime changed) ───────────
    re_solved = cache.update(
        interval=interval_snapshot,
        constraints_df=constraints_df,
        lmp_df=lmp_df if not lmp_df.empty else pd.DataFrame(),
    )

    # ── Step 6: Write to DuckDB (non-blocking) ──────────────────────────
    try:
        from src.storage.duckdb_store import store
        await asyncio.to_thread(
            store.write_interval,
            constraints_df=constraints_df,
            lmp_df=lmp_df,
        )
    except Exception as e:
        # DuckDB store is optional — don't kill the ingest loop on failure
        logger.warning(f"DuckDB write failed (non-fatal): {e}")

    # ── Step 7: Broadcast to WebSocket clients ──────────────────────────
    message = {
        "type": "constraint_update",
        "interval": str(latest_interval_ts),
        "n_binding": len(shadow_prices),
        "regime_changed": re_solved,
        "constraints": interval_snapshot.constraints[:50],  # cap payload size
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    await broadcast(message)

    logger.info(
        f"Ingest cycle complete. "
        f"Interval={latest_interval_ts}, "
        f"Binding={len(shadow_prices)}, "
        f"Re-solved={re_solved}"
    )
    return True


async def ingest_loop() -> None:
    """
    Main ingest loop — runs forever in the background.
    Polls SPP every SPP_POLL_INTERVAL_SECONDS seconds.
    """
    logger.info(
        f"Ingest worker started. "
        f"Poll interval: {SPP_POLL_INTERVAL_SECONDS}s"
    )
    while True:
        try:
            await ingest_once()
        except asyncio.CancelledError:
            logger.info("Ingest worker cancelled — shutting down.")
            break
        except Exception as e:
            logger.error(f"Ingest loop error (will retry): {e}", exc_info=True)

        await asyncio.sleep(SPP_POLL_INTERVAL_SECONDS)
