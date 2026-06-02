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


async def ingest_once(iso: str = "spp") -> bool:
    """
    Run a single scrape → normalize → cache → broadcast cycle for the specified ISO.

    Returns True if data was successfully ingested, False on error.
    """
    from src.analysis.constraint_normalizer import normalize_constraints_df
    iso = iso.lower()

    logger.info(f"[{iso.upper()}] Ingest cycle starting...")

    now = datetime.now()

    if iso == "spp":
        from src.scrapers.spp_rt_constraints_scraper import RTConstraintsScraper
        from src.scrapers.spp_rtlmp_scraper import RTBMLMPScraper

        # ── Step 1: Scrape latest RTBM constraint data (per-interval) ───────
        try:
            bc_scraper = RTConstraintsScraper()
            constraints_df = await _scrape_rt_constraints(bc_scraper, now)
        except Exception as e:
            logger.warning(f"[{iso.upper()}] RT constraint scrape failed: {e}")
            constraints_df = pd.DataFrame()

        # ── Step 2: Scrape latest RTBM LMP (per-interval) ────────────────────
        try:
            lmp_scraper = RTBMLMPScraper()
            lmp_df = await asyncio.to_thread(lmp_scraper.fetch_latest_lmps)
        except Exception as e:
            logger.warning(f"[{iso.upper()}] RT LMP scrape failed: {e}")
            lmp_df = pd.DataFrame()
    elif iso == "miso":
        from src.scrapers.miso_adapter import MISOAdapter
        
        # For MISO, constraints and LMPs are fetched using MISOAdapter
        try:
            miso_adapter = MISOAdapter()
            constraints_df = await asyncio.to_thread(miso_adapter.fetch_latest_constraints)
        except Exception as e:
            logger.warning(f"[{iso.upper()}] RT constraint scrape failed: {e}")
            constraints_df = pd.DataFrame()

        try:
            miso_adapter = MISOAdapter()
            lmp_df = await asyncio.to_thread(miso_adapter.fetch_latest_lmps)
        except Exception as e:
            logger.warning(f"[{iso.upper()}] RT LMP scrape failed: {e}")
            lmp_df = pd.DataFrame()
    else:
        logger.error(f"Unsupported ISO: {iso}")
        return False

    if constraints_df.empty:
        logger.warning(f"[{iso.upper()}] No constraint data available — skipping cycle.")
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
    if "interval" in constraints_df.columns:
        latest_interval_ts = pd.to_datetime(constraints_df["interval"]).max()
        if pd.isnull(latest_interval_ts):
            latest_interval_ts = datetime.now(timezone.utc)
        else:
            if latest_interval_ts.tzinfo is None:
                latest_interval_ts = latest_interval_ts.tz_localize("UTC")
    else:
        latest_interval_ts = datetime.now(timezone.utc)

    # 4a. Get binding constraints for regime key and shadow prices
    binding = constraints_df.copy()
    if "state" in binding.columns:
        binding = binding[binding["state"].astype(str).str.upper() == "BINDING"]
    if "shadow_price" in binding.columns:
        binding = binding[binding["shadow_price"].abs() > 1e-4]

    binding_names = binding["constraint_name"].unique().tolist() if not binding.empty else []
    regime_key = frozenset(binding_names)
    shadow_prices = (
        binding.groupby("constraint_name")["shadow_price"].mean().to_dict()
        if not binding.empty
        else {}
    )

    # 4b. Calculate Loading Volatility Score (rolling 2-hour standard deviation)
    vol_scores = {}
    try:
        from src.storage.duckdb_store import store
        from datetime import timedelta
        two_hours_ago = latest_interval_ts - timedelta(hours=2)
        history_df = store.query_constraints(start=two_hours_ago, end=latest_interval_ts, iso=iso)
        
        # Combine history with current interval
        if not history_df.empty:
            history_df["interval"] = pd.to_datetime(history_df["interval"])
            if history_df["interval"].dt.tz is None:
                history_df["interval"] = history_df["interval"].dt.tz_localize("UTC")
            combined_df = pd.concat([history_df, constraints_df], ignore_index=True)
        else:
            combined_df = constraints_df.copy()

        if not combined_df.empty and "pct_of_limit" in combined_df.columns:
            combined_df["pct_of_limit"] = pd.to_numeric(combined_df["pct_of_limit"], errors="coerce").fillna(0.0)
            vol_scores = combined_df.groupby("constraint_name")["pct_of_limit"].std().fillna(0.0).to_dict()
    except Exception as ex:
        logger.warning(f"[{iso.upper()}] Failed to calculate loading volatility: {ex}")

    # Map volatility scores back to constraints_df so it gets written to DuckDB
    if not constraints_df.empty and "constraint_name" in constraints_df.columns:
        constraints_df["volatility_score"] = constraints_df["constraint_name"].map(vol_scores).fillna(0.0)

    # 4c. Filter constraints to display: BINDING, BREACHED, VIOLATED, or pct_of_limit >= 85
    disp_df = constraints_df.copy()
    if "state" in disp_df.columns:
        disp_df["state_upper"] = disp_df["state"].astype(str).str.upper()
        is_binding_or_breached = disp_df["state_upper"].isin(["BINDING", "BREACHED", "VIOLATED"])
        is_high_pct = disp_df["pct_of_limit"] >= 85.0
        disp_df = disp_df[is_binding_or_breached | is_high_pct].copy()

    # Sort: binding constraints first (or shadow price magnitude descending), then pct_of_limit descending
    if not disp_df.empty:
        disp_df["abs_shadow_price"] = disp_df["shadow_price"].abs()
        disp_df = disp_df.sort_values(by=["abs_shadow_price", "pct_of_limit"], ascending=[False, False])
        
        disp_records = []
        for _, row in disp_df.iterrows():
            name = row["constraint_name"]
            vol = float(vol_scores.get(name, 0.0))
            disp_records.append({
                "interval": row["interval"],
                "constraint_name": name,
                "shadow_price": float(row.get("shadow_price", 0.0)),
                "limit_mw": float(row.get("limit_mw")) if pd.notnull(row.get("limit_mw")) else None,
                "monitored_mw": float(row.get("monitored_mw")) if pd.notnull(row.get("monitored_mw")) else None,
                "pct_of_limit": float(row.get("pct_of_limit")) if pd.notnull(row.get("pct_of_limit")) else None,
                "state": row.get("state"),
                "contingency_name": row.get("contingency_name") if pd.notnull(row.get("contingency_name")) else None,
                "volatility_score": round(vol, 2),
            })
    else:
        disp_records = []

    interval_snapshot = CachedInterval(
        interval=latest_interval_ts,
        constraints=disp_records,
        shadow_prices=shadow_prices,
        lmp_df=lmp_df,
        regime_key=regime_key,
    )

    # ── Step 5: Update SF cache (re-solves if regime changed) ───────────
    iso_cache = cache.for_iso(iso)
    re_solved = iso_cache.update(
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
            iso=iso,
        )
    except Exception as e:
        logger.warning(f"[{iso.upper()}] DuckDB write failed (non-fatal): {e}")

    # ── Step 7: Broadcast to WebSocket clients ──────────────────────────
    message = {
        "type": "constraint_update",
        "iso": iso,
        "interval": str(latest_interval_ts),
        "n_binding": len(shadow_prices),
        "regime_changed": re_solved,
        "constraints": interval_snapshot.constraints[:50],  # cap payload size
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    await broadcast(message)

    logger.info(
        f"[{iso.upper()}] Ingest cycle complete. "
        f"Interval={latest_interval_ts}, "
        f"Binding={len(shadow_prices)}, "
        f"Re-solved={re_solved}"
    )
    return True


async def ingest_loop() -> None:
    """
    Main ingest loop — runs forever in the background.
    Polls active ISOs concurrently.
    """
    logger.info(
        f"Ingest worker started. "
        f"Poll interval: {SPP_POLL_INTERVAL_SECONDS}s"
    )
    while True:
        try:
            # Concurrently run ingest_once for both spp and miso
            await asyncio.gather(
                ingest_once("spp"),
                ingest_once("miso"),
                return_exceptions=True
            )
        except asyncio.CancelledError:
            logger.info("Ingest worker cancelled — shutting down.")
            break
        except Exception as e:
            logger.error(f"Ingest loop error (will retry): {e}", exc_info=True)

        await asyncio.sleep(SPP_POLL_INTERVAL_SECONDS)
