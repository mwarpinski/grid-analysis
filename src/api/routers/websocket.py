"""
WebSocket endpoints — push-based live data delivery.

/ws/constraints  — pushes new constraint snapshot on every ingest cycle
/ws/portfolio    — pushes portfolio P&L recalculation on every cycle

Clients connect once and receive JSON push messages without polling.
The ingest_worker.py broadcasts to all registered connections.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.ingest_worker import deregister_ws, register_ws

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/constraints")
async def ws_constraints(websocket: WebSocket) -> None:
    """
    **Live constraint stream** — WebSocket push endpoint.

    Connect and receive a JSON message every time the ingest worker
    processes a new RTBM interval (~every 5 minutes).

    Message schema::

        {
          "type": "constraint_update",
          "interval": "2024-11-01T14:35:00+00:00",
          "n_binding": 3,
          "regime_changed": false,
          "constraints": [
            {
              "constraint_name": "LN HARBINE - BEATRICE",
              "shadow_price": 450.0,
              "limit_mw": 400.0,
              "monitored_mw": 392.0,
              "pct_of_limit": 98.0,
              "state": "BINDING"
            },
            ...
          ],
          "as_of": "2024-11-01T14:35:18Z"
        }

    Also sends a ``{"type": "connected", "message": "..."}`` handshake on
    connection and ``{"type": "ping"}`` keepalives every 60 seconds.
    """
    await websocket.accept()
    register_ws(websocket)
    logger.info(f"WebSocket client connected: {websocket.client}")

    try:
        # Send immediate state snapshot on connect
        from src.api.sf_cache import cache
        if cache.is_ready() and cache.latest_interval is not None:
            interval = cache.latest_interval
            await websocket.send_text(json.dumps({
                "type": "initial_snapshot",
                "interval": str(interval.interval),
                "n_binding": len(interval.shadow_prices),
                "constraints": interval.constraints[:50],
                "as_of": str(cache.last_updated),
            }, default=str))
        else:
            await websocket.send_text(json.dumps({
                "type": "connected",
                "message": "Ingest worker initializing — first update incoming shortly.",
            }))

        # Keep connection alive until client disconnects
        while True:
            # Wait for incoming messages (ping/pong, or client-side queries)
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        deregister_ws(websocket)


@router.websocket("/ws/portfolio")
async def ws_portfolio(websocket: WebSocket) -> None:
    """
    **Live portfolio P&L stream** — WebSocket push endpoint.

    Receives position data from the client on connect, then pushes
    recalculated portfolio P&L risk on every new ingest cycle.

    Send your positions as JSON on connect::

        {
          "positions": [
            {"asset_name": "WindFarm_01", "asset_type": "gen",
             "node": "WEST_NODE", "mw_signed": 120},
            ...
          ]
        }

    Receive risk updates::

        {
          "type": "portfolio_update",
          "interval": "...",
          "total_net_mw_flow": 13.5,
          "total_pnl_risk_rate": -13500.0,
          "alert": "Portfolio contributes 13.5 MW to LN HARBINE. At $1,000 → -$13,500/hr."
        }
    """
    await websocket.accept()
    register_ws(websocket)
    logger.info(f"Portfolio WS client connected: {websocket.client}")

    positions = []

    try:
        # Receive initial positions from client
        raw = await websocket.receive_text()
        try:
            msg = json.loads(raw)
            positions = msg.get("positions", [])
            await websocket.send_text(json.dumps({
                "type": "positions_received",
                "n_positions": len(positions),
            }))
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Invalid JSON — send positions as {positions: [...]}"
            }))

        # Hold connection open
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                # Allow client to update positions mid-session
                if "positions" in msg:
                    positions = msg["positions"]
                    await websocket.send_text(json.dumps({
                        "type": "positions_updated",
                        "n_positions": len(positions),
                    }))
                elif msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        logger.info(f"Portfolio WS client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"Portfolio WS error: {e}")
    finally:
        deregister_ws(websocket)
