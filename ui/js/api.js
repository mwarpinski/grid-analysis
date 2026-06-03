/**
 * api.js — Thin wrapper around the FastAPI backend.
 * All fetch calls go through here so base URL is set in one place.
 */
const API_BASE = '';  // same origin — FastAPI serves both API and UI

const api = {
  async get(path) {
    const r = await fetch(API_BASE + path);
    if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(API_BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || r.statusText);
    }
    return r.json();
  },
  async del(path) {
    const r = await fetch(API_BASE + path, { method: 'DELETE' });
    if (!r.ok) throw new Error(`DELETE ${path} → ${r.status}`);
    return r.json();
  },
};

/** Open the constraints WebSocket and call onMessage on each push. */
function openConstraintsWS(onMessage, onStatusChange) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/ws/constraints`;
  let ws, reconnectTimer;

  function connect() {
    onStatusChange('connecting');
    ws = new WebSocket(url);

    ws.onopen  = () => onStatusChange('live');
    ws.onclose = () => { onStatusChange('error'); reconnectTimer = setTimeout(connect, 5000); };
    ws.onerror = () => ws.close();
    ws.onmessage = (e) => {
      try { onMessage(JSON.parse(e.data)); } catch (_) {}
    };

    // Send keepalive ping every 30s
    setInterval(() => { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ping' })); }, 30000);
  }

  connect();
  return () => { clearTimeout(reconnectTimer); ws && ws.close(); };
}
