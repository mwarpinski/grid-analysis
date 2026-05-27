/**
 * app.js — Main application bootstrap.
 *
 * Initializes all modules, wires the WebSocket feed,
 * and manages panel navigation.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ── Initialize modules ─────────────────────────────────────────────
  initMap();
  initConstraintsTable();
  initFlowImpactCalculator();
  initPortfolioPanel();
  initReplayPlayer();

  // ── Panel navigation ───────────────────────────────────────────────
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.panel;
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.app-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('panel-' + target).classList.add('active');
      // Resize map when switching back to live panel
      if (target === 'live' && gridMap) {
        setTimeout(() => gridMap.invalidateSize(), 50);
      }
      // Load forecast models when switching to forecast panel
      if (target === 'forecast' && typeof window.forecastPanelActivate === 'function') {
        window.forecastPanelActivate();
      }
    });
  });

  // ── Constraint select → SF panel + zone heatmap ─────────────────────
  window.onConstraintSelect = (name) => {
    // Switch to live panel if not already there
    document.getElementById('panel-live').classList.add('active');
    loadSFPanel(name);       // bar chart in SF panel
    showSFHeatmap(name);     // zone heatmap overlay on map
  };

  // ── WebSocket ─────────────────────────────────────────────────────
  openConstraintsWS(
    (msg) => {
    if (msg.type === 'constraint_update' || msg.type === 'initial_snapshot') {
        updateConstraintsTable(msg);
        if (msg.constraints) updateMapColors(msg.constraints);
        // Update interval label in header
        if (msg.interval) {
          const ts = new Date(msg.interval);
          document.getElementById('last-interval').textContent =
            'Last: ' + ts.toLocaleTimeString();
        }
      }
    },
    (status) => {
      const dot   = document.getElementById('status-indicator');
      const label = document.getElementById('status-label');
      dot.className = 'status-dot status-' + status;
      label.textContent = { connecting: 'Connecting…', live: 'Live', error: 'Reconnecting…' }[status] || status;
    }
  );

  // ── Initial REST poll (fallback if WS hasn't fired yet) ────────────
  setTimeout(async () => {
    try {
      const data = await api.get('/api/v1/constraints/live');
      if (data.constraints && data.constraints.length > 0) {
        updateConstraintsTable({ constraints: data.constraints, interval: data.interval });
      }
    } catch (_) {}
  }, 1500);

});
