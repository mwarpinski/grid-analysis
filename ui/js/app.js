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
  initDARTPanel();

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
      // Load DART spreads when switching to DART panel
      if (target === 'dart' && typeof window.dartPanelActivate === 'function') {
        window.dartPanelActivate();
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

  // ── ISO Selector ───────────────────────────────────────────────────
  window.currentISO = 'spp';
  document.querySelectorAll('.iso-selector-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const iso = btn.dataset.iso;
      if (iso === window.currentISO) return;
      window.currentISO = iso;
      document.querySelectorAll('.iso-selector-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Clear current selected constraint and related UI elements
      document.querySelectorAll('#constraint-tbody tr').forEach(r => r.classList.remove('selected'));
      // Reset SF panel and map heatmap
      const sfContent = document.getElementById('sf-content');
      if (sfContent) {
        sfContent.innerHTML = `
          <div class="sf-placeholder">
            <div class="sf-placeholder-icon">⚡</div>
            <p>Click a constraint in the table<br/>or a line on the map to view<br/>nodal SF exposure.</p>
          </div>`;
      }
      const chartWrap = document.getElementById('sf-chart-wrap');
      if (chartWrap) chartWrap.classList.add('hidden');
      if (typeof clearHeatmap === 'function') clearHeatmap();

      // Trigger immediate REST fetch for constraints
      try {
        const data = await api.get(`/api/v1/constraints/live?iso=${window.currentISO}`);
        updateConstraintsTable({ constraints: data.constraints, interval: data.interval });
      } catch (err) {
        console.error('Failed to switch ISO data:', err);
      }

      // Trigger DART update if panel is active
      const dartPanel = document.getElementById('panel-dart');
      if (dartPanel && dartPanel.classList.contains('active') && typeof window.dartPanelActivate === 'function') {
        window.dartPanelActivate();
      }
    });
  });

  // ── WebSocket ─────────────────────────────────────────────────────
  openConstraintsWS(
    (msg) => {
      // Discard messages from different ISO to prevent data leakage
      if (msg.iso && msg.iso !== window.currentISO) {
        return;
      }
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
      const data = await api.get('/api/v1/constraints/live?iso=' + window.currentISO);
      if (data.constraints && data.constraints.length > 0) {
        updateConstraintsTable({ constraints: data.constraints, interval: data.interval });
      }
    } catch (_) {}
  }, 1500);

});
