/**
 * forecast_panel.js — Phase 13 Shadow Price Forecast Panel
 *
 * Renders:
 *  - Left col:  list of constraints with mini current/next price indicators
 *  - Center:    Chart.js line chart — historical (last 12 intervals from cache)
 *               + forecast horizon with CI band
 *  - Right col: Model metadata (type, MAE, R², training rows)
 *
 * Auto-refreshes every 5 minutes to pick up new ingest data.
 */

/* global Chart, API_BASE */

(function () {
  'use strict';

  const API = window.API_BASE || '';

  // ── State ──────────────────────────────────────────────────────────────
  let _models        = [];          // registry from /api/v1/forecast/models
  let _activeConstraint = null;
  let _chart         = null;        // Chart.js instance
  let _refreshTimer  = null;
  const HORIZON_STEPS = 12;         // 1 hour look-ahead

  // ── DOM refs ────────────────────────────────────────────────────────────
  const elList         = () => document.getElementById('forecast-constraint-list');
  const elCount        = () => document.getElementById('forecast-model-count');
  const elLabel        = () => document.getElementById('forecast-constraint-label');
  const elChartWrap    = () => document.getElementById('forecast-chart-wrap');
  const elChartPlaceholder = () => document.getElementById('forecast-chart-placeholder');
  const elHorizonPills = () => document.getElementById('forecast-horizon-pills');
  const elMetaContent  = () => document.getElementById('forecast-meta-content');
  const elComparison   = () => document.getElementById('forecast-comparison');
  const elRetrainBtn   = () => document.getElementById('forecast-retrain-btn');
  const elRetrainStatus = () => document.getElementById('forecast-retrain-status');

  // ── Helpers ─────────────────────────────────────────────────────────────
  function fmt(v, digits = 2) {
    if (v == null || isNaN(v)) return '—';
    return (v >= 0 ? '+' : '') + Number(v).toFixed(digits);
  }

  function fmtSP(v) {
    if (v == null || isNaN(v)) return '—';
    return '$' + Number(v).toFixed(2) + '/MWh';
  }

  function directionArrow(cur, nxt) {
    if (cur == null || nxt == null) return '—';
    const pct = cur !== 0 ? (nxt - cur) / Math.abs(cur) * 100 : 0;
    if (pct > 5)  return '↑ Increasing';
    if (pct < -5) return '↓ Decreasing';
    return '→ Stable';
  }

  // ── Load model registry ─────────────────────────────────────────────────
  async function loadModels() {
    try {
      const resp = await fetch(`${API}/api/v1/forecast/models`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      _models = data.models || [];
      renderConstraintList();
      const n = _models.length;
      elCount().textContent = `${n} model${n !== 1 ? 's' : ''}`;
    } catch (e) {
      elList().innerHTML = `<div class="forecast-error">Could not load models: ${e.message}</div>`;
    }
  }

  // ── Render constraint list ──────────────────────────────────────────────
  function renderConstraintList() {
    const el = elList();
    if (!_models.length) {
      el.innerHTML = `
        <div class="sf-placeholder">
          <div class="sf-placeholder-icon">🔮</div>
          <p>No models trained yet.<br/>Run the backfill script to populate data.</p>
        </div>`;
      return;
    }

    // Sort: binding first (non-zero predicted), then by MAE ascending
    const sorted = [..._models].sort((a, b) => {
      if (a.fallback !== b.fallback) return a.fallback ? 1 : -1;
      return (a.mae_cv || 999) - (b.mae_cv || 999);
    });

    el.innerHTML = sorted.map(m => {
      const active = m.constraint === _activeConstraint ? 'active' : '';
      let badge = '';
      if (m.fallback) {
        badge = '<span class="badge-pill badge-gray">persistence</span>';
      } else if (m.r2_cv < 0.0) {
        badge = '<span class="badge-pill badge-warning">⚠ Low R²</span>';
      } else {
        badge = '<span class="badge-pill badge-teal">ML</span>';
      }
      const mae = m.mae_cv != null ? `MAE ${m.mae_cv.toFixed(1)}` : '';
      return `
        <div class="forecast-item ${active}" data-constraint="${m.constraint}" id="fc-item-${m.constraint.replace(/[^a-z0-9]/gi,'_')}">
          <div class="forecast-item-name">${m.constraint}</div>
          <div class="forecast-item-meta">${badge} ${mae}</div>
        </div>`;
    }).join('');

    el.querySelectorAll('.forecast-item').forEach(row => {
      row.addEventListener('click', () => {
        const c = row.dataset.constraint;
        selectConstraint(c);
      });
    });
  }

  // ── Select a constraint & load forecast ─────────────────────────────────
  async function selectConstraint(constraint) {
    _activeConstraint = constraint;

    // Update active styles
    document.querySelectorAll('.forecast-item').forEach(r =>
      r.classList.toggle('active', r.dataset.constraint === constraint)
    );

    elLabel().textContent = constraint;
    elChartPlaceholder().classList.add('hidden');
    elChartWrap().classList.remove('hidden');
    elHorizonPills().classList.remove('hidden');
    elComparison().classList.remove('hidden');

    try {
      const resp = await fetch(
        `${API}/api/v1/forecast/${encodeURIComponent(constraint)}?horizon=${HORIZON_STEPS}`
      );
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      renderChart(constraint, data);
      renderMeta(data);
      renderHorizonPills(data.horizon);
    } catch (e) {
      elChartWrap().innerHTML = `<div class="forecast-error">Failed to load forecast: ${e.message}</div>`;
    }
  }

  // ── Chart.js render ─────────────────────────────────────────────────────
  function renderChart(constraint, data) {
    const canvas = document.getElementById('forecast-chart');
    const ctx    = canvas.getContext('2d');

    if (_chart) {
      _chart.destroy();
      _chart = null;
    }

    const horizon = data.horizon || [];
    const curSP   = data.current_shadow_price || 0;
    const nextSP  = data.next_price || 0;

    // Labels: "Now" + each step interval (time portion only)
    const labels = ['Now', ...horizon.map(h => {
      const t = new Date(h.interval);
      return `+${h.step * 5}m`;
    })];

    // Forecast line
    const forecastData = [curSP, ...horizon.map(h => h.shadow_price)];
    const ciHigh       = [curSP, ...horizon.map(h => h.confidence_high)];
    const ciLow        = [curSP, ...horizon.map(h => h.confidence_low)];

    _chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Forecast',
            data: forecastData,
            borderColor: '#38bdf8',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            pointRadius: 4,
            pointBackgroundColor: '#38bdf8',
            tension: 0.35,
            order: 1,
          },
          {
            label: 'CI High',
            data: ciHigh,
            borderColor: 'rgba(56,189,248,0.2)',
            backgroundColor: 'rgba(56,189,248,0.08)',
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0.35,
            fill: '+1',
            order: 2,
          },
          {
            label: 'CI Low',
            data: ciLow,
            borderColor: 'rgba(56,189,248,0.2)',
            backgroundColor: 'rgba(56,189,248,0.08)',
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0.35,
            fill: false,
            order: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            backgroundColor: '#0f172a',
            borderColor: '#334155',
            borderWidth: 1,
            titleColor: '#94a3b8',
            bodyColor: '#e2e8f0',
            callbacks: {
              label: (ctx) => {
                if (ctx.datasetIndex === 0) {
                  return ` ${fmtSP(ctx.parsed.y)}`;
                }
                return null;
              },
              afterBody: (items) => {
                const i = items[0].dataIndex;
                if (i === 0) return '';
                const h = horizon[i - 1];
                if (!h) return '';
                return [`CI: [${fmtSP(h.confidence_low)}, ${fmtSP(h.confidence_high)}]`];
              },
            },
          },
        },
        scales: {
          x: {
            grid:  { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#64748b', font: { size: 11 } },
          },
          y: {
            grid:  { color: 'rgba(255,255,255,0.05)' },
            ticks: {
              color: '#64748b',
              font:  { size: 11 },
              callback: v => `$${v.toFixed(0)}`,
            },
            title: {
              display: true,
              text: 'Shadow Price ($/MWh)',
              color: '#475569',
              font: { size: 11 },
            },
          },
        },
      },
    });
  }

  // ── Horizon pills ───────────────────────────────────────────────────────
  function renderHorizonPills(horizon) {
    const el = elHorizonPills();
    if (!horizon || !horizon.length) {
      el.innerHTML = '';
      return;
    }
    // Show every other step to avoid crowding
    el.innerHTML = horizon
      .filter(h => h.step % 2 === 0 || h.step === 1)
      .map(h => {
        const cls = Math.abs(h.shadow_price) > 0.5 ? 'pill-binding' : 'pill-clear';
        return `<div class="horizon-pill ${cls}">
          <span class="pill-time">+${h.step * 5}m</span>
          <span class="pill-price">${fmtSP(h.shadow_price)}</span>
        </div>`;
      }).join('');
  }

  // ── Model metadata panel ────────────────────────────────────────────────
  function renderMeta(data) {
    const meta = data.model_meta || {};

    // Comparison values
    const curSP  = data.current_shadow_price;
    const nxtSP  = data.next_price;

    document.getElementById('fc-current-sp').textContent = fmtSP(curSP);
    document.getElementById('fc-next-sp').textContent    = fmtSP(nxtSP);

    const dirEl = document.getElementById('fc-direction');
    const dir = directionArrow(curSP, nxtSP);
    dirEl.textContent = dir;
    dirEl.style.color = dir.startsWith('↑') ? '#ef4444'
                      : dir.startsWith('↓') ? '#22c55e'
                      : '#94a3b8';

    if (meta.fallback) {
      document.getElementById('fc-model-type').textContent = '⚠ Persistence (insufficient data)';
      document.getElementById('fc-model-type').style.color = '#f59e0b';
    } else if (meta.r2_cv < 0.0) {
      document.getElementById('fc-model-type').textContent = '⚠ HistGradientBoosting (Low Confidence - Negative R²)';
      document.getElementById('fc-model-type').style.color = '#f97316';
    } else {
      document.getElementById('fc-model-type').textContent = '✓ HistGradientBoosting';
      document.getElementById('fc-model-type').style.color = '#22c55e';
    }

    document.getElementById('fc-mae').textContent  = meta.mae_cv != null ? `$${meta.mae_cv.toFixed(2)}/MWh` : '—';
    document.getElementById('fc-r2').textContent   = meta.r2_cv  != null ? meta.r2_cv.toFixed(3)            : '—';
    document.getElementById('fc-n-rows').textContent = meta.n_training_rows != null ? meta.n_training_rows : '—';

    elComparison().classList.remove('hidden');
  }

  // ── Retrain ─────────────────────────────────────────────────────────────
  async function triggerRetrain() {
    const btn = elRetrainBtn();
    const st  = elRetrainStatus();
    btn.disabled = true;
    st.textContent = 'Retraining…';
    st.style.color = '#f59e0b';

    try {
      const resp = await fetch(`${API}/api/v1/forecast/retrain`, { method: 'POST' });
      const data = await resp.json();
      st.textContent = '✓ Started in background';
      st.style.color = '#22c55e';
      // Reload models after 5s to reflect updated stats
      setTimeout(() => loadModels(), 5000);
    } catch (e) {
      st.textContent = '✗ Failed: ' + e.message;
      st.style.color = '#ef4444';
    } finally {
      btn.disabled = false;
    }
  }

  // ── Auto-refresh ─────────────────────────────────────────────────────────
  function startAutoRefresh() {
    if (_refreshTimer) clearInterval(_refreshTimer);
    _refreshTimer = setInterval(() => {
      loadModels();
      if (_activeConstraint) selectConstraint(_activeConstraint);
    }, 5 * 60 * 1000); // every 5 minutes
  }

  // ── Init ─────────────────────────────────────────────────────────────────
  function init() {
    elRetrainBtn().addEventListener('click', triggerRetrain);
    loadModels();
    startAutoRefresh();
  }

  // Wait for DOM + panel activation
  document.addEventListener('DOMContentLoaded', init);

  // Expose for app.js panel switching
  window.forecastPanelActivate = () => {
    loadModels();
    if (_activeConstraint) selectConstraint(_activeConstraint);
  };

})();
