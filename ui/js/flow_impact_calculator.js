/**
 * flow_impact_calculator.js — What-If bilateral injection form.
 *
 * POSTs to /api/v1/flow-impact and renders a rich result card.
 */

function initFlowImpactCalculator() {
  const mwSlider = document.getElementById('wi-mw');
  const mwLabel  = document.getElementById('wi-mw-label');
  const form     = document.getElementById('whatif-form');

  mwSlider.addEventListener('input', () => {
    mwLabel.textContent = mwSlider.value + ' MW';
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('wi-submit');
    btn.textContent = 'Calculating…';
    btn.disabled = true;

    const body = {
      source_node: document.getElementById('wi-source').value.trim(),
      sink_node:   document.getElementById('wi-sink').value.trim(),
      mw:          parseFloat(mwSlider.value),
      constraint:  document.getElementById('wi-constraint').value.trim(),
      iso:         window.currentISO || 'spp',
    };

    const result = document.getElementById('wi-result');

    try {
      const data = await api.post('/api/v1/flow-impact', body);
      const dir  = data.direction || 'neutral';
      const icons = { exacerbates: '▲ Exacerbates', relieves: '▼ Relieves', neutral: '— Neutral' };

      result.innerHTML = `
        <div class="wi-result-card fade-in">
          <div class="wi-result-direction ${dir}">${icons[dir]}</div>
          <p class="wi-result-message">${escHtml(data.message)}</p>
          <div class="wi-result-metrics">
            <div class="wi-metric">
              <div class="wi-metric-label">ΔFlow on Constraint</div>
              <div class="wi-metric-value">${data.delta_flow_mw != null ? data.delta_flow_mw.toFixed(2) + ' MW' : '—'}</div>
            </div>
            <div class="wi-metric">
              <div class="wi-metric-label">SF Source</div>
              <div class="wi-metric-value">${data.sf_source != null ? data.sf_source.toFixed(4) : '—'}</div>
            </div>
            <div class="wi-metric">
              <div class="wi-metric-label">SF Sink</div>
              <div class="wi-metric-value">${data.sf_sink != null ? data.sf_sink.toFixed(4) : '—'}</div>
            </div>
            <div class="wi-metric">
              <div class="wi-metric-label">MW Injected</div>
              <div class="wi-metric-value">${body.mw} MW</div>
            </div>
          </div>
        </div>`;
    } catch (err) {
      result.innerHTML = `<div class="wi-result-card fade-in"><p style="color:#ef4444">Error: ${escHtml(err.message)}</p></div>`;
    } finally {
      btn.textContent = 'Calculate Impact';
      btn.disabled = false;
    }
  });
}
