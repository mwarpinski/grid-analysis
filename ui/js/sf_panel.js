/**
 * sf_panel.js — Shift factor exposure panel.
 *
 * Calls GET /api/v1/shift-factors/table?constraint=X and renders
 * a horizontal bar chart (positive = red/exacerbates, negative = green/relieves).
 */

let _sfChart = null;

async function loadSFPanel(constraintName) {
  const label = document.getElementById('sf-constraint-label');
  const content = document.getElementById('sf-content');
  const chartWrap = document.getElementById('sf-chart-wrap');

  label.textContent = truncate(constraintName, 28);

  try {
    let data;
    try {
      data = await api.get(
        `/api/v1/shift-factors/table?constraint=${encodeURIComponent(constraintName)}&limit=20&iso=${window.currentISO || 'spp'}`
      );
    } catch (e) {
      // 404 = constraint hasn't bound in the solve window yet — normal, not an error
      const is404 = e.message?.includes('404') || e.status === 404;
      if (is404) {
        content.innerHTML = `
          <div class="sf-placeholder">
            <p style="color:#94a3b8">⏳ No shift factors solved yet for</p>
            <p style="color:#cbd5e1;font-size:11px;word-break:break-all;margin-top:4px">${constraintName}</p>
            <p style="color:#64748b;font-size:11px;margin-top:6px">This constraint hasn't appeared as binding<br>in the current solve window.</p>
          </div>`;
        chartWrap.classList.add('hidden');
        return;
      }
      throw e;  // re-throw real errors
    }

    if (!data.records || data.records.length === 0) {
      content.innerHTML = '<div class="sf-placeholder"><p>No shift factors solved for this constraint yet.</p></div>';
      chartWrap.classList.add('hidden');
      return;
    }

    content.innerHTML = '';
    chartWrap.classList.remove('hidden');

    const sorted = [...data.records].sort((a, b) => Math.abs(b.sf_exposure || b.shift_factor) - Math.abs(a.sf_exposure || a.shift_factor));
    const top = sorted.slice(0, 20);

    const labels = top.map(r => truncate(r.node, 18));
    const values = top.map(r => r.sf_exposure != null ? r.sf_exposure : r.shift_factor);

    const colors = values.map(v => v > 0
      ? 'rgba(239,68,68,0.75)'
      : 'rgba(34,197,94,0.75)'
    );

    if (_sfChart) _sfChart.destroy();

    const ctx = document.getElementById('sf-chart').getContext('2d');
    _sfChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: data.shadow_price != null ? 'SF Exposure ($/MWh)' : 'Shift Factor',
          data: values,
          backgroundColor: colors,
          borderRadius: 3,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => {
                const v = ctx.parsed.x;
                return data.shadow_price != null
                  ? ` $${v.toFixed(2)}/MWh`
                  : ` SF: ${v.toFixed(4)}`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } },
          },
          y: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } },
          },
        },
      },
    });

    highlightMapLine(constraintName);
  } catch (e) {
    content.innerHTML = `<div class="sf-placeholder"><p style="color:#ef4444">${e.message}</p></div>`;
    chartWrap.classList.add('hidden');
  }
}
