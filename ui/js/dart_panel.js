/**
 * dart_panel.js — Day-Ahead vs. Real-Time (DART) arbitrage view controller.
 */

let dartChart = null;

function initDARTPanel() {
  const daysSelect = document.getElementById('dart-days');
  if (daysSelect) {
    daysSelect.addEventListener('change', () => {
      loadDARTData();
    });
  }
}

async function loadDARTData() {
  const badge = document.getElementById('dart-status-badge');
  const tbody = document.getElementById('dart-tbody');
  
  if (badge) {
    badge.textContent = 'Loading…';
    badge.className = 'badge badge-amber';
  }
  
  const days = document.getElementById('dart-days').value;
  const iso = window.currentISO || 'spp';
  
  try {
    const data = await api.get(`/api/v1/constraints/dart-spreads?iso=${iso}&days=${days}`);
    const spreads = data.spreads || [];
    
    // Render Stats
    renderDARTStats(spreads);
    
    // Render Table
    renderDARTTable(spreads);
    
    // Render Chart
    renderDARTChart(spreads);
    
    if (badge) {
      badge.textContent = 'Loaded';
      badge.className = 'badge badge-green';
    }
  } catch (err) {
    console.error('Failed to load DART spreads:', err);
    if (tbody) {
      tbody.innerHTML = `<tr class="empty-row"><td colspan="7" class="text-red">Error loading DART data: ${err.message}</td></tr>`;
    }
    if (badge) {
      badge.textContent = 'Error';
      badge.className = 'badge badge-red';
    }
  }
}

function renderDARTStats(spreads) {
  const totalEl = document.getElementById('dart-stat-total');
  const underpricedEl = document.getElementById('dart-stat-underpriced');
  const maxSpreadEl = document.getElementById('dart-stat-max-spread');
  
  const total = spreads.length;
  const underpriced = spreads.filter(s => s.underpriced_in_dam).length;
  
  let maxSpread = 0;
  spreads.forEach(s => {
    if (s.mean_dart_spread < maxSpread) {
      maxSpread = s.mean_dart_spread;
    }
  });
  
  if (totalEl) totalEl.textContent = total;
  if (underpricedEl) underpricedEl.textContent = underpriced;
  if (maxSpreadEl) maxSpreadEl.textContent = `$${maxSpread.toFixed(2)}`;
}

function renderDARTTable(spreads) {
  const tbody = document.getElementById('dart-tbody');
  if (!tbody) return;
  
  if (spreads.length === 0) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="7">No constraints with shadow prices found in lookback window.</td></tr>';
    return;
  }
  
  tbody.innerHTML = '';
  
  spreads.forEach(s => {
    const tr = document.createElement('tr');
    
    // Apply styling for underpriced
    if (s.underpriced_in_dam) {
      tr.className = 'underpriced-row';
    }
    
    const statusBadge = s.underpriced_in_dam 
      ? '<span class="badge badge-red" style="font-size: 8px;">DAM Underpriced</span>' 
      : '<span class="badge badge-green" style="font-size: 8px;">Aligned</span>';
      
    tr.innerHTML = `
      <td style="font-family: monospace; font-weight: 500;">${s.constraint_name}</td>
      <td style="font-family: monospace; text-align: right; color: var(--red);">$${s.mean_rtbm_shadow_price.toFixed(2)}</td>
      <td style="font-family: monospace; text-align: right; color: var(--amber);">$${s.mean_dam_shadow_price.toFixed(2)}</td>
      <td style="font-family: monospace; text-align: right; font-weight: 600; color: ${s.mean_dart_spread < 0 ? 'var(--red)' : 'var(--green)'};">
        $${s.mean_dart_spread.toFixed(2)}
      </td>
      <td style="font-family: monospace; text-align: right;">${s.rtbm_binding_hours}</td>
      <td style="font-family: monospace; text-align: right;">${s.dam_binding_hours}</td>
      <td style="text-align: center;">${statusBadge}</td>
    `;
    
    tbody.appendChild(tr);
  });
}

function renderDARTChart(spreads) {
  const placeholder = document.getElementById('dart-chart-placeholder');
  const wrap = document.getElementById('dart-chart-wrap');
  const canvas = document.getElementById('dart-chart');
  
  if (!canvas) return;
  
  // Get top 8 constraints with most negative spread (or highest volume of underpricing)
  const topSpreads = spreads.slice(0, 8);
  
  if (topSpreads.length === 0) {
    if (placeholder) placeholder.classList.remove('hidden');
    if (wrap) wrap.classList.add('hidden');
    return;
  }
  
  if (placeholder) placeholder.classList.add('hidden');
  if (wrap) wrap.classList.remove('hidden');
  
  const labels = topSpreads.map(s => s.constraint_name.length > 20 ? s.constraint_name.substring(0, 17) + '...' : s.constraint_name);
  const rtData = topSpreads.map(s => s.mean_rtbm_shadow_price);
  const daData = topSpreads.map(s => s.mean_dam_shadow_price);
  
  if (dartChart) {
    dartChart.destroy();
  }
  
  const ctx = canvas.getContext('2d');
  dartChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'RTBM Mean Price ($/MWh)',
          data: rtData,
          backgroundColor: '#ef4444',
          borderColor: '#ef4444',
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'DAM Mean Price ($/MWh)',
          data: daData,
          backgroundColor: '#f59e0b',
          borderColor: '#f59e0b',
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#94a3b8',
            font: { size: 10, family: 'Inter' }
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { size: 9, family: 'JetBrains Mono' } }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#e2e8f0', font: { size: 10, family: 'JetBrains Mono' } }
        }
      }
    }
  });
}

// Global hook to activate panel switching
window.dartPanelActivate = function() {
  loadDARTData();
};
