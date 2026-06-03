/**
 * constraints_table.js — Live binding constraint table.
 *
 * Connects to /ws/constraints (via api.js openConstraintsWS),
 * renders a sortable table, and highlights near-limit rows in amber/red.
 */

let _constraints = [];
let _sortCol = 'shadow_price';
let _sortDir = -1;   // -1 = descending

function renderConstraintsTable(rows) {
  const minSP  = parseFloat(document.getElementById('filter-min-sp').value)  || 0;
  const minPct = parseFloat(document.getElementById('filter-min-pct').value) || 0;

  let filtered = rows.filter(r =>
    Math.abs(r.shadow_price || 0) >= minSP &&
    (minPct === 0 || (r.pct_of_limit != null && r.pct_of_limit >= minPct))
  );

  filtered.sort((a, b) => {
    const av = a[_sortCol] ?? -Infinity;
    const bv = b[_sortCol] ?? -Infinity;
    return _sortDir * (av < bv ? -1 : av > bv ? 1 : 0);
  });

  const tbody = document.getElementById('constraint-tbody');
  if (filtered.length === 0) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="5">No constraints match filters.</td></tr>';
    return;
  }

  tbody.innerHTML = filtered.map(r => {
    const pct     = r.pct_of_limit != null ? r.pct_of_limit.toFixed(1) + '%' : '—';
    const vol     = r.volatility_score != null ? r.volatility_score.toFixed(2) : '0.00';
    const sp      = r.shadow_price != null ? '$' + r.shadow_price.toLocaleString(undefined, {maximumFractionDigits:0}) : '—';
    const state   = r.state || '—';
    const stateClass = `state-${state.replace(/\s/g,'_')}`;
    const rowClass = r.pct_of_limit >= 95 ? 'row-binding' : r.pct_of_limit >= 85 ? 'row-near-limit-orange' : r.pct_of_limit >= 70 ? 'row-near-limit' : '';
    return `<tr class="${rowClass}" data-name="${escHtml(r.constraint_name)}">
      <td title="${escHtml(r.constraint_name)}">${escHtml(truncate(r.constraint_name,30))}</td>
      <td>${sp}</td>
      <td>${pct}</td>
      <td>${vol}</td>
      <td><span class="state-pill ${stateClass}">${escHtml(state)}</span></td>
    </tr>`;
  }).join('');

  // Row click → select constraint
  tbody.querySelectorAll('tr[data-name]').forEach(row => {
    row.addEventListener('click', () => {
      document.querySelectorAll('#constraint-tbody tr').forEach(r => r.classList.remove('selected'));
      row.classList.add('selected');
      window.onConstraintSelect && window.onConstraintSelect(row.dataset.name);
    });
  });

  // Badge counts
  document.getElementById('binding-count').textContent =
    rows.filter(r => r.state === 'BINDING').length + ' binding';
  document.getElementById('near-limit-count').textContent =
    rows.filter(r => r.pct_of_limit >= 85 && r.state !== 'BINDING').length + ' near limit';
}

function initConstraintsTable() {
  // Sortable headers
  document.querySelectorAll('#constraint-table th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (_sortCol === col) _sortDir *= -1;
      else { _sortCol = col; _sortDir = -1; }
      renderConstraintsTable(_constraints);
    });
  });

  // Filter inputs
  ['filter-min-sp', 'filter-min-pct'].forEach(id => {
    document.getElementById(id).addEventListener('input', () => renderConstraintsTable(_constraints));
  });
}

function updateConstraintsTable(data) {
  if (data.constraints) {
    _constraints = data.constraints;
    renderConstraintsTable(_constraints);
    updateMapColors(_constraints);
  }
}

// Helpers
function escHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function truncate(s, n) { return s.length > n ? s.slice(0, n) + '…' : s; }
