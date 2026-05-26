/**
 * portfolio_panel.js — Portfolio P&L risk aggregation UI.
 *
 * Handles CSV import, row-by-row entry, position upload (POST /api/v1/positions),
 * and risk display (GET /api/v1/portfolio/risk).
 */

let _portfolioRows = [];

function initPortfolioPanel() {
  document.getElementById('portfolio-csv-btn').addEventListener('click', parseCSV);
  document.getElementById('portfolio-add-row').addEventListener('click', addPortfolioRow);
  document.getElementById('portfolio-submit').addEventListener('click', submitPortfolio);
  renderPortfolioRows();
}

function parseCSV() {
  const raw = document.getElementById('portfolio-csv').value.trim();
  const lines = raw.split('\n').filter(Boolean);
  if (lines.length < 2) return;

  const header = lines[0].toLowerCase().split(',').map(s => s.trim());
  const rows = [];
  for (const line of lines.slice(1)) {
    const vals = line.split(',').map(s => s.trim());
    const row = {};
    header.forEach((h, i) => row[h] = vals[i] || '');
    if (row.node) rows.push(row);
  }
  _portfolioRows = rows;
  renderPortfolioRows();
}

function addPortfolioRow() {
  _portfolioRows.push({ asset_name: '', asset_type: 'gen', node: '', mw_signed: '' });
  renderPortfolioRows();
}

function renderPortfolioRows() {
  const container = document.getElementById('portfolio-rows');
  container.innerHTML = _portfolioRows.map((row, i) => `
    <div class="portfolio-row">
      <input type="text"   placeholder="Asset name"  value="${escHtml(row.asset_name || '')}" data-i="${i}" data-f="asset_name" />
      <input type="text"   placeholder="Type"        value="${escHtml(row.asset_type || 'gen')}" data-i="${i}" data-f="asset_type" />
      <input type="text"   placeholder="Node"        value="${escHtml(row.node || '')}" data-i="${i}" data-f="node" />
      <input type="number" placeholder="MW"          value="${escHtml(String(row.mw_signed || ''))}" data-i="${i}" data-f="mw_signed" />
      <button class="remove-btn" data-i="${i}">×</button>
    </div>`).join('');

  container.querySelectorAll('input').forEach(inp => {
    inp.addEventListener('change', e => {
      const i = +e.target.dataset.i, f = e.target.dataset.f;
      _portfolioRows[i][f] = e.target.value;
    });
  });
  container.querySelectorAll('.remove-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      _portfolioRows.splice(+e.target.dataset.i, 1);
      renderPortfolioRows();
    });
  });
}

async function submitPortfolio() {
  const constraint = document.getElementById('portfolio-constraint').value.trim();
  if (!constraint) { alert('Enter a constraint name.'); return; }
  if (_portfolioRows.length === 0) { alert('Add at least one position.'); return; }

  const btn = document.getElementById('portfolio-submit');
  btn.textContent = 'Calculating…'; btn.disabled = true;

  const positions = _portfolioRows.map(r => ({
    asset_name: r.asset_name || 'Unknown',
    asset_type: (r.asset_type || 'gen').toLowerCase(),
    node: r.node,
    mw_signed: parseFloat(r.mw_signed) || 0,
  }));

  const resultEl = document.getElementById('portfolio-result');

  try {
    await api.post('/api/v1/positions', positions);
    const data = await api.get(`/api/v1/portfolio/risk?constraint=${encodeURIComponent(constraint)}`);

    const dir = data.total_net_mw_flow > 0.01 ? 'exacerbates'
               : data.total_net_mw_flow < -0.01 ? 'relieves' : 'neutral';

    // Alert banner
    const alertEl = document.getElementById('portfolio-alert-badge');
    alertEl.className = `alert-banner alert-${dir} fade-in`;
    alertEl.textContent = data.alert_message;
    alertEl.classList.remove('hidden');

    // Risk table
    const rows = data.positions.map(p => `
      <tr>
        <td>${escHtml(p.asset_name)}</td>
        <td>${escHtml(p.asset_type)}</td>
        <td>${escHtml(p.node)}</td>
        <td>${p.mw_signed > 0 ? '+' : ''}${p.mw_signed.toFixed(0)}</td>
        <td>${p.gsf != null ? p.gsf.toFixed(4) : '—'}</td>
        <td>${p.net_mw_flow != null ? (p.net_mw_flow > 0 ? '+' : '') + p.net_mw_flow.toFixed(2) : '—'}</td>
        <td>${p.pnl_risk_rate != null ? '$' + p.pnl_risk_rate.toLocaleString(undefined,{maximumFractionDigits:0}) + '/hr' : '—'}</td>
        <td class="dir-${p.direction || 'neutral'}">${p.direction ? (p.direction==='exacerbates'?'▲':p.direction==='relieves'?'▼':'—') : '—'}</td>
      </tr>`).join('');

    resultEl.innerHTML = `
      <div style="overflow-x:auto;padding:.75rem;width:100%" class="fade-in">
        <table class="portfolio-risk-table">
          <thead>
            <tr><th>Asset</th><th>Type</th><th>Node</th><th>MW</th><th>GSF</th><th>Net MW</th><th>P&L Rate</th><th>Dir</th></tr>
          </thead>
          <tbody>${rows}</tbody>
          <tfoot>
            <tr class="total-row">
              <td colspan="5"><strong>Portfolio Total</strong></td>
              <td><strong>${data.total_net_mw_flow > 0 ? '+' : ''}${data.total_net_mw_flow.toFixed(2)} MW</strong></td>
              <td><strong>$${data.total_pnl_risk_rate.toLocaleString(undefined,{maximumFractionDigits:0})}/hr</strong></td>
              <td class="dir-${dir}">${dir==='exacerbates'?'▲':dir==='relieves'?'▼':'—'}</td>
            </tr>
          </tfoot>
        </table>
      </div>`;
  } catch (e) {
    resultEl.innerHTML = `<div class="wi-result-card fade-in"><p style="color:#ef4444">Error: ${escHtml(e.message)}</p></div>`;
  } finally {
    btn.textContent = 'Calculate Portfolio Risk'; btn.disabled = false;
  }
}
