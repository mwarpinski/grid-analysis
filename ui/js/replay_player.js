/**
 * replay_player.js — DVR-style historical replay UI.
 *
 * Loads interval list from /api/v1/replay/intervals,
 * then scrubs through snapshots via /api/v1/replay/snapshot.
 */

let _replayIntervals = [];
let _replayIdx = 0;
let _replayTimer = null;
let _replayPlaying = false;

function initReplayPlayer() {
  // Default date range: last 24 hours
  const now = new Date();
  const yesterday = new Date(now - 24 * 3600 * 1000);
  document.getElementById('replay-start').value = toLocalInput(yesterday);
  document.getElementById('replay-end').value   = toLocalInput(now);

  document.getElementById('replay-load').addEventListener('click', loadReplayIntervals);
  document.getElementById('replay-play').addEventListener('click', toggleReplayPlay);
  document.getElementById('replay-prev').addEventListener('click', () => stepReplay(-1));
  document.getElementById('replay-next').addEventListener('click', () => stepReplay(1));
  document.getElementById('replay-slider').addEventListener('input', e => {
    _replayIdx = parseInt(e.target.value);
    showReplaySnapshot(_replayIdx);
  });
}

async function loadReplayIntervals() {
  const btn = document.getElementById('replay-load');
  btn.textContent = 'Loading…'; btn.disabled = true;

  const start      = document.getElementById('replay-start').value;
  const end        = document.getElementById('replay-end').value;
  const constraint = document.getElementById('replay-constraint').value.trim();

  try {
    const params = new URLSearchParams({ start, end });
    const data = await api.get('/api/v1/replay/intervals?' + params);

    if (!data.length) {
      alert('No data found for this date range. The ingest worker needs to have been running during that period.');
      return;
    }

    _replayIntervals = data;
    _replayIdx = 0;

    const slider = document.getElementById('replay-slider');
    slider.max   = data.length - 1;
    slider.value = 0;

    document.getElementById('replay-slider-wrap').classList.remove('hidden');
    showReplaySnapshot(0);
  } catch (e) {
    alert('Error loading intervals: ' + e.message);
  } finally {
    btn.textContent = 'Load Intervals'; btn.disabled = false;
  }
}

async function showReplaySnapshot(idx) {
  const iv = _replayIntervals[idx];
  if (!iv) return;

  document.getElementById('replay-interval-label').textContent = new Date(iv.interval).toLocaleString();
  document.getElementById('replay-slider').value = idx;

  const snapshotEl = document.getElementById('replay-snapshot');
  snapshotEl.innerHTML = '<p style="color:#64748b;padding:.75rem">Loading snapshot…</p>';

  try {
    const data = await api.get(
      '/api/v1/replay/snapshot?interval=' + encodeURIComponent(iv.interval)
    );

    if (!data.constraints.length) {
      snapshotEl.innerHTML = '<p style="color:#64748b;padding:.75rem">No binding constraints at this interval.</p>';
      return;
    }

    const rows = data.constraints
      .sort((a, b) => Math.abs(b.shadow_price) - Math.abs(a.shadow_price))
      .map(c => `
        <tr>
          <td>${escHtml(truncate(c.constraint_name, 32))}</td>
          <td>${c.shadow_price != null ? '$' + c.shadow_price.toLocaleString(undefined,{maximumFractionDigits:0}) : '—'}</td>
          <td>${c.pct_of_limit != null ? c.pct_of_limit.toFixed(1) + '%' : '—'}</td>
          <td>${c.limit_mw != null ? c.limit_mw.toFixed(0) + ' MW' : '—'}</td>
          <td><span class="state-pill state-${c.state || ''}">${escHtml(c.state || '—')}</span></td>
        </tr>`).join('');

    snapshotEl.innerHTML = `
      <div class="fade-in" style="overflow-x:auto">
        <table class="data-table">
          <thead>
            <tr><th>Constraint</th><th>Shadow Price</th><th>Loading %</th><th>Limit</th><th>State</th></tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;

    // Also update map
    updateMapColors(data.constraints);
  } catch (e) {
    snapshotEl.innerHTML = `<p style="color:#ef4444;padding:.75rem">Error: ${escHtml(e.message)}</p>`;
  }
}

function stepReplay(delta) {
  _replayIdx = Math.max(0, Math.min(_replayIntervals.length - 1, _replayIdx + delta));
  showReplaySnapshot(_replayIdx);
}

function toggleReplayPlay() {
  const btn = document.getElementById('replay-play');
  if (_replayPlaying) {
    clearInterval(_replayTimer);
    _replayPlaying = false;
    btn.textContent = '▶';
    btn.style.background = 'var(--accent)';
  } else {
    _replayPlaying = true;
    btn.textContent = '⏸';
    btn.style.background = '#f97316';
    const speed = parseInt(document.getElementById('replay-speed').value);
    _replayTimer = setInterval(() => {
      if (_replayIdx >= _replayIntervals.length - 1) {
        toggleReplayPlay(); return;
      }
      _replayIdx++;
      showReplaySnapshot(_replayIdx);
    }, speed);
  }
}

function toLocalInput(d) {
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
