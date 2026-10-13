/**
 * grid_map.js — Leaflet map of where SPP constraints are.
 *
 * Each constraint is a marker at its estimated location (public data only; see
 * src/analysis/geo_crosswalk.py).  These are REGIONAL positions: every marker carries a method, a
 * quality label and an uncertainty radius, shown as a circle on hover/select.  About half of all
 * constraints cannot be placed and do not appear.  Currently binding constraints are colored by
 * shadow price magnitude.
 *
 * When a constraint is selected, a ZONE HEATMAP overlay shows which SPP pricing zones it affects:
 * each zone is a circle sized and colored by its aggregate shift factor
 *   red   = exacerbates congestion (SF > 0)
 *   green = relieves congestion    (SF < 0)
 *   gray  = negligible             (|SF| < 0.005)
 * Zone centroids are publicly available SPP market zone geography.
 */

let gridMap        = null;
let locationsLayer = null;   // one marker per placed constraint
let heatmapLayer   = null;   // zone SF circles — rebuilt on each constraint select
let uncertaintyCircle = null;
let constraintMarkers = {};  // constraintName (lower) → { marker, props }
let _liveFetchAt   = 0;      // throttle for fetching locations of brand-new live constraints

// ── Color helpers ────────────────────────────────────────────────────────────

function shadowPriceColor(sp) {
  const m = Math.abs(sp || 0);
  if (m >= 500) return '#ef4444';
  if (m >= 100) return '#f97316';
  if (m >= 20)  return '#eab308';
  return '#22c55e';
}

const IDLE_STYLE = { radius: 4, color: '#64748b', fillColor: '#475569', weight: 1, opacity: 0.8, fillOpacity: 0.5 };

/**
 * Diverging SF → color scale.
 *   SF < -0.10  → bright green
 *   SF -0.10 to -0.02 → muted green
 *   SF ±0.005   → gray (negligible)
 *   SF 0.02 to 0.10  → muted red
 *   SF > 0.10   → bright red
 */
function sfColor(meanSf) {
  const v = meanSf;
  if (v >=  0.15) return '#ef4444';   // bright red
  if (v >=  0.05) return '#f87171';   // light red
  if (v >=  0.005) return '#fca5a5';  // very light red
  if (v <= -0.15) return '#22c55e';   // bright green
  if (v <= -0.05) return '#4ade80';   // light green
  if (v <= -0.005) return '#86efac';  // very light green
  return '#64748b';                   // slate gray — neutral
}

// ── Map init ─────────────────────────────────────────────────────────────────

function initMap() {
  gridMap = L.map('grid-map', {
    center: [40.5, -98.5],
    zoom: 5,
    zoomControl: true,
    attributionControl: false,
  });

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 12,
  }).addTo(gridMap);

  loadConstraintLocations();
  addApproximationNote();
}

function addApproximationNote() {
  const note = document.createElement('div');
  note.style.cssText = `
    position: absolute; bottom: 8px; left: 8px; z-index: 1000; max-width: 260px;
    background: rgba(15,23,42,0.85); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px;
    padding: 6px 9px; font-size: 10px; color: #94a3b8; pointer-events: none;`;
  note.textContent = 'Approximate locations from public data (regional, not substation-precise). Hover a dot for its uncertainty radius; unplaced constraints are not shown.';
  document.getElementById('grid-map').appendChild(note);
}

// ── Constraint locations ──────────────────────────────────────────────────────

function _addConstraintMarker(c) {
  if (c.lat == null || c.lon == null) return;
  const key = c.constraint_name.toLowerCase();
  if (constraintMarkers[key]) return;
  const approx = c.quality !== 'good';
  const marker = L.circleMarker([c.lat, c.lon], { ...IDLE_STYLE, dashArray: approx ? '2 3' : null });
  marker.bindTooltip(
    `<b>${c.constraint_name}</b><br>${String(c.method).replace('_', ' ')} · ${c.quality}` +
    `<br>±${Math.round(c.radius_km)} km`,
    { sticky: true, className: 'leaflet-tooltip-dark' }
  );
  marker.on('mouseover', () => showUncertainty(c));
  marker.on('mouseout', () => { if (!_selectedName) clearUncertainty(); });
  marker.on('click', () => {
    window.onConstraintSelect && window.onConstraintSelect(c.constraint_name);
  });
  marker.addTo(locationsLayer);
  constraintMarkers[key] = { marker, props: c };
}

async function loadConstraintLocations() {
  locationsLayer = L.layerGroup().addTo(gridMap);
  try {
    const data = await api.get('/api/v1/geo/constraints');
    data.constraints.forEach(_addConstraintMarker);
    console.info(`Constraint locations loaded: ${data.n}`);
  } catch (e) {
    console.info('No constraint locations yet (run scripts/build_geo_crosswalk.py).', e.message);
  }
}

/** Constraints that started binding after the last build have no marker yet: fetch their locations. */
async function refreshLiveLocations() {
  if (Date.now() - _liveFetchAt < 30000) return;
  _liveFetchAt = Date.now();
  try {
    const data = await api.get('/api/v1/geo/live');
    data.constraints.forEach(_addConstraintMarker);
  } catch (e) { /* not ready or no locations; ignore */ }
}

let _selectedName = null;

function showUncertainty(c) {
  clearUncertainty();
  uncertaintyCircle = L.circle([c.lat, c.lon], {
    radius: (c.radius_km || 0) * 1000, color: '#38bdf8', weight: 1, dashArray: '4 4',
    fillColor: '#38bdf8', fillOpacity: 0.06, interactive: false,
  }).addTo(gridMap);
}

function clearUncertainty() {
  if (uncertaintyCircle) { gridMap.removeLayer(uncertaintyCircle); uncertaintyCircle = null; }
}

// ── Live constraint coloring ──────────────────────────────────────────────────

function updateMapColors(constraints) {
  if (!locationsLayer) return;

  const live = {};
  for (const c of constraints) {
    if (c.constraint_name && c.shadow_price) live[c.constraint_name.toLowerCase()] = c;
  }

  let missing = false;
  for (const key of Object.keys(live)) if (!constraintMarkers[key]) missing = true;
  if (missing) refreshLiveLocations();

  for (const [key, { marker, props }] of Object.entries(constraintMarkers)) {
    const c = live[key];
    if (c) {
      const col = shadowPriceColor(c.shadow_price);
      marker.setStyle({ radius: 8, color: '#0f172a', fillColor: col, weight: 1.5, opacity: 1, fillOpacity: 0.9 });
      marker.bringToFront();
      marker.setTooltipContent(
        `<b>${props.constraint_name}</b><br>$${Math.round(c.shadow_price)}/MWh · ${c.state || ''}` +
        `<br>${String(props.method).replace('_', ' ')} · ±${Math.round(props.radius_km)} km`
      );
    } else {
      marker.setStyle(IDLE_STYLE);
    }
  }
}

// ── SF Zone Heatmap overlay ───────────────────────────────────────────────────

/**
 * Fetch shift factor data grouped by SPP zone and render as colored
 * circles.  Each circle is:
 *   - Positioned at the zone's public market centroid
 *   - Colored by direction (red = exacerbates, green = relieves)
 *   - Sized by |mean_sf| (larger = more electrically affected)
 *   - Pulsing if it's the hottest zone
 *
 * Called whenever a constraint is selected from the table or map.
 */
async function showSFHeatmap(constraintName) {
  // Clear previous heatmap
  clearHeatmap();

  let data;
  try {
    data = await api.get(
      `/api/v1/shift-factors/heatmap?constraint=${encodeURIComponent(constraintName)}&iso=${window.currentISO || 'spp'}`
    );
  } catch (e) {
    // Solver not ready or no data — just highlight the line if it exists
    highlightMapLine(constraintName);
    return;
  }

  if (!data.zones || data.zones.length === 0) {
    highlightMapLine(constraintName);
    return;
  }

  // Scale radii: hottest zone = 40px, others proportional, minimum 8px
  const maxAbsSf = Math.max(...data.zones.map(z => Math.abs(z.mean_sf)));
  const scaleR   = (absSf) => Math.max(8, Math.min(48, (absSf / maxAbsSf) * 40));

  heatmapLayer = L.layerGroup();

  data.zones.forEach((zone, idx) => {
    const isHottest = idx === 0;
    const r     = scaleR(Math.abs(zone.mean_sf));
    const color = sfColor(zone.mean_sf);

    // Outer glow ring for hottest zone
    if (isHottest) {
      L.circleMarker([zone.lat, zone.lon], {
        radius: r + 10,
        fillColor: color,
        fillOpacity: 0.12,
        color: color,
        weight: 1,
        opacity: 0.25,
        interactive: false,
        className: 'sf-heatmap-pulse',
      }).addTo(heatmapLayer);
    }

    // Main circle
    const marker = L.circleMarker([zone.lat, zone.lon], {
      radius: r,
      fillColor: color,
      fillOpacity: 0.72,
      color: '#0f172a',
      weight: isHottest ? 2 : 1,
      opacity: 1,
    });

    // Tooltip
    const sp      = data.shadow_price != null ? `$${data.shadow_price.toFixed(0)}/MWh SP` : '';
    const expStr  = zone.exposure != null
      ? `<br>Exposure: <b>${zone.exposure > 0 ? '+' : ''}${zone.exposure.toFixed(2)} $/MWh</b>`
      : '';
    const dirIcon = zone.direction === 'exacerbates' ? '▲ Exacerbates' :
                    zone.direction === 'relieves'     ? '▼ Relieves'    : '● Neutral';
    const dirColor = zone.direction === 'exacerbates' ? '#f87171' :
                     zone.direction === 'relieves'     ? '#4ade80'  : '#94a3b8';

    marker.bindTooltip(`
      <div style="min-width:180px">
        <b style="font-size:12px">${zone.name}</b>
        <div style="color:${dirColor};font-weight:600;margin-top:2px">${dirIcon}</div>
        <div style="color:#94a3b8;font-size:11px;margin-top:4px">
          Mean SF: <b style="color:#e2e8f0">${zone.mean_sf > 0 ? '+' : ''}${zone.mean_sf.toFixed(4)}</b>
          ${expStr}
          <br>${zone.n_nodes} nodes · ${sp}
        </div>
      </div>
    `, { sticky: true, className: 'leaflet-tooltip-dark', offset: [0, -r] });

    marker.on('click', () => {
      window.onConstraintSelect && window.onConstraintSelect(constraintName);
    });

    marker.addTo(heatmapLayer);
  });

  heatmapLayer.addTo(gridMap);

  // Also try to highlight the GeoJSON line (works if name matches)
  highlightMapLine(constraintName);

  // Show a legend
  updateHeatmapLegend(constraintName, data);
}

function clearHeatmap() {
  if (heatmapLayer) {
    heatmapLayer.clearLayers();
    gridMap.removeLayer(heatmapLayer);
    heatmapLayer = null;
  }
  clearHeatmapLegend();

  clearUncertainty();
  _selectedName = null;
}

// ── Legend ────────────────────────────────────────────────────────────────────

function updateHeatmapLegend(constraintName, data) {
  let legend = document.getElementById('sf-map-legend');
  if (!legend) {
    legend = document.createElement('div');
    legend.id = 'sf-map-legend';
    legend.style.cssText = `
      position: absolute; bottom: 32px; right: 12px; z-index: 1000;
      background: rgba(15,23,42,0.92); border: 1px solid rgba(255,255,255,0.08);
      border-radius: 8px; padding: 10px 14px; min-width: 160px;
      font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #cbd5e1;
      pointer-events: none;
    `;
    document.getElementById('grid-map').appendChild(legend);
  }

  const mapped = data.n_nodes_mapped;
  const total  = data.n_nodes_total;
  const pct    = total > 0 ? Math.round(mapped / total * 100) : 0;

  legend.innerHTML = `
    <div style="font-weight:700;color:#f1f5f9;margin-bottom:6px;font-size:12px">
      SF Exposure Map
    </div>
    <div style="color:#94a3b8;font-size:10px;margin-bottom:8px;word-break:break-all">
      ${constraintName}
    </div>
    <div style="display:flex;flex-direction:column;gap:4px">
      <div style="display:flex;align-items:center;gap:6px">
        <span style="width:12px;height:12px;border-radius:50%;background:#ef4444;display:inline-block"></span>
        Exacerbates
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <span style="width:12px;height:12px;border-radius:50%;background:#22c55e;display:inline-block"></span>
        Relieves
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <span style="width:12px;height:12px;border-radius:50%;background:#64748b;display:inline-block"></span>
        Neutral
      </div>
    </div>
    <div style="margin-top:8px;color:#64748b;font-size:10px">
      ${data.zones.length} zones · ${pct}% nodes mapped
    </div>
  `;
}

function clearHeatmapLegend() {
  const legend = document.getElementById('sf-map-legend');
  if (legend) legend.remove();
}

// ── Highlight the selected constraint on the map ─────────────────────────────

function highlightMapLine(constraintName) {
  const entry = constraintMarkers[constraintName.toLowerCase()];
  if (!entry) return;
  _selectedName = constraintName;
  const { props } = entry;
  showUncertainty(props);
  entry.marker.bringToFront();
  gridMap.fitBounds(uncertaintyCircle.getBounds(), { padding: [60, 60], maxZoom: 8 });
}
