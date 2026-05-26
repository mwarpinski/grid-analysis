/**
 * grid_map.js — Leaflet map of the SPP transmission grid.
 *
 * Lines are colored by loading %:
 *   green (<70%) → yellow (70-90%) → orange (90-95%) → red (>95%)
 *
 * When a constraint is selected, a ZONE HEATMAP overlay replaces the
 * default line highlight.  Each SPP pricing zone is drawn as a circle
 * sized and colored by its aggregate shift factor:
 *   red   = exacerbates congestion (SF > 0)
 *   green = relieves congestion    (SF < 0)
 *   gray  = negligible             (|SF| < 0.005)
 *
 * This approach requires no knowledge of individual substation locations.
 * Zone centroids are publicly available SPP market zone geography.
 */

let gridMap        = null;
let linesLayer     = null;
let busLayer       = null;
let heatmapLayer   = null;   // zone SF circles — rebuilt on each constraint select
let constraintLineIndex = {};  // constraintName (lower) → Leaflet polyline

// ── Color helpers ────────────────────────────────────────────────────────────

function loadingColor(pct) {
  if (pct == null) return null;
  if (pct >= 95) return '#ef4444';
  if (pct >= 90) return '#f97316';
  if (pct >= 70) return '#eab308';
  return '#22c55e';
}

function voltageColor(kv) {
  if (kv >= 345) return '#818cf8';
  if (kv >= 230) return '#38bdf8';
  if (kv >= 138) return '#94a3b8';
  return '#475569';
}

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

  loadGeoJSON();
}

// ── GeoJSON base layer ────────────────────────────────────────────────────────

async function loadGeoJSON() {
  try {
    const data = await fetch('/data/spp_only_lines.geojson').then(r => {
      if (!r.ok) throw new Error(`GeoJSON not found (${r.status})`);
      return r.json();
    });

    const lineFeatures  = data.features.filter(f => f.geometry.type === 'LineString');
    const pointFeatures = data.features.filter(f => f.geometry.type === 'Point');

    linesLayer = L.geoJSON(lineFeatures, {
      style: (feature) => {
        const kv = feature.properties?.voltage_kv ?? 115;
        const w  = kv >= 345 ? 2.5 : kv >= 230 ? 1.8 : 1.2;
        return { color: voltageColor(kv), weight: w, opacity: 0.6 };
      },
      onEachFeature: (feature, layer) => {
        const name = feature.properties?.constraint_name;
        if (name) {
          constraintLineIndex[name.toLowerCase()] = layer;
          layer.on('click', () => {
            window.onConstraintSelect && window.onConstraintSelect(name);
          });
          const kv   = feature.properties?.voltage_kv ?? '?';
          const from = feature.properties?.from_bus ?? '';
          const to   = feature.properties?.to_bus ?? '';
          layer.bindTooltip(
            `<b>${name}</b><br>${from} → ${to}<br>${kv} kV`,
            { sticky: true, className: 'leaflet-tooltip-dark' }
          );
        }
      },
    }).addTo(gridMap);

    busLayer = L.geoJSON(pointFeatures, {
      pointToLayer: (feature, latlng) => {
        const kv = feature.properties?.voltage_kv ?? 115;
        const r  = kv >= 345 ? 5 : kv >= 230 ? 3.5 : 2.5;
        return L.circleMarker(latlng, {
          radius: r,
          fillColor: voltageColor(kv),
          color: '#0f172a',
          weight: 1,
          opacity: 1,
          fillOpacity: 0.9,
        });
      },
      onEachFeature: (feature, layer) => {
        const id = feature.properties?.bus_id ?? '';
        const kv = feature.properties?.voltage_kv ?? '';
        layer.bindTooltip(`${id} — ${kv} kV`, { sticky: true, className: 'leaflet-tooltip-dark' });
      },
    }).addTo(gridMap);

    console.info(`SPP topology loaded: ${lineFeatures.length} lines, ${pointFeatures.length} buses`);

  } catch (e) {
    console.info('GeoJSON not available — map will show base tiles only.', e.message);
  }
}

// ── Live constraint coloring ──────────────────────────────────────────────────

function updateMapColors(constraints) {
  if (!linesLayer) return;

  const pctMap = {};
  for (const c of constraints) {
    if (c.constraint_name) pctMap[c.constraint_name.toLowerCase()] = c.pct_of_limit;
  }

  linesLayer.eachLayer(layer => {
    const name = (layer.feature?.properties?.constraint_name || '').toLowerCase();
    const kv   = layer.feature?.properties?.voltage_kv ?? 115;
    const w    = kv >= 345 ? 2.5 : kv >= 230 ? 1.8 : 1.2;
    const pct  = pctMap[name];
    const col  = loadingColor(pct) || voltageColor(kv);
    layer.setStyle({
      color: col,
      weight: pct >= 90 ? w * 2 : w,
      opacity: pct != null ? 0.95 : 0.6,
    });
  });
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
      `/api/v1/shift-factors/heatmap?constraint=${encodeURIComponent(constraintName)}`
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

  // Reset line styles
  if (linesLayer) {
    linesLayer.eachLayer(l => {
      const kv = l.feature?.properties?.voltage_kv ?? 115;
      const w  = kv >= 345 ? 2.5 : kv >= 230 ? 1.8 : 1.2;
      l.setStyle({ color: voltageColor(kv), weight: w, opacity: 0.6 });
    });
  }
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

// ── Fallback line highlight (kept for backward compat) ────────────────────────

function highlightMapLine(constraintName) {
  const key   = constraintName.toLowerCase();
  const layer = constraintLineIndex[key];
  if (layer) {
    layer.setStyle({ color: '#f59e0b', weight: 5, opacity: 1 });
    gridMap.fitBounds(layer.getBounds(), { padding: [40, 40] });
  }
}
