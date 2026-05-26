/**
 * grid_map.js — Leaflet map of the SPP transmission grid.
 *
 * Lines are colored by loading %:
 *   green (<70%) → yellow (70-90%) → orange (90-95%) → red (>95%)
 *
 * Clicking a line fires window.onConstraintSelect(constraintName).
 *
 * TOPOLOGY NOTE: The GeoJSON is a synthetic schematic. Bus positions are
 * regional centroids with ±25 mi jitter — not real substation coordinates.
 */

let gridMap = null;
let linesLayer = null;
let busLayer = null;
let constraintLineIndex = {};   // constraintName (lower) → Leaflet polyline

function loadingColor(pct) {
  if (pct == null) return null;   // fall back to voltage color
  if (pct >= 95) return '#ef4444';
  if (pct >= 90) return '#f97316';
  if (pct >= 70) return '#eab308';
  return '#22c55e';
}

/** Idle line color based on voltage level (schematic convention). */
function voltageColor(kv) {
  if (kv >= 345) return '#818cf8';   // indigo — EHV (345 kV+)
  if (kv >= 230) return '#38bdf8';   // sky — HV (230 kV)
  if (kv >= 138) return '#94a3b8';   // slate — sub-HV
  return '#475569';                   // dark — sub-transmission
}

function initMap() {
  gridMap = L.map('grid-map', {
    center: [40.5, -98.5],   // Nebraska / central SPP
    zoom: 5,
    zoomControl: true,
    attributionControl: false,
  });

  // Dark tiles match the dashboard theme
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 12,
  }).addTo(gridMap);

  loadGeoJSON();
}

async function loadGeoJSON() {
  try {
    const data = await fetch('/data/spp_only_lines.geojson').then(r => {
      if (!r.ok) throw new Error(`GeoJSON not found (${r.status})`);
      return r.json();
    });

    const lineFeatures  = data.features.filter(f => f.geometry.type === 'LineString');
    const pointFeatures = data.features.filter(f => f.geometry.type === 'Point');

    // ── Transmission lines ───────────────────────────────────────────────────
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

    // ── Bus nodes ────────────────────────────────────────────────────────────
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

/** Update line colors based on live constraint data. */
function updateMapColors(constraints) {
  if (!linesLayer) return;

  // Build lookup: constraint_name (lower) → pct_of_limit
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

/** Highlight a specific constraint's line on the map. */
function highlightMapLine(constraintName) {
  // Reset all to voltage-based idle
  if (linesLayer) {
    linesLayer.eachLayer(l => {
      const kv = l.feature?.properties?.voltage_kv ?? 115;
      const w  = kv >= 345 ? 2.5 : kv >= 230 ? 1.8 : 1.2;
      l.setStyle({ color: voltageColor(kv), weight: w, opacity: 0.6 });
    });
  }
  const key   = constraintName.toLowerCase();
  const layer = constraintLineIndex[key];
  if (layer) {
    layer.setStyle({ color: '#f59e0b', weight: 5, opacity: 1 });
    gridMap.fitBounds(layer.getBounds(), { padding: [40, 40] });
  }
}
