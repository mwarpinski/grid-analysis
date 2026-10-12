# SPP Transmission Congestion Monitor

**Real-time shift factor analysis, congestion exposure, and portfolio risk for SPP RTBM traders.**

Built for market makers and asset optimizers who need to know — before the 5-minute interval clears — exactly how much a binding transmission constraint will move the price at every node in the SPP footprint.

---

## What is this?

When a transmission line in the Southwest Power Pool (SPP) runs near its thermal limit, the real-time market (RTBM) prices at nearby nodes can spike from $30/MWh to $5,000/MWh in a single 5-minute dispatch interval. This platform gives traders the tools to anticipate those moves rather than react to them.

### The core problem it solves

The ISO (SPP) publishes Locational Marginal Prices (LMPs) every 5 minutes. Each LMP has three components:

```
LMP = MEC (energy cost) + MCC (congestion cost) + MLCC (loss cost)
```

The congestion component (MCC) is driven by **Generation Shift Factors (GSFs)** — a number between -1 and +1 that describes how much a MW injected at a given node flows across a specific transmission constraint. The formula is:

```
MCC_node = Σ ( GSF[node, constraint] × ShadowPrice[constraint] )
```

SPP does not publish GSFs in real time. This platform derives them statistically from historical binding constraint and LMP data, caches them in memory, and re-solves automatically when the set of binding constraints changes.

### What the platform gives you

| Capability | Description |
|---|---|
| **Live constraint feed** | Auto-refreshing table of every priced RTBM constraint (BINDING or BREACHED) with shadow price and state. Loading % is shown only if SPP publishes it; the RTBM files currently don't. |
| **Shift factor exposure** | GSF × ShadowPrice for every node, ranked by impact — the trader's key exposure metric |
| **Pre-clearing MCC** | Estimated congestion component of LMP computed in-memory before SPP publishes it |
| **Bilateral flow impact** | "If I inject 50 MW at Node A and buy at Node B, how much flow does that add to Constraint X?" |
| **Portfolio P&L risk** | Aggregate all positions against current shift factors → net MW contribution + $/hr run rate |
| **DVR replay** | Scrub back through historical intervals interval-by-interval to backtest positions |
| **SF validation** | MCC back-prediction test: compare our GSFs to SPP's published ground truth |
| **Grid map** | Leaflet map of a *synthetic* SPP schematic. Line names do not correspond to real SPP constraint names yet, so constraints are not placed on the map. |

---

## How to run it locally

### Prerequisites

- Python 3.11+
- A virtual environment with project dependencies installed

### 1. Clone and set up the environment

```bash
git clone git@github.com:mwarpinski/grid-analysis.git
cd grid-analysis

# Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
# or if using Poetry:
poetry install
```

### 2. Generate the grid topology (one-time, already done)

This creates the Leaflet map's `spp_only_lines.geojson` — a synthetic schematic of the SPP
footprint (see [Security note](#security-note-on-grid-topology) below).

```bash
python scripts/generate_spp_topology.py
```

The file is written to `ui/data/spp_only_lines.geojson`. It only needs to be re-run
if you want to regenerate it.

### 3. Start the server

```bash
uvicorn src.api.server:app --reload --port 8000
```

The server will:
1. Run an immediate scrape of SPP RTBM on startup to populate the cache
2. Start the background ingest worker (polls every 290 seconds)
3. Serve the Trading Dashboard at **http://localhost:8000**
4. Expose the Swagger API docs at **http://localhost:8000/docs**

> **Note:** The first startup scrape may take 10–30 seconds and will warn if the current
> interval's file isn't published yet (SPP has a 30–90 second publication lag after each
> 5-minute interval end). This is normal — the server starts with an empty cache and
> populates on the next successful scrape.

### 4. Open the dashboard

Navigate to **http://localhost:8000** in your browser.

The dashboard has four panels accessible via the top navigation:

| Panel | What it shows |
|---|---|
| **Live** | Sortable binding constraint table (WebSocket-fed) + grid map + shift factor bar chart |
| **What-If** | Bilateral injection calculator — type source node, sink node, MW → see flow impact |
| **Portfolio** | Upload positions (CSV or manual entry) → P&L risk aggregation against live constraints |
| **Replay** | DVR-style historical scrubber — pick a date/time, replay interval by interval |

### 5. Run the test suite

```bash
pip install pytest httpx      # not needed with `poetry install`, which includes the dev group
pytest tests/ -v
```

All tests should pass. The suite covers the SF solver, constraint normalizer, regime
detector, scraper column mappings, and API route registration.

---

## Key API endpoints

The full interactive docs are at `/docs`. Highlights:

```
# Market state
GET  /api/v1/constraints/live          Live priced constraints with shadow prices (loading % only if published)
GET  /api/v1/constraints/near-limit    Constraints at ≥ 95% of thermal limit (empty unless loading is published)
GET  /api/v1/status                    Server readiness + cache state + disk usage

# Shift factors
GET  /api/v1/shift-factors?node=X&constraint=Y          Single SF lookup
GET  /api/v1/shift-factors/table?constraint=X&limit=50  Exposure-sorted SF table

# Trading tools
POST /api/v1/flow-impact               Bilateral injection what-if
GET  /api/v1/mcc?node=X               Pre-clearing MCC estimate

# Portfolio
POST /api/v1/positions                Upload positions
GET  /api/v1/portfolio/risk?constraint=X  Aggregate P&L risk

# Historical replay
GET  /api/v1/replay/intervals?start=&end=    Available interval timestamps
GET  /api/v1/replay/snapshot?interval=       Full constraint state at a point in time
GET  /api/v1/replay/pnl?start=&end=          Historical P&L time series for positions

# Validation
GET  /api/v1/validation/summary                One-stop SF quality check
GET  /api/v1/validation/mcc-backtest?days=7    Full MCC back-prediction test
GET  /api/v1/validation/regime-consistency     SF stability within binding regimes
GET  /api/v1/validation/bounds-check           Physical plausibility (|SF| ≤ 1)

# WebSocket feeds (for live UI updates)
WS   /ws/constraints    Push stream — new constraint snapshot every 5-min interval
WS   /ws/portfolio      Push stream — live portfolio P&L recalculated each interval
```

---

## How the shift factors are calculated

### The math

For every 5-minute RTBM interval SPP publishes:
- **Shadow prices** per constraint (from the binding constraint CSV). Binding constraints carry **negative** shadow prices.
- **MCC** per settlement location (from the LMP CSV)

These satisfy the identity (no sign flip — verified at R² ≈ 0.999 on real data):

```
MCC[node, t] = Σ_k ( SF[node, k] × ShadowPrice[k, t] )
```

Shift factors are a property of the network, so **all intervals in the window are pooled**
into one regression, with a shadow price of 0 whenever a constraint is not binding:
- `A` (T × K): shadow price matrix — one row per interval, one column per constraint
- `B` (T × N): observed MCC — one column per node
- `X` (K × N): the unknown shift factors

`A X ≈ B` is solved for all nodes at once by least squares with a very light ridge penalty
(`ShiftFactorCalculator`). Constraints that bind in fewer than `min_intervals` (default 10) intervals are
not estimated and are listed in `unmodeled_constraints`.

> An earlier version split intervals by the *exact set* of binding constraints and solved each set
> separately. On real SPP data that solved almost nothing (on 2026-10-10: 231 distinct sets in 286
> intervals, none with 10 intervals), so it was replaced.

### When does it re-solve?

On startup the cache is seeded from up to 7 days of stored history. After that, whenever the set of
binding constraints changes the solver re-runs over the rolling 7-day window, in a worker thread so the
API stays responsive. A pooled fit over ~2,000 intervals, ~60 constraints and ~1,600 nodes takes about a
second.

### What it cannot do

A shift factor can only be estimated for a constraint that has **already bound** in the window. A constraint
that has just started binding contributes congestion the estimate cannot see. On 2026-10-09/10 that was
37% of shadow-price magnitude (see below). Closing that gap needs network topology (DC power flow), not
more regression. `GET /api/v1/mcc` reports `shadow_price_coverage` so you can see how much of the
current congestion the estimate actually covers.

---

## How the shift factors are validated

> "If your software can prove its predictive accuracy historically, I'll trust it blindly when the screen turns red."

We use SPP's published MCC as ground truth. Only an **out-of-sample** test says anything about predictive accuracy.

### Primary test: out-of-sample hold-out (`GET /api/v1/validation/mcc-holdout`)

Fit on the first part of the history, then reconstruct MCC on later intervals the fit never saw:
```
predicted_MCC = Σ( SF_estimated[node, constraint] × shadow_price_published[constraint] )
actual_MCC    = MCC published by SPP in the LMP file
```
It returns in-sample and out-of-sample R² / MAE, a "predict the mean" baseline, and `unmodeled_share`
(the part of out-of-sample shadow-price magnitude on constraints that never bound during training).

### Secondary checks

| Check | Endpoint | What it tells you |
|---|---|---|
| In-sample reconstruction | `/api/v1/validation/mcc-backtest` | Arithmetic sanity and problem nodes. **In-sample**, so it cannot show predictive accuracy. |
| Fit diagnostics | `/api/v1/validation/regime-consistency` | In-sample R², plus design-matrix rank / condition number (collinear constraints give great R² and meaningless individual shift factors). |
| Bounds check | `/api/v1/validation/bounds-check` | Flags `\|SF\| > 1`. Contingency-based flowgates can legitimately exceed 1, so treat as a prompt to look. |

### Measured accuracy

Measured on real SPP data, 2026-10-05 to 2026-10-10 (RT constraint files + 5-minute LMP files); trained on
10-05..10-08 (59 constraints modeled, 1,620 nodes), scored on the **unseen** days 10-09 and 10-10:

| Scope | R² | MAE ($/MWh) | Predict-zero MAE |
|---|---|---|---|
| All test intervals | 0.63 | 6.03 | 12.96 |
| Only intervals where every binding constraint was seen in training (25% of intervals) | 0.98 | 0.38 | 3.99 |
| Intervals with at least one never-seen binding constraint (75%) | 0.61 | 7.86 | 15.87 |

Takeaways: where the constraints are known the estimate is accurate; most of the error comes from newly
binding constraints, which have no shift factors. Shift factors of known constraints are fairly stable over
days (median |ΔSF| ≈ 0.001), though ~30% of constraints' factors correlate below 0.9 between a 4-day fit and a
later 2-day refit. Longer training windows and a lower `min_intervals` helped modestly (5-day training beat
1-day by ~0.09 R²). These are six days of one season, not a general guarantee — re-measure with
`/mcc-holdout` on your own history.

---

## Data storage

The platform has two storage layers:

### In-memory (process lifetime)
- **SF cache** (`src/api/sf_cache.py`): current `ShiftFactorSolution` + latest shadow prices
- **Rolling history**: 7-day window of constraint and LMP data used for regression
- **RAM footprint**: not measured; the 7-day LMP window is ~2M+ rows of pandas, so expect it to be larger than the Parquet size by a wide margin

### On-disk Parquet (persistent)
Written every 5 minutes to `src/data/timeseries/`:

```
src/data/timeseries/
  constraints/    constraints_YYYYMMDD.parquet   (~1.2 MB/day)
  lmp/            lmp_YYYYMMDD.parquet           (~14 MB/day)
  shift_factors/  shift_factors_YYYYMMDD.parquet (~5–50 MB/day)
```

**Retention policy**: Files older than **90 days** are automatically deleted once per day
(triggered by the ingest worker). At steady state, total disk usage is ~1.5–6 GB.

Check current disk usage:
```bash
# Via API
curl http://localhost:8000/api/v1/status | python -m json.tool

# Or from Python
from src.storage.duckdb_store import store
print(store.storage_stats())
```

---

## Security note on grid topology

The Leaflet map uses a **synthetic schematic** of the SPP transmission network —
not a map of real substation locations. Node positions are:

1. Derived from sub-state regional centroids (e.g. "western Kansas", "northern Oklahoma")
2. Offset by ±25 miles of random jitter
3. Rounded to 1 decimal degree (~7 miles resolution)

This is equivalent to a schematic diagram: accurate enough to show *which region of SPP*
is experiencing congestion, but deliberately unsuitable for locating physical infrastructure.
The topology was generated by `scripts/generate_spp_topology.py`.

---

## Project structure

```
grid-analysis/
├── src/
│   ├── analysis/
│   │   ├── multi_constraint_sf.py      # SF solver (OLS regression)
│   │   ├── constraint_normalizer.py    # SPP name canonicalization
│   │   ├── shift_factor_regimes.py     # Regime grouping
│   │   └── ...
│   ├── api/
│   │   ├── server.py                   # FastAPI app factory
│   │   ├── ingest_worker.py            # Background scrape → cache → broadcast loop
│   │   ├── sf_cache.py                 # In-memory SF + shadow price singleton
│   │   └── routers/
│   │       ├── constraints.py          # /api/v1/constraints/*
│   │       ├── shift_factors.py        # /api/v1/shift-factors/*
│   │       ├── flow_impact.py          # /api/v1/flow-impact
│   │       ├── mcc.py                  # /api/v1/mcc
│   │       ├── positions.py            # /api/v1/positions, /api/v1/portfolio/risk
│   │       ├── replay.py               # /api/v1/replay/*
│   │       ├── validation.py           # /api/v1/validation/*
│   │       └── websocket.py            # /ws/constraints, /ws/portfolio
│   ├── scrapers/
│   │   ├── spp_rt_constraints_scraper.py   # RTBM binding constraint CSV
│   │   ├── spp_rtlmp_scraper.py            # RTBM LMP by settlement location
│   │   └── ...
│   └── storage/
│       └── duckdb_store.py             # Parquet time-series store (90-day retention)
├── ui/                                 # Trading Dashboard (plain HTML + JS)
│   ├── index.html
│   ├── css/dashboard.css
│   ├── data/spp_only_lines.geojson     # Synthetic SPP grid schematic
│   └── js/
│       ├── app.js                      # Bootstrap + navigation
│       ├── constraints_table.js        # Live constraint table (WebSocket)
│       ├── grid_map.js                 # Leaflet map
│       ├── sf_panel.js                 # Shift factor bar chart
│       ├── flow_impact_calculator.js   # Bilateral injection what-if
│       ├── portfolio_panel.js          # Position upload + P&L table
│       └── replay_player.js            # DVR scrubber
├── scripts/
│   └── generate_spp_topology.py        # One-time synthetic GeoJSON generator
├── tests/                              # unit + regression tests
└── pyproject.toml
```

---

## Roadmap & Status

All core architecture phases are complete:

| Phase | Feature | Status |
|---|---|---|
| **1–8** | Core analytical libraries (SF solver, scrapers, normalizer, DuckDB time-series store) | ✅ Complete |
| **9–12** | FastAPI server, WebSocket feeds, Trading Dashboard UI, Portfolio risk engine, and historical DVR replay player | ✅ Complete |
| **13** | **Shadow price forecasting** — Gradient Boosting models with AutoML-based configuration selection (loss and target transform tuning) and persistent fallback logic | ✅ Complete |
| **14** | **LLM constraint resolver** — Automated resolution of ambiguous or truncated constraint names using structured LLM classification | ✅ Complete |
| **15** | **MCP server exposure** — TypeScript/Node-based MCP server exposing key analytical tools (shift factor solver, bilateral flow impact, portfolio risk) to MCP clients | ✅ Complete |
| **16** | **Multi-ISO Expansion** — Extended framework for multi-market trading:<br/>• **SPP**: Real-time & day-ahead constraints, LMPs, shift factors, and ML forecasters.<br/>• **MISO**: Real-time constraints, LMPs, and footprint cache isolation (day-ahead/forecasting pending). | ✅ Complete |
| **17** | **DART Congestion Arbitrage Engine** — Implemented Day-Ahead constraint ingestion, manifest tracking, rolling 30-day DART spreads ($RTBM - DAM$), background scraping, and a dedicated UI settings & matrix dashboard panel with interactive Chart.js visualization | ✅ Complete |

### Future Roadmap

1. **MISO Day-Ahead Scraper Integration**: Build adapter for MISO Day-Ahead binding constraints files to feed into `DuckDBStore` and complete MISO DART analytics.
2. **Forecast-Driven Arbitrage Quality**: Cross-reference calculated DART spreads with predicted future shadow prices from the Gradient Boosted ML forecasters to refine the arbitrage opportunity signals.
3. **Grid Outage Topology Modifiers**: Integrate physical grid changes (such as active CROW outages and flowgate modifiers) directly into the statistical shift factor solver.

