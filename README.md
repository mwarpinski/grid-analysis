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
MCC_node = -Σ ( GSF[node, constraint] × ShadowPrice[constraint] )
```

SPP does not publish GSFs in real time. This platform derives them statistically from historical binding constraint and LMP data, caches them in memory, and re-solves automatically when the set of binding constraints changes.

### What the platform gives you

| Capability | Description |
|---|---|
| **Live constraint feed** | Auto-refreshing table of every binding RTBM constraint with shadow price, loading %, and alarm state |
| **Shift factor exposure** | GSF × ShadowPrice for every node, ranked by impact — the trader's key exposure metric |
| **Pre-clearing MCC** | Estimated congestion component of LMP computed in-memory before SPP publishes it |
| **Bilateral flow impact** | "If I inject 50 MW at Node A and buy at Node B, how much flow does that add to Constraint X?" |
| **Portfolio P&L risk** | Aggregate all positions against current shift factors → net MW contribution + $/hr run rate |
| **DVR replay** | Scrub back through historical intervals interval-by-interval to backtest positions |
| **SF validation** | MCC back-prediction test: compare our GSFs to SPP's published ground truth |
| **Grid map** | Leaflet map of the SPP footprint with lines colored by thermal loading % |

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
pytest tests/ -v
```

All 70 tests should pass. The suite covers the SF solver, constraint normalizer, regime
detector, scraper column mappings, and API route registration.

---

## Key API endpoints

The full interactive docs are at `/docs`. Highlights:

```
# Market state
GET  /api/v1/constraints/live          Live binding constraints with shadow prices + loading %
GET  /api/v1/constraints/near-limit    Constraints at ≥ 95% of thermal limit
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

For each 5-minute RTBM interval where the same set of constraints is binding (a **constraint regime**), SPP publishes:
- **Shadow prices** per constraint (from the binding constraint CSV)
- **MCC** per settlement location (from the LMP CSV)

These satisfy the identity:

```
MCC[node, t] = Σ_k ( SF[node, k] × ShadowPrice[k, t] )
```

With T intervals in a regime and K binding constraints, this is a linear system
`A × x ≈ b` where:
- `A` (T × K): shadow price matrix — each row is one interval's shadow prices
- `b` (T × 1): MCC values for one node across those intervals
- `x` (K × 1): the unknown shift factors SF[node, constraint_1..K]

We solve this with ordinary least squares (`numpy.linalg.lstsq`) for every node
that has LMP data in those intervals. The solver runs once per regime and is
cached in memory — subsequent intervals in the same regime only update shadow prices.

### When does it re-solve?

The **regime detector** tracks the frozenset of binding constraint names. When
the set changes (a new constraint binds or an existing one falls out), a new
regime is detected and the solver re-runs automatically on the rolling 7-day
history window. This happens asynchronously in the background ingest worker
and typically completes in under 500 ms.

---

## How the shift factors are validated

> "If your software can prove its predictive accuracy historically, I'll trust it blindly when the screen turns red."

We validate using SPP's own published data as ground truth, which means the test is completely independent of our estimation approach.

### Primary test: MCC back-prediction (`GET /api/v1/validation/mcc-backtest`)

The key identity is:
```
predicted_MCC = Σ( SF_estimated[node, constraint] × shadow_price_published[constraint] )
actual_MCC    = MCC published by SPP in the LMP file
```

If our shift factors are correct, `predicted_MCC ≈ actual_MCC` for every node in every interval. The endpoint computes this for the past N days (default 7) of Parquet history and returns:

- **Overall R²** — should be > 0.95 for a well-solved regime
- **MAE per MWh** — typical target < $0.10/MWh; investigate if > $0.50
- **Per-node breakdown** — sorted by mean absolute error to identify problem nodes
- **Verdict** — `EXCELLENT / ACCEPTABLE / INVESTIGATE`

This test becomes meaningful after ~1 day of runtime (enough intervals to have statistical power). After 7 days it's fully diagnostic.

### Secondary tests

| Test | Endpoint | What it catches |
|---|---|---|
| **Regime consistency** | `/api/v1/validation/regime-consistency` | SF instability within a regime (topology change mid-regime, noisy data) |
| **Bounds check** | `/api/v1/validation/bounds-check` | `\|SF\| > 1.0` — physically impossible, indicates data quality issues |
| **Summary** | `/api/v1/validation/summary` | All three checks at once, lightweight |

### Expected accuracy

After the solver has seen ≥ 10 intervals per regime (the minimum for the OLS system to be over-determined):

| Node type | Typical R² | Typical MAE |
|---|---|---|
| Nodes near binding constraint | 0.95–0.99 | $0.05–$0.15/MWh |
| Nodes far from constraint | 0.80–0.95 | $0.10–$0.40/MWh |
| Regime with K=1 constraint | 0.99+ | < $0.05/MWh |
| Multi-constraint regime (K>5) | 0.85–0.95 | $0.10–$0.30/MWh |

Nodes with MAE > $1.00/MWh are flagged as problem nodes and returned in `worst_nodes`.

---

## Data storage

The platform has two storage layers:

### In-memory (process lifetime)
- **SF cache** (`src/api/sf_cache.py`): current `ShiftFactorSolution` + latest shadow prices
- **Rolling history**: 7-day window of constraint and LMP data used for regression
- **RAM footprint**: ~60–120 MB

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
├── tests/                              # 70 unit tests
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

