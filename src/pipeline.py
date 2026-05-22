"""
End-to-End Transmission Congestion Analysis Pipeline.

This orchestrates the full workflow:

  1. Scrape  → Pull RT/DA constraints + LMP data for a date range
  2. Normalize → Standardize constraint names via rule-based normalizer
  3. Register  → Resolve raw names to canonical IDs (fuzzy + registry)
  4. Regimes   → Group intervals by binding constraint sets
  5. Solve     → Estimate shift factors via cross-interval regression
  6. Locate    → Map canonical constraints to EIA GeoJSON lines
  7. Export    → Write shift_factors.csv, canonical_constraints.csv,
                 constraint_map.geojson

Usage (CLI)
-----------
::

    # Run for the last 7 days of RT data
    python -m src.pipeline --market rt --days 7

    # Run for a specific date range
    python -m src.pipeline \\
        --market rt \\
        --start 2024-11-01 \\
        --end 2024-11-30 \\
        --output-dir src/data/output

Usage (Python API)
------------------
::

    from src.pipeline import Pipeline, PipelineConfig

    config = PipelineConfig(
        market="rt",
        start_date=datetime(2024, 11, 1),
        end_date=datetime(2024, 11, 30),
        output_dir=Path("src/data/output"),
    )
    pipeline = Pipeline(config)
    results = pipeline.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""

    # Market: 'rt' or 'da'
    market: str = "rt"

    # Date range
    start_date: datetime = field(
        default_factory=lambda: datetime.now() - timedelta(days=7)
    )
    end_date: datetime = field(default_factory=datetime.now)

    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("src/data/output"))

    # Data directory for inputs (GeoJSON, price nodes, etc.)
    data_dir: Path = field(default_factory=lambda: Path("src/data"))

    # Constraint registry DB path
    registry_db: Path = field(
        default_factory=lambda: Path("src/data/constraint_registry.db")
    )

    # Shift factor solver settings
    min_intervals_per_regime: int = 10
    shadow_price_tolerance: float = 1e-4

    # Whether to skip scraping and use cached CSVs already on disk
    use_cached: bool = False

    # Whether to map constraints to EIA GeoJSON (requires spp_only_lines.geojson)
    map_to_geojson: bool = True

    # Logging level
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Pipeline results
# ---------------------------------------------------------------------------

@dataclass
class PipelineResults:
    """Output from a pipeline run."""
    shift_factors_df: pd.DataFrame
    canonical_constraints_df: pd.DataFrame
    regime_summary_df: pd.DataFrame
    pending_review: pd.DataFrame
    output_dir: Path

    def summarize(self) -> None:
        print("\n" + "=" * 64)
        print("PIPELINE RESULTS SUMMARY")
        print("=" * 64)
        print(f"Shift factors computed:     {len(self.shift_factors_df):>8}")
        print(f"Canonical constraints:       {len(self.canonical_constraints_df):>8}")
        print(f"Regimes analyzed:            {len(self.regime_summary_df):>8}")
        print(f"Pending LLM/manual review:   {len(self.pending_review):>8}")
        print(f"Output written to:           {self.output_dir}")
        print("=" * 64)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Orchestrates the full transmission congestion analysis workflow.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._setup_logging()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> PipelineResults:
        """Execute the full pipeline and return results."""
        cfg = self.config
        logger.info(
            f"Starting pipeline: market={cfg.market}, "
            f"{cfg.start_date.date()} → {cfg.end_date.date()}"
        )

        # ── Step 1: Scrape or load cached data ──────────────────────────
        constraints_df, lmp_df = self._step_acquire_data()

        if constraints_df.empty or lmp_df.empty:
            logger.error("No data acquired. Aborting pipeline.")
            return PipelineResults(
                shift_factors_df=pd.DataFrame(),
                canonical_constraints_df=pd.DataFrame(),
                regime_summary_df=pd.DataFrame(),
                pending_review=pd.DataFrame(),
                output_dir=cfg.output_dir,
            )

        # ── Step 2: Normalize constraint names ──────────────────────────
        logger.info("Step 2: Normalizing constraint names...")
        constraints_df = self._step_normalize(constraints_df)

        # ── Step 3: Register & resolve constraint names ──────────────────
        logger.info("Step 3: Resolving constraint names via registry...")
        constraints_df, pending_review = self._step_register(constraints_df)

        # ── Step 4: Analyze regimes ──────────────────────────────────────
        logger.info("Step 4: Analyzing constraint regimes...")
        regime_summary_df = self._step_analyze_regimes(constraints_df)

        # ── Step 5: Calculate shift factors ─────────────────────────────
        logger.info("Step 5: Calculating shift factors...")
        shift_factors_df = self._step_calculate_shift_factors(
            constraints_df, lmp_df
        )

        # ── Step 6: Map to EIA GeoJSON ───────────────────────────────────
        if cfg.map_to_geojson:
            logger.info("Step 6: Mapping constraints to EIA GeoJSON...")
            self._step_locate_constraints(constraints_df)

        # ── Step 7: Export ───────────────────────────────────────────────
        logger.info("Step 7: Writing outputs...")
        canonical_constraints_df = self._step_export(
            shift_factors_df, constraints_df, regime_summary_df, pending_review
        )

        results = PipelineResults(
            shift_factors_df=shift_factors_df,
            canonical_constraints_df=canonical_constraints_df,
            regime_summary_df=regime_summary_df,
            pending_review=pending_review,
            output_dir=cfg.output_dir,
        )
        results.summarize()
        return results

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _step_acquire_data(self):
        """Scrape or load cached constraint + LMP data."""
        cfg = self.config

        if cfg.use_cached:
            return self._load_cached_data()

        if cfg.market == "rt":
            return self._scrape_rt_data()
        elif cfg.market == "da":
            return self._scrape_da_data()
        else:
            raise ValueError(f"Unknown market: {cfg.market!r}. Use 'rt' or 'da'.")

    def _scrape_rt_data(self):
        """Scrape RTBM binding constraints + LMP for the configured date range."""
        from .scrapers.spp_rt_constraints_scraper import RTConstraintsScraper
        from .scrapers.spp_rtlmp_scraper import RTBMLMPScraper

        logger.info("Scraping RT binding constraints...")
        bc_scraper = RTConstraintsScraper()
        constraints_df = bc_scraper.scrape_days_in_range(
            self.config.start_date, self.config.end_date
        )

        if not constraints_df.empty and "State" in constraints_df.columns:
            constraints_df = constraints_df[constraints_df["State"] == "BINDING"]

        logger.info(
            f"  Got {len(constraints_df)} binding constraint rows "
            f"across {constraints_df['Interval'].nunique() if not constraints_df.empty else 0} intervals."
        )

        logger.info("Scraping RTBM LMPs...")
        lmp_scraper = RTBMLMPScraper()
        lmp_df = lmp_scraper.scrape_days_in_range(
            self.config.start_date, self.config.end_date
        )
        logger.info(f"  Got {len(lmp_df)} LMP rows.")

        # Normalize column names to standard schema
        constraints_df = self._normalize_columns(
            constraints_df,
            col_map={
                "Constraint Name": "constraint_name",
                "Shadow Price": "shadow_price",
                "Interval": "interval",
            },
        )
        lmp_df = self._normalize_columns(
            lmp_df,
            col_map={
                "Settlement Location": "settlement_location",
                "Interval": "interval",
                "MCC": "mcc",
            },
        )

        return constraints_df, lmp_df

    def _scrape_da_data(self):
        """Scrape DA binding constraints + LMP for the configured date range."""
        from .scrapers.spp_da_constraints_scraper import DAConstraintsScraper
        from .scrapers.spp_dalmp_scraper import DALMPScraper

        logger.info("Scraping DA binding constraints...")
        bc_scraper = DAConstraintsScraper()
        constraints_df = bc_scraper.scrape_days_in_range(
            self.config.start_date, self.config.end_date
        )

        logger.info("Scraping DA LMPs...")
        lmp_scraper = DALMPScraper()
        lmp_df = lmp_scraper.fetch_mcc_for_range(
            self.config.start_date, self.config.end_date
        )

        constraints_df = self._normalize_columns(
            constraints_df,
            col_map={
                "Constraint Name": "constraint_name",
                "Shadow Price": "shadow_price",
                "Interval": "interval",
            },
        )

        return constraints_df, lmp_df

    def _load_cached_data(self):
        """Load pre-downloaded CSV files from the data directory."""
        cfg = self.config
        constraints_path = cfg.data_dir / "constraints.csv"
        lmp_path = cfg.data_dir / "lmp.csv"

        if not constraints_path.exists() or not lmp_path.exists():
            raise FileNotFoundError(
                f"Cached data not found. Expected:\n"
                f"  {constraints_path}\n  {lmp_path}"
            )

        constraints_df = pd.read_csv(constraints_path)
        lmp_df = pd.read_csv(lmp_path)

        constraints_df = self._normalize_columns(
            constraints_df,
            col_map={
                "Constraint Name": "constraint_name",
                "Shadow Price": "shadow_price",
                "Interval": "interval",
            },
        )
        lmp_df = self._normalize_columns(
            lmp_df,
            col_map={
                "Settlement Location": "settlement_location",
                "Interval": "interval",
                "MCC": "mcc",
            },
        )

        return constraints_df, lmp_df

    def _step_normalize(self, constraints_df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized constraint name columns."""
        from .analysis.constraint_normalizer import normalize_constraints_df

        return normalize_constraints_df(
            constraints_df,
            name_col="constraint_name",
        )

    def _step_register(self, constraints_df: pd.DataFrame):
        """Resolve constraint names to canonical IDs via the registry."""
        from .analysis.constraint_registry import ConstraintRegistry

        registry = ConstraintRegistry(db_path=self.config.registry_db)
        raw_names = constraints_df["constraint_name"].unique().tolist()
        pending = registry.ingest_many(raw_names)

        # Add canonical_id column to constraints_df
        def get_canonical_id(name):
            result = registry.resolve(name)
            return result.canonical_id if result else None

        constraints_df = constraints_df.copy()
        constraints_df["canonical_id"] = constraints_df["constraint_name"].map(
            {n: (registry.resolve(n).canonical_id if registry.resolve(n) else None)
             for n in raw_names}
        )

        pending_df = registry.get_pending_review()
        registry.close()
        return constraints_df, pending_df

    def _step_analyze_regimes(self, constraints_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze and report on binding constraint regimes."""
        from .analysis.shift_factor_regimes import build_regimes, regime_report_to_dataframe

        report = build_regimes(
            constraints_df,
            tolerance=self.config.shadow_price_tolerance,
            min_intervals=self.config.min_intervals_per_regime,
            verbose=True,
        )
        return regime_report_to_dataframe(report)

    def _step_calculate_shift_factors(
        self,
        constraints_df: pd.DataFrame,
        lmp_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run the shift factor regression solver."""
        from .analysis.multi_constraint_sf import calculate_shift_factors

        solution = calculate_shift_factors(
            lmp_df=lmp_df,
            constraints_df=constraints_df,
            min_intervals=self.config.min_intervals_per_regime,
            tolerance=self.config.shadow_price_tolerance,
        )
        return solution.to_dataframe()

    def _step_locate_constraints(self, constraints_df: pd.DataFrame) -> None:
        """Map canonical constraints to EIA GeoJSON features."""
        from .analysis.constraint_locator import ConstraintLocator

        geojson_path = self.config.data_dir / "spp_only_lines.geojson"
        if not geojson_path.exists():
            logger.warning(
                f"spp_only_lines.geojson not found at {geojson_path}. "
                "Skipping GeoJSON mapping. Run src/analysis/spp_filters.py first."
            )
            return

        locator = ConstraintLocator(geojson_path=geojson_path)

        # Get unique canonical constraints
        canonical_constraints = (
            constraints_df[
                ["canonical_id", "canonical_name", "from_sub", "to_sub", "voltage_kv"]
            ]
            .drop_duplicates("canonical_id")
            .to_dict("records")
        )

        results = locator.locate_many(canonical_constraints)
        located = [r for r in results if r.feature_index is not None]

        output_path = self.config.output_dir / "constraint_map.geojson"
        locator.export_located_constraints(results, output_path)

        n_total = len(results)
        n_located = len(located)
        logger.info(
            f"Located {n_located}/{n_total} constraints in EIA GeoJSON "
            f"({100*n_located/max(1,n_total):.1f}% match rate). "
            f"GeoJSON → {output_path}"
        )

    def _step_export(
        self,
        shift_factors_df: pd.DataFrame,
        constraints_df: pd.DataFrame,
        regime_summary_df: pd.DataFrame,
        pending_review: pd.DataFrame,
    ) -> pd.DataFrame:
        """Write output files and return canonical_constraints DataFrame."""
        out = self.config.output_dir

        # Shift factors
        sf_path = out / "shift_factors.csv"
        shift_factors_df.to_csv(sf_path, index=False)
        logger.info(f"Wrote shift factors → {sf_path}")

        # Regime summary
        regime_path = out / "regime_summary.csv"
        regime_summary_df.to_csv(regime_path, index=False)
        logger.info(f"Wrote regime summary → {regime_path}")

        # Pending review
        pending_path = out / "pending_review.csv"
        pending_review.to_csv(pending_path, index=False)
        if len(pending_review) > 0:
            logger.warning(
                f"{len(pending_review)} constraint names need LLM/manual review. "
                f"See {pending_path}"
            )

        # Canonical constraints (unique)
        canonical_df = (
            constraints_df[
                ["canonical_id", "canonical_name", "equipment_type",
                 "from_sub", "to_sub", "voltage_kv", "fingerprint"]
            ]
            .drop_duplicates("canonical_id")
            .reset_index(drop=True)
        )
        canonical_path = out / "canonical_constraints.csv"
        canonical_df.to_csv(canonical_path, index=False)
        logger.info(f"Wrote canonical constraints → {canonical_path}")

        return canonical_df

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_columns(
        df: pd.DataFrame, col_map: dict
    ) -> pd.DataFrame:
        """Rename columns using the provided map (only renames what exists)."""
        existing = {k: v for k, v in col_map.items() if k in df.columns}
        return df.rename(columns=existing)

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="SPP Transmission Congestion Analysis Pipeline"
    )
    parser.add_argument(
        "--market", choices=["rt", "da"], default="rt",
        help="Market to analyze: 'rt' (real-time) or 'da' (day-ahead)"
    )
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of past days to analyze (default: 7)")
    parser.add_argument(
        "--output-dir", default="src/data/output",
        help="Output directory (default: src/data/output)"
    )
    parser.add_argument(
        "--use-cached", action="store_true",
        help="Skip scraping; use cached constraints.csv and lmp.csv"
    )
    parser.add_argument(
        "--min-intervals", type=int, default=10,
        help="Minimum intervals per regime for shift factor estimation"
    )
    parser.add_argument(
        "--no-geojson", action="store_true",
        help="Skip EIA GeoJSON mapping step"
    )
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    end_date = (
        datetime.strptime(args.end, "%Y-%m-%d") if args.end
        else datetime.now()
    )
    start_date = (
        datetime.strptime(args.start, "%Y-%m-%d") if args.start
        else end_date - timedelta(days=args.days)
    )

    config = PipelineConfig(
        market=args.market,
        start_date=start_date,
        end_date=end_date,
        output_dir=Path(args.output_dir),
        use_cached=args.use_cached,
        min_intervals_per_regime=args.min_intervals,
        map_to_geojson=not args.no_geojson,
        log_level=args.log_level,
    )

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    _main()
