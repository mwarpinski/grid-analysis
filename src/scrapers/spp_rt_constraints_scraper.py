"""
RT Binding Constraints Scraper for SPP RTBM.

Captures ALL fields from the RTBM binding constraint CSV, including the
critical line loading columns that feed into REQ-2 (bottleneck prediction):

  - Constraint Name      → the raw constraint identifier
  - Shadow Price         → $/MWh congestion cost (key for SF × SP exposure)
  - Limit                → thermal/stability limit in MW (the "100%")
  - Monitored Value      → actual flow in MW at this interval
  - Percent of Limit     → Monitored Value / Limit × 100  (the "98%" alarm)
  - State                → BINDING | BREACHED | ACTIVATED (priced rows are BINDING or BREACHED)
  - Contingency Name     → the N-1 contingency this constraint monitors

These columns allow real-time line loading % alerts ("ticking time bomb" alarm)
without requiring private state estimator data.
"""

from datetime import datetime
import io
import logging

import pandas as pd
import requests

from .spp_scraper import BaseSPPScraper

logger = logging.getLogger(__name__)

# Columns SPP publishes in RTBM binding constraint CSVs.
# Kept here so we can rename them to consistent snake_case in the output.
_COLUMN_RENAME = {
    # Core fields
    "Constraint Name":           "constraint_name",
    "Constraint Type":           "constraint_type",       # FG, M2M, Flowgate, etc.
    "Contingency Name":          "contingency_name",
    "Shadow Price":              "shadow_price",
    "Limit":                     "limit_mw",
    "Monitored Value":           "monitored_mw",
    # Physical equipment — key for geo-lookup and map display
    # e.g. "LN SPLIT_R - SIOUXFL" or "XFMR MIDJ - MIDJ"
    "Monitored Facility":        "monitored_facility",
    "Contingent Facility":       "contingent_facility",
    # Public NERC facility ID — cross-reference with NERC/EIA databases
    "NERCID":                    "nerc_id",
    # Loading %
    "Percent":                   "pct_of_limit",
    "% of Limit":                "pct_of_limit",
    # Status / timing
    "State":                     "state",
    "TLR Level":                 "tlr_level",
    "Interconnect":              "interconnect",
    "Interval":                  "interval",
    "GMTIntervalEnd":            "gmt_interval_end",
    # Daily BC aggregate file extras
    "Source Limit":              "source_limit_mw",
    "Real Time Effective Limit": "rt_effective_limit_mw",
    "Initial Effective Limit":   "initial_effective_limit_mw",
}

# Threshold below which a "near binding" warning is surfaced (%)
NEAR_BINDING_THRESHOLD_PCT = 95.0


class RTConstraintsScraper(BaseSPPScraper):
    """
    Scrapes SPP RTBM Binding Constraints data.

    Supports two URL modes:
    - ``type="day"``      → daily aggregate CSV  (RTBM-DAILY-BC-YYYYMMDD.csv)
    - ``type="interval"`` → per-5-min-interval CSV (RTBM-BC-YYYYMMDDhhmm.csv)

    Output DataFrame columns (normalized):
        interval, gmt_interval_end, constraint_name, contingency_name,
        shadow_price, limit_mw, monitored_mw, pct_of_limit, state
    """

    def __init__(self) -> None:
        super().__init__(
            "https://portal.spp.org/file-browser-api/download/rtbm-binding-constraints"
        )

    def _build_url(self, date: datetime, type: str = "day") -> str:
        """Build URL for constraints data."""
        if type == "day":
            path = (
                f"/{date.year}/{date.month:02d}/By_Day/"
                f"RTBM-DAILY-BC-{date.strftime('%Y%m%d')}.csv"
            )
        else:
            path = (
                f"/{date.year}/{date.month:02d}/By_Interval/{date.day:02d}/"
                f"RTBM-BC-{date.strftime('%Y%m%d%H%M')}.csv"
            )
        return f"{self.base_url}?path={path}"

    def _process_response(self, response: requests.Response) -> pd.DataFrame:
        """
        Parse the constraints CSV and normalize all columns.

        ``pct_of_limit`` / ``monitored_mw`` are NaN unless SPP supplies them.
        """
        try:
            df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
        except Exception as e:
            logger.error(f"Failed to parse constraints CSV: {e}")
            return pd.DataFrame()

        # SPP headers occasionally carry stray whitespace (e.g. " Contingency Name")
        df.columns = [str(c).strip() for c in df.columns]

        # Rename to normalized snake_case
        df = df.rename(columns={
            k: v for k, v in _COLUMN_RENAME.items() if k in df.columns
        })

        # Ensure limit_mw is populated
        if "limit_mw" not in df.columns or df["limit_mw"].isnull().all():
            if "rt_effective_limit_mw" in df.columns:
                df["limit_mw"] = df["rt_effective_limit_mw"]
            elif "source_limit_mw" in df.columns:
                df["limit_mw"] = df["source_limit_mw"]

        # Parse interval columns to datetime early for simulation seed
        for col in ("interval", "gmt_interval_end"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # SPP's RTBM binding-constraint files do NOT publish loading (no Percent /
        # Monitored Value columns), so ``pct_of_limit`` and ``monitored_mw`` are
        # left empty rather than invented.  They were previously fabricated from
        # a hash of the constraint name and a sine wave, and that fake data fed
        # alerts, the map, volatility scores and the forecaster.
        for col in ("pct_of_limit", "monitored_mw"):
            if col not in df.columns:
                df[col] = float("nan")

        # Numeric coercions
        for col in ("shadow_price", "limit_mw", "monitored_mw", "pct_of_limit"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ------------------------------------------------------------------
    # Convenience filters (used by the pipeline and UI)
    # ------------------------------------------------------------------

    def get_binding_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only rows where state == 'BINDING'."""
        if "state" not in df.columns:
            return df
        return df[df["state"].str.upper() == "BINDING"].copy()

    def get_near_limit(
        self,
        df: pd.DataFrame,
        threshold_pct: float = NEAR_BINDING_THRESHOLD_PCT,
    ) -> pd.DataFrame:
        """
        Return constraints at or above ``threshold_pct`` of their limit.

        This is the "ticking time bomb" list — lines running hot that are
        about to bind.  Default threshold: 95%.
        """
        if "pct_of_limit" not in df.columns:
            logger.warning("pct_of_limit column not available — cannot filter near-limit.")
            return pd.DataFrame()
        return df[df["pct_of_limit"] >= threshold_pct].copy()

    def get_loading_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a summary table sorted by loading % descending.

        Columns: constraint_name, state, shadow_price, limit_mw,
                 monitored_mw, pct_of_limit, near_limit_alarm
        """
        cols = [c for c in [
            "constraint_name", "state", "shadow_price",
            "limit_mw", "monitored_mw", "pct_of_limit",
        ] if c in df.columns]

        summary = df[cols].copy()

        if "pct_of_limit" in summary.columns:
            summary["near_limit_alarm"] = (
                summary["pct_of_limit"] >= NEAR_BINDING_THRESHOLD_PCT
            )
            summary = summary.sort_values("pct_of_limit", ascending=False)

        return summary.reset_index(drop=True)


class MonitoredFlowgateScraper(RTConstraintsScraper):
    """
    Scraper designed for retrieving and processing SPP real-time monitored flowgates.
    Extends RTConstraintsScraper to reuse the underlying API download endpoints.
    """
    pass