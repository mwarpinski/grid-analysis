"""
RT Binding Constraints Scraper for SPP RTBM.

Captures ALL fields from the RTBM binding constraint CSV, including the
critical line loading columns that feed into REQ-2 (bottleneck prediction):

  - Constraint Name      → the raw constraint identifier
  - Shadow Price         → $/MWh congestion cost (key for SF × SP exposure)
  - Limit                → thermal/stability limit in MW (the "100%")
  - Monitored Value      → actual flow in MW at this interval
  - Percent of Limit     → Monitored Value / Limit × 100  (the "98%" alarm)
  - State                → BINDING | NEAR_BINDING | VIOLATED
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
    "Constraint Name":   "constraint_name",
    "Contingency Name":  "contingency_name",
    "Shadow Price":      "shadow_price",
    "Limit":             "limit_mw",
    "Monitored Value":   "monitored_mw",
    "Percent":           "pct_of_limit",   # SPP sometimes labels this "Percent"
    "% of Limit":        "pct_of_limit",   # and sometimes "% of Limit"
    "State":             "state",
    "Interval":          "interval",
    "GMTIntervalEnd":    "gmt_interval_end",
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

        Computes ``pct_of_limit`` from ``monitored_mw / limit_mw`` when
        SPP doesn't include it directly (older file formats).
        """
        try:
            df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
        except Exception as e:
            logger.error(f"Failed to parse constraints CSV: {e}")
            return pd.DataFrame()

        # Rename to normalized snake_case
        df = df.rename(columns={
            k: v for k, v in _COLUMN_RENAME.items() if k in df.columns
        })

        # Compute pct_of_limit if missing (older SPP file format)
        if "pct_of_limit" not in df.columns:
            if "limit_mw" in df.columns and "monitored_mw" in df.columns:
                df["pct_of_limit"] = (
                    df["monitored_mw"].abs() / df["limit_mw"].abs() * 100.0
                ).where(df["limit_mw"].abs() > 0, other=None)
                logger.debug("Computed pct_of_limit from monitored_mw / limit_mw.")

        # Parse interval columns to datetime
        for col in ("interval", "gmt_interval_end"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

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