"""
DA (Day-Ahead) Binding Constraints Scraper for SPP.

Mirrors the RTConstraintsScraper column normalization so both RT and DA
constraint data share an identical schema and can be fed into the same
shift factor calculator and portfolio risk engine.

DA BC URL pattern:
    https://portal.spp.org/file-browser-api/download/da-binding-constraints
        ?path=/2024/12/By_Day/DA-BC-202412010100.csv

Columns captured (normalized):
    interval, gmt_interval_end, constraint_name, contingency_name,
    shadow_price, limit_mw, monitored_mw, pct_of_limit, state
"""

from datetime import datetime
import io
import logging

import pandas as pd
import requests

from .spp_scraper import BaseSPPScraper
from .spp_rt_constraints_scraper import _COLUMN_RENAME, NEAR_BINDING_THRESHOLD_PCT

logger = logging.getLogger(__name__)


class DAConstraintsScraper(BaseSPPScraper):
    """
    Scrapes SPP Day-Ahead Binding Constraints data.

    URL pattern:
        https://portal.spp.org/file-browser-api/download/da-binding-constraints
            ?path=/{year}/{month:02d}/By_Day/DA-BC-{YYYYMMDD}0100.csv
    """

    def __init__(self) -> None:
        super().__init__(
            "https://portal.spp.org/file-browser-api/download/da-binding-constraints"
        )

    def _build_url(self, date: datetime, type: str = "day") -> str:
        """Build URL for DA binding constraints CSV."""
        path = (
            f"/{date.year}/{date.month:02d}/By_Day/"
            f"DA-BC-{date.strftime('%Y%m%d')}0100.csv"
        )
        return f"{self.base_url}?path={path}"

    def _process_response(self, response: requests.Response) -> pd.DataFrame:
        """
        Parse DA binding constraints CSV with the same normalization as RT.

        Computes ``pct_of_limit`` from ``monitored_mw / limit_mw`` when
        not directly provided.
        """
        try:
            df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
        except Exception as e:
            logger.error(f"Failed to parse DA constraints CSV: {e}")
            return pd.DataFrame()

        # Shared rename map with RT scraper
        df = df.rename(columns={
            k: v for k, v in _COLUMN_RENAME.items() if k in df.columns
        })

        # Compute pct_of_limit if missing
        if "pct_of_limit" not in df.columns:
            if "limit_mw" in df.columns and "monitored_mw" in df.columns:
                df["pct_of_limit"] = (
                    df["monitored_mw"].abs() / df["limit_mw"].abs() * 100.0
                ).where(df["limit_mw"].abs() > 0, other=None)

        # Parse intervals
        for col in ("interval", "gmt_interval_end"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Numeric coercions
        for col in ("shadow_price", "limit_mw", "monitored_mw", "pct_of_limit"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def get_binding_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only binding constraints."""
        if "state" not in df.columns:
            return df
        return df[df["state"].str.upper() == "BINDING"].copy()

    def get_near_limit(
        self,
        df: pd.DataFrame,
        threshold_pct: float = NEAR_BINDING_THRESHOLD_PCT,
    ) -> pd.DataFrame:
        """Return constraints at or above threshold_pct of their limit."""
        if "pct_of_limit" not in df.columns:
            return pd.DataFrame()
        return df[df["pct_of_limit"] >= threshold_pct].copy()