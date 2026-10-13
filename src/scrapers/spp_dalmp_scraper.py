"""
DA (Day-Ahead) LMP Scraper for SPP.

SPP publishes DA LMPs (Locational Marginal Prices) for the Day-Ahead market.
This scraper downloads the CSV files and normalizes them to the same schema
used by the RT LMP scraper (RTBMLMPScraper), so both can feed into the
same shift factor calculator.

SPP DA LMP URL pattern (verified against the live portal):
    https://portal.spp.org/file-browser-api/download/da-lmp-by-settlement-location
        ?path=/2026/10/By_Day/DA-LMP-SL-202610100100.csv

Columns in the DA LMP CSV:
    Interval, GMTIntervalEnd, Settlement Location, LMP, MLC, MCC, MEC

Note: DA LMPs are hourly (not 5-minute like RTBM).
"""

from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from .spp_scraper import BaseSPPScraper


class DALMPScraper(BaseSPPScraper):
    """
    Scrapes SPP Day-Ahead LMP data by settlement location.

    URLs follow the pattern:
        https://portal.spp.org/file-browser-api/download/da-lmp-by-settlement-location
            ?path=/{year}/{month:02d}/By_Day/DA-LMP-SL-{YYYYMMDD}0100.csv
    """

    def __init__(self) -> None:
        super().__init__(
            "https://portal.spp.org/file-browser-api/download/da-lmp-by-settlement-location"
        )

    def _build_url(self, date: datetime, type: str = "day") -> str:
        """Build the URL for the DA LMP file for a given date."""
        path = (
            f"/{date.year}/{date.month:02d}/By_Day/"
            f"DA-LMP-SL-{date.strftime('%Y%m%d')}0100.csv"
        )
        return f"{self.base_url}?path={path}"

    def _process_response(self, response: requests.Response) -> pd.DataFrame:
        """
        Parse the DA LMP CSV response and normalize column names.

        Normalizes to the shared schema:
            interval, gmt_interval_end, settlement_location, lmp, mlc, mcc, mec
        """
        content = StringIO(response.content.decode("utf-8"))
        df = pd.read_csv(content)

        # Normalize to datetime
        if "Interval" in df.columns:
            df["Interval"] = pd.to_datetime(df["Interval"])
        if "GMTIntervalEnd" in df.columns:
            df["GMTIntervalEnd"] = pd.to_datetime(df["GMTIntervalEnd"])

        return df

    def fetch_mcc_only(self, date: datetime) -> pd.DataFrame:
        """
        Convenience method to fetch DA LMPs for a date and return only the
        MCC component in the standard format used by ShiftFactorCalculator.

        Returns
        -------
        DataFrame with columns: ``interval``, ``settlement_location``, ``mcc``
        """
        df = self.scrape_day(date)
        if df.empty:
            return pd.DataFrame(columns=["interval", "settlement_location", "mcc"])

        mcc_df = df.rename(
            columns={
                "Interval": "interval",
                "Settlement Location": "settlement_location",
                "MCC": "mcc",
            }
        )[["interval", "settlement_location", "mcc"]]

        return mcc_df

    def fetch_mcc_for_range(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch DA MCC components for a date range.

        Returns
        -------
        DataFrame with columns: ``interval``, ``settlement_location``, ``mcc``
        """
        all_data = []
        for date in pd.date_range(start=start_date, end=end_date):
            self.logger.info(f"Fetching DA LMPs for {date.date()}")
            df = self.fetch_mcc_only(date.to_pydatetime())
            if not df.empty:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame(columns=["interval", "settlement_location", "mcc"])

        return pd.concat(all_data, ignore_index=True)
