from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import requests
from io import StringIO
from .spp_scraper import BaseSPPScraper

class RTBMLMPScraper(BaseSPPScraper):
    def __init__(self):
        super().__init__("https://portal.spp.org/file-browser-api/download/rtbm-lmp-by-location")

    def _build_url(self, date: datetime, type: str = "day") -> str:
        """Build URL for RTBM LMP data.

        SPP does not publish a daily LMP rollup — each 5-minute interval
        has its own file under By_Interval/<day>/.  The ``type`` parameter
        is accepted (to match BaseSPPScraper.fetch_data's call signature)
        but ignored; we always build a per-interval URL.
        """
        # Round down to nearest 5-minute mark so we always get the latest file
        rounded = date.replace(
            minute=(date.minute // 5) * 5, second=0, microsecond=0
        )
        path = (
            f"/{rounded.year}/{rounded.month:02d}"
            f"/By_Interval/{rounded.day:02d}"
            f"/RTBM-LMP-SL-{rounded.strftime('%Y%m%d%H%M')}.csv"
        )
        return f"{self.base_url}?path={path}"

    def _process_response(self, response: requests.Response) -> pd.DataFrame:
        """Process the RTBM LMP CSV response"""
        # Convert bytes to string and create StringIO object
        content = StringIO(response.content.decode('utf-8'))
        df = pd.read_csv(content)
        
        # Convert timestamp columns to datetime
        if 'Interval' in df.columns:
            df['Interval'] = pd.to_datetime(df['Interval'])
        if 'GMTIntervalEnd' in df.columns:
            df['GMTIntervalEnd'] = pd.to_datetime(df['GMTIntervalEnd'])
        return df

    def fetch_latest_lmps(self, max_lookback_intervals: int = 6) -> pd.DataFrame:
        """
        Fetch the most recent published RTBM LMP data.

        SPP labels files in Central time and publishes each interval 30-90 s
        after it ends, so the newest file is often not there yet; walk back up
        to ``max_lookback_intervals`` intervals until one exists.
        """
        now = datetime.now(ZoneInfo("America/Chicago")).replace(tzinfo=None)
        # Round down to nearest 5 minutes since SPP updates every 5 minutes
        current_time = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        for offset in range(max_lookback_intervals + 1):
            df = self.fetch_data(current_time - timedelta(minutes=5 * offset))
            if not df.empty:
                return df
        return pd.DataFrame()