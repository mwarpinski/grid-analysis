from datetime import datetime
import logging
import pandas as pd
import requests
from typing import Optional
from .spp_scraper import BaseSPPScraper

logger = logging.getLogger(__name__)

class MISOAdapter(BaseSPPScraper):
    """
    Scraper / Adapter for MISO market data (LMP and binding constraints).
    Inherits from BaseSPPScraper for API signature compatibility.
    """
    def __init__(self):
        super().__init__("https://public-api.misoenergy.org/api")

    def _build_url(self, date: datetime, type: str = "lmp") -> str:
        """Build the specific API URL based on data type."""
        if type == "constraints":
            return f"{self.base_url}/BindingConstraints/RealTime"
        else:
            return f"{self.base_url}/MarketPricing/GetRealTimeFiveMinExPost/Current"

    def _process_response(self, response: requests.Response) -> pd.DataFrame:
        """Process the response - abstract method requirement."""
        # Overridden in fetch_data for JSON parsing
        return pd.DataFrame()

    def fetch_data(self, date: Optional[datetime] = None, type: str = "lmp") -> pd.DataFrame:
        """Fetch real-time LMP or constraint data from MISO public APIs."""
        url = self._build_url(date or datetime.utcnow(), type)
        self.logger.info(f"Fetching MISO data from {url}")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            if type == "constraints":
                return self._process_constraints(data)
            else:
                return self._process_lmp(data)
        except Exception as e:
            self.logger.error(f"Error fetching MISO data ({type}): {e}")
            return pd.DataFrame()

    def fetch_latest_lmps(self) -> pd.DataFrame:
        """Fetch the most recent MISO RT ex-post LMP data."""
        return self.fetch_data(type="lmp")

    def fetch_latest_constraints(self) -> pd.DataFrame:
        """Fetch the most recent MISO RT binding constraints."""
        return self.fetch_data(type="constraints")

    def _process_constraints(self, data: dict) -> pd.DataFrame:
        """Process constraints JSON response to standard columns."""
        constraints = data.get("Constraint", [])
        if not constraints:
            return pd.DataFrame()
            
        rows = []
        for c in constraints:
            rows.append({
                "constraint_name": c.get("Name"),
                "interval": c.get("Period"),
                "shadow_price": c.get("Price"),
                "state": "BINDING",  # This endpoint only publishes binding/active constraints
                "limit_mw": None,
                "monitored_mw": None,
                "pct_of_limit": 100.0,
                "contingency_name": None
            })
            
        df = pd.DataFrame(rows)
        # Parse numeric columns
        df["shadow_price"] = pd.to_numeric(df["shadow_price"], errors="coerce").abs()
        # Parse intervals and normalize timezone (EST/EDT to UTC timezone-naive)
        df["interval"] = pd.to_datetime(df["interval"])
        try:
            df["interval"] = df["interval"].dt.tz_localize("America/New_York").dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            pass
            
        return df

    def _process_lmp(self, data: dict) -> pd.DataFrame:
        """Process LMP JSON response to standard columns."""
        headers = data.get("headers", [])
        records = data.get("data", [])
        if not headers or not records:
            return pd.DataFrame()
            
        df = pd.DataFrame(records, columns=headers)
        
        # Rename to standard layout
        col_rename = {
            "INTERVAL": "interval",
            "CPNODE": "settlement_location",
            "LMP": "lmp",
            "MLC": "mlc",
            "MCC": "mcc"
        }
        df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})
        
        # Parse numeric columns
        for col in ["lmp", "mlc", "mcc"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        # Parse intervals and normalize timezone (EST/EDT to UTC timezone-naive)
        df["interval"] = pd.to_datetime(df["interval"])
        try:
            df["interval"] = df["interval"].dt.tz_localize("America/New_York").dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            pass
            
        return df
