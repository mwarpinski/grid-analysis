import asyncio
import logging
import io
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
import requests
import pandas as pd
import aiohttp
from bs4 import BeautifulSoup
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 30

class BaseSPPScraper(ABC):
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.logger = logging.getLogger(self.__class__.__name__)
        # Create data directory if it doesn't exist
        self.data_dir = Path("src/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _build_url(self, date: datetime) -> str:
        """Build the specific URL for the data type and date"""
        pass

    @abstractmethod
    def _process_response(self, response: requests.Response) -> pd.DataFrame:
        """Process the response into a pandas DataFrame"""
        pass

    def fetch_data(self, date: datetime, type: str = "day") -> pd.DataFrame:
        """Fetch data for a specific date"""
        url = self._build_url(date, type)
        self.logger.debug(f"Fetching data from {url}")
        try:
            # Without a timeout a stalled connection blocks the ingest thread forever.
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return self._process_response(response)
        except requests.HTTPError as e:
            # A 404 just means the file isn't published (yet); callers walk back.
            level = logging.DEBUG if e.response is not None and e.response.status_code == 404 else logging.ERROR
            self.logger.log(level, f"Error fetching data: {e}")
            return pd.DataFrame()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def scrape_intervals_for_day(self, date: datetime) -> pd.DataFrame:
        """Scrape data for an entire day in 5-minute intervals"""
        start_time = date.replace(hour=0, minute=5, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        current_time = start_time
        
        all_data = []
        while current_time < end_time:
            df = self.fetch_data(current_time)
            if not df.empty:
                all_data.append(df)
            current_time += timedelta(minutes=5)
            
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def scrape_day(self, date: datetime) -> pd.DataFrame:
        """Scrape data for an entire day in 5-minute intervals"""
        df = self.fetch_data(date)
        return df
    
    def scrape_days_in_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        all_data = []
        for date in pd.date_range(start=start_date, end=end_date):
            print(f"Scraping data for {date}")
            df = self.scrape_day(date)
            if not df.empty:
                all_data.append(df)
            
        return pd.concat(all_data) if all_data else pd.DataFrame()

    def save_data(self, df: pd.DataFrame, date: datetime, data_type: str):
        """Save data to CSV file in src/data directory"""
        if df.empty:
            self.logger.warning(f"No data to save for {date}")
            return
            
        filename = f"{data_type}_{date.strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved data to {filepath}")
