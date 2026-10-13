from datetime import datetime

from src.scrapers.spp_dalmp_scraper import DALMPScraper


def test_da_lmp_url_matches_the_live_portal_layout():
    """The dataset slug and file name were both wrong (every request 404'd, silently)."""
    url = DALMPScraper()._build_url(datetime(2026, 10, 10))
    assert url == (
        "https://portal.spp.org/file-browser-api/download/da-lmp-by-settlement-location"
        "?path=/2026/10/By_Day/DA-LMP-SL-202610100100.csv"
    )
