import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.scrapers.spp_rt_constraints_scraper import RTConstraintsScraper, MonitoredFlowgateScraper

class DummyResponse:
    def __init__(self, content):
        self.content = content

def test_rt_constraints_scraper_limit_mapping():
    # Test that RTConstraintsScraper correctly maps rt_effective_limit_mw to limit_mw
    csv_data = (
        "Constraint Name,State,Interval,Real Time Effective Limit\n"
        "LN HARBINE - BEATRICE,ACTIVATED,2026-06-02 12:00:00,100.0\n"
    ).encode("utf-8")

    scraper = RTConstraintsScraper()
    df = scraper._process_response(DummyResponse(csv_data))

    assert "limit_mw" in df.columns
    assert df.loc[0, "limit_mw"] == 100.0


def test_rt_constraints_scraper_does_not_fabricate_loading():
    # SPP's RT binding-constraint files carry no loading columns. The scraper
    # must leave loading empty rather than invent it (it used to hash the name).
    csv_data = (
        "Constraint Name,State,Interval,Real Time Effective Limit\n"
        "LN HARBINE - BEATRICE,ACTIVATED,2026-06-02 12:00:00,100.0\n"
        "LN HARBINE - BEATRICE,BINDING,2026-06-02 12:05:00,100.0\n"
    ).encode("utf-8")

    df = RTConstraintsScraper()._process_response(DummyResponse(csv_data))

    assert df["pct_of_limit"].isna().all()
    assert df["monitored_mw"].isna().all()


def test_rt_constraints_scraper_keeps_published_loading():
    # If a file does publish loading, it is passed through untouched.
    csv_data = (
        "Constraint Name,State,Interval,Real Time Effective Limit,Percent\n"
        "LN HARBINE - BEATRICE,ACTIVATED,2026-06-02 12:00:00,100.0,87.5\n"
    ).encode("utf-8")

    df = RTConstraintsScraper()._process_response(DummyResponse(csv_data))

    assert df.loc[0, "pct_of_limit"] == 87.5


def test_scraper_strips_header_whitespace():
    csv_data = (
        "Constraint Name,State,Interval, Contingency Name\n"
        "LN A - B,BINDING,2026-06-02 12:00:00,7950\n"
    ).encode("utf-8")

    df = RTConstraintsScraper()._process_response(DummyResponse(csv_data))

    assert "contingency_name" in df.columns


def test_monitored_flowgate_scraper_inheritance():
    # Test that MonitoredFlowgateScraper inherits from RTConstraintsScraper
    scraper = MonitoredFlowgateScraper()
    assert isinstance(scraper, RTConstraintsScraper)

def test_volatility_score_calculation():
    # Mock some data to verify how standard deviation (volatility) is computed
    # 2 hours of 5-minute intervals = 24 intervals
    intervals = pd.date_range("2026-06-02 10:00:00", "2026-06-02 12:00:00", freq="5min")
    
    data = []
    # Stable line: pct_of_limit is always 80%
    # Volatile line: pct_of_limit fluctuates
    for i, dt in enumerate(intervals):
        data.append({
            "constraint_name": "LN STABLE - LINE",
            "interval": dt,
            "pct_of_limit": 80.0,
            "state": "ACTIVATED",
            "limit_mw": 100.0
        })
        data.append({
            "constraint_name": "LN VOLATILE - LINE",
            "interval": dt,
            # Vary between 70% and 90%
            "pct_of_limit": 80.0 + 10.0 * np.sin(i * np.pi / 6),
            "state": "ACTIVATED",
            "limit_mw": 100.0
        })
        
    df = pd.DataFrame(data)
    
    # Calculate group std dev
    vol_scores = df.groupby("constraint_name")["pct_of_limit"].std().fillna(0.0).to_dict()
    
    assert vol_scores["LN STABLE - LINE"] == 0.0
    assert vol_scores["LN VOLATILE - LINE"] > 0.0
