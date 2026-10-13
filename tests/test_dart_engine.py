import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient

from src.api.server import app
from src.analysis.dart_engine import calculate_dart_spreads

client = TestClient(app)

def test_calculate_dart_spreads_empty():
    """Test that DART spreads calculation handles empty database returns gracefully."""
    with patch("src.analysis.dart_engine.store") as mock_store:
        mock_store.query_constraints.return_value = pd.DataFrame()
        mock_store.query_da_constraints.return_value = pd.DataFrame()
        
        spreads = calculate_dart_spreads(iso="spp", days=30)
        assert spreads == []

def test_calculate_dart_spreads_calculation():
    """Test DART spreads hourly alignment, averaging, and underpricing logic."""
    # Create sample RTBM constraints
    # SPP labels intervals by their END time: 00:05..01:00 make up hour-ending 01
    # (matching DA label 01:00); 01:05..02:00 make up hour-ending 02.
    rt_data = [
        {"interval": "2026-05-15 00:05:00", "constraint_name": "LINE_A", "shadow_price": -100.0},
        {"interval": "2026-05-15 00:10:00", "constraint_name": "LINE_A", "shadow_price": -200.0}, # Avg = -150.0 for HE01
        {"interval": "2026-05-15 01:55:00", "constraint_name": "LINE_A", "shadow_price": -50.0},  # Avg = -50.0 for HE02
        {"interval": "2026-05-15 00:05:00", "constraint_name": "LINE_B", "shadow_price": -10.0},  # RT avg = -10.0
    ]
    rt_df = pd.DataFrame(rt_data)
    
    # Create sample DAM constraints (hourly)
    da_data = [
        {"interval": "2026-05-15 01:00:00", "constraint_name": "LINE_A", "shadow_price": -50.0}, # Hour 1 DAM
        {"interval": "2026-05-15 02:00:00", "constraint_name": "LINE_A", "shadow_price": -100.0}, # Hour 2 DAM
        {"interval": "2026-05-15 01:00:00", "constraint_name": "LINE_B", "shadow_price": -40.0}, # Hour 1 DAM
    ]
    da_df = pd.DataFrame(da_data)
    
    with patch("src.analysis.dart_engine.store") as mock_store:
        mock_store.query_constraints.return_value = rt_df
        mock_store.query_da_constraints.return_value = da_df
        
        spreads = calculate_dart_spreads(iso="spp", days=30)
        
        # We expect 2 constraints
        assert len(spreads) == 2
        
        # Index by constraint name for assertions
        spreads_map = {s["constraint_name"]: s for s in spreads}
        
        # LINE_A Hour 1: RT = -150.0, DAM = -50.0. Hour 2: RT = -50.0, DAM = -100.0
        # Average RT = (-150 + -50) / 2 = -100.0
        # Average DA = (-50 + -100) / 2 = -75.0
        # Average Spread = (-100 - 0) + (50 - -100) / 2?
        # Aligned hours:
        # Hour 1: RT = -150.0, DA = -50.0 -> Spread = -100.0
        # Hour 2: RT = -50.0, DA = -100.0 -> Spread = 50.0
        # Mean RT = -100.0, Mean DA = -75.0, Mean Spread = -25.0
        assert "LINE_A" in spreads_map
        line_a = spreads_map["LINE_A"]
        assert line_a["mean_rtbm_shadow_price"] == -100.0
        assert line_a["mean_dam_shadow_price"] == -75.0
        assert line_a["mean_dart_spread"] == -25.0
        assert line_a["rtbm_binding_hours"] == 2
        assert line_a["dam_binding_hours"] == 2
        # abs(-100.0) > abs(-75.0) -> True, and RT < 0 -> True
        assert line_a["underpriced_in_dam"] is True
        
        # LINE_B Hour 1: RT = -10.0, DAM = -40.0
        # Mean RT = -10.0, Mean DA = -40.0, Mean Spread = 30.0
        assert "LINE_B" in spreads_map
        line_b = spreads_map["LINE_B"]
        assert line_b["mean_rtbm_shadow_price"] == -10.0
        assert line_b["mean_dam_shadow_price"] == -40.0
        assert line_b["mean_dart_spread"] == 30.0
        # abs(-10.0) > abs(-40.0) -> False
        assert line_b["underpriced_in_dam"] is False

def test_rt_interval_ending_on_the_hour_belongs_to_that_hour_ending():
    """The 01:00 interval is the LAST interval of HE01, so it must pair with DA label 01:00."""
    rt_df = pd.DataFrame([
        {"interval": "2026-05-15 01:00:00", "constraint_name": "LINE_A", "shadow_price": -80.0},
        {"interval": "2026-05-15 01:05:00", "constraint_name": "LINE_A", "shadow_price": -999.0},  # HE02
    ])
    da_df = pd.DataFrame([
        {"interval": "2026-05-15 01:00:00", "constraint_name": "LINE_A", "shadow_price": -50.0},
    ])
    with patch("src.analysis.dart_engine.store") as mock_store:
        mock_store.query_constraints.return_value = rt_df
        mock_store.query_da_constraints.return_value = da_df
        line_a = calculate_dart_spreads(iso="spp", days=30)[0]

    # HE01: RT -80 vs DA -50 (spread -30); HE02: RT -999 vs DA 0 (spread -999). Mean = -514.5
    assert line_a["mean_dart_spread"] == pytest.approx((-30.0 + -999.0) / 2)
    assert line_a["rtbm_binding_hours"] == 2


def test_api_dart_spreads_route():
    """Test that the /api/v1/constraints/dart-spreads endpoint calls the engine and returns JSON."""
    with patch("src.analysis.dart_engine.calculate_dart_spreads") as mock_calc:
        mock_calc.return_value = [
            {
                "constraint_name": "LINE_A",
                "mean_rtbm_shadow_price": -100.0,
                "mean_dam_shadow_price": -75.0,
                "mean_dart_spread": -25.0,
                "rtbm_binding_hours": 2,
                "dam_binding_hours": 2,
                "underpriced_in_dam": True,
            }
        ]
        
        response = client.get("/api/v1/constraints/dart-spreads?iso=spp&days=15")
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["iso"] == "spp"
        assert json_data["days"] == 15
        assert "spreads" in json_data
        assert len(json_data["spreads"]) == 1
        assert json_data["spreads"][0]["constraint_name"] == "LINE_A"


def test_dart_ignores_days_without_day_ahead_data():
    """
    Real-time history usually goes back further than stored Day-Ahead history. RT hours on
    days with no DA file must not be compared against a fabricated DA price of 0, which
    flagged every constraint as 'underpriced'.
    """
    rt_df = pd.DataFrame([
        {"interval": "2026-05-14 12:05:00", "constraint_name": "LINE_A", "shadow_price": -500.0},  # no DA that day
        {"interval": "2026-05-15 00:05:00", "constraint_name": "LINE_A", "shadow_price": -100.0},  # DA exists
    ])
    da_df = pd.DataFrame([
        {"interval": "2026-05-15 01:00:00", "constraint_name": "LINE_A", "shadow_price": -90.0},
    ])
    with patch("src.analysis.dart_engine.store") as mock_store:
        mock_store.query_constraints.return_value = rt_df
        mock_store.query_da_constraints.return_value = da_df
        line_a = calculate_dart_spreads(iso="spp", days=30)[0]

    assert line_a["mean_rtbm_shadow_price"] == -100.0       # the 05-14 hour is excluded
    assert line_a["mean_dart_spread"] == -10.0


def test_dart_returns_nothing_without_day_ahead_data():
    rt_df = pd.DataFrame([{"interval": "2026-05-15 00:05:00", "constraint_name": "A", "shadow_price": -1.0}])
    with patch("src.analysis.dart_engine.store") as mock_store:
        mock_store.query_constraints.return_value = rt_df
        mock_store.query_da_constraints.return_value = pd.DataFrame()
        assert calculate_dart_spreads(iso="spp", days=30) == []
