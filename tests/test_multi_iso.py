import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.scrapers.miso_adapter import MISOAdapter
from src.api.sf_cache import cache
from src.api.server import app

client = TestClient(app)

def test_miso_adapter_parses_response():
    adapter = MISOAdapter()
    
    # Mock the public api response for RT constraints
    mock_constraints_json = {
        "Constraint": [
            {
                "Name": "AMIL.LNODE1_LN_1",
                "Period": "2026-06-02 12:00:00",
                "Price": 150.50
            }
        ]
    }
    
    # Mock the RT LMP response
    mock_lmp_json = {
        "headers": ["INTERVAL", "CPNODE", "LMP", "MLC", "MCC"],
        "data": [
            ["2026-06-02 12:00:00", "AMIL.LNODE1", 45.2, 0.5, -2.3]
        ]
    }
    
    # Patch requests.get to return our mock json responses
    with patch("requests.get") as mock_get:
        # First call for constraints, second for LMPs
        mock_resp_bc = MagicMock()
        mock_resp_bc.status_code = 200
        mock_resp_bc.json.return_value = mock_constraints_json
        
        mock_resp_lmp = MagicMock()
        mock_resp_lmp.status_code = 200
        mock_resp_lmp.json.return_value = mock_lmp_json
        
        mock_get.side_effect = [mock_resp_bc, mock_resp_lmp]
        
        constraints_df = adapter.fetch_latest_constraints()
        lmps_df = adapter.fetch_latest_lmps()
        
        assert not constraints_df.empty
        assert "constraint_name" in constraints_df.columns
        assert constraints_df.loc[0, "constraint_name"] == "AMIL.LNODE1_LN_1"
        assert constraints_df.loc[0, "shadow_price"] == 150.50
        assert constraints_df.loc[0, "limit_mw"] is None
        assert constraints_df.loc[0, "monitored_mw"] is None
        assert constraints_df.loc[0, "state"] == "BINDING"
        
        assert not lmps_df.empty
        assert "settlement_location" in lmps_df.columns
        assert lmps_df.loc[0, "settlement_location"] == "AMIL.LNODE1"
        assert lmps_df.loc[0, "mcc"] == -2.3

def test_multi_iso_sf_cache():
    # Verify that cache registry yields isolated instances
    spp_cache = cache.for_iso("spp")
    miso_cache = cache.for_iso("miso")
    
    assert spp_cache is not miso_cache
    assert cache.for_iso("spp") is spp_cache
    assert cache.for_iso("miso") is miso_cache
    
    # Assert dynamic creation of new namespaces
    new_cache = cache.for_iso("invalid_iso")
    assert new_cache is not spp_cache
    assert new_cache is not miso_cache
    assert cache.for_iso("invalid_iso") is new_cache

def test_api_iso_parameterization():
    # Query constraints with iso parameter
    r1 = client.get("/api/v1/constraints/live?iso=spp")
    assert r1.status_code in (200, 503) # 503 is acceptable if solver is warming up/empty
    
    r2 = client.get("/api/v1/constraints/live?iso=miso")
    assert r2.status_code in (200, 503)

    r3 = client.get("/api/v1/shift-factors/heatmap?constraint=TEST_C&iso=miso")
    assert r3.status_code in (200, 404, 503)
