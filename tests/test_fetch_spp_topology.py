"""SPP topology files: header renames must survive (they were lowercased before being matched)."""

from scripts.fetch_spp_topology import normalize_hub_definitions, normalize_interface_definitions

_HEADER = "Settlement Location Name,Parent PNode,Child PNode,Weighting Factor,Settlement Location Type,BAA\n"


def test_hub_definitions_keep_hub_name(tmp_path):
    f = tmp_path / "Hub_Definitions.csv"
    f.write_text(_HEADER + "HUB_A,HUB_A_H,CHILD1,0.25,HUB,SPP\nHUB_A,HUB_A_H,CHILD2,0.75,HUB,SPP\n")

    df = normalize_hub_definitions(f)

    assert {"hub_name", "child_pnode", "weight"} <= set(df.columns)
    assert df["hub_name"].nunique() == 1
    assert df["weight"].sum() == 1.0


def test_interface_definitions_are_not_empty(tmp_path):
    f = tmp_path / "Interface_Definitions.csv"
    f.write_text(_HEADER + "AEC,SOUC,SOUC15THSTLD3,1,INT,SPP\n")

    df = normalize_interface_definitions(f)

    assert len(df) == 1
    assert df.loc[0, "interface_name"] == "AEC"
    assert df.loc[0, "child_pnode"] == "SOUC15THSTLD3"


def test_bus_mapping_recognises_spp_headers(tmp_path):
    import pandas as pd
    from scripts.fetch_spp_topology import normalize_bus_mapping

    f = tmp_path / "bus.xlsx"
    pd.DataFrame([{
        "SETTLEMENTLOCATION": "CSWS.VOLT.0250", "EMS_AREA": "CSWS", "EMS_STATION": "53CACHE4",
        "PSSE_NUM": 511509.0, "PSSE_NAME": "53CACHE4", "EMS_KV": 138.0, "EMS_UNIT": "V250_DDR_GEN",
        "EMS_MWMAX": 0.3, "TURBINETYPE": "OT - Other", "FUELTYPE1": "OTH - Other Fuel",
    }]).to_excel(f, index=False)

    df = normalize_bus_mapping(f)

    assert df.loc[0, "settlement_location"] == "CSWS.VOLT.0250"
    assert df.loc[0, "ems_station"] == "53CACHE4"
    assert df.loc[0, "fuel_type"] == "OTH - Other Fuel"
    assert "psse_bus_number" in df.columns and "ems_kv" in df.columns
