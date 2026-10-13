import pandas as pd
import pytest

from src.analysis import constraint_identity as ci


def test_monitored_element_is_direction_independent():
    a = ci.parse_monitored("LN TEKAMHO - SUB1226")
    b = ci.parse_monitored("LN SUB1226 - TEKAMHO")
    assert a.kind == "LN" and a.ends == ("SUB1226", "TEKAMHO")
    assert a.key == b.key
    assert ci.parse_monitored("XFMR POTTER_S - POTTER_S").ends == ("POTTER_S", "POTTER_S")
    assert ci.parse_monitored("XFMR A - B").key != ci.parse_monitored("LN A - B").key   # kind matters


def test_unparseable_monitored_facilities_keep_their_raw_text():
    m = ci.parse_monitored("Multi-Element Constraint")
    assert m.kind == "MULTI" and m.ends == ()
    assert ci.parse_monitored("weird").kind == "UNKNOWN"
    assert ci.parse_monitored(None).kind == "UNKNOWN"


def test_contingency_line_with_circuit_and_ids():
    c = ci.parse_contingency("OPPD:SUB3456 NEBRCTY:345:1:11")
    assert c.stations == ("NEBRCTY", "SUB3456") and c.kv == 345.0 and c.owners == ("OPPD",) and not c.base


def test_contingency_transformer_drops_the_transformer_id():
    c = ci.parse_contingency("NSP:LYON_CO XF TR9_3:345 115 34.5:")
    assert c.stations == ("LYON_CO",) and c.kv == 345.0


def test_contingency_irregular_forms():
    assert ci.parse_contingency("                              BASE").base
    assert ci.parse_contingency("").base and ci.parse_contingency(None).base
    named = ci.parse_contingency("ADD NEW EAST")
    assert not named.base and named.stations == () and named.key == "C|ADD NEW EAST"
    truncated = ci.parse_contingency("OKGE:KETCH OMMITCH4 NOWST1 DANFO W")
    assert truncated.kv is None and "KETCH" in truncated.stations
    assert ci.parse_contingency("WFEC:ANADARKO FLETCH3 COMANCH2:138").kv == 138.0
    unit = ci.parse_contingency("SPS:HOBBS:COMBINEDCYCLE")
    assert unit.stations == ("HOBBS", "UNIT:COMBINEDCYCLE") and unit.kv is None       # a generator, not a line
    assert unit.key != ci.parse_contingency("SPS:HOBBS:OTHERUNIT").key


def test_constraint_key_ignores_owner_order_circuit_and_case():
    k1 = ci.constraint_key("LN A - B", "MPS AECI:X Y:161:1:8")
    k2 = ci.constraint_key("ln B - a", "AECI MPS:Y X:161:2:9")
    assert k1 == k2
    assert ci.constraint_key("LN A - B", "BASE") != k1
    assert ci.constraint_key("LN A - B", "MPS:X Y:345:1:8") != k1          # different voltage = different outage


def _bc(rows):
    df = pd.DataFrame(rows, columns=["constraint_name", "monitored_facility", "contingent_facility", "shadow_price", "interval"])
    df["interval"] = pd.to_datetime(df["interval"])
    return df


def test_alias_table_links_rt_and_da_names_for_the_same_element_and_contingency():
    rt = _bc([("TMP527_28137", "LN A - B", "OPPD:X Y:345:1:11", -50.0, "2026-10-10 12:00")])
    da = _bc([("ABCFLOWGATE", "LN B - A", "OPPD:Y X:345:2:3", -20.0, "2026-10-10 13:00"),
              ("DA_ONLY", "LN C - D", "BASE", -5.0, "2026-10-10 13:00")])
    t = ci.build_alias_table(rt, da).set_index("constraint_key")
    both = t[t.in_rt & t.in_da]
    assert len(both) == 1
    row = both.iloc[0]
    assert row["rt_names"] == ["TMP527_28137"] and row["da_names"] == ["ABCFLOWGATE"]
    assert len(t) == 2 and t["in_rt"].sum() == 1 and t["in_da"].sum() == 2


def test_multi_element_constraints_are_not_merged():
    rt = _bc([("M1", "Multi-Element Constraint", "BASE", -1.0, "2026-10-10 12:00"),
              ("M2", "Multi-Element Constraint", "BASE", -1.0, "2026-10-10 12:00")])
    out = ci.add_identity_columns(rt)
    assert out["constraint_key"].nunique() == 2


def test_name_to_key_maps_every_alias():
    rt = _bc([("T1", "LN A - B", "BASE", -1.0, "2026-10-10 12:00")])
    da = _bc([("D1", "LN B - A", "BASE", -1.0, "2026-10-10 13:00")])
    m = ci.name_to_key(rt, da)
    assert m["T1"] == m["D1"]


def test_identity_columns_expose_endpoints():
    out = ci.add_identity_columns(_bc([("T1", "LN TEKAMHO - SUB1226", "BASE", -1.0, "2026-10-10 12:00")]))
    assert (out.loc[0, "from_sub"], out.loc[0, "to_sub"]) == ("SUB1226", "TEKAMHO")
    assert out.loc[0, "element_kind"] == "LN"


def test_orientation_records_the_listed_direction():
    assert ci.parse_monitored("LN A - B").orientation == 1
    assert ci.parse_monitored("LN B - A").orientation == -1
    assert ci.parse_monitored("LN A - B").key == ci.parse_monitored("LN B - A").key      # same element


def test_canonical_sf_aligns_the_two_directions_of_one_line():
    """SOONER->LIBERAL_ and LIBERAL_->SOONER are separate constraints with opposite-sign shift factors."""
    rt = _bc([("FWD", "LN LIBERAL_ - SOONER", "SPS:X Y:345:1:1", -5.0, "2026-10-10 12:00"),
              ("REV", "LN SOONER - LIBERAL_", "SPS:X Y:345:1:1", -5.0, "2026-10-10 12:00")])
    o = ci.name_orientation(rt)
    assert o["FWD"] == 1 and o["REV"] == -1
    X = pd.DataFrame({"FWD": [0.3, -0.1], "REV": [-0.3, 0.1]}, index=["n1", "n2"])      # mirror images
    C = ci.canonical_sf(X, o)
    pd.testing.assert_series_equal(C["FWD"], C["REV"], check_names=False)
    assert ci.add_identity_columns(rt)["constraint_key"].nunique() == 1                  # one alias group


def test_parallel_transformers_with_identical_ends_are_not_merged():
    """XFMR EXIRAWA - EXIRAWA names no particular transformer; only the constraint name does."""
    da = _bc([("EXIRAWA_TRG1_XF", "XFMR EXIRAWA - EXIRAWA", "BASE", 5.0, "2026-10-10 12:00"),
              ("EXIRAWA_TRG2_XF", "XFMR EXIRAWA - EXIRAWA", "BASE", 5.0, "2026-10-10 12:00")])
    assert ci.add_identity_columns(da)["constraint_key"].nunique() == 2
    assert ci.parse_monitored("XFMR EXIRAWA - EXIRAWA").ambiguous
    assert not ci.parse_monitored("XFMR A - B").ambiguous


def test_contingency_keeps_circuit_and_transformer_for_graph_building():
    line = ci.parse_contingency("OPPD:SUB3456 NEBRCTY:345:2:11")
    assert line.circuit == "2" and line.transformer is None and line.stations == ("NEBRCTY", "SUB3456")
    xf = ci.parse_contingency("NSP:LYON_CO XF TR9_3:345 115 34.5:")
    assert xf.transformer == "TR9_3" and xf.stations == ("LYON_CO",) and xf.kv == 345.0
    assert ci.parse_contingency("BASE").circuit is None
    # adding fields must not change keys (aliases/crosswalks rely on them)
    assert ci.parse_contingency("OPPD:SUB3456 NEBRCTY:345:1:11").key == ci.parse_contingency("OPPD:SUB3456 NEBRCTY:345:2:11").key
