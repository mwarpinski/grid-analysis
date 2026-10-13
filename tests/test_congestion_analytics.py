"""Known-answer tests for the congestion analytics (synthetic data with hand-computable results)."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import congestion_analytics as ca

SF = {("N1", "C1"): -0.5, ("N1", "C2"): 0.2,
      ("N2", "C1"): 0.4, ("N2", "C2"): 0.2,
      ("N3", "C1"): 0.0, ("N3", "C2"): -0.3}


def _rt_5min(day="2026-10-10", hours=3, sp_by_hour=None):
    """RT frames over `hours` hour-ending buckets (00:05 .. hours:00), MCC = Σ SF × SP exactly."""
    ivs = pd.date_range(f"{day} 00:05", periods=12 * hours, freq="5min")
    sp_by_hour = sp_by_hour or {}
    bc_rows, lmp_rows = [], []
    for iv in ivs:
        he = iv.ceil("h")
        sps = sp_by_hour.get(he, {})
        for c, sp in sps.items():
            bc_rows.append({"interval": iv, "constraint_name": c, "shadow_price": sp})
        for n in ("N1", "N2", "N3"):
            mcc = sum(SF.get((n, c), 0.0) * sp for c, sp in sps.items())
            lmp_rows.append({"interval": iv, "settlement_location": n, "mcc": mcc})
    return pd.DataFrame(bc_rows), pd.DataFrame(lmp_rows)


def _da(day="2026-10-10", sp_by_hour=None, hours=3):
    hs = pd.date_range(f"{day} 01:00", periods=hours, freq="h")
    bc_rows, lmp_rows = [], []
    for h in hs:
        sps = (sp_by_hour or {}).get(h, {})
        for c, sp in sps.items():
            bc_rows.append({"interval": h, "constraint_name": c, "shadow_price": sp})
        for n in ("N1", "N2", "N3"):
            lmp_rows.append({"interval": h, "settlement_location": n,
                             "mcc": sum(SF.get((n, c), 0.0) * sp for c, sp in sps.items())})
    return pd.DataFrame(bc_rows), pd.DataFrame(lmp_rows)


H1, H2, H3 = (pd.Timestamp(f"2026-10-10 0{i}:00") for i in (1, 2, 3))


def test_hour_ending_alignment():
    s = pd.Series(pd.to_datetime(["2026-10-10 00:05", "2026-10-10 01:00", "2026-10-10 01:05", "2026-10-11 00:00"]))
    assert list(ca.to_hour_ending(s)) == list(pd.to_datetime(
        ["2026-10-10 01:00", "2026-10-10 01:00", "2026-10-10 02:00", "2026-10-11 00:00"]))


def test_nodal_dart_spread_is_rt_minus_da_and_drops_thin_hours():
    rt_bc, rt_lmp = _rt_5min(sp_by_hour={H1: {"C1": -100.0}, H2: {"C1": -100.0}, H3: {"C1": -100.0}})
    da_bc, da_lmp = _da(sp_by_hour={H1: {"C1": -40.0}, H2: {"C1": -40.0}, H3: {"C1": -40.0}})
    # drop 5 of the 12 RT intervals in hour 3 -> only 7 remain (< 10) -> hour 3 excluded
    rt_lmp = rt_lmp[~((rt_lmp["interval"].dt.ceil("h") == H3) & (rt_lmp["interval"].dt.minute < 25)
                      & (rt_lmp["interval"].dt.hour == 2))]
    d = ca.nodal_dart(da_lmp, rt_lmp)

    assert set(d["hour"]) == {H1, H2}
    n1 = d[(d["settlement_location"] == "N1") & (d["hour"] == H1)].iloc[0]
    assert n1["rt_mcc"] == pytest.approx(-0.5 * -100.0)
    assert n1["da_mcc"] == pytest.approx(-0.5 * -40.0)
    assert n1["spread"] == pytest.approx(50.0 - 20.0)


def test_hourly_rt_shadow_price_averages_over_present_intervals():
    ivs = pd.date_range("2026-10-10 00:05", periods=12, freq="5min")
    rows = [{"interval": iv, "constraint_name": "C1", "shadow_price": -100.0} for iv in ivs[:6]]
    rows += [{"interval": iv, "constraint_name": "C2", "shadow_price": 0.0} for iv in ivs]   # keeps 12 intervals present
    out = ca.hourly_shadow_prices_rt(pd.DataFrame(rows))
    assert out.loc[H1, "C1"] == pytest.approx(-50.0)


def test_decomposition_is_exact_when_sfs_are_exact():
    rt_bc, rt_lmp = _rt_5min(hours=2, sp_by_hour={H1: {"C1": -100.0, "C2": -20.0}, H2: {"C1": -50.0}})
    da_bc, da_lmp = _da(hours=2, sp_by_hour={H1: {"C1": -40.0}, H2: {"C1": -80.0, "C2": -10.0}})
    d = ca.nodal_dart(da_lmp, rt_lmp)
    dec = ca.decompose_dart(d, SF, ca.hourly_shadow_prices_rt(rt_bc), ca.hourly_shadow_prices_da(da_bc))

    pd.testing.assert_frame_equal(dec["explained"], dec["actual"], check_names=False, atol=1e-9)
    assert (dec["summary"]["mean_abs_residual"] < 1e-9).all()
    assert dec["unattributed_share"] == 0.0

    # hand check: N1, hour 1: SF[C1]*(−100−−40) + SF[C2]*(−20−0) = −0.5*−60 + 0.2*−20 = 26
    assert dec["explained"].loc[H1, "N1"] == pytest.approx(26.0)
    nd = ca.node_decomposition(dec, "N1")
    assert set(nd.index) == {"C1", "C2"}
    assert nd.loc["C1", "mean_contribution"] == pytest.approx(np.mean([-0.5 * -60.0, -0.5 * 30.0]))


def test_unmodeled_constraint_goes_to_residual_and_is_reported():
    rt_bc, rt_lmp = _rt_5min(hours=1, sp_by_hour={H1: {"C1": -100.0}})
    da_bc, da_lmp = _da(hours=1, sp_by_hour={H1: {"C1": -100.0}})
    rt_bc = pd.concat([rt_bc, pd.DataFrame({"interval": pd.date_range("2026-10-10 00:05", periods=12, freq="5min"),
                                            "constraint_name": "NEW", "shadow_price": -200.0})])
    d = ca.nodal_dart(da_lmp, rt_lmp)
    dec = ca.decompose_dart(d, SF, ca.hourly_shadow_prices_rt(rt_bc), ca.hourly_shadow_prices_da(da_bc))
    assert dec["unattributed_share"] == pytest.approx(1.0)       # all of the RT-DA move is in NEW
    assert "NEW" not in dec["dsp"].columns


def test_constraint_stats_runs_and_rt_only_hours():
    ivs = pd.date_range("2026-10-10 00:05", periods=24, freq="5min")                # HE01 and HE02
    rows = [{"interval": iv, "constraint_name": "C1", "shadow_price": -10.0, "state": "BINDING"}
            for iv in list(ivs[:4]) + list(ivs[8:11])]                              # runs of 4 and 3
    rows += [{"interval": iv, "constraint_name": "C1", "shadow_price": 0.0, "state": "ACTIVATED"} for iv in ivs[4:8]]
    rt_bc = pd.DataFrame(rows)
    da_bc = pd.DataFrame([{"interval": H2, "constraint_name": "C1", "shadow_price": -5.0},
                          {"interval": H1, "constraint_name": "OTHER", "shadow_price": -5.0}])
    st = ca.constraint_stats(rt_bc, da_bc).set_index("constraint_name").loc["C1"]
    assert st["rt_binding_intervals"] == 7
    assert st["mean_run_intervals"] == pytest.approx(3.5)
    assert st["max_run_intervals"] == 4
    assert st["bind_given_activated"] == pytest.approx(7 / 11)
    assert st["rt_binding_hours"] == 1 and st["da_binding_hours"] == 1
    assert st["rt_only_hours"] == 1     # RT bound in HE01, DA only priced it in HE02


def test_best_pairs_orders_by_mcc_and_collapses_duplicates():
    sf = {("A", "C"): -0.40, ("A2", "C"): -0.4001, ("B", "C"): 0.10, ("D", "C"): 0.30}
    p = ca.best_pairs(sf, "C", top=3)
    first = p.iloc[0]
    assert first["high_mcc_node"] in {"A", "A2"} and first["low_mcc_node"] == "D"
    assert first["delta_sf"] == pytest.approx(0.70, abs=1e-3)
    assert len(p) == 1                                  # A/A2 collapsed; no degenerate or mirrored pairs
    assert (p["delta_sf"] > 0).all() and (p["high_mcc_node"] != p["low_mcc_node"]).all()
    assert not {"A", "A2"} <= set(p["high_mcc_node"])
    # With binding SP<0: MCC(high node) − MCC(low node) = (sf_high − sf_low) × SP > 0
    assert (first["sf_high_mcc_node"] - first["sf_low_mcc_node"]) * -100.0 > 0


def test_best_pairs_stability_across_fits():
    sf = {("A", "C"): -0.4, ("D", "C"): 0.3}
    alt_same = {("A", "C"): -0.35, ("D", "C"): 0.28}
    alt_flip = {("A", "C"): 0.2, ("D", "C"): -0.1}
    assert ca.best_pairs(sf, "C", top=1, sf_alt=alt_same).iloc[0]["same_sign"]
    assert not ca.best_pairs(sf, "C", top=1, sf_alt=alt_flip).iloc[0]["same_sign"]


def test_trip_scenario_flow_change_and_ranking():
    t = ca.trip_scenario(SF, "N1", injection_change_mw=-200.0, limits={"C1": 100.0, "C2": 1000.0})
    row = t.set_index("constraint")
    assert row.loc["C1", "delta_flow_mw"] == pytest.approx(-0.5 * -200.0)       # +100 MW
    assert row.loc["C2", "delta_flow_mw"] == pytest.approx(0.2 * -200.0)        # −40 MW
    assert row.loc["C1", "delta_pct_of_limit"] == pytest.approx(100.0)
    assert list(t["constraint"])[0] == "C1"                                     # most stressed relative to limit
    assert ca.trip_scenario(SF, "NOPE", -10).empty


def test_load_trip_is_the_mirror_of_generator_trip():
    gen = ca.trip_scenario(SF, "N2", -100.0).set_index("constraint")["delta_flow_mw"]
    load = ca.trip_scenario(SF, "N2", +100.0).set_index("constraint")["delta_flow_mw"]
    pd.testing.assert_series_equal(gen, -load)


def test_price_impact_matches_compute_mcc():
    from src.analysis.multi_constraint_sf import ShiftFactorSolution
    sol = ShiftFactorSolution()
    sol._flat = dict(SF)
    sp = {"C1": -300.0, "C2": -50.0, "GHOST": -999.0}
    pi = ca.price_impact(SF, sp).set_index("node")
    for n in ("N1", "N2", "N3"):
        assert pi.loc[n, "mcc_change"] == pytest.approx(sol.compute_mcc(n, {"C1": -300.0, "C2": -50.0}))
    assert pi.attrs["unmodeled"] == ["GHOST"]


def test_historical_shadow_price_quantile_uses_severity():
    bc = pd.DataFrame({"constraint_name": "C", "shadow_price": [-10.0, -20.0, -30.0, -40.0, -1000.0]})
    assert ca.historical_shadow_price(bc, ["C"], 0.5)["C"] == pytest.approx(-30.0)
    assert ca.historical_shadow_price(bc, ["C"], 1.0)["C"] == pytest.approx(-1000.0)   # the most severe price


def test_constraint_limits_use_the_static_rating_not_the_adjusted_limit():
    """The RT effective limit is 0/negative on breached rows (e.g. -42.6 on a 450 MW line)."""
    bc = pd.DataFrame({
        "constraint_name": ["C"] * 3,
        "source_limit_mw": [450.0, 450.0, 450.0],
        "rt_effective_limit_mw": [-42.6, 3.6, 436.0],
        "initial_effective_limit_mw": [436.5, 436.5, 436.5],
    })
    assert ca.constraint_limits(bc)["C"] == pytest.approx(450.0)


def test_constraint_stats_can_group_by_facility_across_contingencies():
    ivs = pd.date_range("2026-10-10 00:05", periods=3, freq="5min")
    rt = pd.DataFrame({
        "interval": list(ivs[:2]) + [ivs[2]], "constraint_name": ["C_A", "C_A", "C_B"],
        "shadow_price": [-10.0, -10.0, -10.0], "state": "BINDING", "monitored_facility": "LN X - Y",
    })
    da = pd.DataFrame({"interval": [H1], "constraint_name": ["C_OTHER_CONTINGENCY"],
                       "shadow_price": [-5.0], "monitored_facility": "LN X - Y"})
    by_name = ca.constraint_stats(rt, da).set_index("constraint_name")
    by_fac = ca.constraint_stats(rt, da, group_by="monitored_facility").set_index("monitored_facility")
    assert by_name.loc["C_A", "rt_only_hours"] == 1                  # looks like a DA miss ...
    assert by_fac.loc["LN X - Y", "rt_only_hours"] == 0              # ... but DA priced the same line
    assert by_fac.loc["LN X - Y", "rt_binding_intervals"] == 3
