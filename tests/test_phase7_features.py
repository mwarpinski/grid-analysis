"""
Tests for Phase 7 additions to ShiftFactorSolution:
  - to_dataframe(shadow_prices) → sf_exposure column
  - compute_mcc(node, shadow_prices)
  - flow_impact(source, sink, mw, constraint)
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.multi_constraint_sf import (
    calculate_shift_factors,
    ShiftFactorSolution,
)
from tests.test_shift_factor_calculator import (
    make_single_constraint_data,
    make_multi_constraint_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_solution():
    """Pre-fitted solution with one constraint, known true SFs."""
    constraints_df, lmp_df, true_sfs = make_single_constraint_data(
        n_intervals=80, n_nodes=4, true_sf=0.25, seed=7
    )
    solution = calculate_shift_factors(
        constraints_df=constraints_df,
        lmp_df=lmp_df,
        min_intervals=5,
    )
    return solution, true_sfs


@pytest.fixture
def multi_solution():
    """Pre-fitted solution with two constraints."""
    constraints_df, lmp_df, true_sfs = make_multi_constraint_data(
        n_intervals=120, n_nodes=3, seed=42
    )
    solution = calculate_shift_factors(
        constraints_df=constraints_df,
        lmp_df=lmp_df,
        min_intervals=5,
    )
    return solution, true_sfs


# ---------------------------------------------------------------------------
# to_dataframe with shadow_prices (exposure column)
# ---------------------------------------------------------------------------

class TestToDataframeWithExposure:
    def test_exposure_column_present_when_shadow_prices_given(self, single_solution):
        solution, _ = single_solution
        sp = {"LN TEST - CONSTRAINT": 500.0}
        df = solution.to_dataframe(shadow_prices=sp)
        assert "sf_exposure" in df.columns
        assert "shadow_price" in df.columns

    def test_no_exposure_column_without_shadow_prices(self, single_solution):
        solution, _ = single_solution
        df = solution.to_dataframe()
        assert "sf_exposure" not in df.columns
        assert "shadow_price" not in df.columns

    def test_exposure_equals_sf_times_shadow_price(self, single_solution):
        solution, _ = single_solution
        sp_val = 800.0
        sp = {"LN TEST - CONSTRAINT": sp_val}
        df = solution.to_dataframe(shadow_prices=sp)
        for _, row in df.iterrows():
            expected = row["shift_factor"] * sp_val
            assert abs(row["sf_exposure"] - expected) < 1e-9, (
                f"sf_exposure mismatch: {row['sf_exposure']} != {expected}"
            )

    def test_sorted_by_abs_exposure_descending(self, single_solution):
        solution, _ = single_solution
        sp = {"LN TEST - CONSTRAINT": 1000.0}
        df = solution.to_dataframe(shadow_prices=sp)
        exposures = df["sf_exposure"].abs().tolist()
        assert exposures == sorted(exposures, reverse=True), (
            "Rows should be sorted by |sf_exposure| descending"
        )

    def test_zero_shadow_price_gives_zero_exposure(self, single_solution):
        solution, _ = single_solution
        sp = {"LN TEST - CONSTRAINT": 0.0}
        df = solution.to_dataframe(shadow_prices=sp)
        assert (df["sf_exposure"] == 0.0).all()

    def test_unknown_constraint_in_shadow_prices_is_ignored(self, single_solution):
        solution, _ = single_solution
        sp = {
            "LN TEST - CONSTRAINT": 400.0,
            "NONEXISTENT CONSTRAINT": 9999.0,  # not in solution
        }
        df = solution.to_dataframe(shadow_prices=sp)
        # Should not raise — unknown constraint simply gets exposure=0
        assert "sf_exposure" in df.columns


# ---------------------------------------------------------------------------
# compute_mcc
# ---------------------------------------------------------------------------

class TestComputeMCC:
    def test_mcc_formula_single_constraint(self, single_solution):
        """MCC_node = +(SF × SP), the same convention the SFs are fit with."""
        solution, true_sfs = single_solution
        sp_val = 500.0
        sp = {"LN TEST - CONSTRAINT": sp_val}
        node = "NODE_00"

        mcc = solution.compute_mcc(node, sp)
        sf = solution.get(node, "LN TEST - CONSTRAINT")

        expected_mcc = sf * sp_val
        assert abs(mcc - expected_mcc) < 1e-6, (
            f"compute_mcc={mcc:.4f}, expected {expected_mcc:.4f}"
        )

    def test_mcc_multi_constraint_is_sum(self, multi_solution):
        """With two constraints, MCC = SF1×SP1 + SF2×SP2."""
        solution, _ = multi_solution
        sp = {
            "LN ALPHA - BETA":   200.0,
            "LN GAMMA - DELTA":  -150.0,
        }
        node = "NODE_00"
        mcc = solution.compute_mcc(node, sp)

        sf1 = solution.get(node, "LN ALPHA - BETA")
        sf2 = solution.get(node, "LN GAMMA - DELTA")
        expected = sf1 * 200.0 + sf2 * (-150.0)
        assert abs(mcc - expected) < 1e-6

    def test_compute_mcc_reproduces_observed_mcc(self):
        """
        compute_mcc must predict the MCC the solver was fit on, with the same sign.
        (It once returned the negation of the fit, which no isolated formula test caught.)
        """
        constraints_df, lmp_df, _ = make_single_constraint_data(
            n_intervals=60, n_nodes=3, true_sf=0.25, noise_std=0.0, seed=3
        )
        solution = calculate_shift_factors(
            constraints_df=constraints_df, lmp_df=lmp_df, min_intervals=5
        )
        iv = constraints_df["interval"].iloc[10]
        sp = float(constraints_df.loc[constraints_df["interval"] == iv, "shadow_price"].iloc[0])
        for node in ("NODE_00", "NODE_01", "NODE_02"):
            observed = float(
                lmp_df[(lmp_df["interval"] == iv) & (lmp_df["settlement_location"] == node)]["mcc"].iloc[0]
            )
            predicted = solution.compute_mcc(node, {"LN TEST - CONSTRAINT": sp})
            assert predicted == pytest.approx(observed, rel=1e-3, abs=1e-3)

    def test_mcc_zero_when_no_binding_constraints(self, single_solution):
        solution, _ = single_solution
        mcc = solution.compute_mcc("NODE_00", {})
        assert mcc == 0.0

    def test_mcc_unknown_node_returns_zero(self, single_solution):
        solution, _ = single_solution
        # compute_mcc sums over known constraints; unknown node → all SFs None → sum=0
        mcc = solution.compute_mcc("TOTALLY_UNKNOWN_NODE", {"LN TEST - CONSTRAINT": 500.0})
        assert mcc == 0.0


# ---------------------------------------------------------------------------
# flow_impact (bilateral what-if)
# ---------------------------------------------------------------------------

class TestFlowImpact:
    def test_flow_impact_formula(self, single_solution):
        """ΔFlow = (SF_source - SF_sink) × MW."""
        solution, _ = single_solution
        source = "NODE_00"
        sink = "NODE_01"
        mw = 50.0
        constraint = "LN TEST - CONSTRAINT"

        result = solution.flow_impact(source, sink, mw, constraint)

        sf_source = solution.get(source, constraint)
        sf_sink = solution.get(sink, constraint)
        expected = (sf_source - sf_sink) * mw

        assert result is not None
        assert abs(result - expected) < 1e-9

    def test_positive_result_means_exacerbates(self, single_solution):
        """If SF_source > SF_sink, injection increases constraint flow."""
        solution, _ = single_solution
        # Find a source with higher SF than sink
        nodes = list({node for (node, _) in solution._flat})
        if len(nodes) < 2:
            pytest.skip("Need at least 2 nodes")

        # Sort nodes by SF ascending
        constraint = "LN TEST - CONSTRAINT"
        sfs = {n: solution.get(n, constraint) for n in nodes if solution.get(n, constraint) is not None}
        if len(sfs) < 2:
            pytest.skip("Need SFs for at least 2 nodes")

        sorted_nodes = sorted(sfs, key=sfs.get)
        sink = sorted_nodes[0]    # lowest SF
        source = sorted_nodes[-1] # highest SF

        result = solution.flow_impact(source, sink, 100.0, constraint)
        assert result > 0, "Higher SF source should increase flow (exacerbate)"

    def test_negative_result_means_relieves(self, single_solution):
        """Reversed direction → flow decreases (relieves constraint)."""
        solution, _ = single_solution
        constraint = "LN TEST - CONSTRAINT"
        nodes = list({n for (n, c) in solution._flat if c == constraint})
        if len(nodes) < 2:
            pytest.skip("Need at least 2 nodes")

        sfs = {n: solution.get(n, constraint) for n in nodes}
        sorted_nodes = sorted(sfs, key=sfs.get)
        source = sorted_nodes[0]  # lowest SF injects
        sink = sorted_nodes[-1]   # highest SF withdraws

        result = solution.flow_impact(source, sink, 100.0, constraint)
        assert result < 0, "Lower SF source should decrease flow (relieve)"

    def test_returns_none_for_unknown_source(self, single_solution):
        solution, _ = single_solution
        result = solution.flow_impact(
            "UNKNOWN_SOURCE", "NODE_00", 50.0, "LN TEST - CONSTRAINT"
        )
        assert result is None

    def test_returns_none_for_unknown_sink(self, single_solution):
        solution, _ = single_solution
        result = solution.flow_impact(
            "NODE_00", "UNKNOWN_SINK", 50.0, "LN TEST - CONSTRAINT"
        )
        assert result is None

    def test_returns_none_for_unknown_constraint(self, single_solution):
        solution, _ = single_solution
        result = solution.flow_impact(
            "NODE_00", "NODE_01", 50.0, "NONEXISTENT CONSTRAINT"
        )
        assert result is None

    def test_zero_mw_gives_zero_impact(self, single_solution):
        solution, _ = single_solution
        result = solution.flow_impact(
            "NODE_00", "NODE_01", 0.0, "LN TEST - CONSTRAINT"
        )
        assert result is not None
        assert result == 0.0
