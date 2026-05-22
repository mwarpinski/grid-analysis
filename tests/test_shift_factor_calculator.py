"""
Unit tests for ShiftFactorCalculator.

Tests use synthetic data where ground-truth shift factors are known,
so we can verify the solver produces correct results.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.multi_constraint_sf import (
    ShiftFactorCalculator,
    ShiftFactorSolution,
    calculate_shift_factors,
    summarize_solution,
    validate_shift_factors,
)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_single_constraint_data(
    n_intervals: int = 50,
    n_nodes: int = 5,
    true_sf: float = 0.25,
    shadow_price_mean: float = 100.0,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic data for a single binding constraint.

    MCC[node, t] = true_sf × ShadowPrice[t] + noise

    Returns
    -------
    (constraints_df, lmp_df)
    """
    rng = np.random.default_rng(seed)
    intervals = pd.date_range("2024-11-01 00:05", periods=n_intervals, freq="5min")
    nodes = [f"NODE_{i:02d}" for i in range(n_nodes)]
    true_sfs = {node: true_sf + i * 0.05 for i, node in enumerate(nodes)}

    shadow_prices = rng.normal(shadow_price_mean, 20.0, size=n_intervals)

    constraint_rows = [
        {"interval": iv, "constraint_name": "LN TEST - CONSTRAINT", "shadow_price": sp}
        for iv, sp in zip(intervals, shadow_prices)
    ]

    lmp_rows = []
    for node in nodes:
        sf = true_sfs[node]
        for iv, sp in zip(intervals, shadow_prices):
            mcc = sf * sp + rng.normal(0, noise_std)
            lmp_rows.append(
                {"interval": iv, "settlement_location": node, "mcc": mcc}
            )

    return pd.DataFrame(constraint_rows), pd.DataFrame(lmp_rows), true_sfs


def make_multi_constraint_data(
    n_intervals: int = 100,
    n_nodes: int = 4,
    true_sfs: dict | None = None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Generate synthetic data for two simultaneously binding constraints.

    MCC[node, t] = SF[node, C1] × SP_C1[t] + SF[node, C2] × SP_C2[t]
    """
    rng = np.random.default_rng(seed)
    intervals = pd.date_range("2024-11-01 00:05", periods=n_intervals, freq="5min")
    nodes = [f"NODE_{i:02d}" for i in range(n_nodes)]

    if true_sfs is None:
        true_sfs = {
            node: {
                "LN ALPHA - BETA": 0.3 + i * 0.05,
                "LN GAMMA - DELTA": -0.15 + i * 0.03,
            }
            for i, node in enumerate(nodes)
        }

    sp_c1 = rng.normal(80.0, 15.0, size=n_intervals)
    sp_c2 = rng.normal(50.0, 10.0, size=n_intervals)

    constraint_rows = []
    for iv, s1, s2 in zip(intervals, sp_c1, sp_c2):
        constraint_rows.append(
            {"interval": iv, "constraint_name": "LN ALPHA - BETA", "shadow_price": s1}
        )
        constraint_rows.append(
            {"interval": iv, "constraint_name": "LN GAMMA - DELTA", "shadow_price": s2}
        )

    lmp_rows = []
    for node in nodes:
        sf1 = true_sfs[node]["LN ALPHA - BETA"]
        sf2 = true_sfs[node]["LN GAMMA - DELTA"]
        for iv, s1, s2 in zip(intervals, sp_c1, sp_c2):
            mcc = sf1 * s1 + sf2 * s2
            lmp_rows.append(
                {"interval": iv, "settlement_location": node, "mcc": mcc}
            )

    return pd.DataFrame(constraint_rows), pd.DataFrame(lmp_rows), true_sfs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShiftFactorCalculatorSingleConstraint:
    def test_basic_single_constraint_recovers_true_sf(self):
        """Verify the calculator recovers known shift factors for a single constraint."""
        constraints_df, lmp_df, true_sfs = make_single_constraint_data(
            n_intervals=50, n_nodes=3, true_sf=0.25
        )

        calc = ShiftFactorCalculator(min_intervals=5, tolerance=1e-4)
        solution = calc.fit(constraints_df, lmp_df)
        solution.flatten()

        for node, true_sf in true_sfs.items():
            estimated = solution.get(node, "LN TEST - CONSTRAINT")
            assert estimated is not None, f"No SF for {node}"
            assert abs(estimated - true_sf) < 0.05, (
                f"Node {node}: expected SF ≈ {true_sf:.3f}, got {estimated:.3f}"
            )

    def test_sf_within_physical_bounds(self):
        """All shift factors should be within [-1, 1]."""
        constraints_df, lmp_df, _ = make_single_constraint_data()
        solution = calculate_shift_factors(constraints_df=constraints_df, lmp_df=lmp_df)
        df = solution.to_dataframe()
        assert (df["shift_factor"].abs() <= 1.1).all(), (
            "Shift factors should be approximately within [-1, 1]"
        )

    def test_no_sf_when_insufficient_intervals(self):
        """With min_intervals=50 and only 5 intervals, nothing should be solved."""
        constraints_df, lmp_df, _ = make_single_constraint_data(n_intervals=5)
        calc = ShiftFactorCalculator(min_intervals=50, tolerance=1e-4)
        solution = calc.fit(constraints_df, lmp_df)
        solution.flatten()
        df = solution.to_dataframe()
        assert df.empty, "Should produce no shift factors when below min_intervals"

    def test_to_dataframe_columns(self):
        """Output DataFrame should have required columns."""
        constraints_df, lmp_df, _ = make_single_constraint_data()
        solution = calculate_shift_factors(constraints_df=constraints_df, lmp_df=lmp_df)
        df = solution.to_dataframe()
        assert set(df.columns) == {"node", "constraint", "shift_factor"}


class TestShiftFactorCalculatorMultiConstraint:
    def test_multi_constraint_recovers_true_sfs(self):
        """Verify the multi-constraint solver recovers both known SFs."""
        constraints_df, lmp_df, true_sfs = make_multi_constraint_data(
            n_intervals=100, n_nodes=3, seed=42
        )
        calc = ShiftFactorCalculator(min_intervals=5)
        solution = calc.fit(constraints_df, lmp_df)
        solution.flatten()

        for node, node_sfs in true_sfs.items():
            for constraint, true_sf in node_sfs.items():
                estimated = solution.get(node, constraint)
                assert estimated is not None, f"No SF for ({node}, {constraint})"
                assert abs(estimated - true_sf) < 0.1, (
                    f"({node}, {constraint}): expected {true_sf:.3f}, got {estimated:.3f}"
                )

    def test_regime_identified_correctly(self):
        """With two always-binding constraints, there should be exactly one regime."""
        constraints_df, lmp_df, _ = make_multi_constraint_data(n_intervals=30)
        calc = ShiftFactorCalculator(min_intervals=5)
        solution = calc.fit(constraints_df, lmp_df)
        assert len(solution.regimes) == 1

    def test_r2_scores_are_high_with_clean_data(self):
        """R² should be near 1.0 when there's no noise."""
        constraints_df, lmp_df, _ = make_multi_constraint_data(n_intervals=50)
        calc = ShiftFactorCalculator(min_intervals=5)
        solution = calc.fit(constraints_df, lmp_df)
        for regime in solution.regimes.values():
            for node, r2 in regime.r2_scores.items():
                assert r2 > 0.99, f"R² for {node} = {r2:.4f}, expected > 0.99"


class TestValidateShiftFactors:
    def test_reconstruction_matches_observed(self):
        """Reconstructed MCCs should match observed MCCs when SFs are correct."""
        constraints_df, lmp_df, _ = make_single_constraint_data(n_intervals=30, n_nodes=2)
        solution = calculate_shift_factors(constraints_df=constraints_df, lmp_df=lmp_df)
        sf_df = solution.to_dataframe()

        validation = validate_shift_factors(sf_df, lmp_df, constraints_df, tolerance=1.0)
        flagged = validation[validation["flagged"]].shape[0]
        total = len(validation)
        # With low-noise data, nearly all reconstructions should be accurate
        assert flagged / max(1, total) < 0.05, (
            f"Too many flagged reconstructions: {flagged}/{total}"
        )


class TestSummarizeSolution:
    def test_summary_has_expected_columns(self):
        constraints_df, lmp_df, _ = make_single_constraint_data()
        solution = calculate_shift_factors(constraints_df=constraints_df, lmp_df=lmp_df)
        summary = summarize_solution(solution)
        expected_cols = {
            "constraints", "n_intervals", "n_constraints",
            "n_nodes_solved", "mean_r2", "solved"
        }
        assert set(summary.columns) == expected_cols

    def test_summary_marks_solved_regimes(self):
        constraints_df, lmp_df, _ = make_single_constraint_data(n_intervals=50)
        solution = calculate_shift_factors(
            constraints_df=constraints_df, lmp_df=lmp_df, min_intervals=5
        )
        summary = summarize_solution(solution)
        assert summary["solved"].any(), "At least one regime should be solved"


class TestInputValidation:
    def test_missing_constraint_columns_raises(self):
        bad_df = pd.DataFrame({"interval": [], "wrong_col": []})
        lmp_df = pd.DataFrame({"interval": [], "settlement_location": [], "mcc": []})
        calc = ShiftFactorCalculator()
        with pytest.raises(ValueError, match="missing required columns"):
            calc.fit(bad_df, lmp_df)

    def test_missing_lmp_columns_raises(self):
        constraints_df = pd.DataFrame(
            {"interval": [], "constraint_name": [], "shadow_price": []}
        )
        bad_lmp = pd.DataFrame({"interval": [], "wrong": []})
        calc = ShiftFactorCalculator()
        with pytest.raises(ValueError, match="missing required columns"):
            calc.fit(constraints_df, bad_lmp)

    def test_accepts_uppercase_column_names(self):
        """Calculator should handle SPP's original CamelCase column names."""
        constraints_df, lmp_df, _ = make_single_constraint_data(n_intervals=20, n_nodes=2)
        # Rename to SPP-style
        constraints_df = constraints_df.rename(columns={
            "interval": "Interval",
            "constraint_name": "Constraint Name",
            "shadow_price": "Shadow Price",
        })
        lmp_df = lmp_df.rename(columns={
            "interval": "Interval",
            "settlement_location": "Settlement Location",
            "mcc": "MCC",
        })
        calc = ShiftFactorCalculator(min_intervals=5)
        # Should not raise
        solution = calc.fit(constraints_df, lmp_df)
        assert solution is not None
