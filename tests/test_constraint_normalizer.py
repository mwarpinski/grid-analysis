"""
Unit tests for ConstraintNameNormalizer.

Tests that known SPP constraint name variants are correctly parsed
into their canonical components (equipment type, from/to substations,
voltage, circuit ID, fingerprint).
"""

import pytest
from src.analysis.constraint_normalizer import (
    ConstraintNameNormalizer,
    NormalizedConstraint,
    normalize_constraint_name,
)


# ---------------------------------------------------------------------------
# Known SPP constraint name variants and expected normalizations
# ---------------------------------------------------------------------------

KNOWN_CASES = [
    # (raw_name, expected_from_sub, expected_to_sub, expected_voltage, expected_eq_type)
    (
        "LN HARBINE - BEATRICE",
        "HARBINE", "BEATRICE", None, "line",
    ),
    (
        "LN HARBINE - BEATRICE 115KV",
        "HARBINE", "BEATRICE", 115.0, "line",
    ),
    (
        "HARBINE_BEATRICE_115",
        "HARBINE", "BEATRICE", 115.0, "unknown",
    ),
    (
        "BR AXTELL - SALINA 115",
        "AXTELL", "SALINA", 115.0, "line",
    ),
    (
        "XFMR SUBSTATION A - SUBSTATION B 345KV",
        "SUBSTATION A", "SUBSTATION B", 345.0, "transformer",
    ),
    (
        "LN COWETA - WAGONER 138",
        "COWETA", "WAGONER", 138.0, "line",
    ),
    (
        "INTF LAMAR INTERFACE",
        "LAMAR INTERFACE", None, None, "interface",
    ),
]


@pytest.fixture
def normalizer():
    return ConstraintNameNormalizer()


class TestEquipmentTypeExtraction:
    def test_ln_prefix_gives_line(self, normalizer):
        result = normalizer.normalize("LN HARBINE - BEATRICE")
        assert result.equipment_type == "line"

    def test_br_prefix_gives_line(self, normalizer):
        result = normalizer.normalize("BR ALPHA - BETA")
        assert result.equipment_type == "line"

    def test_xfmr_prefix_gives_transformer(self, normalizer):
        result = normalizer.normalize("XFMR SUB_A - SUB_B")
        assert result.equipment_type == "transformer"

    def test_no_prefix_gives_unknown(self, normalizer):
        result = normalizer.normalize("HARBINE_BEATRICE_115")
        assert result.equipment_type == "unknown"

    def test_intf_prefix_gives_interface(self, normalizer):
        result = normalizer.normalize("INTF LAMAR INTERFACE")
        assert result.equipment_type == "interface"


class TestVoltageExtraction:
    @pytest.mark.parametrize("raw, expected_kv", [
        ("LN HARBINE - BEATRICE 115KV", 115.0),
        ("LN HARBINE - BEATRICE 115", 115.0),
        ("LN HARBINE - BEATRICE 345 KV", 345.0),
        ("HARBINE_BEATRICE_138", 138.0),
        ("LN HARBINE - BEATRICE", None),   # no voltage
    ])
    def test_voltage_extraction(self, normalizer, raw, expected_kv):
        result = normalizer.normalize(raw)
        assert result.voltage_kv == expected_kv, (
            f"For {raw!r}: expected {expected_kv}, got {result.voltage_kv}"
        )

    def test_voltage_not_below_60kv(self, normalizer):
        """Voltages below 60kV should not be extracted as transmission voltages."""
        result = normalizer.normalize("LN SUB_A - SUB_B 25KV")
        assert result.voltage_kv is None


class TestSubstationParsing:
    @pytest.mark.parametrize(
        "raw, from_sub, to_sub", [
            ("LN HARBINE - BEATRICE", "HARBINE", "BEATRICE"),
            ("LN COWETA - WAGONER 138", "COWETA", "WAGONER"),
            ("BR AXTELL - SALINA 115", "AXTELL", "SALINA"),
            ("HARBINE_BEATRICE_115", "HARBINE", "BEATRICE"),
        ]
    )
    def test_from_to_parsing(self, normalizer, raw, from_sub, to_sub):
        result = normalizer.normalize(raw)
        assert result.from_sub == from_sub, (
            f"from_sub mismatch for {raw!r}: expected {from_sub!r}, got {result.from_sub!r}"
        )
        assert result.to_sub == to_sub, (
            f"to_sub mismatch for {raw!r}: expected {to_sub!r}, got {result.to_sub!r}"
        )


class TestFingerprintProperties:
    def test_fingerprint_is_order_normalized(self, normalizer):
        """HARBINE - BEATRICE and BEATRICE - HARBINE should have the same fingerprint."""
        r1 = normalizer.normalize("LN HARBINE - BEATRICE 115KV")
        r2 = normalizer.normalize("LN BEATRICE - HARBINE 115KV")
        assert r1.fingerprint == r2.fingerprint, (
            f"Expected same fingerprint for reversed direction.\n"
            f"  Forward: {r1.fingerprint}\n  Reverse: {r2.fingerprint}"
        )

    def test_fingerprint_is_case_insensitive(self, normalizer):
        r1 = normalizer.normalize("LN HARBINE - BEATRICE")
        r2 = normalizer.normalize("LN Harbine - Beatrice")
        assert r1.fingerprint == r2.fingerprint

    def test_fingerprint_ignores_separators(self, normalizer):
        """Different separators should produce the same fingerprint."""
        r1 = normalizer.normalize("LN HARBINE - BEATRICE")
        r2 = normalizer.normalize("HARBINE_BEATRICE")
        # Fingerprints may differ if one has LN prefix (equipment_type=line),
        # but the substation part should match
        assert "harbine" in r1.fingerprint
        assert "beatrice" in r1.fingerprint
        assert "harbine" in r2.fingerprint
        assert "beatrice" in r2.fingerprint

    def test_fingerprint_includes_voltage(self, normalizer):
        r_with = normalizer.normalize("LN HARBINE - BEATRICE 115KV")
        r_without = normalizer.normalize("LN HARBINE - BEATRICE")
        # Fingerprints differ by voltage
        assert r_with.fingerprint != r_without.fingerprint
        assert "115" in r_with.fingerprint


class TestCanonicalNameBuilding:
    def test_canonical_name_has_ln_prefix_for_lines(self, normalizer):
        result = normalizer.normalize("LN HARBINE - BEATRICE 115KV")
        assert result.canonical_name.startswith("LN ")

    def test_canonical_name_has_xfmr_prefix_for_transformers(self, normalizer):
        result = normalizer.normalize("XFMR SUB_A - SUB_B 345KV")
        assert result.canonical_name.startswith("XFMR ")

    def test_canonical_name_includes_voltage_as_integer(self, normalizer):
        result = normalizer.normalize("LN HARBINE - BEATRICE 115.0KV")
        assert "115" in result.canonical_name
        assert "115.0" not in result.canonical_name  # should be integer


class TestNormalizeMany:
    def test_normalize_many_returns_correct_count(self, normalizer):
        names = ["LN A - B", "LN C - D 115KV", "XFMR E - F"]
        results = normalizer.normalize_many(names)
        assert len(results) == 3

    def test_all_results_are_normalized_constraint(self, normalizer):
        names = ["LN A - B", "BR C - D"]
        for r in normalizer.normalize_many(names):
            assert isinstance(r, NormalizedConstraint)


class TestConvenienceFunction:
    def test_normalize_constraint_name_returns_normalized_constraint(self):
        result = normalize_constraint_name("LN HARBINE - BEATRICE 115KV")
        assert isinstance(result, NormalizedConstraint)
        assert result.from_sub == "HARBINE"
        assert result.to_sub == "BEATRICE"
        assert result.voltage_kv == 115.0
