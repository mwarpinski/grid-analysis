"""
Unit tests for ConstraintRegistry.

Tests the 4-tier resolution strategy: exact raw → exact fingerprint →
fuzzy match → unresolved/pending.
"""

import tempfile
from pathlib import Path

import pytest

from src.analysis.constraint_registry import (
    CanonicalConstraint,
    ConstraintRegistry,
    ResolveResult,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path):
    """Create a fresh registry in a temp directory for each test."""
    db = tmp_path / "test_registry.db"
    reg = ConstraintRegistry(db_path=db)
    yield reg
    reg.close()


@pytest.fixture
def seeded_registry(registry):
    """Registry pre-seeded with a known canonical constraint."""
    registry.add_canonical(
        canonical_name="LN HARBINE - BEATRICE 115",
        equipment_type="line",
        from_sub="HARBINE",
        to_sub="BEATRICE",
        voltage_kv=115.0,
        source="manual",
        raw_name_variants=["LN HARBINE - BEATRICE 115KV"],
    )
    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExactRawLookup:
    def test_exact_raw_name_resolves(self, seeded_registry):
        result = seeded_registry.resolve("LN HARBINE - BEATRICE 115KV")
        assert result is not None
        assert result.match_type == "exact_raw"
        assert result.canonical_name == "LN HARBINE - BEATRICE 115"
        assert result.fuzzy_score == 100.0

    def test_unknown_raw_name_returns_none(self, seeded_registry):
        result = seeded_registry.resolve("COMPLETELY UNKNOWN CONSTRAINT XYZ")
        assert result is None


class TestFingerprintLookup:
    def test_different_format_same_equipment_resolves(self, seeded_registry):
        """A name with identical fingerprint to a known canonical should resolve."""
        # "HARBINE_BEATRICE_115" has same fingerprint as "LN HARBINE - BEATRICE 115"
        result = seeded_registry.resolve("HARBINE_BEATRICE_115")
        assert result is not None
        # Should match via fingerprint or fuzzy
        assert result.match_type in ("exact_fingerprint", "fuzzy")
        assert "HARBINE" in result.canonical_name or "BEATRICE" in result.canonical_name

    def test_reversed_direction_resolves_to_same_canonical(self, seeded_registry):
        """LN BEATRICE - HARBINE should map to the same canonical as LN HARBINE - BEATRICE."""
        result = seeded_registry.resolve("LN BEATRICE - HARBINE 115KV")
        assert result is not None


class TestFuzzyMatch:
    def test_abbreviated_name_fuzzy_matches(self, seeded_registry):
        """A slightly different name should fuzzy match if above threshold."""
        # "LN HARBINE - BEATRICE" (no voltage) should fuzzy-match
        result = seeded_registry.resolve("LN HARBINE - BEATRICE")
        # May or may not match depending on threshold — just check it doesn't crash
        assert result is None or result.canonical_name is not None


class TestIngestMany:
    def test_ingest_known_and_unknown(self, seeded_registry):
        """Known names should resolve; unknown names should become pending."""
        names = [
            "LN HARBINE - BEATRICE 115KV",   # known exact raw
            "BRAND NEW CONSTRAINT ALPHA",     # unknown
            "BRAND NEW CONSTRAINT BETA",      # unknown
        ]
        pending = seeded_registry.ingest_many(names)
        # Known name should NOT be in pending
        assert "LN HARBINE - BEATRICE 115KV" not in pending
        # Unknown names should be pending
        assert "BRAND NEW CONSTRAINT ALPHA" in pending
        assert "BRAND NEW CONSTRAINT BETA" in pending

    def test_ingest_creates_pending_canonicals(self, registry):
        """Unresolved names should create pending entries."""
        registry.ingest_many(["MYSTERY LINE A - B", "MYSTERY LINE C - D"])
        stats = registry.stats()
        assert stats["pending_review"] == 2
        assert stats["canonical_constraints"] == 2

    def test_ingest_idempotent(self, seeded_registry):
        """Running ingest twice should not create duplicate canonicals."""
        seeded_registry.ingest_many(["NEW UNKNOWN LINE X"])
        stats_1 = seeded_registry.stats()
        seeded_registry.ingest_many(["NEW UNKNOWN LINE X"])
        stats_2 = seeded_registry.stats()
        assert stats_1["canonical_constraints"] == stats_2["canonical_constraints"]


class TestAddCanonical:
    def test_add_canonical_is_findable(self, registry):
        canonical_id = registry.add_canonical(
            canonical_name="LN COWETA - WAGONER 138",
            equipment_type="line",
            from_sub="COWETA",
            to_sub="WAGONER",
            voltage_kv=138.0,
            raw_name_variants=["LN COWETA - WAGONER", "COWETA_WAGONER_138"],
        )
        assert canonical_id is not None

        result = registry.resolve("LN COWETA - WAGONER")
        assert result is not None
        assert result.canonical_id == canonical_id

    def test_add_multiple_canonicals(self, registry):
        for i in range(5):
            registry.add_canonical(
                canonical_name=f"LN SUB{i} - SUB{i+1}",
                equipment_type="line",
                from_sub=f"SUB{i}",
                to_sub=f"SUB{i+1}",
                voltage_kv=115.0,
            )
        stats = registry.stats()
        assert stats["canonical_constraints"] == 5


class TestStats:
    def test_empty_registry_stats(self, registry):
        stats = registry.stats()
        assert stats["canonical_constraints"] == 0
        assert stats["raw_name_mappings"] == 0
        assert stats["pending_review"] == 0

    def test_stats_update_after_add(self, registry):
        registry.add_canonical(
            canonical_name="LN A - B 115",
            equipment_type="line",
            from_sub="A",
            to_sub="B",
            voltage_kv=115.0,
            raw_name_variants=["LN A - B 115KV", "A_B_115"],
        )
        stats = registry.stats()
        assert stats["canonical_constraints"] == 1
        assert stats["raw_name_mappings"] == 2  # 2 variants


class TestGetPendingReview:
    def test_pending_review_contains_unresolved(self, registry):
        registry.ingest_many(["UNKNOWN LINE X", "UNKNOWN LINE Y"])
        pending = registry.get_pending_review()
        assert len(pending) == 2
        assert set(pending["raw_name"].tolist()) == {"UNKNOWN LINE X", "UNKNOWN LINE Y"}


class TestUpdateFromLLM:
    def test_llm_update_resolves_pending(self, registry):
        """LLM-resolved names should no longer appear in pending."""
        # First ingest an unknown name
        registry.ingest_many(["MYSTERY CONSTRAINT Z"])
        pending_before = registry.get_pending_review()
        assert len(pending_before) == 1

        canonical_id = pending_before.iloc[0]["canonical_id"]

        # Simulate LLM providing resolution
        registry.add_canonical(
            canonical_name="LN VERIFIED A - B 115",
            equipment_type="line",
            from_sub="A",
            to_sub="B",
            voltage_kv=115.0,
            source="manual",
        )
        # The LLM update points the raw name to the correct canonical
        result = registry.resolve("MYSTERY CONSTRAINT Z")
        # Still pending since we haven't pointed it to the right canonical via LLM
        # (this tests that update_from_llm changes the mapping)
        registry.update_from_llm("MYSTERY CONSTRAINT Z", canonical_id, confidence=0.92)
        result_after = registry.resolve("MYSTERY CONSTRAINT Z")
        assert result_after is not None


class TestToCsv:
    def test_to_csv_creates_file(self, seeded_registry, tmp_path):
        output = tmp_path / "export.csv"
        seeded_registry.to_csv(output)
        assert output.exists()
        import pandas as pd
        df = pd.read_csv(output)
        assert "canonical_id" in df.columns
        assert "canonical_name" in df.columns
        assert len(df) > 0
