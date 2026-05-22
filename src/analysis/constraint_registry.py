"""
Canonical Constraint Registry for SPP Transmission Constraints.

This module provides a persistent lookup table mapping raw SPP constraint names
to canonical equipment IDs.  The registry survives across pipeline runs and
accumulates knowledge over time.

Matching Strategy (in order):
1. Exact raw name match (fastest path — already seen this exact string)
2. Exact fingerprint match (same equipment, different formatting)
3. Fuzzy fingerprint match via RapidFuzz (catches typos, abbreviations)
4. Unresolved → flagged for LLM or manual review

Storage:
- SQLite database: ``constraint_registry.db``
- Two tables: ``canonical`` (one row per canonical constraint) and
  ``raw_name_map`` (many-to-one raw names → canonical)

Usage
-----
::

    registry = ConstraintRegistry("src/data/constraint_registry.db")

    # Resolve a raw name
    result = registry.resolve("LN HARBINE - BEATRICE 115KV")
    if result:
        print(result.canonical_id, result.canonical_name)
    else:
        print("Unresolved — needs LLM or manual review")

    # Bulk ingest a list of raw names from today's scrape
    unresolved = registry.ingest_many(["LN HARBINE - BEATRICE", ...])

    # Export the full registry to CSV
    registry.to_csv("canonical_constraints.csv")
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .constraint_normalizer import ConstraintNameNormalizer, NormalizedConstraint

logger = logging.getLogger(__name__)

# Fuzzy matching threshold (0-100); names scoring >= this are considered matches
_FUZZY_THRESHOLD = 85

# Try to import RapidFuzz; fall back to difflib if unavailable
try:
    from rapidfuzz import fuzz, process as rfuzz_process  # type: ignore
    _HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    _HAS_RAPIDFUZZ = False
    logger.warning(
        "rapidfuzz not installed — falling back to difflib for fuzzy matching. "
        "Install with: pip install rapidfuzz"
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CanonicalConstraint:
    """A canonical constraint entry in the registry."""
    canonical_id: str           # UUID
    canonical_name: str         # Human-readable canonical name
    equipment_type: str
    from_sub: Optional[str]
    to_sub: Optional[str]
    voltage_kv: Optional[float]
    fingerprint: str            # For fuzzy matching
    source: str                 # 'auto', 'llm', 'manual'


@dataclass
class ResolveResult:
    """Result of resolving a raw constraint name."""
    raw_name: str
    canonical_id: str
    canonical_name: str
    match_type: str             # 'exact_raw', 'exact_fingerprint', 'fuzzy', 'manual', 'llm'
    fuzzy_score: Optional[float]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ConstraintRegistry:
    """
    Persistent registry mapping raw SPP constraint names to canonical IDs.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.  Created automatically if it
        doesn't exist.
    fuzzy_threshold : float
        Minimum fuzzy match score (0-100) for auto-resolution.
        Scores below this require LLM or manual review.
    """

    def __init__(
        self,
        db_path: str | Path = "src/data/constraint_registry.db",
        fuzzy_threshold: float = _FUZZY_THRESHOLD,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fuzzy_threshold = fuzzy_threshold
        self._normalizer = ConstraintNameNormalizer()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, raw_name: str) -> Optional[ResolveResult]:
        """
        Resolve a raw constraint name to its canonical entry.

        Returns
        -------
        ResolveResult if matched, None if unresolved.
        """
        # 1. Exact raw name lookup (fastest)
        result = self._lookup_exact_raw(raw_name)
        if result:
            return result

        # 2. Normalize → fingerprint
        norm = self._normalizer.normalize(raw_name)

        # 3. Exact fingerprint match
        result = self._lookup_exact_fingerprint(raw_name, norm)
        if result:
            return result

        # 4. Fuzzy fingerprint match
        result = self._fuzzy_match(raw_name, norm)
        if result:
            return result

        return None  # Unresolved

    def resolve_many(
        self, raw_names: List[str]
    ) -> Tuple[List[ResolveResult], List[str]]:
        """
        Resolve a list of raw names.

        Returns
        -------
        (resolved, unresolved_names)
        """
        resolved = []
        unresolved = []
        for name in raw_names:
            result = self.resolve(name)
            if result:
                resolved.append(result)
            else:
                unresolved.append(name)
        return resolved, unresolved

    def ingest_many(self, raw_names: List[str]) -> List[str]:
        """
        Attempt to resolve a list of raw names, auto-creating canonical entries
        for names that match at fuzzy level or above.

        For names that cannot be resolved automatically:
        - A placeholder canonical entry is created with ``source='pending'``
        - The raw name is flagged for LLM/manual review

        Returns
        -------
        List of raw names that remain unresolved (source='pending').
        """
        pending = []
        for raw_name in raw_names:
            result = self.resolve(raw_name)
            if result:
                # Already known — ensure raw_name is recorded in map
                self._upsert_raw_mapping(raw_name, result.canonical_id, result.match_type)
            else:
                # Auto-create a pending canonical from normalization
                norm = self._normalizer.normalize(raw_name)
                canonical_id = self._auto_create_canonical(norm, source="pending")
                self._upsert_raw_mapping(raw_name, canonical_id, "auto_created")
                pending.append(raw_name)

        self._conn.commit()
        logger.info(
            f"Ingested {len(raw_names)} names: "
            f"{len(raw_names) - len(pending)} resolved, "
            f"{len(pending)} pending review."
        )
        return pending

    def add_canonical(
        self,
        canonical_name: str,
        equipment_type: str,
        from_sub: Optional[str],
        to_sub: Optional[str],
        voltage_kv: Optional[float],
        source: str = "manual",
        raw_name_variants: Optional[List[str]] = None,
    ) -> str:
        """
        Manually add a canonical constraint entry.

        Returns
        -------
        canonical_id (UUID string)
        """
        norm = self._normalizer.normalize(canonical_name)
        canonical_id = str(uuid.uuid4())

        self._conn.execute(
            """
            INSERT OR IGNORE INTO canonical
                (canonical_id, canonical_name, equipment_type, from_sub, to_sub,
                 voltage_kv, fingerprint, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                canonical_id,
                canonical_name,
                equipment_type,
                from_sub,
                to_sub,
                voltage_kv,
                norm.fingerprint,
                source,
            ),
        )

        if raw_name_variants:
            for raw in raw_name_variants:
                self._upsert_raw_mapping(raw, canonical_id, "manual")

        self._conn.commit()
        return canonical_id

    def update_from_llm(
        self,
        raw_name: str,
        canonical_id: str,
        confidence: float,
    ) -> None:
        """
        Record a canonical mapping returned by the LLM resolver.

        Parameters
        ----------
        raw_name : str
            The ambiguous raw name.
        canonical_id : str
            The canonical ID assigned by the LLM.
        confidence : float
            LLM-reported confidence (0–1).
        """
        self._upsert_raw_mapping(raw_name, canonical_id, f"llm:{confidence:.2f}")
        self._conn.commit()

    def to_csv(self, output_path: str | Path) -> None:
        """Export the full registry (canonicals + raw mappings) to CSV."""
        df = pd.read_sql_query(
            """
            SELECT c.canonical_id, c.canonical_name, c.equipment_type,
                   c.from_sub, c.to_sub, c.voltage_kv, c.fingerprint,
                   c.source, r.raw_name, r.match_type
            FROM canonical c
            LEFT JOIN raw_name_map r ON c.canonical_id = r.canonical_id
            """,
            self._conn,
        )
        df.to_csv(output_path, index=False)
        logger.info(f"Exported registry to {output_path}")

    def get_pending_review(self) -> pd.DataFrame:
        """Return all raw names still pending LLM/manual review."""
        return pd.read_sql_query(
            """
            SELECT r.raw_name, c.canonical_id, c.canonical_name,
                   c.fingerprint, c.source
            FROM raw_name_map r
            JOIN canonical c ON r.canonical_id = c.canonical_id
            WHERE c.source = 'pending'
            ORDER BY r.raw_name
            """,
            self._conn,
        )

    def stats(self) -> Dict[str, int]:
        """Return simple statistics about the registry."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM canonical")
        n_canonical = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM raw_name_map")
        n_mappings = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM canonical WHERE source='pending'")
        n_pending = cursor.fetchone()[0]
        return {
            "canonical_constraints": n_canonical,
            "raw_name_mappings": n_mappings,
            "pending_review": n_pending,
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS canonical (
                canonical_id   TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                equipment_type TEXT,
                from_sub       TEXT,
                to_sub         TEXT,
                voltage_kv     REAL,
                fingerprint    TEXT,
                source         TEXT DEFAULT 'auto',
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS raw_name_map (
                raw_name       TEXT NOT NULL,
                canonical_id   TEXT NOT NULL REFERENCES canonical(canonical_id),
                match_type     TEXT,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (raw_name, canonical_id)
            );

            CREATE INDEX IF NOT EXISTS idx_fingerprint
                ON canonical(fingerprint);

            CREATE INDEX IF NOT EXISTS idx_raw_name
                ON raw_name_map(raw_name);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _lookup_exact_raw(self, raw_name: str) -> Optional[ResolveResult]:
        cursor = self._conn.execute(
            """
            SELECT c.canonical_id, c.canonical_name, r.match_type
            FROM raw_name_map r
            JOIN canonical c ON r.canonical_id = c.canonical_id
            WHERE r.raw_name = ?
            LIMIT 1
            """,
            (raw_name,),
        )
        row = cursor.fetchone()
        if row:
            return ResolveResult(
                raw_name=raw_name,
                canonical_id=row[0],
                canonical_name=row[1],
                match_type="exact_raw",
                fuzzy_score=100.0,
            )
        return None

    def _lookup_exact_fingerprint(
        self, raw_name: str, norm: NormalizedConstraint
    ) -> Optional[ResolveResult]:
        cursor = self._conn.execute(
            """
            SELECT canonical_id, canonical_name
            FROM canonical
            WHERE fingerprint = ?
              AND source != 'pending'
            LIMIT 1
            """,
            (norm.fingerprint,),
        )
        row = cursor.fetchone()
        if row:
            self._upsert_raw_mapping(raw_name, row[0], "exact_fingerprint")
            self._conn.commit()
            return ResolveResult(
                raw_name=raw_name,
                canonical_id=row[0],
                canonical_name=row[1],
                match_type="exact_fingerprint",
                fuzzy_score=100.0,
            )
        return None

    def _fuzzy_match(
        self, raw_name: str, norm: NormalizedConstraint
    ) -> Optional[ResolveResult]:
        """Fuzzy match the fingerprint against all known fingerprints."""
        cursor = self._conn.execute(
            "SELECT canonical_id, canonical_name, fingerprint FROM canonical WHERE source != 'pending'"
        )
        rows = cursor.fetchall()
        if not rows:
            return None

        candidates = {row[2]: (row[0], row[1]) for row in rows}
        fingerprints = list(candidates.keys())

        if _HAS_RAPIDFUZZ:
            match = rfuzz_process.extractOne(
                norm.fingerprint,
                fingerprints,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.fuzzy_threshold,
            )
            if match:
                fp, score, _ = match
                canonical_id, canonical_name = candidates[fp]
                self._upsert_raw_mapping(raw_name, canonical_id, f"fuzzy:{score:.0f}")
                self._conn.commit()
                return ResolveResult(
                    raw_name=raw_name,
                    canonical_id=canonical_id,
                    canonical_name=canonical_name,
                    match_type="fuzzy",
                    fuzzy_score=float(score),
                )
        else:
            # difflib fallback
            matches = difflib.get_close_matches(
                norm.fingerprint,
                fingerprints,
                n=1,
                cutoff=self.fuzzy_threshold / 100.0,
            )
            if matches:
                fp = matches[0]
                score = difflib.SequenceMatcher(None, norm.fingerprint, fp).ratio() * 100
                canonical_id, canonical_name = candidates[fp]
                self._upsert_raw_mapping(raw_name, canonical_id, f"fuzzy:{score:.0f}")
                self._conn.commit()
                return ResolveResult(
                    raw_name=raw_name,
                    canonical_id=canonical_id,
                    canonical_name=canonical_name,
                    match_type="fuzzy",
                    fuzzy_score=score,
                )

        return None

    def _auto_create_canonical(
        self, norm: NormalizedConstraint, source: str = "auto"
    ) -> str:
        """Create a new canonical entry from a NormalizedConstraint."""
        canonical_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT OR IGNORE INTO canonical
                (canonical_id, canonical_name, equipment_type, from_sub, to_sub,
                 voltage_kv, fingerprint, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                canonical_id,
                norm.canonical_name,
                norm.equipment_type,
                norm.from_sub,
                norm.to_sub,
                norm.voltage_kv,
                norm.fingerprint,
                source,
            ),
        )
        return canonical_id

    def _upsert_raw_mapping(
        self, raw_name: str, canonical_id: str, match_type: str
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO raw_name_map (raw_name, canonical_id, match_type)
            VALUES (?, ?, ?)
            """,
            (raw_name, canonical_id, match_type),
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def get_registry(
    db_path: str | Path = "src/data/constraint_registry.db",
) -> ConstraintRegistry:
    """Get a ConstraintRegistry instance (creates DB if needed)."""
    return ConstraintRegistry(db_path=db_path)
