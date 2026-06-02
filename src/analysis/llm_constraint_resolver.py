"""
LLM Constraint Resolver — Phase 14
===================================

Decodes SPP RTBM constraint names (e.g. "FPLWDWCLEMEN", "CORNAPTERSUN")
into structured physical metadata:

    {
        "constraint":    "FPLWDWCLEMEN",
        "from_sub":      "FPL Wind (Woodward)",
        "to_sub":        "Clemens",
        "voltage_kv":    345,
        "equipment_type":"LINE",
        "confidence":    "HIGH",
        "reasoning":     "FPL = FPL Wind, WDW = Woodward substation, ..."
    }

SPP constraint names follow an undocumented but consistent encoding:
  - 6-char from-bus abbreviation + 6-char to-bus abbreviation
  - Transformer constraints use "XFR" token
  - Interface ties use "TIES" / "SPPS" prefix
  - Contingency monitors start with TMP, TEMP, or TP (skipped — they're bus IDs)

Scope
-----
Only pure-alpha names (no underscore) are resolved. TMP/TEMP/TP contingency
codes embed PSSE bus numbers — no human-readable resolution is possible without
the full SPP bus database.

Persistence
-----------
Results are written to DuckDB at src/data/topology/resolved_constraints.duckdb.
On subsequent runs, already-resolved names are skipped — the API is only called
once per unique name.

Usage
-----
Standalone::

    python -m src.analysis.llm_constraint_resolver

Via server startup (non-blocking)::

    from src.analysis.llm_constraint_resolver import resolver
    asyncio.create_task(resolver.resolve_all_pending())

REST API (see routers/constraint_resolver.py)::

    GET  /api/v1/resolver/names            — all resolved names
    GET  /api/v1/resolver/names/{name}     — single lookup
    POST /api/v1/resolver/resolve          — trigger on-demand resolution
    GET  /api/v1/resolver/status           — queue depth, API usage stats
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent / "data" / "topology" / "resolved_constraints.db"
_DATA_DIR = Path(__file__).parent.parent / "data" / "timeseries" / "constraints"

# Constraint name patterns that are NOT worth resolving
_SKIP_PREFIXES = ("TMP", "TEMP", "TP")  # contingency monitors with bus IDs
_SKIP_PATTERNS = re.compile(r".*_\d+$")  # anything ending in _<number>

# Rate limiting — be gentle with the API
_BATCH_SIZE = 10          # names per LLM call
_DELAY_BETWEEN_BATCHES = 2.0  # seconds


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ResolvedConstraint:
    constraint:     str
    from_sub:       Optional[str] = None
    to_sub:         Optional[str] = None
    voltage_kv:     Optional[int] = None
    equipment_type: Optional[str] = None   # LINE, TRANSFORMER, INTERFACE, TIE
    confidence:     str = "UNKNOWN"         # HIGH / MEDIUM / LOW / UNKNOWN
    reasoning:      Optional[str] = None
    resolved_at:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_used:     str = "claude-haiku-4-5-20251001"
    error:          Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Persistence layer (SQLite — lightweight, no DuckDB dependency at import)
# ---------------------------------------------------------------------------

class ResolverStore:
    """SQLite-backed cache of resolved constraint names."""

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resolved_constraints (
                    `constraint`    TEXT PRIMARY KEY,
                    from_sub        TEXT,
                    to_sub          TEXT,
                    voltage_kv      INTEGER,
                    equipment_type  TEXT,
                    confidence      TEXT DEFAULT 'UNKNOWN',
                    reasoning       TEXT,
                    resolved_at     TEXT,
                    model_used      TEXT,
                    error           TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resolution_stats (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()

    def get(self, constraint: str) -> Optional[ResolvedConstraint]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM resolved_constraints WHERE `constraint` = ?",
                (constraint,)
            ).fetchone()
        if row is None:
            return None
        return ResolvedConstraint(**dict(row))

    def get_all(self) -> list[ResolvedConstraint]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM resolved_constraints ORDER BY `constraint`"
            ).fetchall()
        return [ResolvedConstraint(**dict(r)) for r in rows]

    def upsert(self, rc: ResolvedConstraint):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO resolved_constraints
                    (`constraint`, from_sub, to_sub, voltage_kv, equipment_type,
                     confidence, reasoning, resolved_at, model_used, error)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(`constraint`) DO UPDATE SET
                    from_sub=excluded.from_sub,
                    to_sub=excluded.to_sub,
                    voltage_kv=excluded.voltage_kv,
                    equipment_type=excluded.equipment_type,
                    confidence=excluded.confidence,
                    reasoning=excluded.reasoning,
                    resolved_at=excluded.resolved_at,
                    model_used=excluded.model_used,
                    error=excluded.error
            """, (
                rc.constraint, rc.from_sub, rc.to_sub, rc.voltage_kv,
                rc.equipment_type, rc.confidence, rc.reasoning,
                rc.resolved_at, rc.model_used, rc.error,
            ))
            conn.commit()

    def upsert_many(self, records: list[ResolvedConstraint]):
        for rc in records:
            self.upsert(rc)

    def already_resolved(self) -> set[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT `constraint` FROM resolved_constraints WHERE error IS NULL"
            ).fetchall()
        return {r[0] for r in rows}

    def increment_stat(self, key: str, by: int = 1):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO resolution_stats (key, value) VALUES (?,?)
                ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT)
            """, (key, str(by), by))
            conn.commit()

    def get_stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT key, value FROM resolution_stats").fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def count(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total    = conn.execute("SELECT COUNT(*) FROM resolved_constraints").fetchone()[0]
            resolved = conn.execute(
                "SELECT COUNT(*) FROM resolved_constraints WHERE error IS NULL AND confidence != 'UNKNOWN'"
            ).fetchone()[0]
            errors   = conn.execute(
                "SELECT COUNT(*) FROM resolved_constraints WHERE error IS NOT NULL"
            ).fetchone()[0]
        return {"total": total, "resolved": resolved, "errors": errors}


# ---------------------------------------------------------------------------
# Constraint name classifier
# ---------------------------------------------------------------------------

def classify_constraint(name: str) -> str:
    """
    Returns the class of a constraint name:
      'contingency'  — TMP/TEMP/TP prefix with bus IDs (skip)
      'flowgate'     — underscore-separated (WTP_XXXXX pattern)
      'named'        — pure-alpha SPP constraint name (resolve these)
    """
    upper = name.upper()
    if upper.startswith(_SKIP_PREFIXES):
        return "contingency"
    if "_" in name:
        return "flowgate"
    return "named"


def collect_unique_names(data_dir: Path = _DATA_DIR) -> set[str]:
    """Scan all parquet constraint files and return the unique named constraint set."""
    try:
        import pandas as pd
    except ImportError:
        return set()

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        return set()

    chunks = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f, columns=["constraint_name"])
            chunks.append(df)
        except Exception:
            continue

    if not chunks:
        return set()

    all_names = pd.concat(chunks)["constraint_name"].dropna().unique()
    return {n for n in all_names if classify_constraint(n) == "named"}


# ---------------------------------------------------------------------------
# LLM resolution engine
# ---------------------------------------------------------------------------

# System prompt that teaches the model SPP constraint naming conventions
_SYSTEM_PROMPT = """You are an expert in SPP (Southwest Power Pool) electricity market operations and power system topology.

SPP RTBM binding constraint names encode the physical transmission elements being monitored. The naming convention concatenates abbreviated substation/equipment names:
- Format: [FROM_STATION_ABBREV][TO_STATION_ABBREV] (typically 5-6 chars each, 10-12 total)
- Transformer constraints often include "XFR" token
- Interface/tie-line constraints may include "TIES", "SPPS", "SPSNM"
- Hub aggregates: "GGS" = Grainbelt Gateway South, "NDEX" = Nebraska Exchange Hub

Common SPP substation abbreviations:
- FPL = FPL (NextEra) wind assets in OK/KS/TX
- WDW = Woodward (OK)
- CLEM/CLEME = Clemens (KS)
- HANWAH = Hanwha (solar, OK)
- FORTIE = Fort Ticonderoga or similar
- KNONHA = Knob Noster to Harrisonville (MO)
- POSSHA = Post Oak to Shawnee
- SMKSUM = Smoke Summit
- FTTXFR = Fort transformer
- GGS = Grainbelt Gateway South
- MRTPAN = Martin to Pawnee
- HUTMRT = Hutton to Martin
- ALCMIR = Alcorn to Miranda
- BRKXF = Brookings transformer
- CIMXF = Cimarron transformer
- NASXFR = Nashville transformer
- NEB456 = Nebraska 456kV bus
- NEBROK = Nebraska to Rokeby
- WAG454 = Waggoner 454 bus
- OSAWEBC/CLESOO = Osawatomie to Webster City / Cleveland to Sooner
- PSOSWE/PCOTIE = PSO to SWEPCO / Pacific Coast tie

For each constraint name, decode it into its constituent parts and provide structured metadata.

Respond ONLY with a JSON array. Each element must have exactly these fields:
{
  "constraint": "<original name>",
  "from_sub": "<from substation or entity, human-readable>",
  "to_sub": "<to substation or entity, human-readable>",
  "voltage_kv": <integer kV class or null>,
  "equipment_type": "<LINE|TRANSFORMER|INTERFACE|TIE|HUB|UNKNOWN>",
  "confidence": "<HIGH|MEDIUM|LOW>",
  "reasoning": "<1-2 sentence explanation of your decoding>"
}

If you cannot decode a name with any confidence, use confidence=LOW and explain what you tried.
"""


def _parse_llm_response(response_text: str, submitted_names: list[str]) -> list[ResolvedConstraint]:
    """Parse LLM JSON response into ResolvedConstraint records."""
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"LLM response JSON parse failed: {e}\nResponse: {text[:500]}")
        # Return error records for all submitted names
        return [
            ResolvedConstraint(constraint=n, error=f"JSON parse failed: {e}")
            for n in submitted_names
        ]

    results = []
    returned_names = set()

    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("constraint", "")
        if not name:
            continue
        returned_names.add(name)

        results.append(ResolvedConstraint(
            constraint=name,
            from_sub=item.get("from_sub"),
            to_sub=item.get("to_sub"),
            voltage_kv=item.get("voltage_kv"),
            equipment_type=item.get("equipment_type", "UNKNOWN"),
            confidence=item.get("confidence", "LOW"),
            reasoning=item.get("reasoning"),
        ))

    # Handle any names the LLM didn't return
    for name in submitted_names:
        if name not in returned_names:
            results.append(ResolvedConstraint(
                constraint=name,
                error="LLM did not return a result for this name",
                confidence="UNKNOWN",
            ))

    return results


# ---------------------------------------------------------------------------
# Main resolver class
# ---------------------------------------------------------------------------

class LLMConstraintResolver:
    """
    Resolves SPP constraint names to physical metadata using Claude.

    Maintains a SQLite cache so each name is only resolved once.
    Designed to run as a background task from the FastAPI server lifespan.
    """

    def __init__(
        self,
        store: Optional[ResolverStore] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.store = store or ResolverStore()
        self.model = model or os.environ.get("RESOLVER_MODEL") or "claude-haiku-4-5-20251001"
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._running = False
        self._last_run: Optional[datetime] = None
        self._resolved_count = 0
        self._error_count = 0

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    def status(self) -> dict:
        counts = self.store.count()
        stats  = self.store.get_stats()
        return {
            "has_api_key":    self.has_api_key,
            "running":        self._running,
            "last_run":       self._last_run.isoformat() if self._last_run else None,
            "resolved_count": self._resolved_count,
            "error_count":    self._error_count,
            "db": counts,
            "api_calls_made": stats.get("api_calls", 0),
            "model":          self.model,
        }

    async def resolve_batch(self, names: list[str]) -> list[ResolvedConstraint]:
        """Call Claude with a batch of constraint names and parse the response."""
        if not self.has_api_key:
            logger.warning("No ANTHROPIC_API_KEY set — skipping LLM resolution.")
            return [
                ResolvedConstraint(constraint=n, error="No API key configured")
                for n in names
            ]

        try:
            import anthropic
        except ImportError:
            return [
                ResolvedConstraint(constraint=n, error="anthropic package not installed")
                for n in names
            ]

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        # Build the user message
        names_block = "\n".join(f"  - {n}" for n in names)
        user_message = (
            f"Decode the following {len(names)} SPP RTBM constraint names into structured metadata:\n\n"
            f"{names_block}\n\n"
            "Return a JSON array with one object per name."
        )

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            self.store.increment_stat("api_calls")
            response_text = response.content[0].text
            return _parse_llm_response(response_text, names)

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return [
                ResolvedConstraint(constraint=n, error=str(e))
                for n in names
            ]

    async def resolve_all_pending(
        self,
        force_refresh: bool = False,
        data_dir: Path = _DATA_DIR,
    ) -> dict:
        """
        Main entry point: scan constraint parquet files, resolve any names
        not already in the cache, and persist results.

        Args:
            force_refresh: If True, re-resolve names even if already cached.
            data_dir: Directory containing constraint parquet files.

        Returns:
            Summary dict with counts.
        """
        if self._running:
            logger.info("Resolver already running — skipping.")
            return {"status": "already_running"}

        self._running = True
        self._last_run = datetime.now(timezone.utc)

        try:
            # Collect all named constraints from history
            all_named = collect_unique_names(data_dir)
            logger.info(f"LLM resolver: found {len(all_named)} named constraints in history.")

            if not all_named:
                return {"status": "no_named_constraints", "total": 0}

            # Filter to unresolved
            if force_refresh:
                pending = list(all_named)
            else:
                already_done = self.store.already_resolved()
                pending = [n for n in all_named if n not in already_done]

            logger.info(
                f"LLM resolver: {len(pending)} pending, "
                f"{len(all_named) - len(pending)} already resolved."
            )

            if not pending:
                return {
                    "status": "all_resolved",
                    "total": len(all_named),
                    "already_resolved": len(all_named),
                    "newly_resolved": 0,
                }

            if not self.has_api_key:
                logger.warning(
                    "ANTHROPIC_API_KEY not set — constraint resolver is inactive. "
                    "Set the key in your .env file or environment to enable resolution."
                )
                return {
                    "status": "no_api_key",
                    "total": len(all_named),
                    "pending": len(pending),
                    "message": "Set ANTHROPIC_API_KEY to enable LLM resolution.",
                }

            # Process in batches
            newly_resolved = 0
            errors = 0
            batches = [
                pending[i:i + _BATCH_SIZE]
                for i in range(0, len(pending), _BATCH_SIZE)
            ]

            for i, batch in enumerate(batches):
                logger.info(
                    f"LLM resolver: processing batch {i+1}/{len(batches)} "
                    f"({len(batch)} names)..."
                )
                results = await self.resolve_batch(batch)
                self.store.upsert_many(results)

                for r in results:
                    if r.error:
                        errors += 1
                    else:
                        newly_resolved += 1

                self._resolved_count += newly_resolved
                self._error_count += errors

                if i < len(batches) - 1:
                    await asyncio.sleep(_DELAY_BETWEEN_BATCHES)

            logger.info(
                f"LLM resolver complete: {newly_resolved} resolved, {errors} errors."
            )

            return {
                "status": "complete",
                "total_named_constraints": len(all_named),
                "newly_resolved": newly_resolved,
                "errors": errors,
                "already_cached": len(all_named) - len(pending),
            }

        except Exception as e:
            logger.error(f"LLM resolver failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
        finally:
            self._running = False

    def lookup(self, constraint: str) -> Optional[ResolvedConstraint]:
        """Synchronous single-name lookup from cache."""
        return self.store.get(constraint)

    def lookup_all(self) -> list[ResolvedConstraint]:
        """Return all resolved constraints from cache."""
        return self.store.get_all()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

resolver = LLMConstraintResolver()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _cli_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Resolve SPP constraint names to physical metadata using Claude."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-resolve names even if already cached."
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Resolve a single constraint name and print the result."
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print resolver status and exit."
    )
    args = parser.parse_args()

    # Load .env if present
    _load_dotenv()

    r = LLMConstraintResolver()

    if args.status:
        import json as _json
        print(_json.dumps(r.status(), indent=2))
        return

    if args.name:
        # Single name resolve
        results = await r.resolve_batch([args.name])
        import json as _json
        print(_json.dumps(results[0].to_dict(), indent=2))
        return

    result = await r.resolve_all_pending(force_refresh=args.force)
    import json as _json
    print(_json.dumps(result, indent=2))


def _load_dotenv():
    """Load .env from the grid-analysis project root if present."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


if __name__ == "__main__":
    _load_dotenv()
    asyncio.run(_cli_main())
