"""
Constraint Name Normalizer for SPP Transmission Constraints.

SPP publishes transmission constraint names in inconsistent formats across:
- RT (RTBM) binding constraint files
- DA binding constraint files
- Different time periods (naming conventions change over the years)

Examples of the same physical line appearing under different names:
    "LN HARBINE - BEATRICE"
    "HARBINE_BEATRICE_115"
    "HRB-BEA 115KV"
    "LN.HARBINE.BEATRICE.1"

This module provides deterministic rule-based normalization that:
1. Strips common prefixes (LN, BR, XFMR, TR, etc.)
2. Normalizes separators and whitespace
3. Extracts voltage (kV) into a separate field
4. Parses "From - To" substation pairs
5. Infers equipment type from prefix/context
6. Produces a canonical fingerprint suitable for fuzzy deduplication

The output of this module feeds into ``constraint_registry.py`` for
persistent canonical ID assignment.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Equipment type prefixes used by SPP in constraint names
_EQUIPMENT_PREFIXES: Dict[str, str] = {
    "LN":   "line",
    "LINE": "line",
    "BR":   "line",      # "branch" — synonym for line in SPP context
    "XFMR": "transformer",
    "TR":   "transformer",
    "TRANS":"transformer",
    "CAP":  "capacitor",
    "SHUNT":"shunt",
    "MON":  "monitored_element",  # monitoring constraint, not a physical device
    "INTF": "interface",           # flowgate / interface constraint
    "INT":  "interface",
}

# Voltage patterns (must come before substation parsing).
# Matches explicit KV suffix ("115KV", "345 KV") OR bare number at end of
# string after a separator ("HARBINE_BEATRICE_115", "LN HARBINE - BEATRICE 115").
_VOLTAGE_PATTERN = re.compile(
    r"(?:[-_.\s]?(\d{2,3}(?:\.\d+)?)\s*K?V\b)"  # with KV/V suffix
    r"|(?:[-_\s](\d{2,3})$)",                      # bare number at end after separator
    re.IGNORECASE,
)

# Separators used between substation names
_SEPARATOR_PATTERN = re.compile(r"[-_\s]+")

# Patterns that indicate a constraint number / circuit ID suffix
_CIRCUIT_SUFFIX_PATTERN = re.compile(
    r"[-_.\s](\d{1})$"  # single trailing digit
)

# Characters to strip from both ends of substation tokens
_STRIP_CHARS = " -_."

# Known "junk" tokens that appear in SPP names but carry no substation info
_JUNK_TOKENS = {
    "RA", "RR", "N", "S", "E", "W",  # directional suffixes sometimes appended
}

# Minimum token length to consider as a substation name fragment
_MIN_TOKEN_LEN = 2


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class NormalizedConstraint:
    """Result of normalizing a raw SPP constraint name."""
    raw_name: str
    equipment_type: str          # 'line', 'transformer', 'interface', 'unknown'
    from_sub: Optional[str]     # from-substation (canonical upper case)
    to_sub: Optional[str]       # to-substation (canonical upper case)
    voltage_kv: Optional[float] # voltage class in kV
    circuit_id: Optional[str]   # circuit number if present
    canonical_name: str          # reconstructed canonical display name
    fingerprint: str             # compact string for fuzzy matching


# ---------------------------------------------------------------------------
# Core normalizer
# ---------------------------------------------------------------------------

class ConstraintNameNormalizer:
    """
    Rule-based deterministic normalizer for SPP constraint names.

    Usage
    -----
    ::

        normalizer = ConstraintNameNormalizer()
        result = normalizer.normalize("LN HARBINE - BEATRICE 115KV")
        # NormalizedConstraint(
        #   raw_name="LN HARBINE - BEATRICE 115KV",
        #   equipment_type="line",
        #   from_sub="HARBINE",
        #   to_sub="BEATRICE",
        #   voltage_kv=115.0,
        #   circuit_id=None,
        #   canonical_name="LN HARBINE - BEATRICE 115",
        #   fingerprint="beatrice_harbine_115_line",
        # )
    """

    def normalize(self, raw_name: str) -> NormalizedConstraint:
        """
        Normalize a single constraint name.

        Parameters
        ----------
        raw_name : str
            The raw constraint name as it appears in SPP data files.

        Returns
        -------
        NormalizedConstraint
        """
        name = self._preprocess(raw_name)

        # 1. Extract equipment type
        equipment_type, name = self._extract_equipment_type(name)

        # 2. Extract voltage
        voltage_kv, name = self._extract_voltage(name)

        # 3. Extract circuit ID
        circuit_id, name = self._extract_circuit_id(name)

        # 4. Parse substation pair
        from_sub, to_sub = self._parse_substations(name)

        # 5. Build canonical name and fingerprint
        canonical_name = self._build_canonical_name(
            equipment_type, from_sub, to_sub, voltage_kv, circuit_id
        )
        fingerprint = self._build_fingerprint(
            equipment_type, from_sub, to_sub, voltage_kv
        )

        return NormalizedConstraint(
            raw_name=raw_name,
            equipment_type=equipment_type,
            from_sub=from_sub,
            to_sub=to_sub,
            voltage_kv=voltage_kv,
            circuit_id=circuit_id,
            canonical_name=canonical_name,
            fingerprint=fingerprint,
        )

    def normalize_many(self, raw_names: List[str]) -> List[NormalizedConstraint]:
        """Normalize a list of constraint names."""
        return [self.normalize(name) for name in raw_names]

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _preprocess(self, name: str) -> str:
        """Unicode normalize, strip, upper-case."""
        name = unicodedata.normalize("NFKC", name)
        return name.strip().upper()

    def _extract_equipment_type(self, name: str) -> Tuple[str, str]:
        """
        Extract and remove the equipment type prefix.

        Returns
        -------
        (equipment_type, remaining_name)
        """
        for prefix, eq_type in _EQUIPMENT_PREFIXES.items():
            # Match prefix at start, followed by space, dot, or underscore
            pattern = re.compile(
                r"^" + re.escape(prefix) + r"[\s._\-]+",
                re.IGNORECASE,
            )
            match = pattern.match(name)
            if match:
                remaining = name[match.end():]
                return eq_type, remaining.strip(_STRIP_CHARS)
        return "unknown", name

    def _extract_voltage(self, name: str) -> Tuple[Optional[float], str]:
        """
        Extract voltage (kV) from the name string.

        Handles:
          - Explicit suffix: "115KV", "345 KV", "138.0 KV"
          - Bare number at end after a separator: "HARBINE_BEATRICE_115",
            "LN HARBINE - BEATRICE 115"

        Returns
        -------
        (voltage_kv, name_without_voltage)
        """
        match = _VOLTAGE_PATTERN.search(name)
        if match:
            # Group 1 = KV-suffix form, Group 2 = bare-number-at-end form
            raw_v = match.group(1) or match.group(2)
            try:
                voltage = float(raw_v)
                # Only accept plausible transmission voltages
                if 60 <= voltage <= 1200:
                    name = name[:match.start()] + name[match.end():]
                    return voltage, name.strip(_STRIP_CHARS)
            except (ValueError, TypeError):
                pass
        return None, name

    def _extract_circuit_id(self, name: str) -> Tuple[Optional[str], str]:
        """
        Extract trailing circuit/line number (single digit).

        Returns
        -------
        (circuit_id, name_without_circuit_suffix)
        """
        match = _CIRCUIT_SUFFIX_PATTERN.search(name)
        if match:
            circuit_id = match.group(1)
            name = name[:match.start()].strip(_STRIP_CHARS)
            return circuit_id, name
        return None, name

    def _parse_substations(self, name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the "From - To" substation pair from the remaining name string.

        Handles separators: " - ", " _ ", ".", "-"
        """
        # Try to split on explicit separator patterns
        for sep in [" - ", " _ ", ".", "_"]:
            parts = name.split(sep, maxsplit=1)
            if len(parts) == 2:
                from_sub = parts[0].strip(_STRIP_CHARS)
                to_sub = parts[1].strip(_STRIP_CHARS)
                if self._is_valid_substation(from_sub) and self._is_valid_substation(to_sub):
                    return from_sub, to_sub

        # Fallback: split on any whitespace run (e.g. "HARBINE BEATRICE")
        parts = name.split()
        parts = [p for p in parts if p not in _JUNK_TOKENS and len(p) >= _MIN_TOKEN_LEN]
        if len(parts) == 2:
            return parts[0], parts[1]
        if len(parts) >= 3:
            # Heuristic: try splitting at midpoint
            mid = len(parts) // 2
            from_sub = " ".join(parts[:mid])
            to_sub = " ".join(parts[mid:])
            return from_sub, to_sub
        if len(parts) == 1:
            return parts[0], None

        # Single-entity constraint (interface, flowgate, etc.)
        if name.strip(_STRIP_CHARS):
            return name.strip(_STRIP_CHARS), None

        return None, None

    def _is_valid_substation(self, name: str) -> bool:
        """Basic validity check for a parsed substation token."""
        if not name or len(name) < _MIN_TOKEN_LEN:
            return False
        junk = {"UNKNOWN", "TAP", "NOT AVAILABLE", "N/A"}
        return not any(name.startswith(j) for j in junk)

    def _build_canonical_name(
        self,
        equipment_type: str,
        from_sub: Optional[str],
        to_sub: Optional[str],
        voltage_kv: Optional[float],
        circuit_id: Optional[str],
    ) -> str:
        """Reconstruct a clean, canonical display name."""
        prefix_map = {
            "line": "LN",
            "transformer": "XFMR",
            "interface": "INTF",
            "unknown": "",
        }
        prefix = prefix_map.get(equipment_type, "")
        parts = [prefix] if prefix else []

        if from_sub and to_sub:
            parts.append(f"{from_sub} - {to_sub}")
        elif from_sub:
            parts.append(from_sub)

        if voltage_kv is not None:
            parts.append(str(int(voltage_kv)))

        if circuit_id:
            parts.append(f"(CKT {circuit_id})")

        return " ".join(parts)

    def _build_fingerprint(
        self,
        equipment_type: str,
        from_sub: Optional[str],
        to_sub: Optional[str],
        voltage_kv: Optional[float],
    ) -> str:
        """
        Build a compact, order-normalized fingerprint for fuzzy deduplication.

        Substation names are sorted alphabetically so that direction doesn't
        matter (HARBINE-BEATRICE == BEATRICE-HARBINE).
        """
        tokens = []

        subs = sorted(
            filter(None, [from_sub, to_sub]),
        )
        # Reduce to alphabetic only, lower-case
        subs_clean = ["".join(c for c in s.lower() if c.isalpha()) for s in subs]
        tokens.extend(subs_clean)

        if voltage_kv is not None:
            tokens.append(str(int(voltage_kv)))

        if equipment_type not in ("unknown", "line"):
            tokens.append(equipment_type)

        return "_".join(tokens)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def normalize_constraint_name(raw_name: str) -> NormalizedConstraint:
    """
    Normalize a single SPP constraint name using default settings.

    Parameters
    ----------
    raw_name : str
        Raw constraint name from SPP data.

    Returns
    -------
    NormalizedConstraint
    """
    return ConstraintNameNormalizer().normalize(raw_name)


def normalize_constraints_df(df: pd.DataFrame, name_col: str = "Constraint Name") -> pd.DataFrame:
    """
    Add normalized constraint columns to a DataFrame in-place.

    Parameters
    ----------
    df : DataFrame
        Must contain a column with constraint names.
    name_col : str
        Column containing raw constraint names.

    Returns
    -------
    DataFrame with added columns:
        ``equipment_type``, ``from_sub``, ``to_sub``, ``voltage_kv``,
        ``circuit_id``, ``canonical_name``, ``fingerprint``
    """
    import pandas as pd  # local import to avoid circular deps at module level

    normalizer = ConstraintNameNormalizer()
    results = df[name_col].apply(normalizer.normalize)

    df = df.copy()
    df["equipment_type"] = results.apply(lambda r: r.equipment_type)
    df["from_sub"]       = results.apply(lambda r: r.from_sub)
    df["to_sub"]         = results.apply(lambda r: r.to_sub)
    df["voltage_kv"]     = results.apply(lambda r: r.voltage_kv)
    df["circuit_id"]     = results.apply(lambda r: r.circuit_id)
    df["canonical_name"] = results.apply(lambda r: r.canonical_name)
    df["fingerprint"]    = results.apply(lambda r: r.fingerprint)

    return df


# ---------------------------------------------------------------------------
# Import guard for pandas (only needed in normalize_constraints_df)
# ---------------------------------------------------------------------------
try:
    import pandas as pd  # noqa: F401
except ImportError:
    pass
