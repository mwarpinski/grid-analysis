"""
Constraint Locator — Maps Canonical SPP Constraints to EIA GeoJSON Lines.

Given a canonical constraint (from_sub, to_sub, voltage_kv), this module
finds the matching transmission line feature in the EIA GeoJSON dataset
(the same ``spp_only_lines.geojson`` used by the rest of the pipeline).

Matching strategy:
1. Exact substation name match (sub1/sub2 in any order) + voltage class match
2. Fuzzy substation name match (handles abbreviations like "HRB" vs "HARBINE")
3. Geographic proximity fallback (if lat/lon data is available for the constraint)

Output:
- GeoJSON Feature with confidence score added to properties
- Summary CSV linking canonical_id → GeoJSON feature index + confidence
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Fuzzy threshold for substation name matching (0-100)
_SUB_FUZZY_THRESHOLD = 75

try:
    from rapidfuzz import fuzz  # type: ignore
    _HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    _HAS_RAPIDFUZZ = False

# Voltage class → typical kV ranges (for matching when exact kV is unavailable)
_VOLT_CLASS_RANGES: Dict[str, Tuple[float, float]] = {
    "UNDER 100":     (0,   100),
    "100-161":       (100, 161),
    "162-275":       (162, 275),
    "276-499":       (276, 499),
    "500 AND ABOVE": (500, 9999),
    "DC":            (0,   9999),
    "NOT AVAILABLE": (0,   9999),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LocateResult:
    """Result of locating a constraint in the EIA GeoJSON."""
    canonical_id: str
    canonical_name: str
    feature_index: Optional[int]       # index in geojson['features']
    match_type: Optional[str]          # 'exact', 'fuzzy', 'proximity', None
    confidence: float                  # 0.0 – 1.0
    matched_sub1: Optional[str]
    matched_sub2: Optional[str]
    matched_voltage: Optional[float]
    geometry: Optional[Dict[str, Any]] # GeoJSON geometry dict, if found


# ---------------------------------------------------------------------------
# Core locator
# ---------------------------------------------------------------------------

class ConstraintLocator:
    """
    Maps canonical SPP constraints to EIA GeoJSON transmission lines.

    Parameters
    ----------
    geojson_path : str or Path
        Path to the filtered SPP GeoJSON (e.g. ``spp_only_lines.geojson``).
    fuzzy_threshold : float
        Minimum fuzzy score (0-100) for substation name matching.
    """

    def __init__(
        self,
        geojson_path: str | Path = "src/data/spp_only_lines.geojson",
        fuzzy_threshold: float = _SUB_FUZZY_THRESHOLD,
    ) -> None:
        self.geojson_path = Path(geojson_path)
        self.fuzzy_threshold = fuzzy_threshold
        self._features: List[Dict[str, Any]] = []
        self._sub_index: Dict[str, List[int]] = {}  # sub_name → [feature indices]
        self._load_geojson()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def locate(
        self,
        canonical_id: str,
        canonical_name: str,
        from_sub: Optional[str],
        to_sub: Optional[str],
        voltage_kv: Optional[float],
    ) -> LocateResult:
        """
        Locate a canonical constraint in the EIA GeoJSON.

        Parameters
        ----------
        canonical_id : str
            UUID from the constraint registry.
        canonical_name : str
            Human-readable canonical name.
        from_sub, to_sub : str or None
            Substation names from the canonical constraint.
        voltage_kv : float or None
            Voltage in kV.

        Returns
        -------
        LocateResult
        """
        if from_sub is None:
            return LocateResult(
                canonical_id=canonical_id,
                canonical_name=canonical_name,
                feature_index=None,
                match_type=None,
                confidence=0.0,
                matched_sub1=None,
                matched_sub2=None,
                matched_voltage=None,
                geometry=None,
            )

        # Try exact match first
        result = self._exact_match(
            canonical_id, canonical_name, from_sub, to_sub, voltage_kv
        )
        if result:
            return result

        # Try fuzzy match
        result = self._fuzzy_match(
            canonical_id, canonical_name, from_sub, to_sub, voltage_kv
        )
        if result:
            return result

        # No match found
        return LocateResult(
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            feature_index=None,
            match_type=None,
            confidence=0.0,
            matched_sub1=None,
            matched_sub2=None,
            matched_voltage=None,
            geometry=None,
        )

    def locate_many(
        self,
        constraints: List[Dict[str, Any]],
    ) -> List[LocateResult]:
        """
        Locate multiple canonical constraints.

        Parameters
        ----------
        constraints : list of dict
            Each dict must have keys: ``canonical_id``, ``canonical_name``,
            ``from_sub``, ``to_sub``, ``voltage_kv``.

        Returns
        -------
        List of LocateResult, one per input constraint.
        """
        return [
            self.locate(
                canonical_id=c.get("canonical_id", ""),
                canonical_name=c.get("canonical_name", ""),
                from_sub=c.get("from_sub"),
                to_sub=c.get("to_sub"),
                voltage_kv=c.get("voltage_kv"),
            )
            for c in constraints
        ]

    def get_line_by_substations(
        self,
        from_sub: str,
        to_sub: str,
        voltage_kv: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Look up a GeoJSON feature by substation pair (and optional voltage).
        Returns the first matching feature, or None.

        This is a convenience helper used by the pipeline and spp_filters.
        """
        result = self.locate(
            canonical_id="",
            canonical_name=f"{from_sub} - {to_sub}",
            from_sub=from_sub,
            to_sub=to_sub,
            voltage_kv=voltage_kv,
        )
        if result.feature_index is not None:
            return self._features[result.feature_index]
        return None

    def export_located_constraints(
        self,
        results: List[LocateResult],
        output_path: str | Path,
    ) -> None:
        """
        Write a GeoJSON FeatureCollection containing all located constraints.

        Each feature has the original EIA geometry enriched with:
        - ``canonical_id``
        - ``canonical_name``
        - ``match_type``
        - ``confidence``
        """
        features = []
        for result in results:
            if result.feature_index is None:
                continue
            feature = dict(self._features[result.feature_index])
            props = dict(feature.get("properties", {}))
            props.update({
                "canonical_id": result.canonical_id,
                "canonical_name": result.canonical_name,
                "match_type": result.match_type,
                "confidence": result.confidence,
            })
            feature["properties"] = props
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}
        with open(output_path, "w") as f:
            json.dump(geojson, f)
        logger.info(f"Wrote {len(features)} located constraints to {output_path}")

    # ------------------------------------------------------------------
    # Internal loading & indexing
    # ------------------------------------------------------------------

    def _load_geojson(self) -> None:
        if not self.geojson_path.exists():
            logger.warning(
                f"GeoJSON not found at {self.geojson_path}. "
                "Run spp_filters.py first to generate it."
            )
            return

        with open(self.geojson_path) as f:
            data = json.load(f)

        # Handle both bare list (spp_only_lines.geojson saves as list) and proper FeatureCollection
        if isinstance(data, list):
            self._features = data
        elif isinstance(data, dict) and "features" in data:
            self._features = data["features"]
        else:
            self._features = []

        # Build substation index: sub_name_upper → [feature indices]
        for i, feature in enumerate(self._features):
            props = feature.get("properties", {})
            for key in ("sub1", "sub2"):
                sub = props.get(key, "")
                if sub:
                    sub_upper = sub.upper().strip()
                    if sub_upper not in self._sub_index:
                        self._sub_index[sub_upper] = []
                    self._sub_index[sub_upper].append(i)

        logger.info(
            f"Loaded {len(self._features)} EIA features, "
            f"{len(self._sub_index)} unique substation names."
        )

    # ------------------------------------------------------------------
    # Matching helpers
    # ------------------------------------------------------------------

    def _exact_match(
        self,
        canonical_id: str,
        canonical_name: str,
        from_sub: str,
        to_sub: Optional[str],
        voltage_kv: Optional[float],
    ) -> Optional[LocateResult]:
        """Try exact (case-insensitive) substation name match."""
        from_upper = from_sub.upper().strip()
        to_upper = to_sub.upper().strip() if to_sub else None

        candidates: List[int] = list(self._sub_index.get(from_upper, []))
        if to_upper:
            to_candidates = set(self._sub_index.get(to_upper, []))
            candidates = [i for i in candidates if i in to_candidates]

        if not candidates:
            return None

        # Filter by voltage if available
        if voltage_kv is not None:
            filtered = self._filter_by_voltage(candidates, voltage_kv)
            if filtered:
                candidates = filtered

        if candidates:
            idx = candidates[0]
            props = self._features[idx].get("properties", {})
            return LocateResult(
                canonical_id=canonical_id,
                canonical_name=canonical_name,
                feature_index=idx,
                match_type="exact",
                confidence=1.0 if len(candidates) == 1 else 0.9,
                matched_sub1=props.get("sub1"),
                matched_sub2=props.get("sub2"),
                matched_voltage=props.get("voltage"),
                geometry=self._features[idx].get("geometry"),
            )
        return None

    def _fuzzy_match(
        self,
        canonical_id: str,
        canonical_name: str,
        from_sub: str,
        to_sub: Optional[str],
        voltage_kv: Optional[float],
    ) -> Optional[LocateResult]:
        """Fuzzy substation name match using token-set ratio."""
        from_upper = from_sub.upper().strip()
        known_subs = list(self._sub_index.keys())

        # Find best fuzzy matches for from_sub
        from_matches = self._fuzzy_find_sub(from_upper, known_subs)
        if not from_matches:
            return None

        # Collect candidate feature indices from best-matching from_sub
        best_from_score, best_from_sub = from_matches[0]
        from_candidates = set(self._sub_index.get(best_from_sub, []))

        if to_sub:
            to_upper = to_sub.upper().strip()
            to_matches = self._fuzzy_find_sub(to_upper, known_subs)
            if to_matches:
                _, best_to_sub = to_matches[0]
                to_candidates = set(self._sub_index.get(best_to_sub, []))
                combined = from_candidates & to_candidates
                if combined:
                    from_candidates = combined
                    best_to_match = best_to_sub
                else:
                    best_to_match = None
            else:
                best_to_match = None
        else:
            best_to_match = None

        candidates = list(from_candidates)

        # Filter by voltage if available
        if voltage_kv is not None:
            filtered = self._filter_by_voltage(candidates, voltage_kv)
            if filtered:
                candidates = filtered

        if candidates:
            idx = candidates[0]
            props = self._features[idx].get("properties", {})
            confidence = (best_from_score / 100.0) * 0.8  # fuzzy is less confident
            return LocateResult(
                canonical_id=canonical_id,
                canonical_name=canonical_name,
                feature_index=idx,
                match_type="fuzzy",
                confidence=confidence,
                matched_sub1=props.get("sub1"),
                matched_sub2=props.get("sub2"),
                matched_voltage=props.get("voltage"),
                geometry=self._features[idx].get("geometry"),
            )
        return None

    def _fuzzy_find_sub(
        self, query: str, candidates: List[str]
    ) -> List[Tuple[float, str]]:
        """Return [(score, substation_name)] sorted descending, above threshold."""
        if _HAS_RAPIDFUZZ:
            from rapidfuzz import process as rfuzz_process  # type: ignore
            results = rfuzz_process.extract(
                query,
                candidates,
                scorer=fuzz.token_set_ratio,
                score_cutoff=self.fuzzy_threshold,
                limit=5,
            )
            return [(score, match) for match, score, _ in results]
        else:
            matches = difflib.get_close_matches(
                query, candidates, n=5, cutoff=self.fuzzy_threshold / 100.0
            )
            return [
                (
                    difflib.SequenceMatcher(None, query, m).ratio() * 100,
                    m,
                )
                for m in matches
            ]

    def _filter_by_voltage(
        self, feature_indices: List[int], voltage_kv: float
    ) -> List[int]:
        """Filter feature indices to those whose voltage class includes voltage_kv."""
        result = []
        for i in feature_indices:
            props = self._features[i].get("properties", {})
            feature_voltage = props.get("voltage")
            volt_class = props.get("voltclass", "NOT AVAILABLE")

            # Try exact voltage match first
            if feature_voltage is not None and feature_voltage != -999999:
                try:
                    if abs(float(feature_voltage) - voltage_kv) < 15:
                        result.append(i)
                        continue
                except (ValueError, TypeError):
                    pass

            # Fall back to volt_class range
            volt_range = _VOLT_CLASS_RANGES.get(volt_class)
            if volt_range and volt_range[0] <= voltage_kv <= volt_range[1]:
                result.append(i)

        return result


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def locate_constraints_from_registry(
    db_path: str | Path,
    geojson_path: str | Path = "src/data/spp_only_lines.geojson",
) -> List[LocateResult]:
    """
    Locate all canonical constraints in the registry against the EIA GeoJSON.

    Parameters
    ----------
    db_path : str or Path
        Path to constraint_registry.db
    geojson_path : str or Path
        Path to the filtered SPP GeoJSON

    Returns
    -------
    List of LocateResult
    """
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    df_canonical = pd.read_sql_query(
        "SELECT canonical_id, canonical_name, from_sub, to_sub, voltage_kv FROM canonical",
        conn,
    )
    conn.close()

    locator = ConstraintLocator(geojson_path=geojson_path)
    return locator.locate_many(df_canonical.to_dict("records"))


try:
    import pandas as pd  # noqa: F401
except ImportError:
    pass
