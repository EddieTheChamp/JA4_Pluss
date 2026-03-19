"""Local exact-match logic with hierarchical JA4+ fallback."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from current.core.models import CandidateMatch, DatabaseRecord, MatchResult, Observation
from current.utils.validation import normalize_casefold, normalize_lookup_text

LOOKUP_PLANS: list[tuple[str, tuple[str, ...]]] = [
    ("ja4_ja4s_ja4t_ja4ts", ("ja4", "ja4s", "ja4t", "ja4ts")),
    ("ja4_ja4s_ja4ts", ("ja4", "ja4s", "ja4ts")),
    ("ja4_ja4s_ja4t", ("ja4", "ja4s", "ja4t")),
    ("ja4_ja4s", ("ja4", "ja4s")),
    ("ja4_ja4ts", ("ja4", "ja4ts")),
    ("ja4_ja4t", ("ja4", "ja4t")),
    ("ja4_only", ("ja4",)),
    ("ja4s_only", ("ja4s",)),
    ("ja4t_only", ("ja4t",)),
    ("ja4ts_only", ("ja4ts",)),
]


class LocalDatabaseMatcher:
    """Exact-match database matcher with explicit fallback ordering."""

    def __init__(self, records: Iterable[DatabaseRecord]):
        self.records = list(records)
        self._indexes: dict[str, dict[str, list[DatabaseRecord]]] = {
            mode: defaultdict(list) for mode, _ in LOOKUP_PLANS
        }
        self._build_indexes()

    @classmethod
    def from_records(cls, records: Iterable[DatabaseRecord]) -> "LocalDatabaseMatcher":
        return cls(records)

    def _build_indexes(self) -> None:
        for record in self.records:
            for mode, fields in LOOKUP_PLANS:
                key = self._record_key(record, fields)
                if key:
                    self._indexes[mode][key].append(record)

    def match_observation(self, observation: Observation) -> MatchResult:
        """Attempt local database lookup using the configured fallback order."""
        available_fields = {
            "ja4": observation.ja4,
            "ja4s": observation.ja4s,
            "ja4t": observation.ja4t,
            "ja4ts": observation.ja4ts,
        }
        if not any(normalize_lookup_text(value) for value in available_fields.values()):
            return MatchResult(
                status="unknown",
                match_mode=None,
                evidence=["Observation did not contain any JA4, JA4S, JA4T or JA4TS values."],
                confidence=0.0,
            )

        evidence: list[str] = []
        for mode, fields in LOOKUP_PLANS:
            if not self._observation_supports_fields(observation, fields):
                continue

            key = self._observation_key(observation, fields)
            matches = list(self._indexes[mode].get(key, []))
            if not matches:
                evidence.append(f"No local database hit for {mode}.")
                continue

            candidates = self._aggregate_candidates(matches)
            status = "unique" if len(candidates) == 1 else "ambiguous"
            evidence.append(
                f"Local database hit for {mode} produced {len(candidates)} candidate(s)."
            )
            return MatchResult(
                status=status,
                match_mode=mode,
                candidates=candidates,
                matched_key=key,
                evidence=evidence,
                confidence=self._calculate_confidence(mode, status),
            )

        return MatchResult(
            status="unknown",
            match_mode=None,
            evidence=evidence or ["No local database matches were found."],
            confidence=0.0,
        )

    def _aggregate_candidates(self, matches: list[DatabaseRecord]) -> list[CandidateMatch]:
        grouped: dict[tuple[str, str], dict[str, object]] = {}
        total_count = 0

        for record in matches:
            application = record.application or "unknown"
            category = record.category or "unknown"
            key = (normalize_casefold(application), normalize_casefold(category))
            grouped.setdefault(
                key,
                {
                    "application": record.application,
                    "category": record.category,
                    "occurrences": 0,
                    "metadata": {},
                },
            )
            grouped[key]["occurrences"] = int(grouped[key]["occurrences"]) + max(record.count, 1)
            total_count += max(record.count, 1)

            metadata = dict(grouped[key]["metadata"])
            for metadata_key, metadata_value in record.metadata.items():
                metadata.setdefault(metadata_key, metadata_value)
            grouped[key]["metadata"] = metadata

        candidates: list[CandidateMatch] = []
        for value in grouped.values():
            occurrences = int(value["occurrences"])
            probability = (occurrences / total_count * 100.0) if total_count else 0.0
            candidates.append(
                CandidateMatch(
                    application=value["application"],
                    category=value["category"],
                    occurrences_in_database=occurrences,
                    probability_percent=round(probability, 2),
                    metadata=dict(value["metadata"]),
                )
            )

        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.occurrences_in_database,
                candidate.application or "",
            ),
            reverse=True,
        )

    def _record_key(self, record: DatabaseRecord, fields: tuple[str, ...]) -> str | None:
        values = [normalize_lookup_text(getattr(record, field)) for field in fields]
        if not all(values):
            return None
        return "|".join(values)

    def _observation_key(self, observation: Observation, fields: tuple[str, ...]) -> str:
        values = [normalize_lookup_text(getattr(observation, field)) for field in fields]
        return "|".join(values)

    def _observation_supports_fields(self, observation: Observation, fields: tuple[str, ...]) -> bool:
        return all(normalize_lookup_text(getattr(observation, field)) for field in fields)

    def _calculate_confidence(self, match_mode: str, status: str) -> float:
        mode_weight = {
            "ja4_ja4s_ja4t_ja4ts": 0.99,
            "ja4_ja4s_ja4ts": 0.96,
            "ja4_ja4s_ja4t": 0.95,
            "ja4_ja4s": 0.9,
            "ja4_ja4ts": 0.85,
            "ja4_ja4t": 0.82,
            "ja4_only": 0.72,
            "ja4s_only": 0.7,
            "ja4t_only": 0.66,
            "ja4ts_only": 0.68,
        }.get(match_mode, 0.5)
        if status == "ambiguous":
            return round(mode_weight * 0.6, 3)
        return round(mode_weight, 3)
