"""FoxIO support adapter.

The repository does not ship a live FoxIO API integration today, so this adapter
uses a local FoxIO-style reference database when available and exposes a stable
interface that can be mocked or replaced by a future HTTP client.
"""

from __future__ import annotations

from pathlib import Path

from current.services.loaders import load_database_records
from current.services.local_matcher import LocalDatabaseMatcher
from current.core.models import FoxIOResult, Observation, PredictionCandidate


class FoxIOAdapter:
    """Support adapter for FoxIO lookups or local reference fallbacks."""

    def __init__(self, matcher: LocalDatabaseMatcher | None = None, source_name: str = "foxio_reference"):
        self.matcher = matcher
        self.source_name = source_name

    @classmethod
    def from_json(cls, path: str | Path | None) -> "FoxIOAdapter":
        """Create an adapter backed by a local FoxIO-style JSON database."""
        if path is None:
            return cls(None)
        file_path = Path(path)
        if not file_path.exists():
            return cls(None)
        records = load_database_records(file_path)
        return cls(LocalDatabaseMatcher.from_records(records), source_name=str(file_path))

    def is_available(self) -> bool:
        return self.matcher is not None

    def lookup(self, observation: Observation) -> FoxIOResult:
        """Perform a support lookup without overriding local exact matches on its own."""
        if self.matcher is None:
            return FoxIOResult(
                status="unavailable",
                evidence=["FoxIO adapter is not configured with an API client or reference database."],
                confidence=0.0,
                source=self.source_name,
            )

        match_result = self.matcher.match_observation(observation)
        if match_result.status == "unknown":
            return FoxIOResult(
                status="unknown",
                evidence=match_result.evidence + ["FoxIO reference database did not contain a supporting match."],
                confidence=0.0,
                source=self.source_name,
            )

        candidates = [
            PredictionCandidate(
                label=candidate.application or "unknown",
                category=candidate.category,
                probability_percent=candidate.probability_percent,
                metadata=candidate.metadata,
            )
            for candidate in match_result.candidates
        ]
        status = "supported" if match_result.status == "unique" else "ambiguous"
        return FoxIOResult(
            status=status,
            candidates=candidates,
            evidence=match_result.evidence + ["FoxIO reference data used as supporting evidence only."],
            confidence=round(match_result.confidence * 0.85, 3),
            source=self.source_name,
        )
