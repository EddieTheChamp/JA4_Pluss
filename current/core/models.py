"""Dataclasses shared across the JA4+ pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Observation:
    """Normalized observation used internally across the pipeline."""

    observation_id: str
    ja4: str | None = None
    ja4s: str | None = None
    ja4t: str | None = None
    ja4ts: str | None = None
    source: str | None = None
    true_application: str | None = None
    true_category: str | None = None
    raw_record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatabaseRecord:
    """Normalized row from a local fingerprint database or training dataset."""

    ja4: str | None = None
    ja4s: str | None = None
    ja4t: str | None = None
    ja4ts: str | None = None
    application: str | None = None
    category: str | None = None
    count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def fingerprint_map(self) -> dict[str, str | None]:
        return {
            "ja4": self.ja4,
            "ja4s": self.ja4s,
            "ja4t": self.ja4t,
            "ja4ts": self.ja4ts,
        }


@dataclass(slots=True)
class CandidateMatch:
    """One candidate produced by local matching, ML or FoxIO."""

    application: str | None
    category: str | None
    occurrences_in_database: int
    probability_percent: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MatchResult:
    """Result from the local exact-match engine."""

    status: str
    match_mode: str | None
    candidates: list[CandidateMatch] = field(default_factory=list)
    matched_key: str | None = None
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "local_database"

    def top_candidate(self) -> CandidateMatch | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "match_mode": self.match_mode,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "matched_key": self.matched_key,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(slots=True)
class PredictionCandidate:
    """Generic prediction candidate used by RF and FoxIO outputs."""

    label: str
    category: str | None
    probability_percent: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RandomForestResult:
    """Structured output from Random Forest inference."""

    status: str
    predicted_label: str | None = None
    predicted_category: str | None = None
    inferred_category: str | None = None
    category_confidence_score: float = 0.0
    category_candidates: list[dict[str, Any]] = field(default_factory=list)
    top_k: list[PredictionCandidate] = field(default_factory=list)
    confidence_score: float = 0.0
    evidence: list[str] = field(default_factory=list)
    source: str = "random_forest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "predicted_label": self.predicted_label,
            "predicted_category": self.predicted_category,
            "inferred_category": self.inferred_category,
            "category_confidence_score": self.category_confidence_score,
            "category_candidates": self.category_candidates,
            "top_k": [candidate.to_dict() for candidate in self.top_k],
            "confidence_score": self.confidence_score,
            "evidence": self.evidence,
            "source": self.source,
        }


@dataclass(slots=True)
class FoxIOResult:
    """Structured result from the FoxIO support adapter."""

    status: str
    candidates: list[PredictionCandidate] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "foxio"

    def top_candidate(self) -> PredictionCandidate | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(slots=True)
class FinalDecision:
    """Final explanation produced by the decision engine."""

    application_prediction: str | None
    category_prediction: str | None
    application_confidence: str
    category_confidence: str
    decision_source: str
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label"] = self.application_prediction
        payload["category"] = self.category_prediction
        payload["confidence"] = self.application_confidence
        return payload


@dataclass(slots=True)
class ClassificationResult:
    """Full per-observation output returned by the pipeline."""

    observation_id: str
    input: dict[str, Any]
    true_application: str | None
    true_category: str | None
    application_prediction: str | None
    category_prediction: str | None
    application_correct: bool | None
    category_correct: bool | None
    application_confidence: str
    category_confidence: str
    decision_source: str
    reasoning: str
    local_db: dict[str, Any]
    random_forest: dict[str, Any]
    foxio: dict[str, Any]
    final_decision: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "input": self.input,
            "true_application": self.true_application,
            "true_category": self.true_category,
            "application_prediction": self.application_prediction,
            "category_prediction": self.category_prediction,
            "application_correct": self.application_correct,
            "category_correct": self.category_correct,
            "application_confidence": self.application_confidence,
            "category_confidence": self.category_confidence,
            "decision_source": self.decision_source,
            "reasoning": self.reasoning,
            "local_db": self.local_db,
            "random_forest": self.random_forest,
            "foxio": self.foxio,
            "final_decision": self.final_decision,
        }
