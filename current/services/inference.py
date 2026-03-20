"""Inference wrapper for JA4+ Random Forest predictions."""

from __future__ import annotations

from collections import defaultdict

from current.core.models import Observation, PredictionCandidate, RandomForestResult
from current.services.category_mapper import infer_category_from_application
from current.services.feature_parser import CATEGORICAL_FEATURES, NUMERIC_FEATURES, observation_to_feature_dict
from current.services.random_forest import JA4RandomForestModel


class RandomForestInference:
    """Thin inference layer that converts observations into RF results."""

    def __init__(self, model: JA4RandomForestModel | None):
        self.model = model

    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, observation: Observation, top_k: int = 5) -> RandomForestResult:
        """Predict the most likely labels for one observation."""
        if self.model is None:
            return RandomForestResult(
                status="unavailable",
                evidence=["Random Forest model is not loaded."],
                confidence_score=0.0,
            )

        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            return RandomForestResult(
                status="unavailable",
                evidence=[f"Random Forest dependencies are unavailable: {exc.name}."],
                confidence_score=0.0,
            )

        raw_features = observation_to_feature_dict(observation)
        encoded_row = {}
        for column in CATEGORICAL_FEATURES:
            mapping = self.model.bundle.categorical_maps[column]
            encoded_row[column] = mapping.get(str(raw_features[column]), -1)
        for column in NUMERIC_FEATURES:
            encoded_row[column] = int(raw_features[column])

        encoded_frame = pd.DataFrame([encoded_row], columns=self.model.bundle.feature_columns)
        probabilities, class_ids = self.model.predict_probabilities(encoded_frame)
        if len(probabilities) == 0:
            return RandomForestResult(
                status="unknown",
                evidence=["Random Forest model did not return any class probabilities."],
                confidence_score=0.0,
            )

        top_positions = probabilities.argsort()[::-1][:top_k]
        top_candidates: list[PredictionCandidate] = []
        for position in top_positions:
            class_id = int(class_ids[position])
            label = self.model.bundle.index_to_label[class_id]
            top_candidates.append(
                PredictionCandidate(
                    label=label,
                    category=self.model.bundle.label_to_category.get(label),
                    probability_percent=round(float(probabilities[position]) * 100.0, 2),
                    metadata={"class_id": class_id},
                )
            )

        best_candidate = top_candidates[0] if top_candidates else None
        if best_candidate is None:
            return RandomForestResult(
                status="unknown",
                evidence=["Random Forest model could not produce a top candidate."],
                confidence_score=0.0,
            )

        category_candidates = _aggregate_category_candidates(top_candidates)
        inferred_category = category_candidates[0]["category"] if category_candidates else None
        category_confidence = round(category_candidates[0]["probability_percent"] / 100.0, 4) if category_candidates else 0.0
        predicted_category = best_candidate.category or infer_category_from_application(best_candidate.label)

        return RandomForestResult(
            status="predicted",
            predicted_label=best_candidate.label,
            predicted_category=predicted_category,
            inferred_category=inferred_category,
            category_confidence_score=category_confidence,
            category_candidates=category_candidates,
            top_k=top_candidates,
            confidence_score=round(best_candidate.probability_percent / 100.0, 4),
            evidence=[
                f"Random Forest used JA4-derived features and optional JA4S/JA4T/JA4TS tokens.",
                f"Top prediction probability was {best_candidate.probability_percent:.2f}%.",
            ],
        )


def _aggregate_category_candidates(top_candidates: list[PredictionCandidate]) -> list[dict[str, object]]:
    """Collapse application-level RF candidates into category-level evidence."""
    category_scores: dict[str, float] = defaultdict(float)
    category_sources: dict[str, list[str]] = defaultdict(list)

    for candidate in top_candidates:
        category = candidate.category or infer_category_from_application(candidate.label)
        if not category or category == "unknown":
            continue
        category_scores[category] += float(candidate.probability_percent)
        category_sources[category].append(candidate.label)

    ordered = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "category": category,
            "probability_percent": round(probability_percent, 2),
            "source_labels": category_sources[category],
        }
        for category, probability_percent in ordered
    ]
