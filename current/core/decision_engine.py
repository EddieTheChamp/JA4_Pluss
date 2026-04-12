"""Decision rules for combining local DB, RF and FoxIO evidence."""

from __future__ import annotations

from current.core.models import CandidateMatch, FinalDecision, FoxIOResult, MatchResult, RandomForestResult
from current.services.category_mapper import infer_category_from_application

STRICT_LOCAL_MODES = {
    "ja4_ja4s_ja4t_ja4ts",
    "ja4_ja4s_ja4ts",
    "ja4_ja4s_ja4t",
    "ja4_ja4s",
}

RF_APPLICATION_ACCEPT_THRESHOLD = 0.60
RF_APPLICATION_HIGH_THRESHOLD = 0.75
RF_CLEAR_MARGIN_PERCENT = 15.0
CATEGORY_ACCEPT_THRESHOLD = 0.55
CATEGORY_HIGH_THRESHOLD = 0.75


class DecisionEngine:
    """Explicit rules for selecting the final output label and fallback category."""

    def decide(
        self,
        *,
        local_result: MatchResult,
        rf_result: RandomForestResult,
        foxio_result: FoxIOResult,
    ) -> FinalDecision:
        """Combine all available signals into one explainable decision."""
        local_status = local_result.status if local_result.status != "unavailable" else "unknown"
        local_top = local_result.top_candidate()

        if local_status == "unique" and local_top is not None:
            category_prediction = self._candidate_category(local_top)
            category_supported = self._foxio_supports_category(foxio_result, category_prediction)
            reasoning = f"Unique local exact match via {local_result.match_mode} was prioritized over ML."
            if self._foxio_supports_label(foxio_result, local_top.application):
                reasoning += " FoxIO reference data supported the same application."
            elif category_supported:
                reasoning += " FoxIO reference data supported the same category."
            return FinalDecision(
                application_prediction=local_top.application,
                category_prediction=category_prediction,
                application_confidence=self._band_for_local(local_result),
                category_confidence="high",
                decision_source="local_database",
                reasoning=reasoning,
            )

        if local_status == "ambiguous" and local_result.candidates:
            resolved_candidate = self._resolve_ambiguous_local_with_rf(local_result, rf_result)
            if resolved_candidate is not None:
                category_prediction = self._candidate_category(resolved_candidate)
                category_supported = self._foxio_supports_category(foxio_result, category_prediction)
                reasoning = (
                    f"Local database produced multiple candidates for {local_result.match_mode}, "
                    f"but Random Forest strongly favored {resolved_candidate.application}."
                )
                if self._foxio_supports_label(foxio_result, resolved_candidate.application):
                    reasoning += " FoxIO supported that exact application."
                elif category_supported:
                    reasoning += " FoxIO supported the same category."
                return FinalDecision(
                    application_prediction=resolved_candidate.application,
                    category_prediction=category_prediction,
                    application_confidence=self._band_for_rf_application(rf_result, category_supported),
                    category_confidence=self._band_for_category_score(
                        rf_result.category_confidence_score,
                        category_supported,
                    ),
                    decision_source="ambiguous_local_match_resolved_by_rf",
                    reasoning=reasoning,
                )

            category_prediction, category_confidence, category_reason = self._infer_category_fallback(
                local_result=local_result,
                rf_result=rf_result,
                foxio_result=foxio_result,
            )
            if category_prediction is not None:
                return FinalDecision(
                    application_prediction=None,
                    category_prediction=category_prediction,
                    application_confidence="low",
                    category_confidence=category_confidence,
                    decision_source="category_fallback",
                    reasoning=category_reason,
                )

        if local_status == "unknown" and self._rf_application_is_accepted(rf_result):
            category_prediction = rf_result.predicted_category or infer_category_from_application(
                rf_result.predicted_label
            )
            category_supported = self._foxio_supports_category(foxio_result, category_prediction)
            reasoning = "No local database match was found, so Random Forest became the primary signal."
            if self._foxio_supports_label(foxio_result, rf_result.predicted_label):
                reasoning += " FoxIO reference data supported the same application."
            elif category_supported:
                reasoning += " FoxIO reference data supported the same category."
            return FinalDecision(
                application_prediction=rf_result.predicted_label,
                category_prediction=category_prediction,
                application_confidence=self._band_for_rf_application(rf_result, category_supported),
                category_confidence=self._band_for_category_score(
                    rf_result.category_confidence_score,
                    category_supported,
                ),
                decision_source="random_forest",
                reasoning=reasoning,
            )

        if local_status == "unknown":
            category_prediction, category_confidence, category_reason = self._infer_category_fallback(
                local_result=local_result,
                rf_result=rf_result,
                foxio_result=foxio_result,
            )
            if category_prediction is not None:
                return FinalDecision(
                    application_prediction=None,
                    category_prediction=category_prediction,
                    application_confidence="low",
                    category_confidence=category_confidence,
                    decision_source="category_fallback",
                    reasoning=category_reason,
                )

        if local_status == "ambiguous" and local_result.candidates:
            candidate_names = ", ".join(
                candidate.application or "unknown" for candidate in local_result.candidates[:5]
            )
            return FinalDecision(
                application_prediction=None,
                category_prediction=None,
                application_confidence="low",
                category_confidence="low",
                decision_source="unknown",
                reasoning=(
                    "Local database remained ambiguous and supporting signals were not strong enough "
                    f"to resolve the candidate set: {candidate_names}."
                ),
            )

        return FinalDecision(
            application_prediction=None,
            category_prediction=None,
            application_confidence="low",
            category_confidence="low",
            decision_source="unknown",
            reasoning="No method produced enough evidence to classify the observation.",
        )

    def _resolve_ambiguous_local_with_rf(
        self,
        local_result: MatchResult,
        rf_result: RandomForestResult,
    ) -> CandidateMatch | None:
        if not self._rf_application_is_accepted(rf_result):
            return None

        candidate_map = {
            (candidate.application or "").casefold(): candidate for candidate in local_result.candidates
        }
        return candidate_map.get((rf_result.predicted_label or "").casefold())

    def _rf_application_is_accepted(self, rf_result: RandomForestResult) -> bool:
        if rf_result.status != "predicted" or not rf_result.predicted_label:
            return False
        if rf_result.confidence_score < RF_APPLICATION_ACCEPT_THRESHOLD:
            return False
        return self._rf_has_clear_margin(rf_result)

    def _rf_has_clear_margin(self, rf_result: RandomForestResult) -> bool:
        if len(rf_result.top_k) < 2:
            return rf_result.confidence_score >= RF_APPLICATION_ACCEPT_THRESHOLD
        top1 = rf_result.top_k[0].probability_percent
        top2 = rf_result.top_k[1].probability_percent
        return (top1 - top2) >= RF_CLEAR_MARGIN_PERCENT or rf_result.confidence_score >= RF_APPLICATION_HIGH_THRESHOLD

    def _infer_category_fallback(
        self,
        *,
        local_result: MatchResult,
        rf_result: RandomForestResult,
        foxio_result: FoxIOResult,
    ) -> tuple[str | None, str, str]:
        local_category = self._shared_local_category(local_result)
        if local_category is not None:
            category_supported = self._foxio_supports_category(foxio_result, local_category)
            confidence = "high" if category_supported else "medium"
            reasoning = (
                "Exact application prediction remained uncertain, but all local candidates "
                f"still point to the same category: {local_category}."
            )
            if category_supported:
                reasoning += " FoxIO supported that category."
            return local_category, confidence, reasoning

        if rf_result.inferred_category and rf_result.category_confidence_score >= CATEGORY_ACCEPT_THRESHOLD:
            category_supported = self._foxio_supports_category(foxio_result, rf_result.inferred_category)
            confidence = self._band_for_category_score(rf_result.category_confidence_score, category_supported)
            reasoning = (
                "Exact application prediction remained uncertain, so the pipeline fell back to the "
                f"strongest category signal from Random Forest top-k predictions: {rf_result.inferred_category}."
            )
            if category_supported:
                reasoning += " FoxIO supported the same category."
            return rf_result.inferred_category, confidence, reasoning

        foxio_category = self._shared_foxio_category(foxio_result)
        if foxio_category is not None:
            return (
                foxio_category,
                "low",
                "Exact application prediction remained uncertain, but FoxIO still supported one category.",
            )

        return None, "low", "No reliable category fallback could be inferred."

    def _shared_local_category(self, local_result: MatchResult) -> str | None:
        categories = {
            self._candidate_category(candidate)
            for candidate in local_result.candidates
            if self._candidate_category(candidate) not in {None, "unknown"}
        }
        if len(categories) == 1:
            return next(iter(categories))
        return None

    def _shared_foxio_category(self, foxio_result: FoxIOResult) -> str | None:
        categories = {
            infer_category_from_application(candidate.label, candidate.category)
            for candidate in foxio_result.candidates
            if infer_category_from_application(candidate.label, candidate.category) != "unknown"
        }
        if len(categories) == 1:
            return next(iter(categories))
        return None

    def _candidate_category(self, candidate: CandidateMatch) -> str:
        return infer_category_from_application(candidate.application, candidate.category)

    def _foxio_supports_label(self, foxio_result: FoxIOResult, label: str | None) -> bool:
        if not label or foxio_result.status not in {"supported", "ambiguous"}:
            return False
        return any(candidate.label.casefold() == label.casefold() for candidate in foxio_result.candidates)

    def _foxio_supports_category(self, foxio_result: FoxIOResult, category: str | None) -> bool:
        if not category or foxio_result.status not in {"supported", "ambiguous"}:
            return False
        return any(
            infer_category_from_application(candidate.label, candidate.category) == category
            for candidate in foxio_result.candidates
        )

    def _band_for_local(self, local_result: MatchResult) -> str:
        if local_result.match_mode in STRICT_LOCAL_MODES:
            return "high"
        return "medium"

    def _band_for_rf_application(self, rf_result: RandomForestResult, category_supported: bool) -> str:
        if category_supported and rf_result.confidence_score >= 0.7:
            return "high"
        if rf_result.confidence_score >= 0.65:
            return "medium"
        return "low"

    def _band_for_category_score(self, score: float, category_supported: bool) -> str:
        if category_supported and score >= CATEGORY_ACCEPT_THRESHOLD:
            return "high"
        if score >= CATEGORY_HIGH_THRESHOLD:
            return "high"
        if score >= CATEGORY_ACCEPT_THRESHOLD:
            return "medium"
        return "low"
