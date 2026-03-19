"""Tests for decision-engine prioritization rules."""

from __future__ import annotations

import unittest

from current.core.models import CandidateMatch, FoxIOResult, MatchResult, PredictionCandidate, RandomForestResult
from current.core.decision_engine import DecisionEngine


class DecisionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = DecisionEngine()

    def test_decision_engine_local_vs_rf_conflict(self) -> None:
        local_result = MatchResult(
            status="unique",
            match_mode="ja4_ja4s",
            candidates=[
                CandidateMatch(
                    application="msedge.exe",
                    category="browser",
                    occurrences_in_database=12,
                    probability_percent=100.0,
                )
            ],
            matched_key="ja4|ja4s",
            evidence=["Strong local exact match."],
            confidence=0.9,
        )
        rf_result = RandomForestResult(
            status="predicted",
            predicted_label="chrome.exe",
            predicted_category="browser",
            top_k=[
                PredictionCandidate(label="chrome.exe", category="browser", probability_percent=92.0),
                PredictionCandidate(label="msedge.exe", category="browser", probability_percent=8.0),
            ],
            confidence_score=0.92,
        )
        foxio_result = FoxIOResult(status="unknown")

        decision = self.engine.decide(
            local_result=local_result,
            rf_result=rf_result,
            foxio_result=foxio_result,
        )

        self.assertEqual(decision.application_prediction, "msedge.exe")
        self.assertEqual(decision.category_prediction, "browser")
        self.assertEqual(decision.decision_source, "local_database")

    def test_category_fallback_for_ambiguous_local_match(self) -> None:
        local_result = MatchResult(
            status="ambiguous",
            match_mode="ja4_ja4s",
            candidates=[
                CandidateMatch(
                    application="msedge.exe",
                    category="browser",
                    occurrences_in_database=12,
                    probability_percent=60.0,
                ),
                CandidateMatch(
                    application="msedgewebview2.exe",
                    category="browser",
                    occurrences_in_database=8,
                    probability_percent=40.0,
                ),
            ],
        )
        rf_result = RandomForestResult(
            status="predicted",
            predicted_label="msedge.exe",
            predicted_category="browser",
            inferred_category="browser",
            category_confidence_score=0.88,
            category_candidates=[
                {"category": "browser", "probability_percent": 88.0, "source_labels": ["msedge.exe", "msedgewebview2.exe"]}
            ],
            top_k=[
                PredictionCandidate(label="msedge.exe", category="browser", probability_percent=54.0),
                PredictionCandidate(label="msedgewebview2.exe", category="browser", probability_percent=34.0),
                PredictionCandidate(label="chrome.exe", category="browser", probability_percent=12.0),
            ],
            confidence_score=0.54,
        )

        decision = self.engine.decide(
            local_result=local_result,
            rf_result=rf_result,
            foxio_result=FoxIOResult(status="unknown"),
        )

        self.assertIsNone(decision.application_prediction)
        self.assertEqual(decision.category_prediction, "browser")
        self.assertEqual(decision.decision_source, "category_fallback")

    def test_category_only_fallback_when_no_exact_match(self) -> None:
        rf_result = RandomForestResult(
            status="predicted",
            predicted_label="OUTLOOK.EXE",
            predicted_category="microsoft_office",
            inferred_category="microsoft_office",
            category_confidence_score=0.93,
            category_candidates=[
                {
                    "category": "microsoft_office",
                    "probability_percent": 93.0,
                    "source_labels": ["OUTLOOK.EXE", "WINWORD.EXE"],
                }
            ],
            top_k=[
                PredictionCandidate(label="OUTLOOK.EXE", category="microsoft_office", probability_percent=49.0),
                PredictionCandidate(label="WINWORD.EXE", category="microsoft_office", probability_percent=44.0),
                PredictionCandidate(label="chrome.exe", category="browser", probability_percent=7.0),
            ],
            confidence_score=0.49,
        )

        decision = self.engine.decide(
            local_result=MatchResult(status="unknown", match_mode=None),
            rf_result=rf_result,
            foxio_result=FoxIOResult(status="unknown"),
        )

        self.assertIsNone(decision.application_prediction)
        self.assertEqual(decision.category_prediction, "microsoft_office")
        self.assertEqual(decision.decision_source, "category_fallback")

    def test_unknown_result_when_no_signal(self) -> None:
        decision = self.engine.decide(
            local_result=MatchResult(status="unknown", match_mode=None),
            rf_result=RandomForestResult(status="unknown"),
            foxio_result=FoxIOResult(status="unknown"),
        )

        self.assertIsNone(decision.application_prediction)
        self.assertIsNone(decision.category_prediction)
        self.assertEqual(decision.decision_source, "unknown")


if __name__ == "__main__":
    unittest.main()
