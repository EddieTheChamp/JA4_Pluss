"""Tests for the end-to-end pipeline flow."""

from __future__ import annotations

import unittest

from current.services.local_matcher import LocalDatabaseMatcher
from current.core.models import (
    DatabaseRecord,
    FoxIOResult,
    Observation,
    PredictionCandidate,
    RandomForestResult,
)
from current.core.classifier import JA4ClassifierPipeline


class StubRFInference:
    def __init__(self, result: RandomForestResult):
        self.result = result

    def predict(self, observation: Observation) -> RandomForestResult:
        return self.result


class StubFoxIOAdapter:
    def __init__(self, result: FoxIOResult):
        self.result = result

    def lookup(self, observation: Observation) -> FoxIOResult:
        return self.result


class PipelineTests(unittest.TestCase):
    def test_pipeline_fallback_to_rf(self) -> None:
        matcher = LocalDatabaseMatcher.from_records(
            [
                DatabaseRecord(
                    ja4="t13d1516h2_local_match",
                    ja4s="t130200_local",
                    application="msedge.exe",
                    category="browser",
                    count=5,
                )
            ]
        )
        rf_result = RandomForestResult(
            status="predicted",
            predicted_label="firefox.exe",
            predicted_category="browser",
            top_k=[
                PredictionCandidate(label="firefox.exe", category="browser", probability_percent=81.0),
                PredictionCandidate(label="chrome.exe", category="browser", probability_percent=11.0),
            ],
            confidence_score=0.81,
        )
        pipeline = JA4ClassifierPipeline(
            local_matcher=matcher,
            rf_inference=StubRFInference(rf_result),
            foxio_adapter=StubFoxIOAdapter(FoxIOResult(status="unavailable")),
        )

        result = pipeline.classify_observation(
            Observation(
                observation_id="obs-rf",
                ja4="t13d1717h2_unknown_xxx",
                ja4s="t130200_unknown",
                true_application="firefox.exe",
                true_category="browser",
            )
        )

        self.assertEqual(result.application_prediction, "firefox.exe")
        self.assertEqual(result.category_prediction, "browser")
        self.assertTrue(result.application_correct)
        self.assertTrue(result.category_correct)
        self.assertEqual(result.decision_source, "random_forest")


if __name__ == "__main__":
    unittest.main()
