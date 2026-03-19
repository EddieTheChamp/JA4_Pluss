"""Tests for category-aware batch reporting."""

from __future__ import annotations

import unittest

from current.io.batch_runner import build_batch_report
from current.core.models import ClassificationResult


class BatchRunnerTests(unittest.TestCase):
    def test_summary_includes_category_accuracy(self) -> None:
        results = [
            ClassificationResult(
                observation_id="obs-1",
                input={"ja4": "a"},
                true_application="msedge.exe",
                true_category="browser",
                application_prediction="msedge.exe",
                category_prediction="browser",
                application_correct=True,
                category_correct=True,
                application_confidence="high",
                category_confidence="high",
                decision_source="local_database",
                reasoning="Exact local match.",
                local_db={"status": "unique", "match_mode": "ja4_ja4s"},
                random_forest={"status": "skipped"},
                foxio={"status": "supported"},
                final_decision={"label": "msedge.exe", "category": "browser", "confidence": "high"},
            ),
            ClassificationResult(
                observation_id="obs-2",
                input={"ja4": "b"},
                true_application="msedgewebview2.exe",
                true_category="browser",
                application_prediction="msedge.exe",
                category_prediction="browser",
                application_correct=False,
                category_correct=True,
                application_confidence="medium",
                category_confidence="high",
                decision_source="ambiguous_local_match_resolved_by_rf",
                reasoning="RF resolved the local ambiguity.",
                local_db={"status": "ambiguous", "match_mode": "ja4_ja4s"},
                random_forest={"status": "predicted"},
                foxio={"status": "supported"},
                final_decision={"label": "msedge.exe", "category": "browser", "confidence": "medium"},
            ),
            ClassificationResult(
                observation_id="obs-3",
                input={"ja4": "c"},
                true_application="OUTLOOK.EXE",
                true_category="microsoft_office",
                application_prediction=None,
                category_prediction="microsoft_office",
                application_correct=False,
                category_correct=True,
                application_confidence="low",
                category_confidence="medium",
                decision_source="category_fallback",
                reasoning="Category fallback used RF top-k.",
                local_db={"status": "unknown", "match_mode": None},
                random_forest={"status": "predicted"},
                foxio={"status": "unknown"},
                final_decision={"label": None, "category": "microsoft_office", "confidence": "low"},
            ),
        ]

        report = build_batch_report(results, input_path="input.json", results_path="results.json")
        summary = report["summary"]

        self.assertEqual(summary["total_records"], 3)
        self.assertEqual(summary["exact_application_correct"], 1)
        self.assertEqual(summary["exact_application_incorrect"], 1)
        self.assertEqual(summary["exact_application_unknown"], 1)
        self.assertEqual(summary["category_correct"], 3)
        self.assertEqual(summary["category_incorrect"], 0)
        self.assertEqual(summary["category_unknown"], 0)
        self.assertEqual(summary["exact_application_accuracy"], 33.33)
        self.assertEqual(summary["category_accuracy"], 100.0)
        self.assertEqual(summary["decision_source_counts"]["category_fallback"], 1)
        self.assertEqual(summary["supporting_signal_counts"]["local_unique"], 1)
        self.assertEqual(summary["supporting_signal_counts"]["local_ambiguous"], 1)


if __name__ == "__main__":
    unittest.main()
