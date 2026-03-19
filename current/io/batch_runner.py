"""Batch execution helpers for pipeline runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from current.io.output_formatter import (
    format_summary_report,
    write_csv_output,
    write_json_output,
    write_json_payload,
    write_text_output,
)
from current.core.classifier import JA4ClassifierPipeline
from current.core.models import ClassificationResult, Observation


def run_batch(
    pipeline: JA4ClassifierPipeline,
    observations: list[Observation],
) -> list[ClassificationResult]:
    """Classify a batch of observations."""
    return pipeline.classify_batch(observations)


def save_batch_results(
    results: list[ClassificationResult],
    output_path: str | Path,
    output_format: str,
) -> Path:
    """Save batch results in the requested output format."""
    if output_format == "json":
        return write_json_output(results, output_path)
    if output_format == "csv":
        return write_csv_output(results, output_path)
    raise ValueError(f"Unsupported output format: {output_format}")


def build_batch_report(
    results: list[ClassificationResult],
    *,
    pipeline: JA4ClassifierPipeline | None = None,
    input_path: str | Path | None = None,
    results_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a structured aggregate report for one batch run."""
    total_records = len(results)
    decision_source_counts = Counter(result.decision_source for result in results)
    local_status_counts = Counter(result.local_db.get("status", "unknown") for result in results)
    rf_status_counts = Counter(result.random_forest.get("status", "unknown") for result in results)
    foxio_status_counts = Counter(result.foxio.get("status", "unknown") for result in results)

    exact_application_correct = sum(1 for result in results if result.application_correct is True)
    exact_application_incorrect = sum(
        1
        for result in results
        if result.application_prediction is not None and result.application_correct is False
    )
    exact_application_unknown = sum(1 for result in results if result.application_prediction is None)

    category_correct = sum(1 for result in results if result.category_correct is True)
    category_incorrect = sum(
        1
        for result in results
        if result.category_prediction is not None and result.category_correct is False
    )
    category_unknown = sum(1 for result in results if result.category_prediction is None)

    summary = {
        "total_records": total_records,
        "exact_application_correct": exact_application_correct,
        "exact_application_incorrect": exact_application_incorrect,
        "exact_application_unknown": exact_application_unknown,
        "exact_application_accuracy": _safe_percent(exact_application_correct, total_records),
        "exact_application_unknown_rate": _safe_percent(exact_application_unknown, total_records),
        "category_correct": category_correct,
        "category_incorrect": category_incorrect,
        "category_unknown": category_unknown,
        "category_accuracy": _safe_percent(category_correct, total_records),
        "category_unknown_rate": _safe_percent(category_unknown, total_records),
        "decision_source_counts": _ordered_counts(
            decision_source_counts,
            ["local_database", "ambiguous_local_match_resolved_by_rf", "random_forest", "category_fallback", "unknown"],
        ),
        "supporting_signal_counts": {
            "local_unique": local_status_counts.get("unique", 0),
            "local_ambiguous": local_status_counts.get("ambiguous", 0),
            "rf_predicted": rf_status_counts.get("predicted", 0),
            "rf_skipped": rf_status_counts.get("skipped", 0),
            "foxio_supported": foxio_status_counts.get("supported", 0),
            "foxio_unknown": foxio_status_counts.get("unknown", 0),
        },
        "interpretation": (
            "The prototype still prioritizes exact application matching first. "
            "Category is now used as a fallback layer when similar fingerprints make exact application "
            "prediction uncertain, which is especially useful for collisions such as msedge.exe versus "
            "msedgewebview2.exe, or OUTLOOK.EXE versus WINWORD.EXE."
        ),
    }

    return {
        "run_metadata": {
            "input_file": str(input_path) if input_path else None,
            "results_file": str(results_path) if results_path else None,
            "local_db_path": pipeline.runtime_paths.get("local_db_path") if pipeline else None,
            "training_dataset_path": pipeline.runtime_paths.get("training_dataset_path") if pipeline else None,
            "foxio_db_path": pipeline.runtime_paths.get("foxio_db_path") if pipeline else None,
            "rf_model_path": pipeline.runtime_paths.get("rf_model_path") if pipeline else None,
        },
        "summary": summary,
        "results": [result.to_dict() for result in results],
    }


def save_batch_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a combined JSON report that contains summary and results."""
    return write_json_payload(report, output_path)


def save_batch_summary(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a human-readable summary report to disk."""
    return write_text_output(format_summary_report(report), output_path)


def _safe_percent(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(value / total * 100.0, 2)


def _ordered_counts(counter: Counter[str], preferred_order: list[str]) -> dict[str, int]:
    payload: dict[str, int] = {}
    for key in preferred_order:
        payload[key] = counter.get(key, 0)
    for key, value in counter.items():
        if key not in payload:
            payload[key] = value
    return payload
