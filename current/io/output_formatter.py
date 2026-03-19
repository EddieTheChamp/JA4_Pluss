"""Output formatting for terminal, JSON and CSV exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from current.core.models import ClassificationResult


def _as_dicts(results: Iterable[ClassificationResult]) -> list[dict]:
    return [result.to_dict() for result in results]


def format_terminal_result(result: ClassificationResult) -> str:
    """Render one classification result for terminal output."""
    final_decision = result.final_decision
    local_status = result.local_db.get("status")
    local_mode = result.local_db.get("match_mode")
    rf_label = result.random_forest.get("predicted_label")
    rf_confidence = result.random_forest.get("confidence_score")

    lines = [
        f"Observation: {result.observation_id}",
        f"True application: {result.true_application or 'unknown'}",
        f"True category: {result.true_category or 'unknown'}",
        f"Application prediction: {result.application_prediction or 'unknown'}",
        f"Category prediction: {result.category_prediction or 'unknown'}",
        f"Application confidence: {result.application_confidence}",
        f"Category confidence: {result.category_confidence}",
        f"Decision source: {result.decision_source}",
        f"Local DB: status={local_status}, mode={local_mode}",
        f"RF: label={rf_label or 'n/a'}, confidence={rf_confidence}",
        f"Reasoning: {result.reasoning}",
    ]
    return "\n".join(lines)


def format_terminal_batch(results: Iterable[ClassificationResult]) -> str:
    """Render many results for terminal output."""
    chunks = [format_terminal_result(result) for result in results]
    return "\n\n".join(chunks)


def write_json_output(results: Iterable[ClassificationResult], output_path: str | Path) -> Path:
    """Write the raw structured output to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_as_dicts(results), handle, indent=2, ensure_ascii=True)
    return path


def write_json_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write an arbitrary JSON payload to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def write_text_output(text: str, output_path: str | Path) -> Path:
    """Write plain text output to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_csv_output(results: Iterable[ClassificationResult], output_path: str | Path) -> Path:
    """Write a compact flattened CSV export for batch execution."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "observation_id",
        "true_application",
        "true_category",
        "application_prediction",
        "category_prediction",
        "application_correct",
        "category_correct",
        "application_confidence",
        "category_confidence",
        "decision_source",
        "reasoning",
        "ja4",
        "ja4s",
        "ja4t",
        "ja4ts",
        "local_status",
        "local_match_mode",
        "rf_predicted_label",
        "rf_predicted_category",
        "rf_confidence_score",
        "foxio_status",
        "final_label",
        "final_category",
        "final_confidence",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "observation_id": result.observation_id,
                    "true_application": result.true_application,
                    "true_category": result.true_category,
                    "application_prediction": result.application_prediction,
                    "category_prediction": result.category_prediction,
                    "application_correct": result.application_correct,
                    "category_correct": result.category_correct,
                    "application_confidence": result.application_confidence,
                    "category_confidence": result.category_confidence,
                    "decision_source": result.decision_source,
                    "reasoning": result.reasoning,
                    "ja4": result.input.get("ja4"),
                    "ja4s": result.input.get("ja4s"),
                    "ja4t": result.input.get("ja4t"),
                    "ja4ts": result.input.get("ja4ts"),
                    "local_status": result.local_db.get("status"),
                    "local_match_mode": result.local_db.get("match_mode"),
                    "rf_predicted_label": result.random_forest.get("predicted_label"),
                    "rf_predicted_category": result.random_forest.get("predicted_category"),
                    "rf_confidence_score": result.random_forest.get("confidence_score"),
                    "foxio_status": result.foxio.get("status"),
                    "final_label": result.final_decision.get("label"),
                    "final_category": result.final_decision.get("category"),
                    "final_confidence": result.final_decision.get("confidence"),
                }
            )
    return path


def format_summary_report(report: dict[str, Any]) -> str:
    """Render a human-readable summary report for batch output."""
    summary = report.get("summary", {})
    run_metadata = report.get("run_metadata", {})
    decision_sources = summary.get("decision_source_counts", {})
    supporting_signals = summary.get("supporting_signal_counts", {})

    lines = [
        "# JA4+ Category-Aware Summary",
        "",
        "## Run Setup",
        "",
        f"- Input file: `{run_metadata.get('input_file')}`",
        f"- Local DB: `{run_metadata.get('local_db_path')}`",
        f"- Training dataset: `{run_metadata.get('training_dataset_path')}`",
        f"- FoxIO DB: `{run_metadata.get('foxio_db_path')}`",
        f"- RF model: `{run_metadata.get('rf_model_path')}`",
        "",
        "## Headline Metrics",
        "",
        f"- Total records: `{summary.get('total_records', 0)}`",
        f"- Exact application correct: `{summary.get('exact_application_correct', 0)}`",
        f"- Exact application incorrect: `{summary.get('exact_application_incorrect', 0)}`",
        f"- Exact application unknown: `{summary.get('exact_application_unknown', 0)}`",
        f"- Exact application accuracy: `{summary.get('exact_application_accuracy', 0.0)}%`",
        f"- Exact application unknown rate: `{summary.get('exact_application_unknown_rate', 0.0)}%`",
        f"- Category correct: `{summary.get('category_correct', 0)}`",
        f"- Category incorrect: `{summary.get('category_incorrect', 0)}`",
        f"- Category unknown: `{summary.get('category_unknown', 0)}`",
        f"- Category accuracy: `{summary.get('category_accuracy', 0.0)}%`",
        f"- Category unknown rate: `{summary.get('category_unknown_rate', 0.0)}%`",
        "",
        "## Decision Sources",
        "",
    ]

    for key, value in decision_sources.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Supporting Signals",
            "",
        ]
    )
    for key, value in supporting_signals.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The active prototype still prioritizes exact application prediction first.",
            "Category is now used as a fallback layer when similar fingerprints make exact application prediction uncertain.",
            "This is especially useful for collisions such as `msedge.exe` versus `msedgewebview2.exe`, or `OUTLOOK.EXE` versus `WINWORD.EXE`.",
        ]
    )
    return "\n".join(lines)
