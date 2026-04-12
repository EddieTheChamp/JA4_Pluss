"""Command-line interface for the JA4+ classification prototype."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from current.services.loaders import (
    build_local_database_from_dataset,
    default_dataset_path,
    default_foxio_db_path,
    default_local_db_path,
    default_model_path,
)
from current.io.batch_runner import build_batch_report, run_batch, save_batch_report, save_batch_results, save_batch_summary
from current.io.input_parser import observation_from_cli_args, parse_input_file
from current.io.output_formatter import format_terminal_batch, format_terminal_result, write_json_output
from current.services.random_forest import JA4RandomForestModel
from current.core.classifier import JA4ClassifierPipeline
from current.utils.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser with subcommands."""
    parser = argparse.ArgumentParser(description="JA4+ network classification prototype")
    parser.add_argument("--log-level", default="WARNING", help="Logging level")

    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser("classify", help="Classify a single observation")
    classify.add_argument("--observation-id", help="Optional observation identifier")
    classify.add_argument("--source", help="Optional observation source")
    classify.add_argument("--ja4", help="JA4 fingerprint")
    classify.add_argument("--ja4s", help="JA4S fingerprint")
    classify.add_argument("--ja4t", help="JA4T fingerprint")
    classify.add_argument("--ja4ts", help="JA4TS fingerprint")
    add_shared_runtime_arguments(classify)
    classify.add_argument("--output", help="Optional JSON file to save the structured result")

    classify_file = subparsers.add_parser("classify-file", help="Classify observations from JSON or CSV")
    classify_file.add_argument("--input", required=True, help="Path to input JSON or CSV file")
    classify_file.add_argument("--output", help="Optional output file for results")
    classify_file.add_argument("--output-format", choices=["json", "csv", "terminal"], default="json")
    classify_file.add_argument("--report-output", help="Optional JSON file for combined summary and results")
    classify_file.add_argument("--summary-output", help="Optional text summary file")
    add_shared_runtime_arguments(classify_file)

    train_rf = subparsers.add_parser("train-rf", help="Train and save a Random Forest model")
    train_rf.add_argument(
        "--dataset",
        default=str(default_dataset_path()) if default_dataset_path() else None,
        required=default_dataset_path() is None,
        help="Path to labeled JSON dataset",
    )
    train_rf.add_argument(
        "--model-output",
        default=str(default_model_path()),
        help="Where to save the trained model bundle",
    )

    build_db = subparsers.add_parser("build-local-db", help="Aggregate a dataset into a local database JSON")
    build_db.add_argument(
        "--dataset",
        default=str(default_dataset_path()) if default_dataset_path() else None,
        required=default_dataset_path() is None,
        help="Path to labeled JSON dataset",
    )
    build_db.add_argument(
        "--output",
        default=str(default_local_db_path()) if default_local_db_path() else str(Path("database") / "local_db.json"),
        help="Where to save the generated database JSON",
    )

    return parser


def add_shared_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common runtime arguments used by classify commands."""
    parser.add_argument(
        "--local-db",
        default=str(default_local_db_path()) if default_local_db_path() else None,
        help="Path to local exact-match database JSON",
    )
    parser.add_argument(
        "--foxio-db",
        default=str(default_foxio_db_path()) if default_foxio_db_path() else None,
        help="Path to FoxIO-style reference database JSON",
    )
    parser.add_argument(
        "--rf-model",
        default=str(default_model_path()),
        help="Path to persisted Random Forest model bundle",
    )
    parser.add_argument(
        "--dataset",
        default=str(default_dataset_path()) if default_dataset_path() else None,
        help="Optional training dataset used when the RF model must be trained automatically",
    )


def run(argv: list[str] | None = None) -> int:
    """Run the CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    log_level = getattr(logging, str(args.log_level).upper(), logging.WARNING)
    configure_logging(log_level)

    if args.command == "train-rf":
        model = JA4RandomForestModel.train_from_json(args.dataset)
        model_path = model.save(args.model_output)
        print(f"Saved Random Forest model to {model_path}")
        return 0

    if args.command == "build-local-db":
        output_path = build_local_database_from_dataset(args.dataset, args.output)
        print(f"Saved local database to {output_path}")
        return 0

    pipeline = JA4ClassifierPipeline.from_paths(
        local_db_path=args.local_db,
        foxio_db_path=args.foxio_db,
        rf_model_path=args.rf_model,
        training_dataset_path=args.dataset,
        auto_train_rf=True,
        auto_build_local_db=False,
    )

    if args.command == "classify":
        observation = observation_from_cli_args(
            observation_id=args.observation_id,
            source=args.source,
            ja4=args.ja4,
            ja4s=args.ja4s,
            ja4t=args.ja4t,
            ja4ts=args.ja4ts,
        )
        result = pipeline.classify_observation(observation)
        print(format_terminal_result(result))
        if args.output:
            write_json_output([result], args.output)
        return 0

    if args.command == "classify-file":
        observations = parse_input_file(args.input)
        results = run_batch(pipeline, observations)
        if args.output_format == "terminal":
            print(format_terminal_batch(results))
        else:
            output_path = args.output or _default_output_path(args.input, args.output_format)
            save_batch_results(results, output_path, args.output_format)
            print(f"Saved {len(results)} result(s) to {output_path}")
            report = build_batch_report(
                results,
                pipeline=pipeline,
                input_path=args.input,
                results_path=output_path,
            )
            report_output = args.report_output or _report_output_path(output_path)
            summary_output = args.summary_output or _summary_output_path(output_path)
            save_batch_report(report, report_output)
            save_batch_summary(report, summary_output)
            print(f"Saved combined report to {report_output}")
            print(f"Saved text summary to {summary_output}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


def _default_output_path(input_path: str | Path, output_format: str) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}_results.{output_format}")


def _report_output_path(results_path: str | Path) -> Path:
    path = Path(results_path)
    return path.with_name(f"{path.stem}_report.json")


def _summary_output_path(results_path: str | Path) -> Path:
    path = Path(results_path)
    return path.with_name(f"{path.stem}_summary.md")
