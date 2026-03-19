"""Loaders and path helpers for local JA4+ databases and datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from current.core.models import DatabaseRecord
from current.services.category_mapper import infer_category_from_application
from current.utils.validation import coalesce, normalize_lookup_text, normalize_optional_text

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_first_existing_path(*candidates: Path) -> Path | None:
    """Return the first path that exists."""
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def default_local_db_path() -> Path | None:
    """Resolve the preferred local database path for the prototype."""
    return find_first_existing_path(
        REPO_ROOT / "data" / "local_db" / "complete_custom_db.json",
        REPO_ROOT / "data" / "local_db" / "complete_custom_db_with_categories.json",
        REPO_ROOT / "Custom Database" / "complete_custom_db_with_categories.json",
        REPO_ROOT / "Custom Database" / "complete_custom_db.json",
        REPO_ROOT / "Custom Database" / "egenlagd_correlated_db.json",
        REPO_ROOT / "Custom Database" / "custom_db.json",
        REPO_ROOT / "Custom Database" / "correlated_ja4_db.json",
        REPO_ROOT / "data" / "local_db" / "egenlagd_correlated_db.json",
        REPO_ROOT / "data" / "local_db" / "custom_db.json",
        REPO_ROOT / "Dictionary" / "Custom_DB" / "egenlagd_correlated_db.json",
        REPO_ROOT / "Create Dictionary" / "custom_db.json",
        REPO_ROOT / "Create Dictionary" / "correlated_ja4_db.json",
    )


def default_foxio_db_path() -> Path | None:
    """Resolve the preferred FoxIO-style support database."""
    return find_first_existing_path(
        REPO_ROOT / "Custom Database" / "firefox_db.json",
        REPO_ROOT / "data" / "references" / "firefox_db.json",
        REPO_ROOT / "Create Dictionary" / "ja4+_db.json",
        REPO_ROOT / "Create Dictionary" / "firefox_db.json",
        REPO_ROOT / "Create Dictionary" / "foxio_db.json",
    )


def default_dataset_path() -> Path | None:
    """Resolve the preferred training dataset for RF training."""
    return find_first_existing_path(
        REPO_ROOT / "data" / "datasets" / "complete_custom_db_with_categories.json",
        REPO_ROOT / "Custom Database" / "complete_custom_db_with_categories.json",
        REPO_ROOT / "data" / "datasets" / "complete_custom_db.json",
        REPO_ROOT / "Custom Database" / "complete_custom_db.json",
        REPO_ROOT / "Custom Database" / "correlated_ja4_db_large.json",
        REPO_ROOT / "Custom Database" / "correlated_ja4_db2.json",
        REPO_ROOT / "Custom Database" / "correlated_ja4_db.json",
        REPO_ROOT / "data" / "datasets" / "correlated_ja4_db_large.json",
        REPO_ROOT / "data" / "datasets" / "correlated_ja4_db2.json",
        REPO_ROOT / "data" / "datasets" / "correlated_ja4_db.json",
        REPO_ROOT / "Create Dictionary" / "correlated_ja4_db_large.json",
        REPO_ROOT / "Create Dictionary" / "correlated_ja4_db2.json",
        REPO_ROOT / "Create Dictionary" / "correlated_ja4_db.json",
    )


def default_model_path() -> Path:
    """Resolve the default path where the RF model is stored."""
    return REPO_ROOT / "data" / "models" / "ja4_random_forest.pkl"


def load_json_payload(path: str | Path) -> Any:
    """Load a JSON file from disk."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_database_row(row: dict[str, Any]) -> DatabaseRecord:
    """Normalize one database row from the existing repo formats."""
    ja4t_value = coalesce(row.get("ja4t_fingerprint"), row.get("ja4t"))
    ja4ts_value = coalesce(row.get("ja4ts_fingerprint"), row.get("ja4ts"), ja4t_value)

    reserved_fields = {
        "application",
        "category",
        "traffic_category",
        "family",
        "ja4",
        "ja4s",
        "ja4t",
        "ja4ts",
        "ja4_fingerprint",
        "ja4s_fingerprint",
        "ja4t_fingerprint",
        "ja4ts_fingerprint",
        "count",
    }

    metadata = {
        key: value
        for key, value in row.items()
        if key not in reserved_fields and value is not None
    }

    count_value = row.get("count", 1)
    try:
        count = int(count_value)
    except (TypeError, ValueError):
        count = 1

    return DatabaseRecord(
        ja4=coalesce(row.get("ja4_fingerprint"), row.get("ja4")),
        ja4s=coalesce(row.get("ja4s_fingerprint"), row.get("ja4s")),
        ja4t=ja4t_value,
        ja4ts=ja4ts_value,
        application=normalize_optional_text(row.get("application")),
        category=infer_category_from_application(
            normalize_optional_text(row.get("application")),
            coalesce(row.get("category"), row.get("traffic_category"), row.get("family")),
        ),
        count=count,
        metadata=metadata,
    )


def load_database_records(path: str | Path) -> list[DatabaseRecord]:
    """Load and normalize database rows from disk."""
    payload = load_json_payload(path)
    if not isinstance(payload, list):
        raise ValueError(f"Database file must contain a list of records: {path}")
    return [normalize_database_row(row) for row in payload if isinstance(row, dict)]


def save_database_records(records: list[DatabaseRecord], output_path: str | Path) -> Path:
    """Persist normalized records to disk in the repository JSON format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for record in records:
        row = {
            "ja4_fingerprint": record.ja4,
            "ja4s_fingerprint": record.ja4s,
            "ja4t_fingerprint": record.ja4t,
            "ja4ts_fingerprint": record.ja4ts,
            "application": record.application,
            "category": record.category,
            "count": record.count,
        }
        row.update(record.metadata)
        payload.append(row)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def build_local_database_from_dataset(dataset_path: str | Path, output_path: str | Path) -> Path:
    """Aggregate a raw dataset into a compact local exact-match database."""
    payload = load_json_payload(dataset_path)
    if not isinstance(payload, list):
        raise ValueError(f"Dataset file must contain a list of records: {dataset_path}")

    grouped_counts: dict[tuple[str, ...], int] = defaultdict(int)
    representative_rows: dict[tuple[str, ...], dict[str, Any]] = {}

    for raw_row in payload:
        if not isinstance(raw_row, dict):
            continue

        normalized = normalize_database_row(raw_row)
        key = (
            normalize_lookup_text(normalized.application),
            normalize_lookup_text(normalized.category),
            normalize_lookup_text(normalized.ja4),
            normalize_lookup_text(normalized.ja4s),
            normalize_lookup_text(normalized.ja4t),
            normalize_lookup_text(normalized.ja4ts),
        )
        if not any(key[2:]):
            continue

        grouped_counts[key] += max(normalized.count, 1)
        representative_rows.setdefault(
            key,
            {
                "ja4_fingerprint": normalized.ja4,
                "ja4s_fingerprint": normalized.ja4s,
                "ja4t_fingerprint": normalized.ja4t,
                "ja4ts_fingerprint": normalized.ja4ts,
                "application": normalized.application,
                "category": normalized.category,
                **normalized.metadata,
            },
        )

    records: list[DatabaseRecord] = []
    for key, row in representative_rows.items():
        row["count"] = grouped_counts[key]
        records.append(normalize_database_row(row))

    return save_database_records(records, output_path)
