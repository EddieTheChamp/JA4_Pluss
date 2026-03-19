"""Input parsing for CLI arguments, JSON files and CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from current.services.category_mapper import infer_category_from_application
from current.services.loaders import load_json_payload
from current.core.models import Observation
from current.utils.validation import coalesce, normalize_lookup_text, normalize_optional_text


def observation_from_mapping(
    payload: dict[str, Any],
    *,
    default_id: str,
    default_source: str,
) -> Observation:
    """Normalize one raw payload into the shared Observation model."""
    ja4t_value = coalesce(payload.get("ja4t"), payload.get("ja4t_fingerprint"))
    true_application = normalize_optional_text(payload.get("application"))
    true_category = coalesce(payload.get("category"), payload.get("traffic_category"), payload.get("family"))
    return Observation(
        observation_id=normalize_lookup_text(
            coalesce(payload.get("observation_id"), payload.get("id"), payload.get("row_id"))
        )
        or default_id,
        ja4=coalesce(payload.get("ja4"), payload.get("ja4_fingerprint")),
        ja4s=coalesce(payload.get("ja4s"), payload.get("ja4s_fingerprint")),
        ja4t=ja4t_value,
        ja4ts=coalesce(payload.get("ja4ts"), payload.get("ja4ts_fingerprint"), ja4t_value),
        source=normalize_optional_text(payload.get("source")) or default_source,
        true_application=true_application,
        true_category=infer_category_from_application(true_application, true_category),
        raw_record=dict(payload),
    )


def parse_json_input(path: str | Path) -> list[Observation]:
    """Parse one JSON observation or a list of observations."""
    file_path = Path(path)
    payload = load_json_payload(file_path)

    if isinstance(payload, dict):
        return [
            observation_from_mapping(
                payload,
                default_id=file_path.stem,
                default_source=str(file_path),
            )
        ]
    if not isinstance(payload, list):
        raise ValueError(f"JSON input must be an object or list of objects: {file_path}")

    observations: list[Observation] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            continue
        observations.append(
            observation_from_mapping(
                row,
                default_id=f"{file_path.stem}-{index}",
                default_source=str(file_path),
            )
        )
    return observations


def parse_csv_input(path: str | Path) -> list[Observation]:
    """Parse observations from a CSV file."""
    file_path = Path(path)
    observations: list[Observation] = []
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            observations.append(
                observation_from_mapping(
                    row,
                    default_id=f"{file_path.stem}-{index}",
                    default_source=str(file_path),
                )
            )
    return observations


def parse_input_file(path: str | Path) -> list[Observation]:
    """Dispatch input parsing based on file extension."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return parse_json_input(file_path)
    if suffix == ".csv":
        return parse_csv_input(file_path)
    raise ValueError(f"Unsupported input file type: {file_path.suffix}")


def observation_from_cli_args(
    *,
    observation_id: str | None,
    ja4: str | None,
    ja4s: str | None,
    ja4t: str | None,
    ja4ts: str | None,
    source: str | None,
) -> Observation:
    """Create an Observation from individual CLI options."""
    payload = {
        "observation_id": observation_id,
        "ja4": ja4,
        "ja4s": ja4s,
        "ja4t": ja4t,
        "ja4ts": ja4ts,
        "source": source,
    }
    return observation_from_mapping(payload, default_id="cli-observation", default_source="cli")
