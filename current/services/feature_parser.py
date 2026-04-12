"""Feature parsing for JA4+ observations used by Random Forest."""

from __future__ import annotations

from typing import Any

from current.core.models import DatabaseRecord, Observation
from current.utils.validation import normalize_lookup_text

CATEGORICAL_FEATURES = [
    "protocol",
    "tls_version",
    "sni_indicator",
    "alpn",
    "ciphers_hash",
    "extensions_hash",
    "ja4s",
    "ja4t",
    "ja4ts",
]
NUMERIC_FEATURES = [
    "num_ciphers",
    "num_extensions",
    "has_ja4s",
    "has_ja4t",
    "has_ja4ts",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
MISSING_TOKEN = "__missing__"


def parse_ja4_fingerprint(ja4_value: str | None) -> dict[str, Any]:
    """Parse the compact JA4 string into a feature dictionary."""
    ja4_text = normalize_lookup_text(ja4_value)
    defaults = {
        "protocol": MISSING_TOKEN,
        "tls_version": MISSING_TOKEN,
        "sni_indicator": MISSING_TOKEN,
        "num_ciphers": 0,
        "num_extensions": 0,
        "alpn": MISSING_TOKEN,
        "ciphers_hash": MISSING_TOKEN,
        "extensions_hash": MISSING_TOKEN,
    }
    if not ja4_text or "_" not in ja4_text:
        return defaults

    parts = ja4_text.split("_")
    if len(parts) != 3:
        return defaults

    prefix = parts[0].ljust(10, "0")
    return {
        "protocol": prefix[0] or MISSING_TOKEN,
        "tls_version": prefix[1:3] or MISSING_TOKEN,
        "sni_indicator": prefix[3] or MISSING_TOKEN,
        "num_ciphers": int(prefix[4:6]) if prefix[4:6].isdigit() else 0,
        "num_extensions": int(prefix[6:8]) if prefix[6:8].isdigit() else 0,
        "alpn": prefix[8:10] or MISSING_TOKEN,
        "ciphers_hash": parts[1] or MISSING_TOKEN,
        "extensions_hash": parts[2] or MISSING_TOKEN,
    }


def observation_to_feature_dict(observation: Observation) -> dict[str, Any]:
    """Convert one normalized observation into a flat feature mapping."""
    parsed = parse_ja4_fingerprint(observation.ja4)
    parsed.update(
        {
            "ja4s": normalize_lookup_text(observation.ja4s) or MISSING_TOKEN,
            "ja4t": normalize_lookup_text(observation.ja4t) or MISSING_TOKEN,
            "ja4ts": normalize_lookup_text(observation.ja4ts) or MISSING_TOKEN,
            "has_ja4s": 1 if normalize_lookup_text(observation.ja4s) else 0,
            "has_ja4t": 1 if normalize_lookup_text(observation.ja4t) else 0,
            "has_ja4ts": 1 if normalize_lookup_text(observation.ja4ts) else 0,
        }
    )
    return parsed


def record_to_training_sample(record: DatabaseRecord) -> dict[str, Any] | None:
    """Convert a database record into a training sample for RF."""
    if not record.application:
        return None

    observation = Observation(
        observation_id="training-record",
        ja4=record.ja4,
        ja4s=record.ja4s,
        ja4t=record.ja4t,
        ja4ts=record.ja4ts,
        source="training-dataset",
        raw_record=record.metadata,
    )
    sample = observation_to_feature_dict(observation)
    sample["label"] = record.application
    sample["category"] = record.category
    sample["weight"] = max(record.count, 1)
    return sample


def records_to_training_samples(records: list[DatabaseRecord]) -> list[dict[str, Any]]:
    """Convert normalized records into RF training samples."""
    samples: list[dict[str, Any]] = []
    for record in records:
        sample = record_to_training_sample(record)
        if sample is not None:
            samples.append(sample)
    return samples
