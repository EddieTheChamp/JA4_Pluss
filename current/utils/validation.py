"""Validation and normalization helpers shared across the prototype."""

from __future__ import annotations

from typing import Any


def normalize_optional_text(value: Any) -> str | None:
    """Return stripped text or ``None`` when the value is effectively empty."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_lookup_text(value: Any) -> str:
    """Return stripped text or an empty string for lookups and key generation."""
    return normalize_optional_text(value) or ""


def normalize_casefold(value: Any) -> str:
    """Normalize text for case-insensitive comparisons."""
    return normalize_lookup_text(value).casefold()


def coalesce(*values: Any) -> str | None:
    """Return the first non-empty normalized string from the given values."""
    for value in values:
        normalized = normalize_optional_text(value)
        if normalized is not None:
            return normalized
    return None


def is_present(value: Any) -> bool:
    """Check whether a lookup field contains meaningful content."""
    return bool(normalize_optional_text(value))
