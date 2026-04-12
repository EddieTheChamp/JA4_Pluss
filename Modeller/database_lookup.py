"""Load and query the local JA4+ database.

Single source of truth: Custom Database/complete_custom_db.json
All other scripts import from here — never load the JSON directly.
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from models import DatabaseRecord, infer_category

# ── hardcoded dataset path ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]  # Modeller/ → repo root
DB_PATH = _ROOT / "Custom Database" / "complete_custom_db.json"

# ── module-level cache (loaded once per process) ──────────────────────────────
_CACHE: list[DatabaseRecord] | None = None


def load_db() -> list[DatabaseRecord]:
    """Load the database from disk (cached after first call)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    with DB_PATH.open("r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)
    _CACHE = [_normalize(row) for row in raw if isinstance(row, dict)]
    print(f"[database_lookup] Loaded {len(_CACHE)} records from {DB_PATH.name}")
    return _CACHE


def _normalize(row: dict[str, Any]) -> DatabaseRecord:
    """Convert a raw JSON row into a DatabaseRecord."""
    ja4 = row.get("ja4_fingerprint") or row.get("ja4")
    ja4s = row.get("ja4s_fingerprint") or row.get("ja4s")
    ja4_string = row.get("ja4_fingerprint_string")
    ja4s_string = row.get("ja4s_fingerprint_string")
    ja4t = row.get("ja4t_fingerprint") or row.get("ja4t")
    ja4ts = row.get("ja4ts_fingerprint") or row.get("ja4ts") or ja4t
    application = (row.get("application") or "").strip() or None
    explicit_cat = row.get("category") or row.get("traffic_category") or row.get("family")
    category = infer_category(application, explicit_cat)
    count_raw = row.get("count", 1)
    try:
        count = int(count_raw)
    except (TypeError, ValueError):
        count = 1

    reserved = {"application", "category", "traffic_category", "family", "ja4",
                "ja4s", "ja4t", "ja4ts", "ja4_fingerprint", "ja4s_fingerprint",
                "ja4_fingerprint_string", "ja4s_fingerprint_string",
                "ja4t_fingerprint", "ja4ts_fingerprint", "count"}
    metadata = {k: v for k, v in row.items() if k not in reserved and v is not None}

    return DatabaseRecord(
        ja4=ja4, ja4s=ja4s, ja4_string=ja4_string, ja4s_string=ja4s_string,
        ja4t=ja4t, ja4ts=ja4ts,
        application=application, category=category,
        count=count, metadata=metadata,
    )


# ── query helpers ─────────────────────────────────────────────────────────────

def find_by_ja4(ja4: str) -> list[DatabaseRecord]:
    """Return all records that match the given JA4 string exactly."""
    key = (ja4 or "").strip().lower()
    return [r for r in load_db() if (r.ja4 or "").strip().lower() == key]


def find_by_ja4_and_ja4s(ja4: str, ja4s: str) -> list[DatabaseRecord]:
    """Return all records matching both JA4 and JA4S."""
    k4 = (ja4 or "").strip().lower()
    ks = (ja4s or "").strip().lower()
    return [
        r for r in load_db()
        if (r.ja4 or "").strip().lower() == k4
        and (r.ja4s or "").strip().lower() == ks
    ]


def train_test_split(seed: int = 42, test_ratio: float = 0.2) -> tuple[list[DatabaseRecord], list[DatabaseRecord]]:
    """Return a reproducible 80/20 train/test split of the database.

    Groups records by application, then splits each group individually
    so every application is represented in both splits.
    """
    import random
    rng = random.Random(seed)

    # group by application
    groups: dict[str, list[DatabaseRecord]] = defaultdict(list)
    unlabeled: list[DatabaseRecord] = []
    for record in load_db():
        if record.application:
            groups[record.application].append(record)
        else:
            unlabeled.append(record)

    train: list[DatabaseRecord] = []
    test: list[DatabaseRecord] = []

    for app_records in groups.values():
        shuffled = list(app_records)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    return train, test
