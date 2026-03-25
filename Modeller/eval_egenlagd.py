"""Iteration 2: Evaluate the custom (Egenlagd) dictionary against the test split.

Builds an in-memory exact-match index from the 80% training split of
complete_custom_db.json and tests on the 20% test split.

Fallback priority:
    ja4+ja4s+ja4t+ja4ts → ja4+ja4s+ja4ts → ja4+ja4s+ja4t → ja4+ja4s → ja4 only

Usage:
    python eval_egenlagd.py
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import database_lookup
from models import DatabaseRecord

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent  # Modeller/
RESULTS_DIR = _ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "egenlagd_results.json"

# Lookup plans in priority order (most strict → least strict)
_LOOKUP_PLANS: list[tuple[str, tuple[str, ...]]] = [
    ("ja4_ja4s_ja4t_ja4ts", ("ja4", "ja4s", "ja4t", "ja4ts")),
    ("ja4_ja4s_ja4ts",      ("ja4", "ja4s", "ja4ts")),
    ("ja4_ja4s_ja4t",       ("ja4", "ja4s", "ja4t")),
    ("ja4_ja4s",            ("ja4", "ja4s")),
    ("ja4_ja4ts",           ("ja4", "ja4ts")),
    ("ja4_ja4t",            ("ja4", "ja4t")),
    ("ja4_only",            ("ja4",)),
]


def _field(record: DatabaseRecord, name: str) -> str:
    return ((getattr(record, name) or "").strip().lower())


def build_index(train_records: list[DatabaseRecord]) -> dict[str, dict[str, list[DatabaseRecord]]]:
    """Build multi-key index from training records."""
    index: dict[str, dict[str, list[DatabaseRecord]]] = {
        mode: defaultdict(list) for mode, _ in _LOOKUP_PLANS
    }
    for record in train_records:
        for mode, fields in _LOOKUP_PLANS:
            values = [_field(record, f) for f in fields]
            if all(values):
                key = "|".join(values)
                index[mode][key].append(record)
    return index


def lookup(index: dict, record: DatabaseRecord) -> tuple[str, list[str], int, str]:
    """Find best match for a test record using the priority fallback chain.
    Returns (top_prediction, top_k_apps, matches_count, match_mode).
    """
    for mode, fields in _LOOKUP_PLANS:
        values = [_field(record, f) for f in fields]
        if not all(values):
            continue
        key = "|".join(values)
        hits = index[mode].get(key, [])
        if not hits:
            continue

        # Aggregate by application, weighted by count
        app_counts: dict[str, int] = defaultdict(int)
        for hit in hits:
            if hit.application:
                app_counts[hit.application] += max(hit.count, 1)

        if not app_counts:
            continue

        ranked = sorted(app_counts.keys(), key=lambda a: app_counts[a], reverse=True)
        return ranked[0], ranked, len(ranked), mode

    return "Unknown", [], 0, "none"


def run() -> list[dict]:
    """Run the Egenlagd evaluation. Returns result records."""
    train_records, test_records = database_lookup.train_test_split(seed=42, test_ratio=0.2)
    print(f"[eval_egenlagd] Training on {len(train_records)} records...")
    index = build_index(train_records)
    print(f"[eval_egenlagd] Evaluating {len(test_records)} test records...")

    results = []
    correct = 0
    for record in test_records:
        true_app = record.application or "Unknown"
        prediction, top_k, matches_count, match_mode = lookup(index, record)
        is_correct = prediction.lower() == true_app.lower()
        if is_correct:
            correct += 1
        results.append({
            "true_app": true_app,
            "prediction": prediction,
            "top_k": top_k[:5],
            "matches_count": matches_count,
            "match_mode": match_mode,
            "correct": is_correct,
            "ja4": record.ja4,
        })

    total = len(results)
    acc = correct / total * 100 if total else 0
    print(f"\n[eval_egenlagd] Results: {correct}/{total} correct — Top-1 Accuracy: {acc:.1f}%")

    unique_count = sum(1 for r in results if r["matches_count"] == 1)
    collision_count = sum(1 for r in results if r["matches_count"] > 1)
    unknown_count = sum(1 for r in results if r["matches_count"] == 0)
    print(f"           Unique: {unique_count} | Collisions: {collision_count} | Unknown: {unknown_count}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[eval_egenlagd] Saved {len(results)} results → {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    run()
