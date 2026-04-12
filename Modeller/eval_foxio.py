"""Iteration 1: Evaluate the public FoxIO JA4+ database against our test data.

Loads the public FoxIO DB (Dictionary/ja4+_db.json), queries it with the 20% test
split from complete_custom_db.json, and saves results for compare_all.py.

Usage:
    python eval_foxio.py
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import database_lookup

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent          # Modeller/
FOXIO_DB_PATH = _ROOT.parent / "Dictionary" / "ja4+_db.json"  # repo root
RESULTS_DIR = _ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "foxio_results.json"


def load_foxio_db(path: Path) -> dict[str, list[dict]]:
    """Load FoxIO DB into a ja4-keyed lookup dict."""
    if not path.exists():
        raise FileNotFoundError(f"FoxIO database not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw: list[dict] = json.load(f)
    index: dict[str, list[dict]] = defaultdict(list)
    for entry in raw:
        ja4_key = (entry.get("ja4_fingerprint") or entry.get("ja4") or "").strip().lower()
        if ja4_key:
            index[ja4_key].append(entry)
    print(f"[eval_foxio] FoxIO DB loaded: {len(raw)} entries, {len(index)} unique JA4 keys")
    return index


def query_foxio(foxio_index: dict[str, list[dict]], ja4: str | None) -> tuple[str, list[str], int]:
    """Query FoxIO index for one fingerprint.
    Returns (prediction, top_k_list, matches_count).
    """
    key = (ja4 or "").strip().lower()
    hits = foxio_index.get(key, [])
    if not hits:
        return "Unknown", [], 0

    # Collect application names from hits
    apps: list[str] = []
    for h in hits:
        app = h.get("application") or h.get("Application") or ""
        if app:
            apps.append(app)

    if not apps:
        return "Unknown", [], len(hits)

    # Rank by frequency
    counts: dict[str, int] = defaultdict(int)
    for a in apps:
        counts[a] += 1
    ranked = sorted(counts.keys(), key=lambda a: counts[a], reverse=True)
    return ranked[0], ranked, len(ranked)


def run() -> list[dict]:
    """Run the FoxIO evaluation on the 20% test split. Returns result records."""
    foxio_index = load_foxio_db(FOXIO_DB_PATH)
    _, test_records = database_lookup.train_test_split(seed=42, test_ratio=0.2)
    print(f"[eval_foxio] Evaluating {len(test_records)} test records...")

    results = []
    correct = 0
    for record in test_records:
        true_app = record.application or "Unknown"
        prediction, top_k, matches_count = query_foxio(foxio_index, record.ja4)
        is_correct = prediction.lower() == true_app.lower()
        if is_correct:
            correct += 1
        results.append({
            "true_app": true_app,
            "prediction": prediction,
            "top_k": top_k[:5],
            "matches_count": matches_count,
            "correct": is_correct,
            "ja4": record.ja4,
        })

    total = len(results)
    acc = correct / total * 100 if total else 0
    print(f"\n[eval_foxio] Results: {correct}/{total} correct — Top-1 Accuracy: {acc:.1f}%")

    # Collision breakdown
    unique_count = sum(1 for r in results if r["matches_count"] == 1)
    collision_count = sum(1 for r in results if r["matches_count"] > 1)
    unknown_count = sum(1 for r in results if r["matches_count"] == 0)
    print(f"           Unique: {unique_count} | Collisions: {collision_count} | Unknown: {unknown_count}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[eval_foxio] Saved {len(results)} results → {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    run()
