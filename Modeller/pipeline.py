"""Combined JA4+ classifier pipeline.

Runs all three classifiers in priority order — same logic as the existing
current/core/classifier.py + current/core/decision_engine.py, written as
plain functions (no classes).

Priority order:
    1. Egenlagd exact match  → unique hit → return immediately
    2. Ambiguous exact match → use RF to pick the best candidate
    3. No local match        → RF becomes primary (if confidence ≥ 0.60)
    4. FoxIO                 → used as supporting evidence / category fallback at every step

Usage:
    python pipeline.py
    python pipeline.py --ja4 "t13d1516h2_aaa_bbb" --ja4s "t130200_1301_ccc"
"""

from __future__ import annotations
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import database_lookup
import ja4_parser
from models import (
    ClassificationResult,
    DatabaseRecord,
    FinalDecision,
    Observation,
    infer_category,
)

# ── constants ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent          # Modeller/
MODEL_PATH = _ROOT / "data" / "models" / "ja4_rf.pkl"
FOXIO_DB_PATH = _ROOT / "data" / "models" / "ja4+_db.json"  # internal data model path

RF_ACCEPT_THRESHOLD = 0.60   # minimum RF confidence to trust the prediction
RF_HIGH_THRESHOLD = 0.75     # high-confidence band
RF_MARGIN_PERCENT = 15.0     # minimum gap between top-1 and top-2 RF candidates

# Lookup plans in priority order
_LOOKUP_PLANS: list[tuple[str, tuple[str, ...]]] = [
    ("ja4_ja4s_ja4t_ja4ts", ("ja4", "ja4s", "ja4t", "ja4ts")),
    ("ja4_ja4s_ja4ts",      ("ja4", "ja4s", "ja4ts")),
    ("ja4_ja4s_ja4t",       ("ja4", "ja4s", "ja4t")),
    ("ja4_ja4s",            ("ja4", "ja4s")),
    ("ja4_ja4ts",           ("ja4", "ja4ts")),
    ("ja4_ja4t",            ("ja4", "ja4t")),
    ("ja4_only",            ("ja4",)),
]
_STRICT_MODES = {"ja4_ja4s_ja4t_ja4ts", "ja4_ja4s_ja4ts", "ja4_ja4s_ja4t", "ja4_ja4s"}


# ── index builders (built once, cached in module-level vars) ──────────────────
_EGENLAGD_INDEX: dict | None = None
_FOXIO_INDEX: dict | None = None
_RF_BUNDLE: dict | None = None


def _field(record: DatabaseRecord, name: str) -> str:
    return (getattr(record, name) or "").strip().lower()


def _get_egenlagd_index() -> dict:
    global _EGENLAGD_INDEX
    if _EGENLAGD_INDEX is not None:
        return _EGENLAGD_INDEX
    train_records, _ = database_lookup.train_test_split(seed=42, test_ratio=0.2)
    index: dict[str, dict[str, list[DatabaseRecord]]] = {
        mode: defaultdict(list) for mode, _ in _LOOKUP_PLANS
    }
    for record in train_records:
        for mode, fields in _LOOKUP_PLANS:
            values = [_field(record, f) for f in fields]
            if all(values):
                index[mode]["|".join(values)].append(record)
    _EGENLAGD_INDEX = index
    print("[pipeline] Egenlagd index built.")
    return _EGENLAGD_INDEX


def _get_foxio_index() -> dict:
    global _FOXIO_INDEX
    if _FOXIO_INDEX is not None:
        return _FOXIO_INDEX
    if not FOXIO_DB_PATH.exists():
        print("[pipeline] FoxIO DB not found — FoxIO support disabled.")
        _FOXIO_INDEX = {}
        return _FOXIO_INDEX
    with FOXIO_DB_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    index: dict[str, list[dict]] = defaultdict(list)
    for entry in raw:
        key = (entry.get("ja4_fingerprint") or entry.get("ja4") or "").strip().lower()
        if key:
            index[key].append(entry)
    _FOXIO_INDEX = index
    print("[pipeline] FoxIO index loaded.")
    return _FOXIO_INDEX


def _get_rf_bundle():
    global _RF_BUNDLE
    if _RF_BUNDLE is not None:
        return _RF_BUNDLE
    if not MODEL_PATH.exists():
        print("[pipeline] RF model not found — RF disabled. Run eval_random_forest.py first.")
        return None
    with MODEL_PATH.open("rb") as f:
        _RF_BUNDLE = pickle.load(f)
    print("[pipeline] RF model loaded.")
    return _RF_BUNDLE


# ── egenlagd exact match ──────────────────────────────────────────────────────

def _egenlagd_lookup(obs: Observation) -> dict:
    """Run the Egenlagd exact-match lookup. Returns a result dict."""
    index = _get_egenlagd_index()
    obs_fields = {"ja4": obs.ja4, "ja4s": obs.ja4s, "ja4t": obs.ja4t, "ja4ts": obs.ja4ts}

    for mode, fields in _LOOKUP_PLANS:
        values = [(obs_fields.get(f) or "").strip().lower() for f in fields]
        if not all(values):
            continue
        key = "|".join(values)
        hits = index.get(mode, {}).get(key, [])
        if not hits:
            continue

        app_counts: dict[str, int] = defaultdict(int)
        cat_map: dict[str, str] = {}
        for hit in hits:
            if hit.application:
                app_counts[hit.application] += max(hit.count, 1)
                cat_map[hit.application] = hit.category or "unknown"

        if not app_counts:
            continue

        ranked = sorted(app_counts.keys(), key=lambda a: app_counts[a], reverse=True)
        status = "unique" if len(ranked) == 1 else "ambiguous"
        confidence = 0.9 if mode in _STRICT_MODES else 0.7
        if status == "ambiguous":
            confidence *= 0.6
        return {
            "status": status, "match_mode": mode,
            "candidates": ranked, "top_app": ranked[0],
            "top_category": cat_map.get(ranked[0], "unknown"),
            "confidence": round(confidence, 3),
        }

    return {"status": "unknown", "match_mode": None, "candidates": [], "top_app": None, "top_category": None, "confidence": 0.0}


# ── RF inference ──────────────────────────────────────────────────────────────

def _rf_predict(obs: Observation) -> dict:
    """Run RF inference. Returns a result dict."""
    bundle = _get_rf_bundle()
    if bundle is None:
        return {"status": "unavailable", "prediction": None, "top_k": [], "confidence": 0.0, "category": None}

    try:
        import pandas as pd
    except ImportError:
        return {"status": "unavailable", "prediction": None, "top_k": [], "confidence": 0.0, "category": None}

    raw = ja4_parser.observation_to_features(obs)
    encoded = {}
    for col in bundle["feature_columns"]:
        if col in ja4_parser.CATEGORICAL_FEATURES:
            mapping = bundle["categorical_maps"].get(col, {})
            encoded[col] = mapping.get(str(raw.get(col, ja4_parser.MISSING)), -1)
        else:
            encoded[col] = int(raw.get(col, 0))

    frame = pd.DataFrame([encoded], columns=bundle["feature_columns"])
    probs = bundle["classifier"].predict_proba(frame)[0]
    class_ids = bundle["classifier"].classes_
    top_pos = probs.argsort()[::-1][:5]
    top_k = [(bundle["idx_to_label"][int(class_ids[i])], round(float(probs[i]) * 100, 2)) for i in top_pos]

    best_label, best_prob_pct = top_k[0]
    best_conf = best_prob_pct / 100.0
    best_category = bundle["label_to_category"].get(best_label) or infer_category(best_label)

    return {
        "status": "predicted",
        "prediction": best_label,
        "top_k": [t[0] for t in top_k],
        "confidence": round(best_conf, 4),
        "category": best_category,
        "top_k_with_prob": top_k,
    }


# ── FoxIO support ─────────────────────────────────────────────────────────────

def _foxio_support(obs: Observation) -> dict:
    """Query FoxIO for supporting evidence."""
    index = _get_foxio_index()
    key = (obs.ja4 or "").strip().lower()
    hits = index.get(key, [])
    if not hits:
        return {"status": "unknown", "candidates": [], "category": None}

    apps: dict[str, int] = defaultdict(int)
    for h in hits:
        app = h.get("application") or h.get("Application") or ""
        if app:
            apps[app] += 1

    ranked = sorted(apps.keys(), key=lambda a: apps[a], reverse=True)
    return {
        "status": "supported" if len(ranked) == 1 else "ambiguous",
        "candidates": ranked,
        "category": infer_category(ranked[0]) if ranked else None,
    }


# ── decision engine ───────────────────────────────────────────────────────────

def _rf_clear_enough(rf: dict) -> bool:
    """Check if RF confidence clears the acceptance threshold with a clear margin."""
    if rf["status"] != "predicted" or rf["confidence"] < RF_ACCEPT_THRESHOLD:
        return False
    top_k_probs = rf.get("top_k_with_prob", [])
    if len(top_k_probs) >= 2:
        gap = top_k_probs[0][1] - top_k_probs[1][1]
        return gap >= RF_MARGIN_PERCENT or rf["confidence"] >= RF_HIGH_THRESHOLD
    return rf["confidence"] >= RF_ACCEPT_THRESHOLD


def decide(local: dict, rf: dict, foxio: dict) -> FinalDecision:
    """Combine all signals into one final decision — refined logic for thesis."""

    # 1 ── Unique exact match → highest trust
    if local["status"] == "unique":
        app = local["top_app"]
        cat = local["top_category"] or infer_category(app)
        foxio_supports = app and any(c.lower() == app.lower() for c in foxio.get("candidates", []))
        confidence = "high" if local["match_mode"] in _STRICT_MODES else "medium"
        reasoning = f"Unique Egenlagd exact match via {local['match_mode']}."
        if foxio_supports:
            reasoning += " FoxIO also confirmed this application."
        return FinalDecision(
            application_prediction=app,
            category_prediction=cat,
            application_confidence=confidence,
            category_confidence="high",
            decision_source="egenlagd_exact",
            reasoning=reasoning,
        )

    # 2 ── Ambiguous exact match → try RF to resolve
    if local["status"] == "ambiguous" and local["candidates"] and _rf_clear_enough(rf):
        candidates_lower = {c.lower(): c for c in local["candidates"]}
        rf_pred = (rf["prediction"] or "").lower()
        if rf_pred in candidates_lower:
            resolved = candidates_lower[rf_pred]
            cat = rf["category"] or infer_category(resolved)
            foxio_supports = any(c.lower() == resolved.lower() for c in foxio.get("candidates", []))
            confidence = "high" if foxio_supports and rf["confidence"] >= RF_HIGH_THRESHOLD else "medium"
            return FinalDecision(
                application_prediction=resolved,
                category_prediction=cat,
                application_confidence=confidence,
                category_confidence="high",
                decision_source="ambiguous_resolved_by_rf",
                reasoning=(
                    f"Egenlagd returned {len(local['candidates'])} candidates via {local['match_mode']}. "
                    f"RF resolved to {resolved} with {rf['confidence']*100:.1f}% confidence."
                ),
            )

    # 3 ── No local match → RF as primary signal
    if local["status"] in ("unknown", "ambiguous") and _rf_clear_enough(rf):
        app = rf["prediction"]
        cat = rf["category"] or infer_category(app)
        foxio_ok = foxio.get("status") in ("supported", "ambiguous")
        confidence = "high" if foxio_ok and rf["confidence"] >= RF_HIGH_THRESHOLD else "medium"
        reasoning = f"No strong local match — RF predicts {app} with {rf['confidence']*100:.1f}% confidence."
        if foxio_ok:
            reasoning += " FoxIO provided supporting evidence."
        return FinalDecision(
            application_prediction=app,
            category_prediction=cat,
            application_confidence=confidence,
            category_confidence="medium",
            decision_source="random_forest",
            reasoning=reasoning,
        )

    # 4 ── Category fallback — try to at least agree on a category
    if local["candidates"]:
        categories = {infer_category(c) for c in local["candidates"]} - {"unknown"}
        if len(categories) == 1:
            cat = next(iter(categories))
            return FinalDecision(
                application_prediction=None,
                category_prediction=cat,
                application_confidence="low",
                category_confidence="high",
                decision_source="category_fallback_local",
                reasoning=f"Application ambiguous across {len(local['candidates'])} candidates but all share category: {cat}.",
            )

    foxio_cat = foxio.get("category")
    if foxio_cat and foxio_cat != "unknown":
        return FinalDecision(
            application_prediction=None,
            category_prediction=foxio_cat,
            application_confidence="low",
            category_confidence="low",
            decision_source="category_fallback_foxio",
            reasoning="No application-level match found; FoxIO indicated a category.",
        )

    return FinalDecision(
        application_prediction=None,
        category_prediction=None,
        application_confidence="none",
        category_confidence="none",
        decision_source="unknown",
        reasoning="No classifier produced enough evidence to classify this observation.",
    )


def classify(ja4: str | None, ja4s: str | None = None,
             ja4t: str | None = None, ja4ts: str | None = None,
             ja4_string: str | None = None, ja4s_string: str | None = None,
             observation_id: str = "query") -> ClassificationResult:
    """Classify one fingerprint through the full pipeline."""
    obs = Observation(observation_id=observation_id,
                      ja4=ja4, ja4s=ja4s, ja4t=ja4t, ja4ts=ja4ts,
                      ja4_string=ja4_string, ja4s_string=ja4s_string)
    local = _egenlagd_lookup(obs)
    rf = _rf_predict(obs)
    foxio = _foxio_support(obs)

    decision = decide(local, rf, foxio)

    return ClassificationResult(
        observation_id=observation_id,
        ja4=ja4,
        true_application=None,  # unknown at query time
        true_category=None,
        predicted_application=decision.application_prediction,
        predicted_category=decision.category_prediction,
        is_correct=None,
        confidence=decision.application_confidence,
        decision_source=decision.decision_source,
        reasoning=decision.reasoning,
        model_details={
            "local": local,
            "rf": rf,
            "foxio": foxio
        }
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JA4+ combined pipeline classifier")
    parser.add_argument("--ja4",  default=None)
    parser.add_argument("--ja4s", default=None)
    parser.add_argument("--ja4_string", default=None)
    parser.add_argument("--ja4s_string", default=None)
    parser.add_argument("--ja4t", default=None)
    parser.add_argument("--ja4ts", default=None)
    args = parser.parse_args()

    if args.ja4:
        result = classify(args.ja4, args.ja4s, args.ja4t, args.ja4ts, args.ja4_string, args.ja4s_string)
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # Demo: classify the first 3 test records
        _, test = database_lookup.train_test_split(seed=42)
        print("=== Pipeline demo (first 3 test records) ===\n")
        for record in test[:3]:
            result = classify(record.ja4, record.ja4s, record.ja4t, record.ja4ts, record.ja4_string, record.ja4s_string)
            print(f"True app  : {record.application}")
            print(f"Predicted : {result.predicted_application} ({result.confidence})")
            print(f"Category  : {result.predicted_category}")
            print(f"Source    : {result.decision_source}")
            print(f"Reasoning : {result.reasoning}\n")
