"""Combined JA4+ classifier pipeline.

This is the core of the project. It runs three independent classifiers and
then combines their results into one final answer.

─────────────────────────────────────────────────────────────────────────────
HOW THE THREE CLASSIFIERS WORK
─────────────────────────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1. EGENLAGD EXACT MATCH                                                │
  │     Looks up the fingerprint in our own custom database.                │
  │     Tries progressively looser combinations (ja4+ja4s+ja4t+ja4ts →     │
  │     ja4 alone). Returns "unique", "ambiguous", or "unknown".            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  2. RANDOM FOREST (RF)                                                  │
  │     A pre-trained ML model. Parses the JA4 strings into features, runs  │
  │     predict_proba(), and returns a ranked list of applications with      │
  │     confidence percentages.                                             │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  3. FOXIO                                                               │
  │     Looks up ja4 in the FoxIO open-source threat-intel database.        │
  │     Used as supporting evidence — does not decide on its own.           │
  └─────────────────────────────────────────────────────────────────────────┘

PRIORITY ORDER (decide() function):
  1. Unique Egenlagd exact match → use it directly (highest trust)
  2. Ambiguous Egenlagd match + RF agrees → RF picks between candidates
  3. No local match + RF is confident enough → RF is the primary answer
  4. All else fails → try to agree at least on a category (local or FoxIO)
  5. Nothing works → return "unknown"

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

from . import database_lookup
from . import ja4_parser
from .models import (
    ClassificationResult,
    DatabaseRecord,
    FinalDecision,
    Observation,
    infer_category,
)

# ── file paths ────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parent          # pipeline_model/
MODEL_PATH     = _ROOT.parent / "data" / "models" / "ja4_rf.pkl"
FOXIO_DB_PATH  = _ROOT.parent / "data" / "models" / "ja4+_db.json"

# ── confidence thresholds ─────────────────────────────────────────────────────
RF_ACCEPT_THRESHOLD = 0.60   # RF must be at least this confident to be trusted
RF_HIGH_THRESHOLD   = 0.75   # above this → "high" confidence band
RF_MARGIN_PERCENT   = 15.0   # the top prediction must beat #2 by this many % points
                             # (prevents accepting a coin-flip between two apps)

# ── lookup plans ──────────────────────────────────────────────────────────────
# Each plan is (name, fields_to_use).  We try them in order — most specific
# first (all four fingerprints) down to least specific (ja4 alone).
# The intuition: the more fingerprints match, the more confident we can be.
_LOOKUP_PLANS: list[tuple[str, tuple[str, ...]]] = [
    ("ja4_ja4s_ja4t_ja4ts", ("ja4", "ja4s", "ja4t", "ja4ts")),
    ("ja4_ja4s_ja4ts",      ("ja4", "ja4s", "ja4ts")),
    ("ja4_ja4s_ja4t",       ("ja4", "ja4s", "ja4t")),
    ("ja4_ja4s",            ("ja4", "ja4s")),
    ("ja4_ja4ts",           ("ja4", "ja4ts")),
    ("ja4_ja4t",            ("ja4", "ja4t")),
    ("ja4_only",            ("ja4",)),
]
# Plans that use both client and server fingerprints — higher-confidence group.
_STRICT_MODES = {"ja4_ja4s_ja4t_ja4ts", "ja4_ja4s_ja4ts", "ja4_ja4s_ja4t", "ja4_ja4s"}


# ── lazy-loaded singletons ────────────────────────────────────────────────────
# Each index / model is loaded from disk exactly once and then cached here.
_EGENLAGD_INDEX: dict | None = None   # the local DB structured for fast lookup
_FOXIO_INDEX:    dict | None = None   # the FoxIO DB structured for fast lookup
_RF_BUNDLE:      dict | None = None   # the pickled Random Forest + metadata


def _field(record: DatabaseRecord, name: str) -> str:
    """Read a field from a DatabaseRecord, normalised to lowercase with no spaces."""
    return (getattr(record, name) or "").strip().lower()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — EGENLAGD EXACT MATCH
# ═════════════════════════════════════════════════════════════════════════════

def _get_egenlagd_index() -> dict:
    """Build (once) the Egenlagd lookup index from the 80% training split.

    The index is a nested dict:
      index[mode][key] = [list of matching DatabaseRecord objects]

    where key = "ja4_value|ja4s_value|..." — the fingerprint values joined by '|'.
    We use the training split only so evaluation on the test split is fair.
    """
    global _EGENLAGD_INDEX
    if _EGENLAGD_INDEX is not None:
        return _EGENLAGD_INDEX

    train_records, _ = database_lookup.train_test_split(seed=42, test_ratio=0.2)

    # Pre-allocate one sub-dict per lookup plan.
    index: dict[str, dict[str, list[DatabaseRecord]]] = {
        mode: defaultdict(list) for mode, _ in _LOOKUP_PLANS
    }
    # Index every training record under every applicable plan.
    for record in train_records:
        for mode, fields in _LOOKUP_PLANS:
            values = [_field(record, f) for f in fields]
            if all(values):   # only index if all required fields are non-empty
                index[mode]["|".join(values)].append(record)

    _EGENLAGD_INDEX = index
    print("[pipeline] Egenlagd index built.")
    return _EGENLAGD_INDEX


def _egenlagd_lookup(obs: Observation) -> dict:
    """Run the Egenlagd exact-match lookup for one observation.

    Tries every lookup plan in priority order, stopping at the first hit.

    Returns a dict with:
      status       — "unique" (one app matched), "ambiguous" (multiple),
                     or "unknown" (no match at all)
      match_mode   — which lookup plan produced the hit (e.g. "ja4_ja4s")
      candidates   — list of application names, most frequent first
      top_app      — the best candidate
      top_category — category of the best candidate
      confidence   — float 0-1, base confidence assigned by the plan type
    """
    index     = _get_egenlagd_index()
    obs_fields = {"ja4": obs.ja4, "ja4s": obs.ja4s, "ja4t": obs.ja4t, "ja4ts": obs.ja4ts}

    for mode, fields in _LOOKUP_PLANS:
        # Build the lookup key for this plan.
        values = [(obs_fields.get(f) or "").strip().lower() for f in fields]
        if not all(values):
            continue   # observation is missing one of the required fingerprints
        key  = "|".join(values)
        hits = index.get(mode, {}).get(key, [])
        if not hits:
            continue   # no records match this fingerprint combination

        # Tally occurrences per application, weighted by the record's count.
        app_counts: dict[str, int] = defaultdict(int)
        cat_map:    dict[str, str] = {}
        for hit in hits:
            if hit.application:
                app_counts[hit.application] += max(hit.count, 1)
                cat_map[hit.application] = hit.category or "unknown"

        if not app_counts:
            continue   # no labeled hits in this cluster

        # Sort apps by total observed count → most common app listed first.
        ranked = sorted(app_counts.keys(), key=lambda a: app_counts[a], reverse=True)
        status = "unique" if len(ranked) == 1 else "ambiguous"

        # Assign base confidence depending on how many fingerprints matched.
        confidence = 0.9 if mode in _STRICT_MODES else 0.7
        if status == "ambiguous":
            confidence *= 0.6   # ambiguity reduces our certainty

        return {
            "status": status, "match_mode": mode,
            "candidates": ranked, "top_app": ranked[0],
            "top_category": cat_map.get(ranked[0], "unknown"),
            "confidence": round(confidence, 3),
        }

    # Tried all plans and found nothing.
    return {
        "status": "unknown", "match_mode": None,
        "candidates": [], "top_app": None,
        "top_category": None, "confidence": 0.0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — RANDOM FOREST INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def _get_rf_bundle():
    """Load (once) the pickled RF model bundle from disk.

    The bundle is a dict saved by eval_random_forest.py with keys:
      classifier       — the trained sklearn RandomForestClassifier
      feature_columns  — ordered list of feature names the model expects
      categorical_maps — dict of {feature → {value_str → int}} label encodings
      idx_to_label     — dict of {int index → application name string}
      label_to_category — dict of {application name → category}
    """
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


def _rf_predict(obs: Observation) -> dict:
    """Run Random Forest inference for one observation.

    Steps:
      1. Parse the JA4 strings into a flat feature dict (via ja4_parser).
      2. Label-encode categorical features using the bundle's mappings.
      3. Build a one-row pandas DataFrame in the exact column order the model
         was trained on.
      4. Call predict_proba() → get a probability for every known application.
      5. Return the top-5 predictions with their probabilities.

    Returns a dict with:
      status         — "predicted" or "unavailable" (model not loaded)
      prediction     — the top-1 application name
      top_k          — list of top-5 application names
      top_k_with_prob — list of (app_name, probability_percent) tuples
      confidence     — float 0-1 (top-1 probability)
      category       — category of the top-1 prediction
    """
    bundle = _get_rf_bundle()
    if bundle is None:
        return {"status": "unavailable", "prediction": None, "top_k": [], "confidence": 0.0, "category": None}

    try:
        import pandas as pd
    except ImportError:
        return {"status": "unavailable", "prediction": None, "top_k": [], "confidence": 0.0, "category": None}

    # Step 1: parse JA4 strings into a raw feature dict.
    raw = ja4_parser.observation_to_features(obs)

    # Step 2: encode each feature the way the model expects it.
    encoded = {}
    for col in bundle["feature_columns"]:
        if col in ja4_parser.CATEGORICAL_FEATURES:
            # Categorical → look up the integer code from training-time mapping.
            # -1 means "unseen value" (the model was not trained on this string).
            mapping = bundle["categorical_maps"].get(col, {})
            encoded[col] = mapping.get(str(raw.get(col, ja4_parser.MISSING)), -1)
        else:
            # Numeric → cast to int (binary flags and counts).
            encoded[col] = int(raw.get(col, 0))

    # Step 3: build the DataFrame and run inference.
    frame = pd.DataFrame([encoded], columns=bundle["feature_columns"])
    probs     = bundle["classifier"].predict_proba(frame)[0]
    class_ids = bundle["classifier"].classes_

    # Step 4: sort by probability descending, keep top 5.
    top_pos = probs.argsort()[::-1][:5]
    top_k   = [
        (bundle["idx_to_label"][int(class_ids[i])], round(float(probs[i]) * 100, 2))
        for i in top_pos
    ]

    best_label, best_prob_pct = top_k[0]
    best_conf     = best_prob_pct / 100.0
    best_category = bundle["label_to_category"].get(best_label) or infer_category(best_label)

    return {
        "status":          "predicted",
        "prediction":      best_label,
        "top_k":           [t[0] for t in top_k],
        "confidence":      round(best_conf, 4),
        "category":        best_category,
        "top_k_with_prob": top_k,   # raw (app, %) tuples — useful for eval
    }


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — FOXIO SUPPORT
# ═════════════════════════════════════════════════════════════════════════════

def _get_foxio_index() -> dict:
    """Load (once) the FoxIO database and index it by ja4 hash.

    FoxIO is an open-source threat-intel database of known JA4 fingerprints.
    We use it as corroborating evidence, not as the primary classifier.
    """
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


def _foxio_support(obs: Observation) -> dict:
    """Query FoxIO for supporting evidence for the given observation.

    We look up only by the ja4 hash (FoxIO does not have ja4s/t/ts).

    Returns a dict with:
      status     — "supported" (one app in FoxIO), "ambiguous" (several),
                   or "unknown" (not in FoxIO)
      candidates — list of application names found in FoxIO for this ja4
      category   — inferred category of the top FoxIO candidate
    """
    index = _get_foxio_index()
    key   = (obs.ja4 or "").strip().lower()
    hits  = index.get(key, [])
    if not hits:
        return {"status": "unknown", "candidates": [], "category": None}

    # Count how many FoxIO entries mention each application name.
    apps: dict[str, int] = defaultdict(int)
    for h in hits:
        app = h.get("application") or h.get("Application") or ""
        if app:
            apps[app] += 1

    ranked = sorted(apps.keys(), key=lambda a: apps[a], reverse=True)
    return {
        "status":     "supported" if len(ranked) == 1 else "ambiguous",
        "candidates": ranked,
        "category":   infer_category(ranked[0]) if ranked else None,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — DECISION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _rf_clear_enough(rf: dict) -> bool:
    """Return True if the RF result is trustworthy enough to act on.

    Two conditions must hold:
      a) RF confidence >= RF_ACCEPT_THRESHOLD (0.60)
      b) Either the gap between #1 and #2 is >= RF_MARGIN_PERCENT (15 pp)
         OR the confidence is already in the high band (>= 0.75).
    This prevents acting on "51% vs 49%" coin-flip predictions.
    """
    if rf["status"] != "predicted" or rf["confidence"] < RF_ACCEPT_THRESHOLD:
        return False
    top_k_probs = rf.get("top_k_with_prob", [])
    if len(top_k_probs) >= 2:
        gap = top_k_probs[0][1] - top_k_probs[1][1]   # gap in percentage points
        return gap >= RF_MARGIN_PERCENT or rf["confidence"] >= RF_HIGH_THRESHOLD
    return rf["confidence"] >= RF_ACCEPT_THRESHOLD


def decide(local: dict, rf: dict, foxio: dict) -> FinalDecision:
    """Combine the three classifier signals into exactly one final answer.

    Parameters
    ----------
    local   — result dict from _egenlagd_lookup()
    rf      — result dict from _rf_predict()
    foxio   — result dict from _foxio_support()

    Decision waterfall (first matching branch wins):
    ─────────────────────────────────────────────────
    Branch 1 — Unique Egenlagd exact match
      Only one application in the DB matches this fingerprint.
      This is the most reliable signal; return it immediately.
      FoxIO agreement bumps reasoning but doesn't change confidence level.

    Branch 2 — Ambiguous Egenlagd match resolved by RF
      Multiple apps matched, but the RF is confident AND its top pick is
      one of the Egenlagd candidates.  RF acts as a tie-breaker.
      FoxIO agreement + high RF confidence → "high" confidence.

    Branch 3 — No local match, RF is the primary signal
      The fingerprint wasn't seen during training, but RF is confident.
      FoxIO as supporting evidence can lift to "high" confidence.

    Branch 4 — Category fallback (local then FoxIO)
      We can't name the application, but maybe all local candidates share
      the same category (e.g. all are "browser"), so we report that.
      If that's empty too, ask FoxIO for its category.

    Branch 5 — Give up
      Nothing produced enough evidence. Return "unknown".
    """

    # ── Branch 1: Unique exact match ─────────────────────────────────────────
    if local["status"] == "unique":
        app = local["top_app"]
        cat = local["top_category"] or infer_category(app)
        # Check whether FoxIO independently lists the same application.
        foxio_supports = app and any(
            c.lower() == app.lower() for c in foxio.get("candidates", [])
        )
        confidence = "high" if local["match_mode"] in _STRICT_MODES else "medium"
        reasoning  = f"Unique Egenlagd exact match via {local['match_mode']}."
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

    # ── Branch 2: Ambiguous local match, RF resolves it ──────────────────────
    if local["status"] == "ambiguous" and local["candidates"] and _rf_clear_enough(rf):
        # Build a lower-case → original-case map of the Egenlagd candidates.
        candidates_lower = {c.lower(): c for c in local["candidates"]}
        rf_pred = (rf["prediction"] or "").lower()
        if rf_pred in candidates_lower:   # RF agrees with one of the Egenlagd hits
            resolved = candidates_lower[rf_pred]
            cat = rf["category"] or infer_category(resolved)
            foxio_supports = any(
                c.lower() == resolved.lower() for c in foxio.get("candidates", [])
            )
            confidence = (
                "high"
                if foxio_supports and rf["confidence"] >= RF_HIGH_THRESHOLD
                else "medium"
            )
            return FinalDecision(
                application_prediction=resolved,
                category_prediction=cat,
                application_confidence=confidence,
                category_confidence="high",
                decision_source="ambiguous_resolved_by_rf",
                reasoning=(
                    f"Egenlagd returned {len(local['candidates'])} candidates via "
                    f"{local['match_mode']}. RF resolved to {resolved} with "
                    f"{rf['confidence']*100:.1f}% confidence."
                ),
            )

    # ── Branch 3: No local match — use RF as primary ──────────────────────────
    if local["status"] in ("unknown", "ambiguous") and _rf_clear_enough(rf):
        app     = rf["prediction"]
        cat     = rf["category"] or infer_category(app)
        foxio_ok = foxio.get("status") in ("supported", "ambiguous")
        confidence = (
            "high"
            if foxio_ok and rf["confidence"] >= RF_HIGH_THRESHOLD
            else "medium"
        )
        reasoning = (
            f"No strong local match — RF predicts {app} with "
            f"{rf['confidence']*100:.1f}% confidence."
        )
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

    # ── Branch 4a: All local candidates share a category ─────────────────────
    if local["candidates"]:
        # Infer a category for every candidate and check if they all agree.
        categories = {infer_category(c) for c in local["candidates"]} - {"unknown"}
        if len(categories) == 1:
            cat = next(iter(categories))
            return FinalDecision(
                application_prediction=None,
                category_prediction=cat,
                application_confidence="low",
                category_confidence="high",
                decision_source="category_fallback_local",
                reasoning=(
                    f"Application ambiguous across {len(local['candidates'])} "
                    f"candidates but all share category: {cat}."
                ),
            )

    # ── Branch 4b: FoxIO can at least tell us the category ───────────────────
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

    # ── Branch 5: Give up ─────────────────────────────────────────────────────
    return FinalDecision(
        application_prediction=None,
        category_prediction=None,
        application_confidence="none",
        category_confidence="none",
        decision_source="unknown",
        reasoning="No classifier produced enough evidence to classify this observation.",
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def classify(
    ja4:        str | None,
    ja4s:       str | None = None,
    ja4t:       str | None = None,
    ja4ts:      str | None = None,
    ja4_string: str | None = None,
    ja4s_string: str | None = None,
    observation_id: str = "query",
) -> ClassificationResult:
    """Classify one set of JA4 fingerprints through the full pipeline.

    This is the function all eval scripts and the GUI call.

    Parameters
    ----------
    ja4, ja4s, ja4t, ja4ts   — short fingerprint hashes
    ja4_string, ja4s_string  — long human-readable forms (richer RF features)
    observation_id           — label to identify this row in results

    Returns
    -------
    ClassificationResult
        predicted_application, predicted_category, confidence, decision_source,
        reasoning, and model_details (raw sub-results from all three stages).
    """
    # Pack all fingerprints into the typed input container.
    obs = Observation(
        observation_id=observation_id,
        ja4=ja4, ja4s=ja4s, ja4t=ja4t, ja4ts=ja4ts,
        ja4_string=ja4_string, ja4s_string=ja4s_string,
    )

    # Run all three classifiers independently.
    local  = _egenlagd_lookup(obs)
    rf     = _rf_predict(obs)
    foxio  = _foxio_support(obs)

    # Combine the three outputs into one final answer.
    decision = decide(local, rf, foxio)

    return ClassificationResult(
        observation_id=observation_id,
        ja4=ja4,
        true_application=None,   # unknown at query time — only set during eval
        true_category=None,
        predicted_application=decision.application_prediction,
        predicted_category=decision.category_prediction,
        is_correct=None,
        confidence=decision.application_confidence,
        decision_source=decision.decision_source,
        reasoning=decision.reasoning,
        model_details={
            "local": local,
            "rf":    rf,
            "foxio": foxio,
        },
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
# python pipeline.py --ja4 "t13d1516h2_aaa_bbb"   → classify one fingerprint
# python pipeline.py                                → demo on first 3 test rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JA4+ combined pipeline classifier")
    parser.add_argument("--ja4",        default=None)
    parser.add_argument("--ja4s",       default=None)
    parser.add_argument("--ja4_string", default=None)
    parser.add_argument("--ja4s_string",default=None)
    parser.add_argument("--ja4t",       default=None)
    parser.add_argument("--ja4ts",      default=None)
    args = parser.parse_args()

    if args.ja4:
        result = classify(
            args.ja4, args.ja4s, args.ja4t, args.ja4ts,
            args.ja4_string, args.ja4s_string,
        )
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # No arguments supplied → run a quick demo.
        _, test = database_lookup.train_test_split(seed=42)
        print("=== Pipeline demo (first 3 test records) ===\n")
        for record in test[:3]:
            result = classify(
                record.ja4, record.ja4s, record.ja4t, record.ja4ts,
                record.ja4_string, record.ja4s_string,
            )
            print(f"True app  : {record.application}")
            print(f"Predicted : {result.predicted_application} ({result.confidence})")
            print(f"Category  : {result.predicted_category}")
            print(f"Source    : {result.decision_source}")
            print(f"Reasoning : {result.reasoning}\n")
