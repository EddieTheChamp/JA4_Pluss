"""Parse the JA4 fingerprint string into modular features.

JA4 format:  <prefix10chars>_<ciphers_hash>_<extensions_hash>
Prefix:  [0]   protocol   (t = TLS, q = QUIC, d = DTLS, ...)
         [1:3] TLS version (13 = 1.3, 12 = 1.2, …)
         [3]   SNI indicator (d = domain, i = IP)
         [4:6] num_ciphers  (2-digit zero-padded)
         [6:8] num_extensions (2-digit zero-padded)
         [8:10] ALPN first two chars (h2, h1, 00, …)
"""

from __future__ import annotations
from typing import Any

from models import DatabaseRecord, Observation

MISSING = "__missing__"

CATEGORICAL_FEATURES = [
    "protocol", "tls_version", "sni_indicator", "alpn",
    "ja4t", "ja4ts",
]
NUMERIC_FEATURES = ["num_ciphers", "num_extensions", "has_ja4s", "has_ja4t", "has_ja4ts"]
# FEATURE_COLUMNS will be determined dynamically
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def parse_ja4(ja4: str | None, ja4_string: str | None = None) -> dict[str, Any]:
    """Parse one JA4 string into a flat feature dictionary."""
    defaults: dict[str, Any] = {
        "protocol": MISSING, "tls_version": MISSING,
        "sni_indicator": MISSING, "num_ciphers": 0,
        "num_extensions": 0, "alpn": MISSING,
    }
    text = (ja4_string or ja4 or "").strip().lower()
    if not text or "_" not in text:
        return defaults
    parts = text.split("_")

    prefix = parts[0].ljust(10, "0")
    features = {
        "protocol":        prefix[0]    or MISSING,
        "tls_version":     prefix[1:3]  or MISSING,
        "sni_indicator":   prefix[3]    or MISSING,
        "num_ciphers":     int(prefix[4:6]) if prefix[4:6].isdigit() else 0,
        "num_extensions":  int(prefix[6:8]) if prefix[6:8].isdigit() else 0,
        "alpn":            prefix[8:10] or MISSING,
    }

    if ja4_string and len(parts) >= 3:
        ciphers = parts[1].split(",")
        for c in ciphers:
            if c: features[f"cipher_{c}"] = 1
            
        extensions = parts[2].split(",")
        for e in extensions:
            if e: features[f"ext_{e}"] = 1
            
        if len(parts) >= 4:
            sigs = parts[3].split(",")
            for s in sigs:
                if s: features[f"sig_{s}"] = 1

    return features


def parse_ja4s(ja4s_string: str | None) -> dict[str, Any]:
    features = {}
    if not ja4s_string:
        return features
    text = ja4s_string.strip().lower()
    parts = text.split("_")
    if len(parts) >= 2:
        ciphers = parts[1].split(",")
        for c in ciphers:
            if c: features[f"ja4s_cipher_{c}"] = 1
    if len(parts) >= 3:
        extensions = parts[2].split(",")
        for e in extensions:
            if e: features[f"ja4s_ext_{e}"] = 1
    return features


def observation_to_features(obs: Observation) -> dict[str, Any]:
    """Convert an Observation into the full flat feature dict used by RF."""
    features = parse_ja4(obs.ja4, obs.ja4_string)
    if obs.ja4s_string:
        features.update(parse_ja4s(obs.ja4s_string))
        
    features.update({
        "ja4t":    (obs.ja4t or "").strip().lower() or MISSING,
        "ja4ts":   (obs.ja4ts or "").strip().lower() or MISSING,
        "has_ja4s": 1 if obs.ja4s else 0,
        "has_ja4t": 1 if obs.ja4t else 0,
        "has_ja4ts": 1 if obs.ja4ts else 0,
    })
    return features


def record_to_training_sample(record: DatabaseRecord) -> dict[str, Any] | None:
    """Convert a DatabaseRecord into an RF training sample dict, or None if unlabeled."""
    if not record.application:
        return None
    obs = Observation(
        observation_id="train",
        ja4=record.ja4, ja4s=record.ja4s,
        ja4_string=record.ja4_string, ja4s_string=record.ja4s_string,
        ja4t=record.ja4t, ja4ts=record.ja4ts,
        source="training",
    )
    sample = observation_to_features(obs)
    sample["label"] = record.application
    sample["category"] = record.category
    sample["weight"] = max(record.count, 1)
    return sample


def records_to_training_samples(records: list[DatabaseRecord]) -> list[dict[str, Any]]:
    """Convert a list of records into RF training samples (skips unlabeled)."""
    return [s for r in records if (s := record_to_training_sample(r)) is not None]
