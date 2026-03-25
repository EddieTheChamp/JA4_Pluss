"""Shared dataclasses for the JA4+ pipeline."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Observation:
    """One fingerprint observation to classify."""
    observation_id: str
    ja4: str | None = None
    ja4s: str | None = None
    ja4_string: str | None = None
    ja4s_string: str | None = None
    ja4t: str | None = None
    ja4ts: str | None = None
    true_application: str | None = None
    true_category: str | None = None
    source: str | None = None
    raw_record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatabaseRecord:
    """One normalized row from the local fingerprint database."""
    ja4: str | None = None
    ja4s: str | None = None
    ja4_string: str | None = None
    ja4s_string: str | None = None
    ja4t: str | None = None
    ja4ts: str | None = None
    application: str | None = None
    category: str | None = None
    count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateMatch:
    """One candidate application from an exact-match lookup."""
    application: str | None
    category: str | None
    occurrences_in_database: int
    probability_percent: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ── category inference ────────────────────────────────────────────────────────

_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("malware", ("cobalt", "sliver", "meterpreter", "beacon", "empire", "metasploit")),
    ("security", ("defender", "antivirus", "msmpeng", "smartscreen", "sentinel",
                  "crowdstrike", "falcon", "carbonblack", "sophos", "symantec",
                  "mcafee", "kaspersky", "eset", "avast", "avg", "bitdefender")),
    ("browser", ("chrome.exe", "chrome", "msedge.exe", "msedge", "edgewebview",
                 "webview2", "firefox.exe", "firefox", "opera.exe", "opera",
                 "brave.exe", "brave", "microsoftedgeupdate", "code.exe")),
    ("microsoft_office", ("outlook.exe", "outlook", "winword.exe", "winword",
                          "excel.exe", "excel", "powerpnt.exe", "powerpnt",
                          "onenote.exe", "onenote", "officec2rclient",
                          "filecoauth", "m365copilot")),
    ("messaging", ("teams.exe", "teams", "ms-teams", "discord.exe", "discord",
                   "slack.exe", "slack", "telegram.exe", "telegram",
                   "skype.exe", "skype", "signal.exe", "signal")),
    ("system", ("svchost.exe", "svchost", "lsass.exe", "lsass",
                "services.exe", "services", "wininit.exe", "wininit",
                "explorer.exe", "explorer", "backgroundtaskhost",
                "wermgr.exe", "wermgr", "pwsh.exe", "powershell",
                "tailscaled", "onedrive", "jotta", "unknown process")),
]


def infer_category(application: str | None, explicit_category: str | None = None) -> str:
    """Return a normalized category string from an application name."""
    if explicit_category:
        return explicit_category.strip().lower().replace("-", "_").replace(" ", "_")
    if not application:
        return "unknown"
    app = application.strip().lower()
    for category, patterns in _CATEGORY_RULES:
        if any(p in app for p in patterns):
            return category
    return "unknown"


@dataclass(slots=True)
class FinalDecision:
    """Final explanation produced by the decision engine."""
    application_prediction: str | None
    category_prediction: str | None
    application_confidence: str  # high, medium, low, none
    category_confidence: str     # high, medium, low
    decision_source: str
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClassificationResult:
    """Full per-observation output for reporting."""
    observation_id: str
    ja4: str | None
    true_application: str | None
    true_category: str | None
    predicted_application: str | None
    predicted_category: str | None
    is_correct: bool | None
    confidence: str
    decision_source: str
    reasoning: str
    # Raw sub-results (simplified)
    model_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
