"""Category helpers shared across the active JA4+ pipeline."""

from __future__ import annotations

from current.utils.validation import normalize_lookup_text

CATEGORY_PRIORITY_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "malware",
        (
            "cobalt",
            "sliver",
            "meterpreter",
            "beacon",
            "empire",
            "metasploit",
        ),
    ),
    (
        "security",
        (
            "defender",
            "antivirus",
            "msmpeng",
            "smartscreen",
            "sentinel",
            "crowdstrike",
            "falcon",
            "carbonblack",
            "sophos",
            "symantec",
            "mcafee",
            "kaspersky",
            "eset",
            "avast",
            "avg",
            "bitdefender",
        ),
    ),
    (
        "browser",
        (
            "chrome.exe",
            "chrome",
            "msedge.exe",
            "msedge",
            "edgewebview",
            "webview2",
            "firefox.exe",
            "firefox",
            "opera.exe",
            "opera",
            "brave.exe",
            "brave",
            "microsoftedgeupdate",
            "code.exe",
        ),
    ),
    (
        "microsoft_office",
        (
            "outlook.exe",
            "outlook",
            "winword.exe",
            "winword",
            "excel.exe",
            "excel",
            "powerpnt.exe",
            "powerpnt",
            "onenote.exe",
            "onenote",
            "officec2rclient",
            "filecoauth",
            "m365copilot",
        ),
    ),
    (
        "messaging",
        (
            "teams.exe",
            "teams",
            "ms-teams",
            "discord.exe",
            "discord",
            "slack.exe",
            "slack",
            "telegram.exe",
            "telegram",
            "skype.exe",
            "skype",
            "signal.exe",
            "signal",
        ),
    ),
    (
        "system",
        (
            "svchost.exe",
            "svchost",
            "lsass.exe",
            "lsass",
            "services.exe",
            "services",
            "wininit.exe",
            "wininit",
            "explorer.exe",
            "explorer",
            "backgroundtaskhost",
            "wermgr.exe",
            "wermgr",
            "pwsh.exe",
            "powershell",
            "tailscaled",
            "onedrive",
            "jotta",
            "unknown process",
        ),
    ),
]


def normalize_category_label(value: str | None) -> str | None:
    """Normalize an explicit category value into the active label format."""
    normalized = normalize_lookup_text(value)
    if not normalized:
        return None
    return normalized.replace("-", "_").replace(" ", "_")


def infer_category_from_application(application_name: str | None, explicit_category: str | None = None) -> str:
    """Infer a normalized category label from an application name."""
    normalized_category = normalize_category_label(explicit_category)
    if normalized_category:
        return normalized_category

    application = normalize_lookup_text(application_name)
    if not application:
        return "unknown"

    for category, patterns in CATEGORY_PRIORITY_RULES:
        if any(pattern in application for pattern in patterns):
            return category
    return "unknown"
