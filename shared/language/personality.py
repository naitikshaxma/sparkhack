from __future__ import annotations

from typing import Optional


VALID_TONES = {"formal", "friendly", "assistant-like"}


def normalize_tone(value: Optional[str], default: str = "assistant-like") -> str:
    tone = (value or "").strip().lower()
    if tone in VALID_TONES:
        return tone
    return default if default in VALID_TONES else "assistant-like"


def apply_tone(text: str, tone: str, language: str) -> str:
    content = (text or "").strip()
    if not content:
        return content

    normalized_tone = normalize_tone(tone)
    lang = (language or "en").strip().lower()

    if normalized_tone == "formal":
        prefix = "Kindly note: " if lang == "en" else "कृपया ध्यान दें: "
        return f"{prefix}{content}"

    if normalized_tone == "friendly":
        prefix = "Sure, " if lang == "en" else "ज़रूर, "
        return f"{prefix}{content}"

    # assistant-like default
    prefix = "Assistant: " if lang == "en" else "सहायक: "
    return f"{prefix}{content}"
