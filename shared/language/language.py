from typing import Optional
import re


DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
HINGLISH_HINTS = {
    "kya",
    "haan",
    "nahi",
    "kripya",
    "kaise",
    "loan",
    "yojana",
    "pension",
    "aadhaar",
    "samjhao",
    "bataye",
    "batayein",
}


def normalize_language_code(language: Optional[str], default: str = "en") -> str:
    value = (language or "").strip().lower()
    if value.startswith("en"):
        return "en"
    if value.startswith("hi"):
        return "hi"

    fallback = (default or "en").strip().lower()
    return "en" if fallback.startswith("en") else "hi"


def detect_text_language(text: Optional[str], default: str = "en") -> str:
    content = (text or "").strip()
    if not content:
        return normalize_language_code(default, default="en")
    if DEVANAGARI_PATTERN.search(content):
        return "hi"
    return "en"


def detect_input_language(text: Optional[str], default: str = "en") -> str:
    content = (text or "").strip()
    if not content:
        return normalize_language_code(default, default="en")

    lowered = content.lower()
    if DEVANAGARI_PATTERN.search(content):
        return "hi"

    if any(token in lowered for token in HINGLISH_HINTS):
        return "hi"

    return "en"
