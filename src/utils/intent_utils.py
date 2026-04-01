from __future__ import annotations

import re
from typing import Any, Dict

from backend.src.data.schemes_loader import SCHEME_DATA, SCHEME_KEYWORDS

_HINDI_RE = re.compile(r"[\u0900-\u097F]")
_ELIGIBILITY_KEYWORDS = ("eligibility", "eligible", "पात्रता", "पात्र", "योग्य")
_APPLY_KEYWORDS = ("how to apply", "apply", "application", "register", "आवेदन", "कैसे अप्लाई", "कैसे आवेदन")


def is_hindi(language: str, text: str) -> bool:
    lang = str(language or "").strip().lower()
    return lang.startswith("hi") or bool(_HINDI_RE.search(str(text or "")))


def detect_scheme(text: str) -> str | None:
    query = str(text or "").lower()
    for scheme, keywords in SCHEME_KEYWORDS.items():
        if any(keyword.lower() in query for keyword in keywords):
            return scheme
    return None


def detect_query_intent(text: str) -> str:
    query = str(text or "").lower()
    if any(k in query for k in _ELIGIBILITY_KEYWORDS):
        return "eligibility"
    if any(k in query for k in _APPLY_KEYWORDS):
        return "apply"
    return "scheme_info"


def build_intent_payload(text: str, language: str) -> Dict[str, Any]:
    scheme = detect_scheme(text)
    if scheme:
        intent_type = detect_query_intent(text)
        is_hi = is_hindi(language, text)
        scheme_data = SCHEME_DATA[scheme]

        if intent_type == "eligibility":
            message = scheme_data["eligibility_hi"] if is_hi else scheme_data["eligibility_en"]
        elif intent_type == "apply":
            message = scheme_data["apply_hi"] if is_hi else scheme_data["apply_en"]
        else:
            message = scheme_data["description_hi"] if is_hi else scheme_data["description_en"]

        return {
            "success": True,
            "type": intent_type,
            "scheme": scheme,
            "message": message,
            "confidence": 1.0,
        }

    unknown_message = (
        "Sorry, I do not have knowledge about this scheme right now. Please ask about a scheme from my supported hardcoded list, and I will help you fully."
        if not is_hindi(language, text)
        else "माफ कीजिए, अभी इस योजना के बारे में मेरे पास जानकारी उपलब्ध नहीं है। कृपया मेरी समर्थित हार्डकोड सूची में से योजना पूछें, मैं पूरी मदद करूंगा।"
    )
    return {
        "success": True,
        "type": "error",
        "message": unknown_message,
        "confidence": 1.0,
    }
