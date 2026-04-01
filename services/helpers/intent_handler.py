import re
from typing import Optional, Set


AMBIGUOUS_WORDS: Set[str] = {"maybe", "around", "approx", "lagbhag", "approximately"}
UNCLEAR_WORDS: Set[str] = {"hmm", "uh", "hello", "helo", "sun", "listen", "something", "kuch", "pata nahi", "not sure"}
GENERIC_HELP_PATTERNS: Set[str] = {
    "loan batao",
    "loan",
    "scheme batao",
    "scheme",
    "madad chahiye",
    "help",
    "yojana",
    "help chahiye",
    "yojana batao",
    "koi scheme",
    "kuch scheme",
}
CORRECTION_PATTERNS: Set[str] = {
    "wrong",
    "change",
    "update",
    "edit",
    "not correct",
    "गलत",
    "बदल",
    "सुधार",
}


def is_ambiguous_input(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    return any(word in text for word in AMBIGUOUS_WORDS)


def is_unclear_input(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    if len(text) <= 2:
        return True
    return text in UNCLEAR_WORDS


def is_generic_help_query(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    if not text:
        return False
    if len(text.split()) <= 2 and text in {"loan", "scheme", "help", "yojana", "madad"}:
        return True
    if text in GENERIC_HELP_PATTERNS:
        return True
    return any(pattern in text for pattern in GENERIC_HELP_PATTERNS)


def is_correction_request(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in CORRECTION_PATTERNS)


def looks_like_field_value(field_name: Optional[str], user_input: str) -> bool:
    value = (user_input or "").strip()
    if not value or not field_name:
        return False

    if field_name == "phone":
        return bool(re.fullmatch(r"\D*\d\D*\d\D*\d\D*\d\D*\d\D*\d\D*\d\D*\d\D*\d\D*\d\D*", value))

    if field_name == "aadhaar_number":
        digits = re.sub(r"\D", "", value)
        return len(digits) == 12

    if field_name == "annual_income":
        candidate = value.replace(",", "")
        return bool(re.fullmatch(r"\d+(\.\d+)?", candidate))

    if field_name == "full_name":
        lowered = value.lower()
        if "?" in lowered or "kya" in lowered or "what" in lowered or "scheme" in lowered or "yojana" in lowered:
            return False
        return bool(re.fullmatch(r"[A-Za-z\s.'-]{2,80}", value))

    return False
