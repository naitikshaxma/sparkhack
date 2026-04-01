from __future__ import annotations

from typing import Any, Dict, Tuple

from backend.text_normalizer import normalize_text


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _income_matches(user_income: Any, income_limit: Any) -> Tuple[bool, str]:
    if income_limit in (None, "", 0, "0"):
        return True, "No strict income limit for this scheme."

    u = _to_number(user_income)
    l = _to_number(income_limit)
    if u is None or l is None:
        return True, "Income data incomplete; treated as potentially eligible."

    if u <= l:
        return True, "Income appears within the scheme threshold."
    return False, "Income appears above the scheme threshold."


def _normalize_user_type(value: Any) -> str:
    text = normalize_text(str(value or ""))
    if not text:
        return ""
    if any(token in text for token in {"farmer", "kisan", "agri"}):
        return "farmer"
    if any(token in text for token in {"student", "scholar"}):
        return "student"
    if any(token in text for token in {"business", "shop", "entrepreneur", "startup"}):
        return "business"
    if text in {"general", "all", "public"}:
        return "general"
    return text


def _target_user_matches(user_type: Any, target_user: Any) -> Tuple[bool, str]:
    target = _normalize_user_type(target_user)
    utype = _normalize_user_type(user_type)

    if not target:
        return True, "No strict user-type restriction found."
    if target == "general":
        return True, "Scheme is open to general users."
    if not utype:
        return True, "User type missing; treated as potentially eligible."
    if target in utype or utype in target:
        return True, "User type aligns with scheme target group."
    return False, "User type does not strongly align with scheme target group."


def check_eligibility(user_profile: Dict[str, Any], scheme: Dict[str, Any]) -> Dict[str, Any]:
    profile = user_profile or {}
    target_ok, target_reason = _target_user_matches(profile.get("user_type"), scheme.get("target_user"))
    income_ok, income_reason = _income_matches(profile.get("annual_income") or profile.get("income_range"), scheme.get("income_limit"))

    score = 0.0
    if target_ok:
        score += 0.55
    if income_ok:
        score += 0.45
    score = max(0.0, min(1.0, score))

    eligible = score >= 0.5
    reasons = [target_reason, income_reason]
    return {
        "eligible": eligible,
        "score": round(score, 3),
        "reason": " ".join(reasons),
    }
