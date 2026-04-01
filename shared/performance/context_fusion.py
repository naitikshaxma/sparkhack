from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_intent(intent: Any) -> str:
    return _safe_text(intent).lower().replace(" ", "_")


def build_context_fusion(
    *,
    current_intent: Optional[str],
    previous_intent: Optional[str],
    user_profile: Optional[Dict[str, Any]],
    need_category: Optional[str],
    history_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compact context object used by decision and ranking layers."""
    profile = dict(user_profile or {})
    fused = {
        "current_intent": _normalize_intent(current_intent),
        "previous_intent": _normalize_intent(previous_intent),
        "need_category": _safe_text(need_category).lower(),
        "user_profile": profile,
        "history_summary": _safe_text(history_summary),
    }

    profile_hints: List[str] = []
    for key in ("user_type", "income_range", "location", "occupation"):
        value = _safe_text(profile.get(key))
        if value:
            profile_hints.append(f"{key}:{value.lower()}")

    intent_hints = [value for value in [fused["current_intent"], fused["previous_intent"], fused["need_category"]] if value]

    fused["intent_hints"] = intent_hints
    fused["profile_hints"] = profile_hints
    fused["search_hints"] = intent_hints + profile_hints
    return fused


def adaptive_confidence_thresholds(
    *,
    query: str,
    past_confidence: Optional[float],
    intent_type: Optional[str],
) -> Dict[str, float]:
    """Return adaptive low/high confidence thresholds.

    Lower threshold when query is long and specific, increase it for short/ambiguous prompts.
    """
    text = _safe_text(query)
    tokens = [token for token in text.split() if token]
    token_count = len(tokens)

    low = 0.60
    high = 0.80

    if token_count <= 3:
        low += 0.08
        high += 0.06
    elif token_count >= 12:
        low -= 0.06
        high -= 0.05

    intent = _normalize_intent(intent_type)
    if intent in {"scheme_query", "intent_scheme_query", "eligibility", "information"}:
        low -= 0.02
    elif intent in {"unknown", "clarify", "generic"}:
        low += 0.04
        high += 0.03

    if past_confidence is not None:
        try:
            value = max(0.0, min(1.0, float(past_confidence)))
            if value < 0.45:
                low += 0.04
                high += 0.04
            elif value > 0.85:
                low -= 0.03
                high -= 0.03
        except (TypeError, ValueError):
            pass

    low = max(0.45, min(0.78, low))
    high = max(low + 0.08, min(0.92, high))
    return {"low": round(low, 3), "high": round(high, 3)}
