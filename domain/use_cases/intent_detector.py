from __future__ import annotations

from typing import Optional, Tuple

from backend.services.intent_service import detect_intent_and_mode as _detect_intent_and_mode
from backend.services.intent_service import is_followup_info_query as _is_followup_info_query


def detect_intent_and_mode(
    query: str,
    predicted_intent: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Tuple[str, str]:
    return _detect_intent_and_mode(query, predicted_intent=predicted_intent, confidence=confidence)


def is_followup_info_query(query: str) -> bool:
    return _is_followup_info_query(query)
