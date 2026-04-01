from __future__ import annotations

from typing import List, Optional

from backend.services.intent_service import detect_multi_intents as _detect_multi_intents
from backend.services.intent_service import resolve_intent_decision as _resolve_intent_decision


def detect_multi_intents(text: str) -> List[str]:
    return _detect_multi_intents(text)


def resolve_intent_decision(
    raw_intent: str,
    raw_confidence: float,
    text: str,
    session_context: Optional[dict] = None,
) -> dict:
    return _resolve_intent_decision(
        raw_intent=raw_intent,
        raw_confidence=raw_confidence,
        text=text,
        session_context=session_context,
    )
