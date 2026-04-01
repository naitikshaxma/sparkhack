from __future__ import annotations

from typing import Any, Dict

from fastapi import Request

from backend.src.utils.intent_utils import build_intent_payload


async def handle_intent(request: Request) -> Dict[str, Any]:
    payload = await request.json()
    text = str((payload or {}).get("text") or "").strip()
    language = str((payload or {}).get("language") or request.headers.get("x-language") or "en")
    return build_intent_payload(text, language)
