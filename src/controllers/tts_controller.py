from __future__ import annotations

from typing import Any, Dict

from fastapi import Request

from backend.src.utils.tts_utils import generate_tts


async def handle_tts(request: Request) -> Dict[str, Any]:
    payload = await request.json()
    text = str((payload or {}).get("text") or "").strip()
    language = str((payload or {}).get("language") or request.headers.get("x-language") or "en")
    return {"audio_base64": await generate_tts(text, language)}
