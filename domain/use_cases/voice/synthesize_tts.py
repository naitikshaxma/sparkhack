from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException

from ....infrastructure.session.voice_state_store import clear_interrupt, is_interrupted, set_voice_state
from backend.shared.language.personality import apply_tone, normalize_tone
from backend.shared.security.privacy import redact_sensitive_text


async def synthesize_tts(
    *,
    text: str,
    normalized_text: str,
    body_language: Optional[str],
    header_language: Optional[str],
    tone: Optional[str],
    session_id: Optional[str],
    default_tone: str,
    tts_service: Any,
    timings: Optional[Dict[str, Any]],
    resolve_auto_language_fn: Callable[[Optional[str], Optional[str], str], str],
) -> Dict[str, Any]:
    request_language = resolve_auto_language_fn(body_language, header_language, text)
    resolved_tone = normalize_tone(tone or default_tone, default=default_tone)
    toned_text = apply_tone(text, resolved_tone, request_language)
    safe_text = redact_sensitive_text(toned_text)
    session_key = (session_id or "").strip()

    if session_key:
        if is_interrupted(session_key):
            clear_interrupt(session_key)
        set_voice_state(session_key, "speaking")

    try:
        audio_base64 = await tts_service.synthesize_async(
            text=safe_text,
            language=request_language,
            timings=timings or {},
        )
        if not audio_base64:
            raise HTTPException(status_code=500, detail="TTS generation failed.")
    finally:
        if session_key:
            set_voice_state(session_key, "idle")

    return {
        "response_text": safe_text,
        "audio_base64": f"data:audio/mp3;base64,{audio_base64}",
        "user_input_length": len(normalized_text),
    }
