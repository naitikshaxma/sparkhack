from __future__ import annotations

import base64
import os
from typing import Optional

from backend.src.utils.intent_utils import is_hindi

_DUMMY_MP3_BASE64 = "//NExAAAAANIAAAAAExBTUUzLjEwMKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"


async def _edge_tts_base64(text: str, voice: str) -> Optional[str]:
    try:
        import edge_tts  # Lazy import keeps startup lightweight.
    except Exception:
        return None

    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                chunks.append(chunk.get("data", b""))
        audio_bytes = b"".join(chunks)
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
        return None
    return None


def resolve_voice(language: str, text: str) -> str:
    if is_hindi(language, text):
        return os.getenv("VOICE_HI", "hi-IN-SwaraNeural")
    return os.getenv("VOICE_EN", "en-IN-NeerjaNeural")


async def generate_tts(text: str, language: str) -> str:
    safe_text = str(text or "").strip() or "Ready"
    voice = resolve_voice(language, safe_text)
    audio_base64 = await _edge_tts_base64(safe_text, voice)
    if audio_base64:
        return audio_base64
    return _DUMMY_MP3_BASE64
