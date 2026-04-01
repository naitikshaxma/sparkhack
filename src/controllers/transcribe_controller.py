from __future__ import annotations

from typing import Any, Dict

from fastapi import Request


async def handle_transcribe(request: Request) -> Dict[str, Any]:
    language = str(request.headers.get("x-language") or "en").strip() or "en"
    transcript = str(request.headers.get("x-live-transcript") or "").strip()

    if not transcript:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            transcript = str(payload.get("transcript") or payload.get("text") or "").strip()

    if not transcript:
        return {
            "success": False,
            "transcript": "",
            "text": "",
            "error": "No speech detected",
        }

    return {
        "success": True,
        "language": language,
        "transcript": transcript,
        "text": transcript,
    }
