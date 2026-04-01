from __future__ import annotations

from fastapi import APIRouter, Request

from backend.src.controllers.intent_controller import handle_intent
from backend.src.controllers.transcribe_controller import handle_transcribe
from backend.src.controllers.tts_controller import handle_tts

router = APIRouter()


@router.get("/")
def root() -> dict[str, object]:
    return {"success": True, "status": "ok", "service": "voice-os-backend"}


@router.get("/health")
def health() -> dict[str, object]:
    return {"success": True, "status": "ok"}


@router.get("/api/health")
def health_api() -> dict[str, object]:
    return health()


@router.get("/api/v1/health")
def health_api_v1() -> dict[str, object]:
    return health()


@router.post("/api/intent")
async def intent(request: Request) -> dict[str, object]:
    return await handle_intent(request)


@router.post("/api/v1/intent")
async def intent_v1(request: Request) -> dict[str, object]:
    return await handle_intent(request)


@router.post("/api/tts")
async def tts(request: Request) -> dict[str, object]:
    return await handle_tts(request)


@router.post("/api/v1/tts")
async def tts_v1(request: Request) -> dict[str, object]:
    return await handle_tts(request)


@router.post("/api/transcribe")
async def transcribe(request: Request) -> dict[str, object]:
    return await handle_transcribe(request)


@router.post("/api/v1/transcribe")
async def transcribe_v1(request: Request) -> dict[str, object]:
    return await handle_transcribe(request)
