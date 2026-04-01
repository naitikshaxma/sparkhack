"""
backend/api/v1/routes/voice_ws.py

WebSocket endpoint for the async voice pipeline.

Flow:
  1. Client connects and sends JSON payload
  2. API enqueues a job → Redis job queue
  3. Worker processes: STT → Intent → RAG → TTS
  4. Worker stores result → Redis (voice_os:result_data:{job_id})
  5. Worker publishes → Redis Pub/Sub (voice_os:result:{job_id})
  6. API WebSocket subscribes and streams result to client

Reconnect / result-fetch:
  - Client sends {"fetch_result": true, "job_id": "..."} to retrieve a stored result
    even if the WebSocket dropped and reconnected.

Fallback:
  - If Redis is unavailable, runs full pipeline synchronously in the API process.
"""
from __future__ import annotations

import json
import uuid
import base64
from binascii import Error as BinasciiError
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.logger import log_event
from backend.routes.response_utils import RESPONSE_REDACTION_SKIP_KEYS
from backend.shared.security.privacy import redact_sensitive_payload, redact_sensitive_text

try:
    from backend.infrastructure.queue.redis_queue import (
        enqueue_job,
        get_job_status,
        get_result,
        make_job,
    )
except Exception:  # pragma: no cover - optional async backend
    enqueue_job = None
    get_job_status = None
    get_result = None
    make_job = None

try:
    from backend.infrastructure.pubsub.redis_pubsub import subscribe_result
except Exception:  # pragma: no cover - optional async backend
    subscribe_result = None

router = APIRouter(tags=["voice-ws"])

# How long the WebSocket waits for the worker to publish a result
JOB_TIMEOUT_SECONDS = float(90)
MAX_AUDIO_BYTES = 5 * 1024 * 1024


def _queue_backend_available() -> bool:
    return all(callable(fn) for fn in (enqueue_job, get_job_status, get_result, make_job))


def _pubsub_backend_available() -> bool:
    return callable(subscribe_result)


# ── Sync fallback (no Redis / local dev) ─────────────────────────────────────────
async def _process_sync_fallback(
    session_id: str,
    transcript: str,
    language: str,
) -> dict:
    """Run the full pipeline inline when Redis is unavailable."""
    import asyncio
    try:
        from backend.services.conversation_service import ConversationService
        from backend.services.tts_service import TTSService

        conv_service = ConversationService()
        tts_service  = TTSService()

        conversation_result = await asyncio.to_thread(
            conv_service.process, session_id, transcript, language, False
        )
        response_text = (
            conversation_result.get("voice_text")
            or conversation_result.get("response_text", "")
        )
        safe_response_text = redact_sensitive_text(response_text)
        audio_b64: Optional[str] = None
        try:
            audio_b64 = await tts_service.synthesize_async(safe_response_text, language)
        except Exception:
            pass

        return {
            "status":        "ok",
            "job_id":        str(uuid.uuid4()),
            "session_id":    session_id,
            "transcript":    redact_sensitive_text(transcript),
            "response_text": safe_response_text,
            "audio_base64":  audio_b64,
            "conversation":  conversation_result,
            "fallback_mode": True,
        }
    except Exception as exc:
        return {
            "status":        "error",
            "error":         str(exc),
            "session_id":    session_id,
            "fallback_mode": True,
        }


# ── WebSocket endpoint ────────────────────────────────────────────────────────────
@router.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket voice pipeline endpoint.

    Client sends JSON:
        {"text": "...", "language": "hi"}        — process new query
        {"cancel": true}                          — cancel / no-op
        {"fetch_result": true, "job_id": "..."}   — retrieve stored result after reconnect

    Server sends JSON events:
        {"type": "ack",    "job_id": "..."}
        {"type": "status", "job_id": "...", "status": "processing"}
        {"type": "result", "payload": {...}}
        {"type": "error",  "error":  "...", "job_id": "..."}
        {"type": "done",   "job_id": "..."}
        {"type": "cancelled"}
    """
    await websocket.accept()
    log_event("ws_voice_connect", session_id=session_id, status="success")

    async def _send_json_safe(payload: dict) -> None:
        safe_payload = redact_sensitive_payload(payload, skip_keys=RESPONSE_REDACTION_SKIP_KEYS)
        await websocket.send_json(safe_payload)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_json_safe({"type": "error", "error": "invalid_json"})
                continue

            # ── Cancellation ─────────────────────────────────────────────────────
            if msg.get("cancel"):
                await _send_json_safe({"type": "cancelled"})
                continue

            # ── Reconnect: fetch stored result by job_id ──────────────────────────
            if msg.get("fetch_result"):
                job_id = (msg.get("job_id") or "").strip()
                if not job_id:
                    await _send_json_safe({"type": "error", "error": "missing job_id"})
                    continue

                if not (callable(get_job_status) and callable(get_result)):
                    await _send_json_safe({
                        "type": "error",
                        "error": "async_result_lookup_unavailable",
                        "job_id": job_id,
                    })
                    continue

                # Check status first
                status = get_job_status(job_id)
                if status in ("pending", "processing"):
                    await _send_json_safe({"type": "status", "job_id": job_id, "status": status})
                    continue

                stored = get_result(job_id)
                if stored:
                    await _send_json_safe({"type": "result", "payload": stored})
                    await _send_json_safe({"type": "done", "job_id": job_id})
                else:
                    await _send_json_safe({
                        "type":    "error",
                        "error":   "result_not_found",
                        "job_id":  job_id,
                        "message": "Result expired or job_id unknown.",
                    })
                continue

            # ── New query ─────────────────────────────────────────────────────────
            transcript = (msg.get("text") or "").strip()
            language   = (msg.get("language") or "en").strip()
            audio_b64  = msg.get("audio_base64")
            audio_format = (msg.get("audio_format") or "audio/webm").strip().lower()

            audio_bytes_list = []
            if audio_b64:
                if not isinstance(audio_b64, str):
                    await _send_json_safe({"type": "error", "error": "invalid_audio_encoding"})
                    continue
                try:
                    audio_bytes = base64.b64decode(audio_b64, validate=True)
                except (BinasciiError, ValueError):
                    await _send_json_safe({"type": "error", "error": "invalid_audio_base64"})
                    continue

                if not audio_bytes:
                    await _send_json_safe({"type": "error", "error": "empty_audio_payload"})
                    continue

                if len(audio_bytes) > MAX_AUDIO_BYTES:
                    await _send_json_safe({"type": "error", "error": "audio_too_large"})
                    continue

                audio_bytes_list = list(audio_bytes)

            if not transcript and not audio_bytes_list:
                await _send_json_safe({"type": "error", "error": "empty_text_and_audio"})
                continue
            
            payload = {"transcript": transcript, "language": language}
            if audio_bytes_list:
                payload["audio_bytes"] = audio_bytes_list
                payload["audio_format"] = audio_format

            redis_available = False
            if _queue_backend_available():
                job = make_job(
                    session_id=session_id,
                    job_type="pipeline",
                    payload=payload,
                )
                job_id = job["job_id"]
                # Enqueue (returns False = Redis unavailable -> use sync fallback)
                redis_available = bool(enqueue_job(job))
            else:
                job_id = str(uuid.uuid4())

            await _send_json_safe({"type": "ack", "job_id": job_id})

            # ── Sync fallback ─────────────────────────────────────────────────────
            if not redis_available:
                if not transcript:
                    await _send_json_safe({
                        "type": "error",
                        "error": "audio_processing_requires_async_worker",
                        "job_id": job_id,
                    })
                    await _send_json_safe({"type": "done", "job_id": job_id})
                    continue
                log_event("ws_sync_fallback", session_id=session_id, job_id=job_id)
                result = await _process_sync_fallback(session_id, transcript, language)
                await _send_json_safe({"type": "result", "payload": result})
                await _send_json_safe({"type": "done", "job_id": job_id})
                continue

            # ── Async path: subscribe to Pub/Sub, stream result ───────────────────
            got_result = False
            if _pubsub_backend_available():
                async for result in subscribe_result(job_id, timeout_seconds=JOB_TIMEOUT_SECONDS):
                    await _send_json_safe({"type": "result", "payload": result})
                    got_result = True
                    break

            if not got_result:
                # Pub/Sub timed out — check if result was stored anyway
                # (handles the case where worker finished but pub/sub message was lost)
                stored = get_result(job_id) if callable(get_result) else None
                if stored:
                    log_event(
                        "ws_result_recovered_from_storage",
                        session_id=session_id,
                        job_id=job_id,
                    )
                    await _send_json_safe({"type": "result", "payload": stored})
                else:
                    await _send_json_safe({
                        "type":    "error",
                        "error":   "timeout",
                        "job_id":  job_id,
                        "message": "Worker did not respond in time. Use fetch_result to retry.",
                    })

            await _send_json_safe({"type": "done", "job_id": job_id})

    except WebSocketDisconnect:
        log_event("ws_voice_disconnect", session_id=session_id, status="success")
    except Exception as exc:
        log_event(
            "ws_voice_error",
            session_id=session_id,
            status="failure",
            error_type=type(exc).__name__,
        )
        try:
            await _send_json_safe({"type": "error", "error": str(exc)})
        except Exception:
            pass
