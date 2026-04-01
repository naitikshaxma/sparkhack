import asyncio
import os
import time

from backend.core.logger import log_event
from backend.infrastructure.ml.whisper_service import transcribe_audio

DEFAULT_STT_TIMEOUT_SECONDS = float(os.getenv("STT_TIMEOUT_SECONDS", "30"))
ALLOW_STT_FALLBACK = (os.getenv("ALLOW_STT_FALLBACK", "1").strip() not in {"0", "false", "False"})
STT_FALLBACK_TRANSCRIPT = (os.getenv("STT_FALLBACK_TRANSCRIPT") or "Transcription unavailable").strip() or "Transcription unavailable"


def _fallback_or_raise(exc: Exception) -> str:
    if not ALLOW_STT_FALLBACK:
        raise exc
    return STT_FALLBACK_TRANSCRIPT


class STTService:
    def transcribe(self, audio_bytes: bytes, language: str, suffix: str, timings: dict | None = None) -> str:
        """Sync entry point — only call from a thread pool, never directly in an async function."""
        start = time.perf_counter()
        log_event("stt_service_start", endpoint="stt_service", status="success")
        try:
            transcript = transcribe_audio(audio_bytes, language=language, source_suffix=suffix)
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            if timings is not None:
                timings["stt_ms"] = elapsed_ms
            log_event("stt_service_success", endpoint="stt_service", status="success", response_time_ms=elapsed_ms)
            return transcript
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            if timings is not None:
                timings["stt_ms"] = elapsed_ms
            log_event(
                "stt_service_failure",
                level="error",
                endpoint="stt_service",
                status="failure",
                error_type=type(exc).__name__,
                response_time_ms=elapsed_ms,
            )
            return _fallback_or_raise(exc)

    async def transcribe_async(
        self,
        audio_bytes: bytes,
        language: str,
        suffix: str,
        timings: dict | None = None,
        timeout: float | None = None,
    ) -> str:
        """
        Async entry point — runs STT in thread pool with a hard timeout.
        Raises RuntimeError on timeout, re-raises original exception on failure.
        """
        start = time.perf_counter()
        effective_timeout = timeout if timeout is not None else DEFAULT_STT_TIMEOUT_SECONDS
        log_event("stt_service_async_start", endpoint="stt_service", status="success")
        try:
            transcript = await asyncio.wait_for(
                asyncio.to_thread(transcribe_audio, audio_bytes, language, suffix),
                timeout=effective_timeout,
            )
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            if timings is not None:
                timings["stt_ms"] = elapsed_ms
            log_event("stt_service_async_success", endpoint="stt_service", status="success", response_time_ms=elapsed_ms)
            return transcript
        except asyncio.TimeoutError:
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            if timings is not None:
                timings["stt_ms"] = elapsed_ms
            log_event(
                "stt_service_async_timeout",
                level="error",
                endpoint="stt_service",
                status="timeout",
                response_time_ms=elapsed_ms,
            )
            return _fallback_or_raise(RuntimeError(f"STT transcribe_async timed out after {effective_timeout}s"))
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            if timings is not None:
                timings["stt_ms"] = elapsed_ms
            log_event(
                "stt_service_async_failure",
                level="error",
                endpoint="stt_service",
                status="failure",
                error_type=type(exc).__name__,
                response_time_ms=elapsed_ms,
            )
            return _fallback_or_raise(exc)
