from typing import Any, Dict

from backend.infrastructure.ml.bert_service import get_intent_model_status
from backend.core.intent_analytics import get_intent_metrics
from backend.core.logger import log_event
from backend.core.metrics import get_metrics_snapshot, get_public_metrics
from backend.services.rag_service import get_rag_status
from ..infrastructure.session.session_store import get_session_store_status
from backend.infrastructure.ml.whisper_service import get_whisper_status


class SystemService:
    def health(self) -> Dict[str, Any]:
        log_event("system_health_start", endpoint="system_service", status="success")
        intent_status = get_intent_model_status()
        whisper_status = get_whisper_status() or {}
        rag_status = get_rag_status() or {}
        obs = get_metrics_snapshot()

        payload = {
            "schema_version": "2026-03-22",
            "status": "ok",
            "redis": get_session_store_status(),
            "uptime_seconds": obs.get("uptime_seconds", 0),
            "request_count": obs.get("total_requests", 0),
            "error_rate": obs.get("error_rate", 0.0),
            "intent_model": {
                "status": "loaded" if intent_status.get("loaded") else "fallback",
                "loaded": bool(intent_status.get("loaded", False)),
                "fallback_enabled": bool(intent_status.get("fallback_enabled", True)),
                "error": intent_status.get("error"),
            },
            "stt": {
                "status": "ok" if whisper_status.get("model_loaded", False) else "degraded",
                "provider": "whisper",
            },
            "tts": {
                "status": "ok",
                "provider": "gtts",
            },
            "ocr": {
                "status": "ok",
                "provider": "tesseract+local-parser",
            },
            "whisper": {
                "model_name": whisper_status.get("model_name", "unknown"),
                "model_loaded": bool(whisper_status.get("model_loaded", False)),
                "ffmpeg_available": bool(whisper_status.get("ffmpeg_available", False)),
                "default_language": whisper_status.get("default_language") or "en",
            },
            "rag": {
                "dataset_path": rag_status.get("dataset_path"),
                "total_schemes": int(rag_status.get("total_schemes", 0)),
                "loaded": bool(rag_status.get("loaded", False)),
            },
        }
        log_event("system_health_success", endpoint="system_service", status="success")
        return payload

    def metrics(self) -> Dict[str, Any]:
        log_event("system_metrics_start", endpoint="system_service", status="success")
        public = get_public_metrics()
        payload = {
            **public,
            "observability": get_metrics_snapshot(),
            "intent": get_intent_metrics(),
            "session_store": get_session_store_status(),
        }
        log_event("system_metrics_success", endpoint="system_service", status="success")
        return payload

    def status(self) -> Dict[str, Any]:
        log_event("system_status_start", endpoint="system_service", status="success")
        health = self.health()
        obs = get_metrics_snapshot()
        payload = {
            "service": "voice-os-bharat",
            "status": health.get("status", "unknown"),
            "uptime_seconds": obs.get("uptime_seconds", 0),
            "request_count": obs.get("total_requests", 0),
            "error_rate": obs.get("error_rate", 0.0),
            "components": {
                "intent": health.get("intent_model", {}),
                "whisper": health.get("whisper", {}),
                "stt": health.get("stt", {}),
                "tts": health.get("tts", {}),
                "ocr": health.get("ocr", {}),
                "rag": health.get("rag", {}),
                "session_store": health.get("redis"),
            },
        }
        log_event("system_status_success", endpoint="system_service", status="success")
        return payload
