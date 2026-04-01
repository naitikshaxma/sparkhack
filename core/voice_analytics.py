from __future__ import annotations

import threading
import time
import os


_LOCK = threading.RLock()
_ANALYTICS_ENABLED = (os.getenv("ENABLE_VOICE_ANALYTICS") or "0").strip().lower() in {"1", "true", "yes"}
_SESSION: dict[str, dict] = {}
_GLOBAL = {
    "interruption_frequency": 0,
    "retry_patterns": 0,
}


def _bucket(session_id: str) -> dict:
    key = (session_id or "").strip() or "anonymous"
    with _LOCK:
        data = _SESSION.get(key)
        if data is None:
            data = {
                "interruption_frequency": 0,
                "retry_patterns": 0,
                "stt_signal_score_sum": 0.0,
                "stt_signal_score_count": 0,
                "latency_perception_ms_sum": 0.0,
                "latency_perception_count": 0,
                "last_event_ts": 0.0,
            }
            _SESSION[key] = data
        return data


def record_interruption(session_id: str) -> None:
    if not _ANALYTICS_ENABLED:
        return
    with _LOCK:
        _GLOBAL["interruption_frequency"] = int(_GLOBAL.get("interruption_frequency", 0)) + 1
        data = _bucket(session_id)
        data["interruption_frequency"] = int(data.get("interruption_frequency", 0)) + 1
        data["last_event_ts"] = time.time()


def record_retry(session_id: str) -> None:
    if not _ANALYTICS_ENABLED:
        return
    with _LOCK:
        _GLOBAL["retry_patterns"] = int(_GLOBAL.get("retry_patterns", 0)) + 1
        data = _bucket(session_id)
        data["retry_patterns"] = int(data.get("retry_patterns", 0)) + 1
        data["last_event_ts"] = time.time()


def record_stt_signal(session_id: str, score: float) -> None:
    if not _ANALYTICS_ENABLED:
        return
    bounded = max(0.0, min(1.0, float(score)))
    with _LOCK:
        data = _bucket(session_id)
        data["stt_signal_score_sum"] = float(data.get("stt_signal_score_sum", 0.0)) + bounded
        data["stt_signal_score_count"] = int(data.get("stt_signal_score_count", 0)) + 1
        data["last_event_ts"] = time.time()


def record_latency_perception(session_id: str, elapsed_ms: float) -> None:
    if not _ANALYTICS_ENABLED:
        return
    bounded = max(0.0, float(elapsed_ms))
    with _LOCK:
        data = _bucket(session_id)
        data["latency_perception_ms_sum"] = float(data.get("latency_perception_ms_sum", 0.0)) + bounded
        data["latency_perception_count"] = int(data.get("latency_perception_count", 0)) + 1
        data["last_event_ts"] = time.time()


def snapshot(session_id: str | None = None) -> dict:
    if not _ANALYTICS_ENABLED:
        return {"global": {"interruption_frequency": 0, "retry_patterns": 0}, "sessions_tracked": 0}
    with _LOCK:
        if session_id:
            data = dict(_bucket(session_id))
            stt_count = int(data.get("stt_signal_score_count", 0))
            latency_count = int(data.get("latency_perception_count", 0))
            data["stt_signal_score_avg"] = round(float(data.get("stt_signal_score_sum", 0.0)) / stt_count, 4) if stt_count else 0.0
            data["latency_perception_ms_avg"] = round(float(data.get("latency_perception_ms_sum", 0.0)) / latency_count, 2) if latency_count else 0.0
            return {"session_id": (session_id or "").strip(), **data}

        return {
            "global": dict(_GLOBAL),
            "sessions_tracked": len(_SESSION),
        }
