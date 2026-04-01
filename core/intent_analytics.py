from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict


_METRICS_PATH = Path(__file__).resolve().parent / "intent_metrics.json"
_LOCK = threading.Lock()
_ANALYTICS_ENABLED = (os.getenv("ENABLE_INTENT_ANALYTICS") or "0").strip().lower() in {"1", "true", "yes"}
_METRICS: Dict[str, Any] = {
    "intent_frequency": {},
    "fallback_frequency": 0,
    "unknown_intents": {},
    "low_confidence_cases": 0,
}


def _safe_write_metrics() -> None:
    try:
        _METRICS_PATH.write_text(json.dumps(_METRICS, indent=2), encoding="utf-8")
    except Exception:
        # Metrics persistence should never break request handling.
        return


def record_intent_event(
    intent: str,
    confidence: float,
    fallback_used: bool,
    low_confidence: bool,
    raw_intent: str = "",
) -> None:
    if not _ANALYTICS_ENABLED:
        return
    with _LOCK:
        intent_frequency = _METRICS.setdefault("intent_frequency", {})
        intent_frequency[intent] = int(intent_frequency.get(intent, 0)) + 1

        if fallback_used:
            _METRICS["fallback_frequency"] = int(_METRICS.get("fallback_frequency", 0)) + 1

        if low_confidence:
            _METRICS["low_confidence_cases"] = int(_METRICS.get("low_confidence_cases", 0)) + 1

        normalized_raw = (raw_intent or "").strip().lower()
        if normalized_raw in {"", "unknown", "unrecognized"}:
            unknown_intents = _METRICS.setdefault("unknown_intents", {})
            key = normalized_raw or "empty"
            unknown_intents[key] = int(unknown_intents.get(key, 0)) + 1

        _safe_write_metrics()


def get_intent_metrics() -> Dict[str, Any]:
    if not _ANALYTICS_ENABLED:
        return {
            "intent_frequency": {},
            "fallback_frequency": 0,
            "unknown_intents": {},
            "low_confidence_cases": 0,
        }
    with _LOCK:
        return {
            "intent_frequency": dict(_METRICS.get("intent_frequency", {})),
            "fallback_frequency": int(_METRICS.get("fallback_frequency", 0)),
            "unknown_intents": dict(_METRICS.get("unknown_intents", {})),
            "low_confidence_cases": int(_METRICS.get("low_confidence_cases", 0)),
        }
