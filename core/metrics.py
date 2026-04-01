from __future__ import annotations

import threading
import time
import os
from typing import Any, Dict


_LOCK = threading.Lock()
_METRICS_ENABLED = (os.getenv("ENABLE_CORE_METRICS") or "0").strip().lower() in {"1", "true", "yes"}
_START_TIME = time.time()
_TOTAL_REQUESTS = 0
_SUCCESS_REQUESTS = 0
_FAILURE_REQUESTS = 0
_TOTAL_LATENCY_MS = 0.0
_ERROR_TYPES: dict[str, int] = {}
_FALLBACK_FREQUENCY = 0
_AUTOMATION_ATTEMPTS = 0
_AUTOMATION_SUCCESSES = 0
_AUTOMATION_FAILURES = 0
_AUTOMATION_FALLBACK_USED = 0
_GENERIC_COUNTERS: dict[str, int] = {}
_GENERIC_TIMINGS: dict[str, dict[str, float]] = {}


def increment_counter(name: str, amount: int = 1) -> None:
    if not _METRICS_ENABLED:
        return
    key = (name or "").strip() or "unknown_counter"
    delta = max(0, int(amount))
    with _LOCK:
        _GENERIC_COUNTERS[key] = int(_GENERIC_COUNTERS.get(key, 0)) + delta


def record_timing(name: str, value: float) -> None:
    if not _METRICS_ENABLED:
        return
    key = (name or "").strip() or "unknown_timing"
    try:
        numeric = max(0.0, float(value))
    except (TypeError, ValueError):
        numeric = 0.0
    with _LOCK:
        bucket = _GENERIC_TIMINGS.setdefault(key, {"count": 0.0, "sum": 0.0})
        bucket["count"] = float(bucket.get("count", 0.0)) + 1.0
        bucket["sum"] = float(bucket.get("sum", 0.0)) + numeric


def record_request(*, response_time_ms: float, success: bool) -> None:
    if not _METRICS_ENABLED:
        return
    global _TOTAL_REQUESTS, _SUCCESS_REQUESTS, _FAILURE_REQUESTS, _TOTAL_LATENCY_MS
    with _LOCK:
        _TOTAL_REQUESTS += 1
        _TOTAL_LATENCY_MS += max(0.0, float(response_time_ms))
        if success:
            _SUCCESS_REQUESTS += 1
        else:
            _FAILURE_REQUESTS += 1
    increment_counter("total_requests", 1)
    record_timing("response_time_ms", response_time_ms)


def record_error(error_type: str) -> None:
    if not _METRICS_ENABLED:
        return
    key = (error_type or "unknown_error").strip() or "unknown_error"
    with _LOCK:
        _ERROR_TYPES[key] = int(_ERROR_TYPES.get(key, 0)) + 1
    increment_counter("total_errors", 1)


def record_fallback() -> None:
    if not _METRICS_ENABLED:
        return
    global _FALLBACK_FREQUENCY
    with _LOCK:
        _FALLBACK_FREQUENCY += 1
    increment_counter("fallback_count", 1)


def record_automation_result(*, success: bool, fallback_used: bool = False) -> None:
    if not _METRICS_ENABLED:
        return
    global _AUTOMATION_ATTEMPTS, _AUTOMATION_SUCCESSES, _AUTOMATION_FAILURES, _AUTOMATION_FALLBACK_USED
    with _LOCK:
        _AUTOMATION_ATTEMPTS += 1
        if success:
            _AUTOMATION_SUCCESSES += 1
            increment_counter("automation_success_count", 1)
        else:
            _AUTOMATION_FAILURES += 1
            increment_counter("automation_failure_count", 1)
        if fallback_used:
            _AUTOMATION_FALLBACK_USED += 1


def get_metrics_snapshot() -> Dict[str, Any]:
    if not _METRICS_ENABLED:
        return {
            "uptime_seconds": max(0, int(time.time() - _START_TIME)),
            "total_requests": 0,
            "success_requests": 0,
            "failure_requests": 0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "error_rate": 0.0,
            "average_latency_ms": 0.0,
            "fallback_frequency": 0,
            "fallback_rate": 0.0,
            "automation_attempts": 0,
            "automation_successes": 0,
            "automation_failures": 0,
            "automation_success_rate": 0.0,
            "automation_fallback_rate": 0.0,
            "error_types": {},
            "counters": {},
        }
    with _LOCK:
        total_requests = _TOTAL_REQUESTS
        success_requests = _SUCCESS_REQUESTS
        failure_requests = _FAILURE_REQUESTS
        total_latency_ms = _TOTAL_LATENCY_MS
        average_latency_ms = (total_latency_ms / total_requests) if total_requests else 0.0
        success_rate = (success_requests / total_requests) if total_requests else 0.0
        failure_rate = (failure_requests / total_requests) if total_requests else 0.0
        fallback_rate = (_FALLBACK_FREQUENCY / total_requests) if total_requests else 0.0
        automation_success_rate = (_AUTOMATION_SUCCESSES / _AUTOMATION_ATTEMPTS) if _AUTOMATION_ATTEMPTS else 0.0
        automation_fallback_rate = (_AUTOMATION_FALLBACK_USED / _AUTOMATION_ATTEMPTS) if _AUTOMATION_ATTEMPTS else 0.0

        return {
            "uptime_seconds": max(0, int(time.time() - _START_TIME)),
            "total_requests": total_requests,
            "success_requests": success_requests,
            "failure_requests": failure_requests,
            "success_rate": round(success_rate, 4),
            "failure_rate": round(failure_rate, 4),
            "error_rate": round(failure_rate, 4),
            "average_latency_ms": round(average_latency_ms, 2),
            "fallback_frequency": int(_FALLBACK_FREQUENCY),
            "fallback_rate": round(fallback_rate, 4),
            "automation_attempts": int(_AUTOMATION_ATTEMPTS),
            "automation_successes": int(_AUTOMATION_SUCCESSES),
            "automation_failures": int(_AUTOMATION_FAILURES),
            "automation_success_rate": round(automation_success_rate, 4),
            "automation_fallback_rate": round(automation_fallback_rate, 4),
            "error_types": dict(_ERROR_TYPES),
            "counters": dict(_GENERIC_COUNTERS),
        }


def get_public_metrics() -> Dict[str, Any]:
    snapshot = get_metrics_snapshot()
    total_requests = int(snapshot.get("total_requests", 0))
    total_errors = int(snapshot.get("failure_requests", 0))
    fallback_count = int(snapshot.get("fallback_frequency", 0))
    automation_success = int(snapshot.get("automation_successes", 0))
    automation_failure = int(snapshot.get("automation_failures", 0))
    automation_attempts = automation_success + automation_failure

    error_rate = (total_errors / total_requests) if total_requests else 0.0
    fallback_rate = (fallback_count / total_requests) if total_requests else 0.0
    automation_success_rate = (automation_success / automation_attempts) if automation_attempts else 0.0

    return {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "avg_response_time": round(float(snapshot.get("average_latency_ms", 0.0)), 2),
        "fallback_count": fallback_count,
        "automation_success_count": automation_success,
        "automation_failure_count": automation_failure,
        "error_rate": round(error_rate, 4),
        "fallback_rate": round(fallback_rate, 4),
        "automation_success_rate": round(automation_success_rate, 4),
    }
