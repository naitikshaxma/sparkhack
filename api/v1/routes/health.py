"""
backend/api/v1/routes/health.py

Production health check endpoints.

  GET /api/v1/health  — shallow liveness probe (always fast)
  GET /api/v1/ready   — deep readiness probe (checks Redis + worker heartbeats)
  GET /api/v1/sys/metrics — Prometheus text format metrics
"""
from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, JSONResponse

router = APIRouter(tags=["observability"])


# ── Liveness ──────────────────────────────────────────────────────────────────────
@router.get("/health", summary="Liveness probe")
def health() -> Dict[str, Any]:
    """
    Simple liveness check. Returns 200 if the API process is alive.
    Should NEVER depend on external services — just confirms the process is up.
    """
    return {
        "status": "ok",
        "timestamp": time.time(),
        "service": "voice-os-bharat",
    }


# ── Readiness ─────────────────────────────────────────────────────────────────────
@router.get("/ready", summary="Readiness probe")
def ready() -> JSONResponse:
    """
    Deep readiness check. Returns:
      200 — all dependencies healthy
      503 — one or more dependencies degraded (API still running but not ready)

    Checks:
      - Redis connectivity
      - At least one live worker heartbeat
      - Queue backpressure (warns if queue is deep)
    """
    checks: Dict[str, Any] = {}
    healthy = True

    # ── Redis ────────────────────────────────────────────────────────────────
    try:
        from backend.infrastructure.queue.redis_queue import _get_redis, queue_length
    except ModuleNotFoundError:
        checks["redis"] = {"status": "disabled", "detail": "Queue backend disabled for MVP"}
    else:
        try:
            r = _get_redis()
            if r is None:
                checks["redis"] = {
                    "status": "fallback_mode",
                    "detail": "Redis unavailable; API fallback path is active",
                    "queue_length": 0,
                }
            else:
                r.ping()
                q_len = queue_length()
                checks["redis"] = {"status": "ok", "queue_length": q_len}
                if q_len > 500:
                    checks["redis"]["warning"] = "queue depth > 500"
        except Exception as exc:
            checks["redis"] = {"status": "error", "detail": str(exc)}
            healthy = False

    # ── Worker heartbeats ─────────────────────────────────────────────────────
    try:
        live_workers = _count_live_workers()
        checks["workers"] = {
            "status": "ok" if live_workers > 0 else "no_workers",
            "live_count": live_workers,
        }
        if live_workers == 0:
            checks["workers"]["detail"] = "No worker heartbeat found — jobs will queue but not process"
            # Degraded — warn but don't fail (API can still accept + queue jobs)
    except Exception as exc:
        checks["workers"] = {"status": "error", "detail": str(exc)}

    # ── Response ──────────────────────────────────────────────────────────────
    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if healthy else "degraded",
            "timestamp": time.time(),
            "checks": checks,
        },
    )


def _count_live_workers() -> int:
    """Count workers that have sent a heartbeat recently (key still exists = TTL not expired)."""
    try:
        from backend.infrastructure.queue.redis_queue import _get_redis
        r = _get_redis()
        if r is None:
            return 0
        # Workers write voice_os:worker:{id}:heartbeat with TTL=60s
        keys = list(r.keys("voice_os:worker:*:heartbeat"))
        return len(keys)
    except Exception:
        return 0


# ── Prometheus metrics ─────────────────────────────────────────────────────────────
@router.get(
    "/sys/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
    include_in_schema=True,
)
def prometheus_metrics() -> str:
    """
    Prometheus text exposition format.
    Compatible with Prometheus scraping, Grafana, and any OpenMetrics-aware tool.

    Metrics included:
      - voice_os_queue_length (gauge)
      - voice_os_dead_letter_length (gauge)
      - voice_os_jobs_processed_total (counter)
      - voice_os_jobs_failed_total (counter)
      - voice_os_latency_stt_avg_ms (gauge)
      - voice_os_latency_intent_rag_avg_ms (gauge)
      - voice_os_latency_tts_avg_ms (gauge)
      - voice_os_live_workers (gauge)
      ... and all other tracked counters
    """
    try:
        from backend.infrastructure.monitoring.metrics import prometheus_text
        base = prometheus_text()
    except ModuleNotFoundError:
        base = (
            "# HELP voice_os_metrics_disabled Monitoring module disabled\n"
            "# TYPE voice_os_metrics_disabled gauge\n"
            "voice_os_metrics_disabled 1\n"
        )
    except Exception:
        base = (
            "# HELP voice_os_metrics_error Monitoring exporter failed\n"
            "# TYPE voice_os_metrics_error gauge\n"
            "voice_os_metrics_error 1\n"
        )

    # Append worker count as a gauge
    live = _count_live_workers()
    extra = (
        "# HELP voice_os_live_workers Workers with active heartbeat\n"
        "# TYPE voice_os_live_workers gauge\n"
        f"voice_os_live_workers {live}\n"
    )
    return base + extra
