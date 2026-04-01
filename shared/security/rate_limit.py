from __future__ import annotations

import os
import threading
import time
from typing import Dict, List

try:
    import redis
except Exception:  # pragma: no cover - graceful fallback when redis extras are unavailable
    redis = None

_BUCKETS: Dict[str, List[float]] = {}
_LOCK = threading.Lock()
_REDIS_CLIENT = None
_REDIS_INIT_LOCK = threading.Lock()


def _get_redis_client():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    if redis is None:
        return None

    redis_url = (os.getenv("REDIS_URL") or "redis://localhost:6379/0").strip()
    try:
        with _REDIS_INIT_LOCK:
            if _REDIS_CLIENT is None:
                _REDIS_CLIENT = redis.Redis.from_url(redis_url, socket_timeout=0.2, socket_connect_timeout=0.2)
        return _REDIS_CLIENT
    except Exception:
        return None


def _allow_request_redis(key: str, limit: int, window: int, now: float) -> bool | None:
    client = _get_redis_client()
    if client is None:
        return None

    bucket_key = f"rl:{key}"
    try:
        pipeline = client.pipeline(transaction=True)
        pipeline.zremrangebyscore(bucket_key, 0, now - window)
        pipeline.zcard(bucket_key)
        _, count = pipeline.execute()
        if int(count or 0) >= limit:
            return False

        pipeline = client.pipeline(transaction=True)
        member = f"{now:.6f}:{time.monotonic_ns()}"
        pipeline.zadd(bucket_key, {member: now})
        pipeline.expire(bucket_key, max(window, 1))
        pipeline.execute()
        return True
    except Exception:
        # Fall back to in-memory limiter when Redis is unavailable.
        return None


def allow_request(key: str, *, max_requests: int, window_seconds: int) -> bool:
    cleaned_key = (key or "").strip() or "anonymous"
    limit = max(1, int(max_requests))
    window = max(1, int(window_seconds))
    now = time.time()

    redis_allowed = _allow_request_redis(cleaned_key, limit, window, now)
    if redis_allowed is False:
        return False

    # Redis path succeeded and accepted this request.
    if redis_allowed is True:
        return True

    with _LOCK:
        bucket = _BUCKETS.get(cleaned_key, [])
        cutoff = now - window
        bucket = [ts for ts in bucket if ts >= cutoff]
        if len(bucket) >= limit:
            _BUCKETS[cleaned_key] = bucket
            return False
        bucket.append(now)
        _BUCKETS[cleaned_key] = bucket
        return True
