from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict

from backend.auth import get_current_user_id
from backend.core.config import get_settings
from backend.shared.security.privacy import sanitize_session_payload
from ...shared.session import session_manager as legacy_session_manager

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None


logger = logging.getLogger(__name__)
SETTINGS = get_settings()
SESSION_TTL_SECONDS = max(1, int(SETTINGS.session_ttl_seconds))
_REDIS_CLIENT = None


def _ensure_redis_client():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    if redis is None:
        raise RuntimeError("redis package is not installed")
    client = redis.Redis.from_url(SETTINGS.redis_url, decode_responses=True)
    client.ping()
    _REDIS_CLIENT = client
    return _REDIS_CLIENT


def _session_key(session_id: str) -> str:
    sid = str(session_id or "").strip() or "anonymous"
    return f"session:{sid}"


def _attach_user_context(session_data: Dict[str, Any]) -> Dict[str, Any]:
    current_user_id = str(get_current_user_id() or "").strip()
    if current_user_id and str(session_data.get("user_id") or "").strip() != current_user_id:
        session_data["user_id"] = current_user_id
    return session_data


def _read_session(session_id: str) -> Dict[str, Any] | None:
    client = _ensure_redis_client()
    raw = client.get(_session_key(session_id))
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        logger.warning("Corrupt redis session payload for session_id=%s", session_id)
    return None


def _write_session(session_id: str, session_data: Dict[str, Any]) -> None:
    client = _ensure_redis_client()
    client.setex(_session_key(session_id), SESSION_TTL_SECONDS, json.dumps(session_data))


def get_async_session_lock(session_id: str) -> asyncio.Lock:
    return legacy_session_manager.get_async_session_lock(session_id)


def create_session(session_id: str) -> Dict[str, Any]:
    session = legacy_session_manager._default_session(session_id)
    session = _attach_user_context(session)
    _write_session(session_id, session)
    return session


def get_session(session_id: str) -> Dict[str, Any]:
    session = _read_session(session_id)
    if not session:
        return create_session(session_id)

    before_user = str(session.get("user_id") or "").strip()
    session = _attach_user_context(session)
    after_user = str(session.get("user_id") or "").strip()
    if after_user != before_user:
        _write_session(session_id, session)
    return session


def update_session(session_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
    session_data = _attach_user_context(sanitize_session_payload(dict(session_data or {})))
    session_data["updated_at"] = time.time()
    _write_session(session_id, session_data)
    return session_data


def delete_session(session_id: str) -> None:
    client = _ensure_redis_client()
    client.delete(_session_key(session_id))


def get_session_store_status() -> str:
    return "redis"


def cleanup_expired_sessions() -> int:
    return 0
