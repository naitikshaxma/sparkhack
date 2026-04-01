import asyncio
import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

from backend.core.config import get_settings
from backend.infrastructure.database.connection import db_session_scope
from backend.infrastructure.database import ConversationHistory, Session as DBSession
from backend.auth import get_current_user_id
from sqlalchemy import text
from backend.shared.session.form_schema import DEFAULT_SCHEME_NAME, get_fields_for_scheme, resolve_scheme_name
from backend.shared.security.privacy import redact_sensitive_data, sanitize_session_payload
from backend.infrastructure.ml.scheme_registry import find_schemes_in_text, get_scheme_registry

load_dotenv()

SETTINGS = get_settings()

REDIS_URL = SETTINGS.redis_url
SESSION_TTL_SECONDS = SETTINGS.session_ttl_seconds
SESSION_STORE_BACKEND = (os.getenv("SESSION_STORE_BACKEND") or "auto").strip().lower()

logger = logging.getLogger(__name__)

_ASYNC_LOCKS: Dict[tuple[int, str], asyncio.Lock] = {}
_ASYNC_LOCK_LAST_USED: Dict[tuple[int, str], float] = {}
_ASYNC_LOCK_GUARD = threading.Lock()
_MAX_ASYNC_LOCKS = max(100, int((os.getenv("MAX_ASYNC_LOCKS") or "4000").strip() or "4000"))


STATE_IDLE = "IDLE"
STATE_QUERY = "QUERY"
STATE_APPLY_START = "APPLY_START"
STATE_COLLECTING_INFO = "COLLECTING_INFO"
STATE_CONFIRMATION = "CONFIRMATION"
STATE_COMPLETED = "COMPLETED"

APPLY_INTENTS = {"apply", "apply_loan", "apply_scheme", "start_application"}

DEFAULT_REQUIRED_FIELDS = ["name", "aadhaar", "phone"]

AADHAAR_RE = re.compile(r"(?<!\d)(\d{12})(?!\d)")
PHONE_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")
NAME_RE = re.compile(r"(?:my name is|name is|i am|mera naam)\s*[:\-]?\s*([A-Za-z\u0900-\u097F\s.'-]{2,80})", re.IGNORECASE)

YES_WORDS = {"yes", "haan", "ha", "ji", "confirm", "confirmed", "sahi", "correct"}
NO_WORDS = {"no", "nahi", "nahin", "galat", "wrong", "change"}


def initialize_session_structure(session: Dict[str, Any]) -> Dict[str, Any]:
    session.setdefault("state", STATE_IDLE)
    session.setdefault("current_scheme", None)
    session.setdefault("current_scheme_name", None)
    session.setdefault("collected_fields", {})
    session.setdefault("missing_fields", [])
    session.setdefault("last_intent", None)
    return session


def reset_state_machine(session: Dict[str, Any]) -> None:
    session["state"] = STATE_IDLE
    session["current_scheme"] = None
    session["current_scheme_name"] = None
    session["collected_fields"] = {}
    session["missing_fields"] = []


def _load_scheme_configs() -> List[Dict[str, Any]]:
    registry = get_scheme_registry() or {}
    scheme_names = [str(name or "").strip() for name in registry.get("schemes", []) if str(name or "").strip()]
    if not scheme_names:
        return []

    configs: List[Dict[str, Any]] = []
    for name in scheme_names:
        scheme_id = re.sub(r"\s+", "_", name.strip().lower())
        keyword_tokens = [token for token in re.split(r"\s+", name.strip().lower()) if len(token) >= 3]
        configs.append(
            {
                "id": scheme_id,
                "name": name,
                "keywords": keyword_tokens or [name.strip().lower()],
                "required_fields": list(DEFAULT_REQUIRED_FIELDS),
                "category": "general",
            }
        )
    return configs


def _required_fields_for_scheme(scheme: Optional[Dict[str, Any]]) -> List[str]:
    if not scheme:
        return list(DEFAULT_REQUIRED_FIELDS)
    fields = scheme.get("required_fields")
    if isinstance(fields, list):
        normalized = [str(field).strip() for field in fields if str(field).strip()]
        if normalized:
            return normalized
    return list(DEFAULT_REQUIRED_FIELDS)


def _scheme_by_id(scheme_id: str) -> Optional[Dict[str, Any]]:
    target = (scheme_id or "").strip().lower()
    if not target:
        return None
    for scheme in _load_scheme_configs():
        current_id = str(scheme.get("id") or "").strip().lower()
        scheme_name = str(scheme.get("name") or "").strip().lower()
        if current_id == target or scheme_name == target:
            return scheme
    return None


def _is_yes(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return lowered in YES_WORDS or any(token in lowered for token in YES_WORDS)


def _is_no(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return lowered in NO_WORDS or any(token in lowered for token in NO_WORDS)


def _scheme_matches(text: str) -> List[Dict[str, Any]]:
    lowered = (text or "").strip().lower()
    if not lowered:
        return []
    registry = get_scheme_registry() or {}
    max_limit = max(10, int(registry.get("total") or 0))
    hits = find_schemes_in_text(lowered, limit=max_limit)
    return [item for item in hits if str(item.get("scheme") or "").strip()]


def build_scheme_clarification(matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(matches) <= 1:
        return None
    options = [str(item.get("scheme") or "").strip() for item in matches if str(item.get("scheme") or "").strip()]
    if len(options) <= 1:
        return None
    return {
        "type": "clarification",
        "message": "Multiple schemes found. Please clarify.",
        "options": options[:5],
    }


def detect_scheme(text: str) -> Optional[Dict[str, Any]]:
    try:
        matches = _scheme_matches(text)
        if len(matches) != 1:
            return None

        scheme_name = str(matches[0].get("scheme") or "").strip()
        if not scheme_name:
            return None

        scheme_id = re.sub(r"\s+", "_", scheme_name.lower())
        return {
            "id": scheme_id,
            "name": scheme_name,
            "keywords": [token for token in re.split(r"\s+", scheme_name.lower()) if len(token) >= 3],
            "required_fields": list(DEFAULT_REQUIRED_FIELDS),
            "category": "general",
        }
    except Exception:
        logger.exception("detect_scheme failed; returning None")
        return None


def detect_scheme_from_text(text: str) -> Optional[str]:
    match = detect_scheme(text)
    if not match:
        return None
    scheme_id = str(match.get("id") or "").strip()
    return scheme_id or None


def extract_fields(text: str) -> Dict[str, str]:
    content = (text or "").strip()
    lowered = content.lower()
    extracted: Dict[str, str] = {}

    aadhaar_match = AADHAAR_RE.search(content)
    if aadhaar_match:
        extracted["aadhaar"] = aadhaar_match.group(1)

    phone_match = PHONE_RE.search(content)
    if phone_match:
        extracted["phone"] = phone_match.group(1)

    age_match = re.search(r"(?:age|umra?|umar)\D{0,8}(\d{1,3})", lowered, re.IGNORECASE)
    if age_match:
        extracted["age"] = age_match.group(1)

    name_match = NAME_RE.search(content)
    if name_match:
        extracted["name"] = name_match.group(1).strip()
    elif any(token in lowered for token in {"name", "mera naam", "i am"}):
        tokens = [token for token in re.split(r"\s+", content) if token]
        if len(tokens) >= 2:
            fallback_name = " ".join(tokens[-2:]).strip(" .,")
            if fallback_name:
                extracted.setdefault("name", fallback_name)

    return extracted


def apply_state_transition(
    session: Dict[str, Any],
    user_input: str,
    detected_intent: str,
) -> Dict[str, Any]:
    initialize_session_structure(session)
    previous_state = str(session.get("state") or STATE_IDLE)
    state = previous_state
    intent = (detected_intent or "").strip().lower()
    text = (user_input or "").strip()

    transition = f"{previous_state}->{previous_state}"
    handled = False
    response_text = ""
    action = ""
    session_complete = False

    if state == STATE_IDLE:
        if intent in APPLY_INTENTS:
            session["state"] = STATE_APPLY_START
            transition = f"{STATE_IDLE}->{STATE_APPLY_START}"
            state = STATE_APPLY_START
        elif intent in {"scheme_query", "general_query"}:
            session["state"] = STATE_QUERY
            transition = f"{STATE_IDLE}->{STATE_QUERY}"
            return {
                "handled": False,
                "state_transition": transition,
                "current_state": session["state"],
                "collected_fields": dict(session.get("collected_fields", {})),
                "missing_fields": list(session.get("missing_fields", [])),
                "session_complete": False,
                "action": "",
                "response_text": "",
            }

    if state == STATE_APPLY_START:
        scheme_match = detect_scheme(text)
        clarification = build_scheme_clarification(_scheme_matches(text))
        if not scheme_match and session.get("current_scheme"):
            scheme_match = _scheme_by_id(str(session.get("current_scheme") or ""))

        if not scheme_match and clarification:
            handled = True
            action = "state_clarify_scheme"
            transition = f"{STATE_APPLY_START}->{STATE_APPLY_START}"
            response_text = str(clarification.get("message") or "Multiple schemes found. Please clarify.")
            return {
                "handled": handled,
                "state_transition": transition,
                "current_state": session.get("state", STATE_IDLE),
                "current_scheme": session.get("current_scheme"),
                "collected_fields": dict(session.get("collected_fields", {})),
                "missing_fields": list(session.get("missing_fields", [])),
                "session_complete": session_complete,
                "action": action,
                "response_text": response_text,
                "clarification": clarification,
            }

        if not scheme_match:
            handled = True
            action = "state_select_scheme"
            transition = f"{STATE_APPLY_START}->{STATE_APPLY_START}"
            response_text = "Please tell me the scheme name you want to apply for."
            return {
                "handled": handled,
                "state_transition": transition,
                "current_state": session.get("state", STATE_IDLE),
                "current_scheme": session.get("current_scheme"),
                "collected_fields": dict(session.get("collected_fields", {})),
                "missing_fields": list(session.get("missing_fields", [])),
                "session_complete": session_complete,
                "action": action,
                "response_text": response_text,
            }

        scheme_id = str(scheme_match.get("id") or "").strip()
        scheme_name = str(scheme_match.get("name") or scheme_id).strip()
        required_fields = _required_fields_for_scheme(scheme_match)
        session["current_scheme"] = scheme_id or session.get("current_scheme")
        session["current_scheme_name"] = scheme_name or session.get("current_scheme_name")
        session["missing_fields"] = list(required_fields)
        session["state"] = STATE_COLLECTING_INFO
        transition = f"{transition.split('->')[0]}->{STATE_COLLECTING_INFO}" if "->" in transition and transition.startswith(STATE_IDLE) else f"{STATE_APPLY_START}->{STATE_COLLECTING_INFO}"
        handled = True
        action = "state_collecting_info"
        response_text = f"I can help you apply for {scheme_name}. Please share: {', '.join(required_fields)}."

    elif state == STATE_COLLECTING_INFO:
        handled = True
        action = "state_collecting_info"
        extracted = extract_fields(text)
        collected = dict(session.get("collected_fields") or {})
        missing = list(session.get("missing_fields") or [])

        for key, value in extracted.items():
            if value:
                collected[key] = value
                if key in missing:
                    missing.remove(key)

        session["collected_fields"] = collected
        session["missing_fields"] = missing

        if not missing:
            session["state"] = STATE_CONFIRMATION
            transition = f"{STATE_COLLECTING_INFO}->{STATE_CONFIRMATION}"
            summary = ", ".join(f"{k}: {v}" for k, v in collected.items())
            response_text = f"I captured your details ({summary}). Please confirm to proceed."
            action = "state_confirmation"
        else:
            transition = f"{STATE_COLLECTING_INFO}->{STATE_COLLECTING_INFO}"
            response_text = f"Thanks. Please provide missing fields: {', '.join(missing)}."

    elif state == STATE_CONFIRMATION:
        handled = True
        if _is_yes(text):
            session["state"] = STATE_COMPLETED
            transition = f"{STATE_CONFIRMATION}->{STATE_COMPLETED}"
            session_complete = True
            action = "state_completed"
            response_text = "Great. Your application details are confirmed and saved."
            reset_state_machine(session)
            transition = f"{STATE_CONFIRMATION}->{STATE_COMPLETED}->{STATE_IDLE}"
        elif _is_no(text):
            session["state"] = STATE_COLLECTING_INFO
            transition = f"{STATE_CONFIRMATION}->{STATE_COLLECTING_INFO}"
            action = "state_collecting_info"
            missing = list(session.get("missing_fields") or [])
            if not missing:
                scheme = _scheme_by_id(str(session.get("current_scheme") or ""))
                missing = _required_fields_for_scheme(scheme)
                session["missing_fields"] = missing
            response_text = f"No problem. Please re-share: {', '.join(missing)}."
        else:
            transition = f"{STATE_CONFIRMATION}->{STATE_CONFIRMATION}"
            action = "state_confirmation"
            response_text = "Please reply with yes or no to confirm your details."

    return {
        "handled": handled,
        "state_transition": transition,
        "current_state": session.get("state", STATE_IDLE),
        "current_scheme": session.get("current_scheme"),
        "collected_fields": dict(session.get("collected_fields", {})),
        "missing_fields": list(session.get("missing_fields", [])),
        "session_complete": session_complete,
        "action": action,
        "response_text": response_text,
    }


def _session_key(session_id: str) -> str:
    return (session_id or "").strip() or "anonymous"


def get_async_session_lock(session_id: str) -> asyncio.Lock:
    key = _session_key(session_id)
    loop_id = id(asyncio.get_running_loop())
    lock_key = (loop_id, key)
    with _ASYNC_LOCK_GUARD:
        lock = _ASYNC_LOCKS.get(lock_key)
        if lock is None:
            lock = asyncio.Lock()
            _ASYNC_LOCKS[lock_key] = lock
        _ASYNC_LOCK_LAST_USED[lock_key] = time.time()

        if len(_ASYNC_LOCKS) > _MAX_ASYNC_LOCKS:
            # Drop unlocked, least-recently-used locks.
            candidates = sorted(_ASYNC_LOCK_LAST_USED.items(), key=lambda item: item[1])
            for candidate_key, _ in candidates:
                candidate = _ASYNC_LOCKS.get(candidate_key)
                if candidate is None:
                    _ASYNC_LOCK_LAST_USED.pop(candidate_key, None)
                    continue
                if candidate.locked():
                    continue
                _ASYNC_LOCKS.pop(candidate_key, None)
                _ASYNC_LOCK_LAST_USED.pop(candidate_key, None)
                if len(_ASYNC_LOCKS) <= _MAX_ASYNC_LOCKS:
                    break
    return lock

class SessionStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def set(self, session_id: str, session_data: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, session_id: str) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def status(self) -> str:
        raise NotImplementedError


class MemorySessionStore(SessionStore):
    def __init__(self, ttl_seconds: int) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._expires_at: Dict[str, float] = {}
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._lock = threading.RLock()

    def _cleanup_expired_locked(self) -> None:
        now = time.time()
        expired_ids = [session_id for session_id, expiry in self._expires_at.items() if expiry <= now]
        for session_id in expired_ids:
            self._sessions.pop(session_id, None)
            self._expires_at.pop(session_id, None)

    def cleanup_expired(self) -> int:
        with self._lock:
            before = len(self._sessions)
            self._cleanup_expired_locked()
            return max(0, before - len(self._sessions))

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._cleanup_expired_locked()
            return self._sessions.get(session_id)

    def set(self, session_id: str, session_data: Dict[str, Any]) -> None:
        with self._lock:
            self._cleanup_expired_locked()
            self._sessions[session_id] = session_data
            self._expires_at[session_id] = time.time() + self._ttl_seconds

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._expires_at.pop(session_id, None)

    @property
    def status(self) -> str:
        return "memory"


class RedisSessionStore(SessionStore):
    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        if redis is None:
            raise RuntimeError("redis package is not installed")
        self._ttl = ttl_seconds
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._client.ping()

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        raw = self._client.get(f"session:{session_id}")
        if not raw:
            return None
        return json.loads(raw)

    def set(self, session_id: str, session_data: Dict[str, Any]) -> None:
        self._client.setex(f"session:{session_id}", self._ttl, json.dumps(session_data))

    def delete(self, session_id: str) -> None:
        self._client.delete(f"session:{session_id}")

    @property
    def status(self) -> str:
        return "redis"


class SqlAlchemySessionStore(SessionStore):
    def __init__(self) -> None:
        with db_session_scope() as db:
            db.execute(text("SELECT 1"))

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with db_session_scope() as db:
            record = db.get(DBSession, session_id)
            if record is None:
                return None

            state = record.state_json if isinstance(record.state_json, dict) else {}
            state = dict(state)
            state["session_id"] = session_id
            if record.user_id is not None:
                state["user_id"] = str(record.user_id)

            history_rows = (
                db.query(ConversationHistory)
                .filter(ConversationHistory.session_id == session_id)
                .order_by(ConversationHistory.timestamp.asc())
                .all()
            )
            if history_rows:
                state["conversation_history"] = [
                    {
                        "role": row.role,
                        "content": row.message,
                    }
                    for row in history_rows
                ][-10:]

            return state

    def set(self, session_id: str, session_data: Dict[str, Any]) -> None:
        payload = sanitize_session_payload(dict(session_data))
        history_items = list(payload.get("conversation_history") or [])

        with db_session_scope() as db:
            record = db.get(DBSession, session_id)
            if record is None:
                record = DBSession(
                    session_id=session_id,
                    state_json=payload,
                    updated_at=self._utcnow(),
                )
                db.add(record)
            else:
                record.state_json = payload
                record.updated_at = self._utcnow()

            user_id = str(payload.get("user_id") or "").strip()
            if user_id.isdigit():
                record.user_id = int(user_id)

            db.flush()

            db.query(ConversationHistory).filter(ConversationHistory.session_id == session_id).delete(synchronize_session=False)
            for item in history_items[-50:]:
                role = str(item.get("role") or "assistant").strip()[:32]
                message = redact_sensitive_data(str(item.get("content") or item.get("message") or "").strip())
                if not message:
                    continue
                db.add(
                    ConversationHistory(
                        session_id=session_id,
                        role=role,
                        message=message,
                        timestamp=self._utcnow(),
                    )
                )

    def delete(self, session_id: str) -> None:
        with db_session_scope() as db:
            db.query(ConversationHistory).filter(ConversationHistory.session_id == session_id).delete(synchronize_session=False)
            record = db.get(DBSession, session_id)
            if record is not None:
                db.delete(record)

    @property
    def status(self) -> str:
        return "postgresql"


def _build_store() -> SessionStore:
    if SESSION_STORE_BACKEND in {"postgres", "postgresql", "sqlalchemy", "db"}:
        logger.info("Session store backend forced to postgresql")
        return SqlAlchemySessionStore()

    if SESSION_STORE_BACKEND == "memory":
        logger.info("Session store backend forced to memory")
        return MemorySessionStore(SESSION_TTL_SECONDS)

    if SESSION_STORE_BACKEND == "redis":
        return RedisSessionStore(REDIS_URL, SESSION_TTL_SECONDS)

    try:
        return SqlAlchemySessionStore()
    except Exception:
        logger.warning("PostgreSQL unavailable, trying redis session store")

    try:
        return RedisSessionStore(REDIS_URL, SESSION_TTL_SECONDS)
    except Exception:
        logger.warning("Redis unavailable, using in-memory session store")
        return MemorySessionStore(SESSION_TTL_SECONDS)


_store: SessionStore = _build_store()


def _default_session(session_id: str) -> Dict[str, Any]:
    selected_scheme = resolve_scheme_name(DEFAULT_SCHEME_NAME)
    dynamic_fields = get_fields_for_scheme(selected_scheme)
    first_field = dynamic_fields[0] if dynamic_fields else None
    return {
        "session_id": session_id,
        "form_type": "loan_application",
        "selected_scheme": selected_scheme,
        "language": "en",
        "user_profile": {},
        "user_need_profile": {
            "user_type": None,
            "income_range": None,
            "need_category": None,
        },
        "field_completion": {field: False for field in dynamic_fields},
        "next_field": first_field,
        "conversation_history": [],
        "history_summary": "",
        "onboarding_done": False,
        "past_need_confidence": None,
        "rejected_schemes": [],
        "accepted_scheme": None,
        "last_recommendation_reason": None,
        "learning_profile": {
            "rejected_counts": {},
            "accepted_counts": {},
        },
        "last_completed_field_index": -1,
        "confirmation_done": False,
        "confirmation_state": "pending",
        "session_complete": False,
        "user_id": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def _attach_user_context(session_data: Dict[str, Any]) -> Dict[str, Any]:
    current_user_id = get_current_user_id()
    if current_user_id and str(session_data.get("user_id") or "").strip() != current_user_id:
        session_data["user_id"] = current_user_id
    return session_data


def create_session(session_id: str) -> Dict[str, Any]:
    session = _attach_user_context(_default_session(session_id))
    _store.set(session_id, session)
    logger.info("Session created: %s", session_id)
    return session


def get_session(session_id: str) -> Dict[str, Any]:
    session = _store.get(session_id)
    if not session:
        return create_session(session_id)
    before_user = str(session.get("user_id") or "").strip()
    session = _attach_user_context(session)
    after_user = str(session.get("user_id") or "").strip()
    if after_user != before_user:
        _store.set(session_id, session)
    return session


def update_session(session_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
    session_data = _attach_user_context(sanitize_session_payload(session_data))
    session_data["updated_at"] = time.time()
    _store.set(session_id, session_data)
    return session_data


def delete_session(session_id: str) -> None:
    _store.delete(session_id)


def get_session_store_status() -> str:
    return _store.status


def cleanup_expired_sessions() -> int:
    if isinstance(_store, MemorySessionStore):
        return _store.cleanup_expired()
    return 0
