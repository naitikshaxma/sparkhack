import re
import logging
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urljoin

from ..intents import ACTION_INTENTS, INFO_INTENTS, INTENT_SCHEME_QUERY
from backend.services.rag_service import recommend_schemes, recommend_schemes_with_reasons, retrieve_scheme_with_recommendations
from backend.domain.engines.decision import detect_user_need
from ..response_formatter import (
    build_quick_actions,
    build_recommendation_quick_actions,
    build_scheme_details,
    format_info_text,
)
from backend.core.metrics import record_fallback
from backend.core.logger import log_event
from backend.shared.language.language import normalize_language_code
from backend.shared.security.privacy import fingerprint_text, redact_sensitive_data, redact_sensitive_text
from backend.shared.performance.context_fusion import adaptive_confidence_thresholds, build_context_fusion
from backend.shared.session.form_schema import (
    get_default_scheme_for_category,
    get_field_question,
    get_fields_for_scheme,
    get_form_type_for_scheme,
    get_next_field,
    resolve_scheme_name,
    validate_field,
)
from ..infrastructure.session.session_store import create_session, delete_session, get_session, update_session
from backend.shared.validators.validator import validate
from ..text_normalizer import normalize_for_intent
from backend.shared.validators.input_validator import validate_input as security_validate_input
from .agent_service import run_agent
from .intent_service import IntentService, detect_intent_and_mode, is_followup_info_query
from .helpers.response_builder import (
    build_response_payload,
    display_aligned_text,
    format_response as format_response_helper,
    merge_control_actions,
    micro_latency_ack,
    short_answer as short_answer_helper,
)
from .helpers.intent_handler import (
    is_ambiguous_input,
    is_correction_request,
    is_generic_help_query,
    is_unclear_input,
    looks_like_field_value,
)
from .helpers.rag_handler import (
    adaptive_recommendation_limit,
    apply_recommendation_continuity,
    recommendation_suffix,
    smart_clarification_message,
)
from backend.shared.session.session_manager import (
    APPLY_INTENTS,
    STATE_IDLE,
    apply_state_transition,
    initialize_session_structure,
)
from backend.infrastructure.ml.scheme_registry import find_schemes_in_text, get_scheme_registry
from backend.infrastructure.database.connection import db_session_scope
from backend.infrastructure.database import ConversationHistory, Session as SessionModel
from backend.core.config import get_settings
import requests

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None


logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 10
MAX_SEMANTIC_MEMORY_ITEMS = 12
MAX_TEXT_INPUT_CHARS = max(64, int((os.getenv("MAX_TEXT_INPUT_CHARS") or "500").strip() or "500"))
AMBIGUOUS_WORDS = {"maybe", "around", "approx", "lagbhag", "approximately"}
YES_WORDS = {"yes", "ok", "okay", "haan", "ha", "haan ji", "ji", "theek hai", "ठीक है", "correct", "sahi", "right", "confirm", "confirmed"}
NO_WORDS = {"no", "nahin", "nahi", "galat", "wrong", "not correct", "cancel"}
UNCLEAR_WORDS = {"hmm", "uh", "hello", "helo", "sun", "listen", "something", "kuch", "pata nahi", "not sure"}
GENERIC_HELP_PATTERNS = {
    "loan batao",
    "loan",
    "scheme batao",
    "scheme",
    "madad chahiye",
    "help",
    "yojana",
    "help chahiye",
    "yojana batao",
    "koi scheme",
    "kuch scheme",
}

FIELD_LABELS = {
    "name": {"en": "Name", "hi": "नाम"},
    "full_name": {"en": "Full Name", "hi": "पूरा नाम"},
    "phone": {"en": "Phone", "hi": "मोबाइल नंबर"},
    "aadhaar": {"en": "Aadhaar", "hi": "आधार"},
    "aadhaar_number": {"en": "Aadhaar", "hi": "आधार नंबर"},
    "annual_income": {"en": "Annual Income", "hi": "वार्षिक आय"},
    "land_holding_acres": {"en": "Land Holding", "hi": "भूमि होल्डिंग"},
    "farmer_id": {"en": "Farmer ID", "hi": "किसान आईडी"},
    "health_card_number": {"en": "Health Card Number", "hi": "हेल्थ कार्ड नंबर"},
    "family_size": {"en": "Family Size", "hi": "परिवार का आकार"},
    "residential_status": {"en": "Residential Status", "hi": "आवासीय स्थिति"},
    "property_ownership": {"en": "Property Ownership", "hi": "संपत्ति स्वामित्व"},
}

NUMBER_ENTITY_RE = re.compile(r"\b\d{4,}\b")
INTENT_SERVICE = IntentService()
DIALOGUE_STATES = {"idle", "collecting_info", "confirming", "completed"}
EXTRACTION_AUTO_FILL_THRESHOLD = 0.72
MAX_INVALID_ATTEMPTS_PER_FIELD = 3
RATE_LIMIT_WINDOW_SECONDS = 5.0
RATE_LIMIT_MAX_REQUESTS = 5
MAX_RESPONSE_WORDS = 300
AUTOFILL_SERVICE_URL = (os.getenv("AUTOFILL_SERVICE_URL") or "http://127.0.0.1:8089").strip()
AUTOFILL_HEALTH_TIMEOUT_SECONDS = 1.0
AUTOFILL_REQUEST_TIMEOUT_SECONDS = 2.5
CORRECTION_PATTERNS = {
    "wrong",
    "change",
    "update",
    "edit",
    "not correct",
    "गलत",
    "बदल",
    "सुधार",
}

_history_persist_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="history-persist")
_autofill_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="autofill")
_settings = get_settings()
_rate_limit_redis_client = None
MVP_PIPELINE_ENABLED = (os.getenv("MVP_PIPELINE_ENABLED") or "1").strip().lower() not in {"0", "false", "no"}
ML_RAG_TIMEOUT_SECONDS = min(1.5, max(0.2, float((os.getenv("ML_RAG_TIMEOUT_SECONDS") or "1.5").strip() or "1.5")))


def _simple_fallback_text(user_input: str, language: str) -> Tuple[str, str]:
    lowered = str(user_input or "").lower()
    if "kisan" in lowered or "किसान" in str(user_input or ""):
        if language == "hi":
            return (
                "आपकी query किसान-related योजना से जुड़ सकती है। कृपया scheme का सटीक नाम या और विवरण बताएं।",
                "",
            )
        return (
            "Your query may relate to a farmer-focused scheme. Please share the exact scheme name or more details.",
            "",
        )
    if language == "hi":
        return (
            "मैं आपकी मदद के लिए तैयार हूँ। कृपया योजना का नाम या अपनी जरूरत बताएं, जैसे PM Kisan या आयुष्मान।",
            "",
        )
    return (
        "I am ready to help. Please share a scheme name or your need, like PM Kisan or Ayushman.",
        "",
    )


def _run_intent_with_timeout(cleaned_input: str) -> Dict[str, Any]:
    coroutine = INTENT_SERVICE.detect_async(cleaned_input, debug=False, timings={})
    return asyncio.run(asyncio.wait_for(coroutine, timeout=ML_RAG_TIMEOUT_SECONDS))


def _run_rag_with_timeout(cleaned_input: str, language: str, scheme_hint: str) -> Tuple[Optional[Dict[str, Any]], List[str], bool]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            retrieve_scheme_with_recommendations,
            cleaned_input,
            language,
            3,
            None,
            None,
            {"scheme_name": scheme_hint} if scheme_hint else None,
            None,
            None,
        )
        return future.result(timeout=ML_RAG_TIMEOUT_SECONDS)


def _get_rate_limit_redis_client():
    global _rate_limit_redis_client
    if _rate_limit_redis_client is not None:
        return _rate_limit_redis_client
    if redis is None:
        return None
    try:
        client = redis.Redis.from_url(_settings.redis_url, decode_responses=True)
        client.ping()
        _rate_limit_redis_client = client
        return _rate_limit_redis_client
    except Exception:
        return None


def _rate_limit_subject(session_id: str, session: Optional[Dict[str, Any]]) -> str:
    user_id = str((session or {}).get("user_id") or "").strip()
    if user_id:
        return f"user:{user_id}"
    return f"anon:{session_id}"


def _is_rate_limited(rate_key: str) -> bool:
    client = _get_rate_limit_redis_client()
    if client is None:
        return False
    redis_key = f"rate_limit:{rate_key}"
    try:
        raw_count = cast(Any, client.incr(redis_key))
        request_count = int(raw_count if isinstance(raw_count, int) else (raw_count or 0))
        if request_count == 1:
            client.expire(redis_key, int(RATE_LIMIT_WINDOW_SECONDS))
        return request_count > RATE_LIMIT_MAX_REQUESTS
    except Exception:
        return False


def _summarize_to_max_words(text: str, max_words: int = MAX_RESPONSE_WORDS) -> str:
    content = str(text or "").strip()
    if not content:
        return ""
    words = content.split()
    if len(words) <= max_words:
        return content
    trimmed = " ".join(words[:max_words]).rstrip(" ,.;:")
    return f"{trimmed}..."


def _autofill_fallback_message(language: str) -> str:
    if language == "hi":
        return "Aapka form ready hai. Kripya review karke submit karein."
    return "Your form is ready. Please review and submit."


def _is_autofill_command(user_input: str) -> bool:
    text = str(user_input or "").strip().lower()
    return text in {"auto fill form", "autofill", "auto-fill"}


def _autofill_health_ok() -> bool:
    health_url = urljoin(AUTOFILL_SERVICE_URL.rstrip("/") + "/", "health")
    try:
        response = requests.get(health_url, timeout=AUTOFILL_HEALTH_TIMEOUT_SECONDS)
        return response.status_code == 200
    except Exception:
        return False


def _call_autofill_service(session: Dict[str, Any]) -> Dict[str, Any]:
    autofill_url = urljoin(AUTOFILL_SERVICE_URL.rstrip("/") + "/", "autofill")
    payload = {
        "session": session,
    }
    response = requests.post(autofill_url, json=payload, timeout=AUTOFILL_REQUEST_TIMEOUT_SECONDS)
    if response.status_code >= 400:
        raise RuntimeError(f"autofill_http_{response.status_code}")
    data = response.json() if response.content else {}
    if isinstance(data, dict):
        return data
    return {}


def _run_autofill_with_timeout(session: Dict[str, Any], language: str) -> Dict[str, str]:
    if not _autofill_health_ok():
        return {
            "status": "failed",
            "message": _autofill_fallback_message(language),
        }

    future = _autofill_executor.submit(_call_autofill_service, dict(session or {}))
    try:
        result = future.result(timeout=AUTOFILL_REQUEST_TIMEOUT_SECONDS + 0.2)
        service_status = str((result or {}).get("status") or "").strip().lower()
        if service_status == "success":
            if language == "hi":
                return {
                    "status": "success",
                    "message": "Auto-fill complete. Kripya review karke submit karein.",
                }
            return {
                "status": "success",
                "message": "Auto-fill complete. Please review and submit.",
            }
        return {
            "status": "failed",
            "message": _autofill_fallback_message(language),
        }
    except Exception:
        return {
            "status": "failed",
            "message": _autofill_fallback_message(language),
        }


def _apply_response_length_control(payload: Dict[str, Any], max_words: int = MAX_RESPONSE_WORDS) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    for key in ("response_text", "voice_text", "confirmation", "explanation", "next_step"):
        if isinstance(payload.get(key), str):
            payload[key] = _summarize_to_max_words(payload[key], max_words=max_words)
    return payload


def _build_rate_limit_response(session_id: str, language: str, session: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text = "Please wait a moment before sending more requests." if language != "hi" else "कृपया थोड़ी देर रुककर फिर से अनुरोध भेजें।"
    return _build_response(
        session_id=session_id,
        response_text=text,
        field_name=None,
        validation_passed=True,
        validation_error=None,
        session_complete=False,
        mode="clarify",
        action="rate_limited",
        session=session,
        quick_actions=[],
        voice_text=text,
    )


def _persist_user_history(
    *,
    session_id: str,
    user_id: int,
    query_text: str,
    response_text: str,
    detected_scheme: str,
    intent: str,
) -> None:
    try:
        safe_query = redact_sensitive_data(query_text)
        safe_response = redact_sensitive_data(response_text)
        with db_session_scope() as db:
            existing_session = db.get(SessionModel, str(session_id or "")[:128])
            if existing_session is None:
                db.add(
                    SessionModel(
                        session_id=str(session_id or "")[:128],
                        user_id=user_id,
                        state_json={},
                    )
                )
                db.flush()

            db.add(
                ConversationHistory(
                    session_id=str(session_id or "")[:128],
                    user_id=user_id,
                    query=safe_query[:4000],
                    response=safe_response[:8000],
                    detected_scheme=detected_scheme[:255] if detected_scheme else None,
                    intent=intent[:128] if intent else None,
                    role="assistant",
                    message=safe_response[:4000],
                )
            )
    except Exception as exc:
        log_event(
            "conversation_history_persist_failure",
            level="warning",
            endpoint="conversation_service",
            status="failure",
            session_id=str(session_id or ""),
            user_id=str(user_id),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _persist_user_history_async(session_id: str, session: Dict[str, Any], user_input: str, result: Dict[str, Any]) -> None:
    user_id_raw = str(session.get("user_id") or "").strip()
    if not user_id_raw.isdigit():
        return

    scheme_detection = result.get("scheme_detection") or (result.get("debug") or {}).get("scheme_detection") or {}
    selected_scheme = str(
        scheme_detection.get("selected_scheme")
        or session.get("selected_scheme")
        or session.get("last_scheme")
        or ""
    ).strip()
    intent = str(result.get("primary_intent") or "").strip()
    response_text = str(result.get("response_text") or result.get("voice_text") or "").strip()
    query_text = str(user_input or "").strip()
    if not query_text and not response_text:
        return

    _persist_user_history(
        session_id=session_id,
        user_id=int(user_id_raw),
        query_text=query_text,
        response_text=response_text,
        detected_scheme=selected_scheme,
        intent=intent,
    )


def _normalize_history_user_id(session: Dict[str, Any]) -> Optional[int]:
    user_id_raw = str(session.get("user_id") or "").strip()
    if not user_id_raw.isdigit():
        return None
    return int(user_id_raw)


def _extract_category_from_scheme_name(scheme_name: str) -> str:
    text = str(scheme_name or "").strip().lower()
    if not text:
        return "general"

    category_keywords = {
        "agriculture": {"kisan", "krishi", "farmer", "crop", "agri"},
        "housing": {"housing", "house", "home", "awas", "rental"},
        "health": {"health", "bima", "insurance", "medical", "ayush", "care"},
        "education": {"student", "scholarship", "education", "vidya", "school"},
        "employment": {"employment", "job", "skill", "startup", "business", "self"},
        "finance": {"loan", "credit", "finance", "pension", "savings"},
    }

    for category, keywords in category_keywords.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "general"


def _fetch_user_history_context(session: Dict[str, Any]) -> Dict[str, Any]:
    user_id = _normalize_history_user_id(session)
    if user_id is None:
        return {"last_scheme": "", "top_category": "", "recent_schemes": []}

    with db_session_scope() as db:
        rows = (
            db.query(ConversationHistory)
            .filter(ConversationHistory.user_id == user_id)
            .order_by(ConversationHistory.timestamp.desc())
            .limit(30)
            .all()
        )

    if not rows:
        return {"last_scheme": "", "top_category": "", "recent_schemes": []}

    recent_schemes: List[str] = []
    category_counts: Dict[str, int] = {}
    last_scheme = ""

    for row in rows:
        scheme = str(getattr(row, "detected_scheme", "") or "").strip()
        if not scheme:
            continue
        if not last_scheme:
            last_scheme = scheme
        if scheme not in recent_schemes:
            recent_schemes.append(scheme)
        category = _extract_category_from_scheme_name(scheme)
        category_counts[category] = int(category_counts.get(category, 0)) + 1

    top_category = ""
    if category_counts:
        top_category = max(category_counts.items(), key=lambda item: item[1])[0]

    return {
        "last_scheme": last_scheme,
        "top_category": top_category,
        "recent_schemes": recent_schemes[:5],
    }


def _has_explicit_scheme_reference(user_input: str) -> bool:
    matches = _detect_scheme_mentions(user_input, limit=3)
    return any(float(item.get("score") or 0.0) >= 0.72 for item in matches)


def _is_vague_scheme_reference(user_input: str) -> bool:
    return is_vague_reference(user_input)


def is_vague_reference(text: str) -> bool:
    text = str(text or "").strip().lower()
    if not text:
        return False
    vague_markers = {
        "uska",
        "uske",
        "uski",
        "that",
        "that scheme",
        "that one",
        "same scheme",
        "woh",
        "wo",
        "wo scheme",
        "woh scheme",
        "us wali",
    }
    return text in vague_markers or any(marker in text for marker in vague_markers)


def _is_context_info_followup(text: str) -> bool:
    query = str(text or "").strip().lower()
    if not query:
        return False
    tokens = {
        "eligibility",
        "eligible",
        "documents",
        "document",
        "benefits",
        "benefit",
        "process",
        "apply process",
        "पात्र",
        "पात्रता",
        "दस्तावेज",
        "लाभ",
    }
    return any(token in query for token in tokens)


def _build_returning_user_prompt(language: str, scheme_name: str) -> str:
    scheme = str(scheme_name or "").strip()
    if language == "hi":
        return f"पिछली बार आपने {scheme} के बारे में पूछा था। क्या वहीं से जारी रखें?"
    return f"Last time you asked about {scheme}. Continue?"


def _session_fields(session: Dict[str, Any]) -> List[str]:
    selected_scheme = session.get("selected_scheme")
    if get_form_type_for_scheme(selected_scheme) == "generic" and bool(session.get("_force_minimal_generic_fields", False)):
        return ["full_name", "phone", "aadhaar_number"]
    return get_fields_for_scheme(selected_scheme)


def _detect_user_type(text: str) -> Optional[str]:
    query = (text or "").strip().lower()
    if any(token in query for token in {"farmer", "kisan", "किसान"}):
        return "farmer"
    if any(token in query for token in {"student", "विद्यार्थी", "छात्र"}):
        return "student"
    if any(token in query for token in {"business", "shop", "व्यापार", "कारोबार"}):
        return "business"
    return None


def _detect_income_range(text: str) -> Optional[str]:
    query = (text or "").strip().lower()
    if any(token in query for token in {"below", "under", "less than", "कम", "below 2", "under 2"}):
        return "low"
    if any(token in query for token in {"between", "mid", "मध्यम"}):
        return "mid"
    if any(token in query for token in {"high", "above", "more than", "ज्यादा"}):
        return "high"
    return None


def _update_user_need_profile(session: Dict[str, Any], user_input: str, need_category: Optional[str] = None) -> Dict[str, Optional[str]]:
    profile = dict(session.get("user_need_profile", {}))
    detected_user_type = _detect_user_type(user_input)
    detected_income_range = _detect_income_range(user_input)

    if detected_user_type:
        profile["user_type"] = detected_user_type
    if detected_income_range:
        profile["income_range"] = detected_income_range
    if need_category:
        profile["need_category"] = need_category

    profile.setdefault("user_type", None)
    profile.setdefault("income_range", None)
    profile.setdefault("need_category", None)
    session["user_need_profile"] = profile
    return profile


def _sanitize_user_profile_for_rag(profile: Dict[str, Optional[str]]) -> Dict[str, str]:
    return {key: value for key, value in profile.items() if isinstance(value, str) and value.strip()}


def _session_feedback(session: Dict[str, Any]) -> Dict[str, object]:
    learning_profile = session.get("learning_profile") or {}
    return {
        "rejected_schemes": list(session.get("rejected_schemes", [])),
        "accepted_scheme": session.get("accepted_scheme"),
        "accepted_category": (session.get("user_need_profile") or {}).get("need_category"),
        "rejected_counts": dict(learning_profile.get("rejected_counts", {})),
        "accepted_counts": dict(learning_profile.get("accepted_counts", {})),
    }


def _maybe_update_feedback_from_input(session: Dict[str, Any], user_input: str) -> None:
    text = (user_input or "").strip().lower()
    last_scheme = str(session.get("last_scheme") or "").strip()
    if not last_scheme:
        return

    rejected_markers = {"not this", "nahi", "nahin", "no", "another", "different scheme"}
    if any(marker in text for marker in rejected_markers):
        rejected = set(session.get("rejected_schemes", []))
        rejected.add(last_scheme)
        session["rejected_schemes"] = sorted(rejected)
        learning = session.setdefault("learning_profile", {"rejected_counts": {}, "accepted_counts": {}})
        rejected_counts = learning.setdefault("rejected_counts", {})
        key = str(last_scheme).strip().lower()
        rejected_counts[key] = int(rejected_counts.get(key, 0)) + 1


def _mark_accepted_scheme(session: Dict[str, Any], scheme_name: str) -> None:
    selected = str(scheme_name or "").strip()
    if not selected:
        return
    session["accepted_scheme"] = selected
    learning = session.setdefault("learning_profile", {"rejected_counts": {}, "accepted_counts": {}})
    accepted_counts = learning.setdefault("accepted_counts", {})
    key = selected.lower()
    accepted_counts[key] = int(accepted_counts.get(key, 0)) + 1


def _summarize_history_messages(messages: List[Dict[str, str]]) -> str:
    snippets: List[str] = []
    for item in messages[-6:]:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        snippets.append(f"{role}:{content[:80]}")
    return " | ".join(snippets)[:400]


def _push_clarification(session: Dict[str, Any], context: str) -> None:
    stack = session.setdefault("clarification_stack", [])
    entry = (context or "").strip()
    if not entry:
        return
    stack.append(entry)
    session["clarification_stack"] = stack[-5:]


def _pop_clarification(session: Dict[str, Any]) -> str:
    stack = session.setdefault("clarification_stack", [])
    if not stack:
        return ""
    context = str(stack.pop() or "").strip()
    session["clarification_stack"] = stack
    return context


def _safe_reset_session(session_id: str, language: str) -> Dict[str, Any]:
    delete_session(session_id)
    fresh = create_session(session_id)
    fresh = _normalize_session_state(fresh)
    fresh["language"] = language
    fresh["dialogue_state"] = "collecting_info"
    return fresh


def _trim_history(session: Dict[str, Any]) -> None:
    history = session.setdefault("conversation_history", [])
    if len(history) > MAX_HISTORY_MESSAGES:
        overflow = history[:-MAX_HISTORY_MESSAGES]
        existing_summary = str(session.get("history_summary") or "").strip()
        delta_summary = _summarize_history_messages(overflow)
        if delta_summary:
            session["history_summary"] = (f"{existing_summary} | {delta_summary}" if existing_summary else delta_summary)[:800]
        session["conversation_history"] = history[-MAX_HISTORY_MESSAGES:]


def _append_history(session: Dict[str, Any], role: str, content: str) -> None:
    if not content:
        return
    session.setdefault("conversation_history", []).append({"role": role, "content": redact_sensitive_text(content)})
    _trim_history(session)


def _detect_scheme_mentions(text: str, limit: int = 5) -> List[Dict[str, Any]]:
    forced = _forced_scheme_from_query(text)
    if forced:
        return [{"scheme": forced, "score": 1.0}]
    return find_schemes_in_text(text, limit=limit)


def _extract_explicit_scheme_phrase(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""
    lowered = source.lower()
    patterns = (
        r"(?:apply\s+for|application\s+for|enroll\s+for)\s+([a-z0-9][a-z0-9\s\-]{2,80})",
        r"(?:for)\s+([a-z0-9][a-z0-9\s\-]{2,80})\s+(?:scheme|yojana)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        phrase = re.sub(r"\b(scheme|yojana)\b", "", match.group(1)).strip(" .,!?:;-\t")
        phrase = re.sub(r"\s+", " ", phrase)
        if len(phrase) >= 3:
            return phrase
    return ""


def _prefer_explicit_scheme_match(text: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    query = str(text or "").strip().lower()
    if not query or not candidates:
        return None

    for item in candidates:
        scheme = str(item.get("scheme") or "").strip().lower()
        if scheme and scheme in query:
            return item

    explicit_phrase = _extract_explicit_scheme_phrase(query)
    if explicit_phrase:
        for item in candidates:
            scheme = str(item.get("scheme") or "").strip().lower()
            if not scheme:
                continue
            if explicit_phrase in scheme or scheme in explicit_phrase:
                return item
    return None


def _has_scheme_signal(text: str) -> bool:
    query = str(text or "").strip().lower()
    if not query:
        return False

    generic_tokens = {"scheme", "yojana", "apply", "eligibility", "eligible", "application"}
    if any(token in query for token in generic_tokens):
        return True

    return bool(_detect_scheme_mentions(query, limit=1))


def _is_broad_discovery_request(text: str) -> bool:
    query = str(text or "").strip().lower()
    if not query:
        return False
    markers = {
        "scheme",
        "yojana",
        "for family",
        "family",
        "kisan",
        "farmer",
        "batao",
        "suggest",
        "recommend",
    }
    return any(marker in query for marker in markers)


def _forced_scheme_from_query(text: str) -> str:
    return ""


def _fast_scheme_info_response(query: str, language: str, scheme_name: str) -> Dict[str, str]:
    normalized_query = str(query or "").strip().lower()
    lang = normalize_language_code(language, default="en")

    asks_process = any(token in normalized_query for token in {"kaise", "kya karna", "banega", "milega", "apply", "process", "application"})
    asks_documents = any(token in normalized_query for token in {"document", "documents", "dastavez", "kya chahiye", "required"})
    asks_benefits = any(token in normalized_query for token in {"kitna", "benefit", "benefits", "amount", "paisa"})

    if asks_documents:
        if lang == "hi":
            explanation = f"{scheme_name} के लिए आमतौर पर आधार, पता प्रमाण और आय संबंधी दस्तावेज़ चाहिए होते हैं।"
            next_step = "अगर चाहें तो मैं आवेदन के step-by-step process भी बताऊँ।"
        else:
            explanation = f"For {scheme_name}, you usually need Aadhaar, address proof, and income-related documents."
            next_step = "I can also share the step-by-step application process."
    elif asks_benefits:
        if lang == "hi":
            explanation = f"{scheme_name} के लाभ राज्य और पात्रता पर निर्भर करते हैं, लेकिन यह योजना आर्थिक/सेवा सहायता प्रदान करती है।"
            next_step = "अगर चाहें तो मैं eligibility और apply process भी बता दूँ।"
        else:
            explanation = f"Benefits under {scheme_name} depend on state rules and eligibility, but the scheme provides meaningful support."
            next_step = "I can also share eligibility and the apply process."
    elif asks_process:
        if lang == "hi":
            explanation = (
                f"{scheme_name} के लिए process: 1. eligibility check करें। "
                "2. required documents तैयार करें। 3. portal या CSC से apply/register करें।"
            )
            next_step = "अगर चाहें तो मैं documents की छोटी checklist भी दे सकता हूँ।"
        else:
            explanation = (
                f"Process for {scheme_name}: 1. Check eligibility. "
                "2. Keep required documents ready. 3. Apply/register on the portal or via CSC."
            )
            next_step = "I can also share a short documents checklist."
    else:
        if lang == "hi":
            explanation = f"{scheme_name} एक सरकारी योजना है जो पात्र नागरिकों को लाभ और सहायता प्रदान करती है।"
            next_step = "आप eligibility, documents, benefits या application process पूछ सकते हैं।"
        else:
            explanation = f"{scheme_name} is a government scheme that offers benefits and support to eligible citizens."
            next_step = "You can ask about eligibility, documents, benefits, or the application process."

    return {
        "confirmation": scheme_name,
        "explanation": explanation,
        "next_step": next_step,
    }


def _resolve_apply_target_scheme(
    session: Dict[str, Any],
    cleaned_input: str,
    mentioned_schemes: List[Dict[str, Any]],
    category_hint: Optional[str] = None,
) -> str:
    forced_scheme = _forced_scheme_from_query(cleaned_input)
    if forced_scheme:
        return resolve_scheme_name(forced_scheme)

    explicit_match = _prefer_explicit_scheme_match(cleaned_input, mentioned_schemes)
    if explicit_match and str(explicit_match.get("scheme") or "").strip():
        return resolve_scheme_name(explicit_match.get("scheme"))

    explicit_phrase = _extract_explicit_scheme_phrase(cleaned_input)
    if explicit_phrase:
        return resolve_scheme_name(explicit_phrase)

    if mentioned_schemes:
        top_candidate = max(mentioned_schemes, key=lambda item: float(item.get("score") or 0.0))
        if str(top_candidate.get("scheme") or "").strip():
            return resolve_scheme_name(top_candidate.get("scheme"))

    selected_default = resolve_scheme_name(get_default_scheme_for_category("general"))
    existing = resolve_scheme_name(session.get("selected_scheme") or session.get("current_scheme") or session.get("last_scheme"))
    if existing and existing != selected_default:
        return existing

    return resolve_scheme_name(get_default_scheme_for_category(category_hint))


def _scheme_detection_debug(session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    registry = get_scheme_registry()
    context = session or {}
    selected = str(
        context.get("selected_scheme")
        or context.get("current_scheme")
        or context.get("last_scheme")
        or ""
    ).strip()
    return {
        "input": str(context.get("_scheme_detection_input") or ""),
        "candidates": list(context.get("_scheme_detection_candidates") or []),
        "selected_scheme": selected,
        "decision": str(context.get("_scheme_detection_decision") or "none"),
        "total_available": int(registry.get("total", 0)),
    }


def _safety_debug(session: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    context = session or {}
    return {
        "low_confidence": bool(context.get("_safety_low_confidence", False)),
        "ambiguous": bool(context.get("_safety_ambiguous", False)),
        "fallback_triggered": bool(context.get("_safety_fallback_triggered", False)),
    }


def _extract_entities(text: str) -> Dict[str, List[str]]:
    content = (text or "").strip()
    if not content:
        return {"schemes": [], "numbers": []}

    candidates = _detect_scheme_mentions(content, limit=5)
    schemes = [str(item.get("scheme") or "").lower() for item in candidates if str(item.get("scheme") or "").strip()]
    numbers = [match.group(0) for match in NUMBER_ENTITY_RE.finditer(content)]
    return {
        "schemes": sorted(set(schemes))[:5],
        "numbers": sorted(set(numbers))[:5],
    }


def _update_semantic_memory(session: Dict[str, Any], user_input: str, response: Dict[str, Any], intent: str) -> None:
    entities = _extract_entities(user_input)
    semantic = session.setdefault("semantic_memory", [])
    semantic.append(
        {
            "ts": int(time.time()),
            "intent": (intent or "").strip(),
            "entities": entities,
            "user_input": (user_input or "").strip()[:200],
            "assistant_summary": (response.get("voice_text") or response.get("response_text") or "")[:240],
        }
    )
    session["semantic_memory"] = semantic[-MAX_SEMANTIC_MEMORY_ITEMS:]
    # Fast-access context hints for future turns.
    if entities.get("schemes"):
        session["memory_last_scheme_entities"] = entities["schemes"]
    session["memory_last_intent"] = (intent or "").strip()


def update_semantic_memory(session: Dict[str, Any], user_input: str, response: Dict[str, Any], intent: str) -> None:
    _update_semantic_memory(session, user_input, response, intent)


def _build_response(
    session_id: str,
    response_text: str,
    field_name: Optional[str],
    validation_passed: bool,
    session_complete: bool,
    validation_error: Optional[str] = None,
    mode: str = "action",
    action: Optional[str] = None,
    session: Optional[Dict[str, Any]] = None,
    scheme_details: Optional[Dict[str, Any]] = None,
    voice_text: Optional[str] = None,
    quick_actions: Optional[List[Dict[str, str]]] = None,
    recommended_schemes: Optional[List[str]] = None,
    autofill_status: Optional[str] = None,
) -> Dict[str, Any]:
    form_type = "generic"
    if isinstance(session, dict):
        form_type = get_form_type_for_scheme(session.get("selected_scheme") or session.get("last_scheme"))

    payload = build_response_payload(
        session_id=session_id,
        response_text=response_text,
        field_name=field_name,
        validation_passed=validation_passed,
        session_complete=session_complete,
        validation_error=validation_error,
        mode=mode,
        action=action,
        session=session,
        scheme_details=scheme_details,
        voice_text=voice_text,
        quick_actions=quick_actions,
        recommended_schemes=recommended_schemes,
        field_labels=FIELD_LABELS,
        session_fields=_session_fields(session or {}) if session else [],
    )
    scheme_debug = _scheme_detection_debug(session)
    safety_debug = _safety_debug(session)
    payload["scheme_detection"] = scheme_debug
    payload["safety"] = safety_debug
    payload["form_type"] = form_type
    payload["apply_flow_forced"] = bool((session or {}).get("_apply_flow_forced", False))
    payload["confirmation_handled"] = bool((session or {}).get("_confirmation_handled", False))
    payload["context_applied"] = bool((session or {}).get("_context_applied", False))
    payload["autofill_status"] = str(autofill_status) if autofill_status is not None else None
    debug_payload = payload.setdefault("debug", {})
    debug_payload["scheme_detection"] = scheme_debug
    debug_payload["safety"] = safety_debug
    debug_payload["form_type"] = form_type
    debug_payload["apply_flow_forced"] = bool((session or {}).get("_apply_flow_forced", False))
    debug_payload["confirmation_handled"] = bool((session or {}).get("_confirmation_handled", False))
    debug_payload["context_applied"] = bool((session or {}).get("_context_applied", False))
    debug_payload["autofill_status"] = str(autofill_status) if autofill_status is not None else None
    return payload


def format_response(text: str, language: str) -> str:
    return format_response_helper(text, language)


def _normalize_mixed_input_text(user_input: str) -> str:
    text = str(user_input or "").strip()
    if not text:
        return ""
    text = text.replace("।", ".")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_affirmative(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    if not text:
        return False
    if text in YES_WORDS:
        return True
    markers = {"yes", "ok", "okay", "haan", "theek", "ठीक", "confirm", "sahi"}
    return any(marker in text for marker in markers)


def _is_negative(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    if not text:
        return False
    if text in NO_WORDS:
        return True
    markers = {"no", "nah", "nahi", "nahin", "गलत", "wrong", "cancel"}
    return any(marker in text for marker in markers)


def _micro_latency_ack(language: str) -> str:
    return micro_latency_ack(language)


def _merge_control_actions(language: str, quick_actions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return merge_control_actions(language, quick_actions)


def _is_short_query(text: str) -> bool:
    tokens = [token for token in (text or "").strip().split() if token]
    return len(tokens) <= 4


def _display_aligned_text(text: str, language: str) -> str:
    return display_aligned_text(text, language)


def _short_answer(text: str, language: str) -> str:
    return short_answer_helper(text, language)


def _recommendation_confirmation_prompt(language: str) -> str:
    if language == "hi":
        return "क्या ये सुझाव सही लग रहे हैं, या मैं इन्हें और बेहतर करके दिखाऊँ?"
    return "Do these suggestions look right, or should I refine them further?"


def _confidence_explanation_line(language: str, reason: str) -> str:
    detail = (reason or "").strip()
    if language == "hi":
        return f"मैंने ये सुझाव आपकी बात और प्रोफ़ाइल के आधार पर दिए हैं। {detail}".strip()
    return f"I suggested this based on your request and profile context. {detail}".strip()


def _closing_summary(session: Dict[str, Any], language: str) -> str:
    scheme = str(session.get("last_scheme") or session.get("accepted_scheme") or "").strip()
    if language == "hi":
        if scheme:
            return f"आज हमने {scheme} पर आपकी मदद पूरी की। अगर चाहें तो अगले कदम में भी मैं साथ हूँ।"
        return "आज की प्रक्रिया आराम से पूरी हो गई। अगर चाहें तो मैं आगे भी मदद के लिए हूँ।"
    if scheme:
        return f"We completed {scheme} together. I can help you with the next step too."
    return "We completed this step smoothly. I am here if you need anything else."


def _detect_language(user_input: str) -> str:
    text = (user_input or "").strip().lower()
    hindi_hinglish_tokens = {
        "kya",
        "mera",
        "meri",
        "aap",
        "kripya",
        "haan",
        "nahi",
        "ji",
        "aadhaar",
        "batayein",
        "wapas",
        "jao",
    }
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    if any(token in text for token in hindi_hinglish_tokens):
        return "hi"
    return "en"


def _is_restart_command(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    return text in {"restart", "reset", "start over", "new form", "phir se", "dobara"}


def _is_go_back_command(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    return text in {"go back", "back", "wapas jao", "pichla", "previous"}


def _is_skip_command(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    return text in {"skip", "chhodo", "chod do", "aage bado", "next"}


def _resolve_quick_action_input(user_input: str, language: str, session: Dict[str, Any]) -> str:
    raw = (user_input or "").strip()
    if not raw:
        return raw

    lowered = raw.lower()
    if lowered.startswith("recommend_scheme:"):
        scheme = raw.split(":", 1)[1].strip()
        return scheme or raw

    last_scheme = session.get("last_scheme")
    mapping = {
        "need_information": "जानकारी चाहिए" if language == "hi" else "Need information",
        "start_application": "आवेदन शुरू करें" if language == "hi" else "Start application",
        "apply_now": "आवेदन शुरू करें" if language == "hi" else "Apply now",
        "more_info": "और जानकारी" if language == "hi" else "More info",
        "show_eligibility": (
            f"{last_scheme} पात्रता" if language == "hi" and last_scheme else
            f"{last_scheme} eligibility" if last_scheme else
            "पात्रता बताएं" if language == "hi" else
            "Show eligibility"
        ),
        "confirm_yes": "हाँ" if language == "hi" else "yes",
        "confirm_no": "नहीं" if language == "hi" else "no",
        "next_step": "next",
        "application_status": "status",
        "continue_flow": "next",
        "refine_suggestions": "different scheme",
        "apply_now_direct": "start application",
        "restart_session": "restart",
        "auto_fill_form": "auto fill form",
    }
    return mapping.get(lowered, raw)


def _reset_session_state(session: Dict[str, Any]) -> Dict[str, Any]:
    # Kept for compatibility, but hard reset now uses delete + create.
    selected_scheme = resolve_scheme_name(session.get("selected_scheme") or session.get("last_scheme"))
    if get_form_type_for_scheme(selected_scheme) == "generic" and bool(session.get("_force_minimal_generic_fields", False)):
        dynamic_fields = ["full_name", "phone", "aadhaar_number"]
    else:
        dynamic_fields = get_fields_for_scheme(selected_scheme)
    field_completion = {field: False for field in dynamic_fields}
    session["selected_scheme"] = selected_scheme
    session["user_profile"] = {}
    session["field_completion"] = field_completion
    session["next_field"] = dynamic_fields[0] if dynamic_fields else None
    session["session_complete"] = False
    session["conversation_history"] = []
    session["last_completed_field_index"] = -1
    session["confirmation_done"] = False
    session["confirmation_state"] = "pending"
    return session


def _normalize_session_state(session: Dict[str, Any]) -> Dict[str, Any]:
    # Protect against corrupted or partially missing session payloads.
    selected_scheme = resolve_scheme_name(session.get("selected_scheme") or session.get("last_scheme"))
    session["selected_scheme"] = selected_scheme
    if get_form_type_for_scheme(selected_scheme) == "generic" and bool(session.get("_force_minimal_generic_fields", False)):
        dynamic_fields = ["full_name", "phone", "aadhaar_number"]
    else:
        dynamic_fields = get_fields_for_scheme(selected_scheme)

    session.setdefault("user_profile", {})
    session.setdefault("field_completion", {field: False for field in dynamic_fields})
    session["field_completion"] = {
        field: bool(session.get("field_completion", {}).get(field, False))
        for field in dynamic_fields
    }

    session.setdefault("conversation_history", [])
    session.setdefault("session_complete", False)
    session.setdefault("confirmation_done", False)
    session.setdefault("confirmation_state", "pending")
    session.setdefault("last_completed_field_index", -1)
    session.setdefault("ocr_extracted", {"fields": [], "confidence": 0.0})
    session.setdefault("ocr_confirmation_pending", False)
    session.setdefault("ocr_pending_fields", [])
    session.setdefault("action_confirmation_pending", False)
    session.setdefault("user_need_profile", {"user_type": None, "income_range": None, "need_category": None})
    session.setdefault("history_summary", "")
    session.setdefault("past_need_confidence", None)
    session.setdefault("onboarding_done", False)
    session.setdefault("rejected_schemes", [])
    session.setdefault("accepted_scheme", None)
    session.setdefault("last_recommendation_reason", None)
    session.setdefault("learning_profile", {"rejected_counts": {}, "accepted_counts": {}})
    session.setdefault("dialogue_state", "idle")
    session.setdefault("clarification_pending", False)
    session.setdefault("clarification_context", "")
    session.setdefault("clarification_stack", [])
    session.setdefault("invalid_attempts", {})
    session.setdefault("extraction_conflicts", {})

    next_field = session.get("next_field")
    if next_field not in dynamic_fields and next_field is not None:
        next_field = get_next_field(session)
    if next_field is None and not session.get("session_complete", False):
        next_field = get_next_field(session)
    session["next_field"] = next_field
    initialize_session_structure(session)
    return session


def _sync_state_machine_fields_to_profile(session: Dict[str, Any]) -> None:
    collected = dict(session.get("collected_fields") or {})
    if not collected:
        return

    mapping = {
        "name": "full_name",
        "phone": "phone",
        "aadhaar": "aadhaar_number",
    }
    profile = session.setdefault("user_profile", {})
    completion = session.setdefault("field_completion", {})
    for key, value in collected.items():
        target = mapping.get(key)
        if not target or not value:
            continue
        profile[target] = str(value)
        completion[target] = True


def _update_dialogue_state(session: Dict[str, Any]) -> str:
    current = str(session.get("dialogue_state") or "idle").strip().lower()
    if current not in DIALOGUE_STATES:
        current = "idle"

    next_field = get_next_field(session)
    confirmation_state = str(session.get("confirmation_state") or "pending").strip().lower()

    if session.get("session_complete") or confirmation_state == "confirmed":
        current = "completed"
    elif confirmation_state == "pending" and next_field is None:
        current = "confirming"
    elif next_field is not None:
        current = "collecting_info"
    else:
        current = "idle"

    session["dialogue_state"] = current
    return current


def _is_correction_request(user_input: str) -> bool:
    return is_correction_request(user_input)


def _unique_candidates(candidates: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    seen: Dict[str, float] = {}
    for value, conf in candidates:
        key = str(value or "").strip()
        if not key:
            continue
        seen[key] = max(float(conf), seen.get(key, 0.0))
    return [{"value": key, "confidence": round(score, 3)} for key, score in seen.items()]


def _extract_multi_field_values(text: str) -> Dict[str, List[Dict[str, Any]]]:
    content = (text or "").strip()
    lowered = content.lower()
    extracted: Dict[str, List[Dict[str, Any]]] = {}

    phones = re.findall(r"(?<!\d)(\d{10})(?!\d)", content)
    if phones:
        phone_conf = 0.94 if len(set(phones)) == 1 else 0.76
        extracted["phone"] = _unique_candidates([(value, phone_conf) for value in phones])

    aadhaars = re.findall(r"(?<!\d)(\d{12})(?!\d)", content)
    if aadhaars:
        aadhaar_conf = 0.96 if len(set(aadhaars)) == 1 else 0.74
        extracted["aadhaar_number"] = _unique_candidates([(value, aadhaar_conf) for value in aadhaars])

    incomes = re.findall(r"(?:income|आय|salary|कमाई)\D{0,12}(\d+(?:,\d{3})*(?:\.\d+)?)", lowered)
    if incomes:
        extracted["annual_income"] = _unique_candidates([(value.replace(",", ""), 0.86) for value in incomes])

    name_match = re.search(r"(?:my name is|i am|mera naam|name[:\s]|मेरा नाम|नाम)\s*[:\-]?\s*([A-Za-z\u0900-\u097F\s.'-]{2,80})", content, re.IGNORECASE)
    if name_match:
        extracted["full_name"] = _unique_candidates([(name_match.group(1).strip(), 0.84)])

    return extracted


def detect_information_input(user_input: str) -> bool:
    text = str(user_input or "").strip()
    if not text:
        return False

    lowered = text.lower()
    if re.fullmatch(r"\d{10}|\d{12}", re.sub(r"\D", "", text)):
        return True

    info_keywords = (
        "aadhaar",
        "aadhar",
        "phone",
        "mobile",
        "my name",
        "name is",
        "income",
        "salary",
        "मेरा नाम",
        "आधार",
        "मोबाइल",
        "फोन",
        "आय",
    )
    if any(token in lowered for token in info_keywords):
        return True

    extracted = _extract_multi_field_values(text)
    return bool(extracted)


def _apply_info_detection_to_profile(session: Dict[str, Any], user_input: str, language: str) -> None:
    extracted = _extract_multi_field_values(user_input)
    if not extracted:
        return

    profile = session.setdefault("user_profile", {})
    completion = session.setdefault("field_completion", {})
    active_fields = set(_session_fields(session))

    for field, candidates in extracted.items():
        if field not in active_fields:
            continue
        values = [str(item.get("value") or "").strip() for item in candidates if str(item.get("value") or "").strip()]
        if not values:
            continue
        candidate = values[0]
        validated = validate_field(field, candidate, language=language)
        if not validated.get("valid"):
            continue
        profile[field] = str(validated.get("normalized") or "")
        completion[field] = True

    next_field = get_next_field(session)
    session["next_field"] = next_field
    if next_field is None:
        session["dialogue_state"] = "confirming"
    else:
        session["dialogue_state"] = "collecting_info"


def _apply_extracted_fields(session: Dict[str, Any], extracted: Dict[str, List[Dict[str, Any]]], language: str) -> Dict[str, Any]:
    active_fields = _session_fields(session)
    completion = session.setdefault("field_completion", {})
    profile = session.setdefault("user_profile", {})
    applied: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    conflicts: Dict[str, List[str]] = {}
    low_confidence: Dict[str, float] = {}

    for field, candidates in extracted.items():
        if field not in active_fields or completion.get(field):
            continue

        values = [str(item.get("value") or "").strip() for item in candidates if str(item.get("value") or "").strip()]
        if len(set(values)) > 1:
            conflicts[field] = sorted(set(values))[:4]
            continue

        if not candidates:
            continue

        best = max(candidates, key=lambda item: float(item.get("confidence") or 0.0))
        best_value = str(best.get("value") or "").strip()
        best_confidence = float(best.get("confidence") or 0.0)

        if best_confidence < EXTRACTION_AUTO_FILL_THRESHOLD:
            low_confidence[field] = best_confidence
            continue

        result = validate_field(field, best_value, language=language)
        if result.get("valid"):
            profile[field] = str(result.get("normalized") or "")
            completion[field] = True
            applied[field] = str(result.get("normalized") or "")
        else:
            errors[field] = str(result.get("error_message") or "")

    if applied:
        last_field = None
        for field in active_fields:
            if completion.get(field):
                last_field = field
        if last_field in active_fields:
            session["last_completed_field_index"] = active_fields.index(last_field)

    return {
        "applied": applied,
        "errors": errors,
        "conflicts": conflicts,
        "low_confidence": low_confidence,
    }


def merge_ocr_data(session: Dict[str, Any], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    session = _normalize_session_state(session)
    updated_fields = []
    user_profile = session.setdefault("user_profile", {})
    field_completion = session.setdefault("field_completion", {})

    for field in ["full_name", "aadhaar_number"]:
        value = extracted_data.get(field)
        if value is None:
            continue
        if field_completion.get(field, False) or user_profile.get(field):
            continue

        is_valid, normalized_value, _ = validate(field, str(value))
        if not is_valid:
            continue

        user_profile[field] = normalized_value
        field_completion[field] = True
        updated_fields.append(field)

    if updated_fields:
        active_fields = _session_fields(session)
        session["last_completed_field_index"] = max(active_fields.index(f) for f in updated_fields if f in active_fields)

    confidence_raw = extracted_data.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.0

    session["ocr_extracted"] = {"fields": updated_fields, "confidence": confidence}
    session["ocr_pending_fields"] = list(updated_fields)
    session["ocr_confirmation_pending"] = bool(updated_fields)
    session["next_field"] = get_next_field(session)
    session["session_complete"] = False
    session["confirmation_done"] = False
    session["confirmation_state"] = "pending"
    return session


def _build_ocr_confirmation_text(session: Dict[str, Any], ocr_data: Dict[str, Any], language: str) -> str:
    profile = session.get("user_profile", {})
    name = profile.get("full_name") or "-"
    aadhaar = profile.get("aadhaar_number") or "-"
    dob = ocr_data.get("date_of_birth") or "-"
    if language == "hi":
        return (
            "मैंने आपका दस्तावेज़ स्कैन किया:\n"
            f"नाम: {name}\n"
            f"आधार: {aadhaar}\n"
            f"जन्म तिथि: {dob}\n"
            "क्या यह सही है?"
        )
    return (
        "I scanned your document:\n"
        f"Name: {name}\n"
        f"Aadhaar: {aadhaar}\n"
        f"Date of Birth: {dob}\n"
        "Is this correct?"
    )


def get_ocr_confirmation_message(session: Dict[str, Any], ocr_data: Dict[str, Any], language: str) -> str:
    return _build_ocr_confirmation_text(session, ocr_data, language)


def _handle_ocr_confirmation(session_id: str, session: Dict[str, Any], user_input: str, language: str) -> Dict[str, Any]:
    cleaned_input = _normalize_mixed_input_text(user_input).lower()

    if _is_affirmative(cleaned_input):
        session["ocr_confirmation_pending"] = False
        session["ocr_pending_fields"] = []

        next_field = get_next_field(session)
        session["next_field"] = next_field
        if next_field is None:
            session["confirmation_state"] = "pending"
            confirmation_text = _build_confirmation_summary(session, language)
            _append_history(session, "assistant", confirmation_text)
            update_session(session_id, session)
            return _build_response(session_id, confirmation_text, None, True, False, None, session=session)

        question = get_field_question(next_field, language)
        _append_history(session, "assistant", question)
        update_session(session_id, session)
        return _build_response(session_id, question, next_field, True, False, None, session=session)

    if _is_negative(cleaned_input) or _is_go_back_command(cleaned_input) or "correct" in cleaned_input:
        for field in session.get("ocr_pending_fields", []):
            session.setdefault("user_profile", {}).pop(field, None)
            session.setdefault("field_completion", {})[field] = False

        session["ocr_confirmation_pending"] = False
        session["ocr_pending_fields"] = []
        session["next_field"] = get_next_field(session)
        session["session_complete"] = False

        next_field = session.get("next_field")
        question = get_field_question(next_field, language)
        guide = (
            f"ठीक है, हम इसे मैन्युअली भरते हैं। {question}"
            if language == "hi"
            else f"Sure, let us fill it manually. {question}"
        )
        _append_history(session, "assistant", guide)
        update_session(session_id, session)
        return _build_response(session_id, guide, next_field, True, False, None, session=session)

    prompt = (
        "यदि विवरण सही हैं तो कृपया हाँ कहें, अन्यथा नहीं कहें।"
        if language == "hi"
        else "Please reply yes if details are correct, otherwise say no."
    )
    _append_history(session, "assistant", prompt)
    update_session(session_id, session)
    return _build_response(session_id, prompt, None, False, False, "ocr_confirmation_pending", session=session)


def _is_ambiguous_input(user_input: str) -> bool:
    return is_ambiguous_input(user_input)


def _is_unclear_input(user_input: str) -> bool:
    return is_unclear_input(user_input)


def _is_generic_help_query(user_input: str) -> bool:
    return is_generic_help_query(user_input)


def _is_apply_intent_signal(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    if not text:
        return False
    markers = {
        "apply",
        "application",
        "apply scheme",
        "start application",
        "start form",
        "fill form",
        "enroll",
        "registration",
        "register",
        "start_application",
        "apply_now",
    }
    return any(marker in text for marker in markers)


def _recommendation_suffix(language: str, recommendations: List[str]) -> str:
    return recommendation_suffix(language, recommendations)


def _smart_clarification_message(language: str, recommendations: List[str], user_input: str = "") -> str:
    return smart_clarification_message(language, recommendations, user_input)


def _adaptive_recommendation_limit(confidence: float, low_threshold: float, high_threshold: float) -> int:
    return adaptive_recommendation_limit(confidence, low_threshold, high_threshold)


def _apply_recommendation_continuity(session: Dict[str, Any], recommendations: List[str]) -> List[str]:
    return apply_recommendation_continuity(session, recommendations)


def _looks_like_field_value(field_name: Optional[str], user_input: str) -> bool:
    return looks_like_field_value(field_name, user_input)


def _build_confirmation_summary(session: Dict[str, Any], language: str) -> str:
    profile = session.get("user_profile", {})
    active_fields = _session_fields(session)
    parts: List[str] = []
    for field in active_fields:
        label = FIELD_LABELS.get(field, {}).get(language, field)
        value = profile.get(field)
        parts.append(f"{label}: {value if value not in {None, ''} else '-'}")

    joined = ", ".join(parts)

    if language == "hi":
        return f"मैं पुष्टि करता हूँ: {joined}. क्या यह सही है?"
    return f"Let me confirm: {joined}. Is this correct?"


def _move_to_previous_field(session: Dict[str, Any]) -> Optional[str]:
    last_index = int(session.get("last_completed_field_index", -1))
    active_fields = _session_fields(session)
    if last_index < 0:
        return session.get("next_field")
    if last_index >= len(active_fields):
        last_index = len(active_fields) - 1
    if last_index < 0:
        return session.get("next_field")

    previous_field = active_fields[last_index]
    session.setdefault("user_profile", {}).pop(previous_field, None)
    session.setdefault("field_completion", {})[previous_field] = False
    session["next_field"] = previous_field
    session["session_complete"] = False
    session["confirmation_done"] = False
    session["confirmation_state"] = "pending"
    session["last_completed_field_index"] = max(-1, last_index - 1)
    return previous_field


def _confirmation_handler(
    session_id: str,
    session: Dict[str, Any],
    user_input: str,
    language: str,
) -> Dict[str, Any]:
    cleaned_input = _normalize_mixed_input_text(user_input).lower()
    session["_confirmation_handled"] = True

    if _is_restart_command(cleaned_input):
        fresh = _safe_reset_session(session_id, language)
        question = get_field_question(fresh.get("next_field"), language)
        _append_history(fresh, "assistant", question)
        update_session(session_id, fresh)
        return _build_response(session_id, question, fresh.get("next_field"), True, False, None, session=fresh)

    if _is_go_back_command(cleaned_input):
        previous = _move_to_previous_field(session)
        question = get_field_question(previous, language)
        _append_history(session, "assistant", question)
        update_session(session_id, session)
        return _build_response(session_id, question, previous, True, False, None, session=session)

    if _is_affirmative(cleaned_input):
        session["confirmation_done"] = True
        session["confirmation_state"] = "confirmed"
        session["session_complete"] = True
        session["dialogue_state"] = "completed"
        done_text = "आपका फॉर्म सफलतापूर्वक पूरा हो गया है।" if language == "hi" else "Your form is completed successfully."
        _append_history(session, "assistant", done_text)
        update_session(session_id, session)
        return _build_response(session_id, done_text, None, True, True, None, session=session)

    if _is_negative(cleaned_input) or "change" in cleaned_input:
        session["confirmation_done"] = False
        session["confirmation_state"] = "pending"
        previous = _move_to_previous_field(session)
        session["dialogue_state"] = "collecting_info"
        question = get_field_question(previous, language)
        guide = (
            f"ठीक है, हम इसे ठीक करते हैं। {question}"
            if language == "hi"
            else f"Sure, let us correct it. {question}"
        )
        _append_history(session, "assistant", guide)
        update_session(session_id, session)
        return _build_response(session_id, guide, previous, True, False, None, session=session)

    prompt = (
        "Please reply with yes to confirm, or use go back/restart to make corrections."
        if language == "en"
        else "कृपया पुष्टि के लिए हाँ कहें, या संशोधन के लिए वापस जाएँ/पुनः प्रारंभ करें।"
    )
    _append_history(session, "assistant", prompt)
    update_session(session_id, session)
    return _build_response(session_id, prompt, None, False, False, "confirmation_pending", session=session)


def _validation_error_message(field: str, error_message: str, language: str) -> str:
    if field == "phone":
        return (
            "मोबाइल नंबर सही नहीं है। उदाहरण: 9876543210"
            if language == "hi"
            else "That mobile number looks invalid. Example: 9876543210"
        )
    if field == "aadhaar_number":
        return (
            "आधार नंबर सही नहीं है। उदाहरण: 123412341234"
            if language == "hi"
            else "That Aadhaar number looks invalid. Example: 123412341234"
        )
    if field == "annual_income":
        return (
            "आय सिर्फ अंकों में बताएं। उदाहरण: 250000"
            if language == "hi"
            else "Please share income in numbers only. Example: 250000"
        )
    if language == "hi":
        return "इनपुट अमान्य है, कृपया दोबारा बताएं।"
    return "Invalid input. Please try again."


def _clarification_message(language: str) -> str:
    if language == "hi":
        return "आप चाहें तो पहले योजना समझ लेते हैं, या अभी आवेदन शुरू कर सकते हैं।"
    return "We can first review the scheme details, or start your application right away."


def _action_start_confirmation_message(language: str) -> str:
    if language == "hi":
        return "मैं आवेदन फॉर्म शुरू कर दूँ? बस हाँ या नहीं कह दीजिए।"
    return "Shall I start your application form now? Just say yes or no."
    
def _resolve_extraction_conflicts(session: Dict[str, Any], user_input: str, language: str) -> bool:
    conflicts = dict(session.get("extraction_conflicts") or {})
    if not conflicts:
        return False

    field = next(iter(conflicts.keys()))
    options = [str(v).strip() for v in conflicts.get(field, []) if str(v).strip()]
    text = (user_input or "").strip()
    selected = None

    ordinal_map = {
        "first": 0,
        "1st": 0,
        "पहला": 0,
        "pehla": 0,
        "second": 1,
        "2nd": 1,
        "दूसरा": 1,
        "dusra": 1,
        "third": 2,
        "3rd": 2,
        "तीसरा": 2,
        "teesra": 2,
    }

    number_match = re.search(r"\b([1-9])\b", text)
    if number_match:
        idx = int(number_match.group(1)) - 1
        if 0 <= idx < len(options):
            selected = options[idx]

    lowered_text = text.lower()
    if selected is None:
        for token, idx in ordinal_map.items():
            if token in lowered_text and 0 <= idx < len(options):
                selected = options[idx]
                break

    if not selected:
        for option in options:
            if option and option in text:
                selected = option
                break

    # Correction override: "not this, use that" prefers latest explicit value.
    override_match = re.search(r"(?:not this|गलत|nahi)\s*,?\s*(?:use|choose|select|lo|ले)\s+(.+)$", lowered_text)
    if override_match and options:
        tail = override_match.group(1).strip()
        for option in options:
            if option.lower() in tail:
                selected = option
                break
        if selected is None:
            selected = options[-1]

    if not selected:
        option_text = "; ".join(f"{i + 1}. {value}" for i, value in enumerate(options))
        prompt = (
            f"मुझे {FIELD_LABELS.get(field, {}).get('hi', field)} के लिए कई मान मिले। सही विकल्प चुनें: {option_text}"
            if language == "hi"
            else f"I found multiple values for {FIELD_LABELS.get(field, {}).get('en', field)}. Please choose: {option_text}"
        )
        session["pending_conflict_prompt"] = prompt
        return False

    validated = validate_field(field, selected, language=language)
    if not validated.get("valid"):
        return False

    session.setdefault("user_profile", {})[field] = str(validated.get("normalized") or "")
    session.setdefault("field_completion", {})[field] = True
    conflicts.pop(field, None)
    session["extraction_conflicts"] = conflicts
    session.pop("pending_conflict_prompt", None)
    return True


def handle_conversation(session_id: str, user_input: str, language: Optional[str] = None, debug: bool = False) -> Dict[str, Any]:
    try:
        session = get_session(session_id)
    except Exception:
        session = create_session(session_id)

    session = _normalize_session_state(session)

    validated_input = security_validate_input(user_input or "", max_chars=MAX_TEXT_INPUT_CHARS)
    if not validated_input.is_valid:
        raise ValueError(validated_input.rejected_reason or "Invalid user input.")

    # Use normalized text to avoid double escaping when upstream routes already sanitize payloads.
    cleaned_input = _normalize_mixed_input_text(validated_input.normalized_text)
    user_history_context = _fetch_user_history_context(session)
    history_last_scheme = str(user_history_context.get("last_scheme") or "").strip()
    if history_last_scheme and not str(session.get("last_scheme") or "").strip():
        session["last_scheme"] = history_last_scheme
    last_scheme = str(session.get("last_scheme") or session.get("selected_scheme") or session.get("current_scheme") or "").strip()
    session["_context_applied"] = False
    if is_vague_reference(cleaned_input) and last_scheme and not _has_explicit_scheme_reference(cleaned_input):
        cleaned_input = f"{last_scheme} {cleaned_input}".strip()
        session["_context_applied"] = True
    context_applied = bool(session.get("_context_applied", False))

    _update_dialogue_state(session)
    session["info_detected"] = False
    session["_confirmation_handled"] = False

    pending_next_field = get_next_field(session)
    if pending_next_field is None and not bool(session.get("session_complete")):
        session["dialogue_state"] = "confirming"
        session["confirmation_state"] = "pending"

    if str(session.get("dialogue_state") or "").strip().lower() == "confirming":
        entry_language = normalize_language_code(language or session.get("language") or _detect_language(cleaned_input), default="en")
        session["language"] = entry_language
        _append_history(session, "user", cleaned_input)
        return _confirmation_handler(session_id, session, cleaned_input, entry_language)

    entry_language = normalize_language_code(language or session.get("language") or _detect_language(cleaned_input), default="en")
    session["language"] = entry_language

    if context_applied and _is_context_info_followup(cleaned_input):
        session["onboarding_done"] = True
        session["last_intent"] = INTENT_SCHEME_QUERY
        session["last_secondary_intents"] = []
        session["last_action"] = "info"
        session["_intent_debug"] = {
            "primary_intent": INTENT_SCHEME_QUERY,
            "confidence": 0.95,
            "fallback_used": False,
            "secondary_intents": [],
            "raw_model_output": None,
            "normalized_intent": INTENT_SCHEME_QUERY,
            "context_used": True,
            "model_used": False,
            "source": "context_followup_resolution",
        }
        referenced_scheme = str(session.get("last_scheme") or session.get("selected_scheme") or "this scheme").strip() or "this scheme"
        if entry_language == "hi":
            response_text = (
                f"{referenced_scheme} के लिए पात्रता और दस्तावेज़ मैं साझा कर सकता हूँ। "
                "Eligibility, required documents और application process देखिए।"
            )
        else:
            response_text = (
                f"For {referenced_scheme}, I can share eligibility and documents. "
                "Please check eligibility, required documents, and application process."
            )
        _append_history(session, "user", cleaned_input)
        _append_history(session, "assistant", response_text)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=response_text,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=bool(session.get("session_complete", False)),
            mode="info",
            action="ask_to_apply_or_more_info",
            session=session,
            quick_actions=build_quick_actions(entry_language, "info", "ask_to_apply_or_more_info", session.get("last_scheme"), bool(session.get("session_complete", False))),
            voice_text=response_text,
        )

    if _is_autofill_command(cleaned_input):
        _append_history(session, "user", cleaned_input)
        if session.get("session_complete"):
            autofill_result = _run_autofill_with_timeout(session, entry_language)
            autofill_status = str((autofill_result or {}).get("status") or "failed").strip().lower()
            if autofill_status not in {"success", "failed", "skipped"}:
                autofill_status = "failed"
            auto_msg = str((autofill_result or {}).get("message") or _autofill_fallback_message(entry_language))
            _append_history(session, "assistant", auto_msg)
            update_session(session_id, session)
            return _build_response(
                session_id=session_id,
                response_text=auto_msg,
                field_name=None,
                validation_passed=True,
                validation_error=None,
                session_complete=True,
                mode="action",
                action="auto_fill_form",
                session=session,
                quick_actions=build_quick_actions(entry_language, "action", "auto_fill_form", session.get("last_scheme"), True),
                voice_text=auto_msg,
                autofill_status=autofill_status,
            )

        auto_msg = (
            "Auto-fill शुरू करने से पहले 1-2 जानकारी और चाहिए। चलिए, उसे पूरा करते हैं।"
            if entry_language == "hi"
            else "Before auto-fill, I need 1-2 more details. Let us finish those first."
        )
        _append_history(session, "assistant", auto_msg)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=auto_msg,
            field_name=session.get("next_field") or get_next_field(session),
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="action",
            action="continue_form",
            session=session,
            quick_actions=build_quick_actions(entry_language, "action", "continue_form", session.get("last_scheme"), False),
            voice_text=auto_msg,
            autofill_status="skipped",
        )

    if detect_information_input(user_input):
        session["info_detected"] = True
        session["onboarding_done"] = True
        session["last_intent"] = "provide_information"
        session["last_secondary_intents"] = []
        session["last_action"] = "action"
        session["_intent_debug"] = {
            "primary_intent": "provide_information",
            "confidence": 0.95,
            "fallback_used": False,
            "secondary_intents": [],
            "raw_model_output": None,
            "normalized_intent": "provide_information",
            "context_used": False,
            "model_used": False,
            "source": "rule_based_info_detection",
        }
        _apply_info_detection_to_profile(session, cleaned_input, entry_language)

        response_text = (
            get_field_question(session.get("next_field"), entry_language)
            if session.get("next_field")
            else _build_confirmation_summary(session, entry_language)
        )
        _append_history(session, "user", cleaned_input)
        _append_history(session, "assistant", response_text)
        update_session(session_id, session)

        return _build_response(
            session_id=session_id,
            response_text=response_text,
            field_name=session.get("next_field"),
            validation_passed=True,
            validation_error=None,
            session_complete=bool(session.get("session_complete", False)),
            mode="action",
            action="collect_information",
            session=session,
            quick_actions=build_quick_actions(entry_language, "action", "collect_information", session.get("last_scheme"), False),
            voice_text=response_text,
        )

    # Cold-start onboarding for first interaction in a new session.
    lowered_first_turn = cleaned_input.lower()
    is_intentful_first_turn = _has_scheme_signal(lowered_first_turn) or _is_apply_intent_signal(cleaned_input)
    should_show_onboarding = (
        is_generic_help_query(cleaned_input)
        or lowered_first_turn in {"hi", "hello", "help", "madad"}
        or len(lowered_first_turn.split()) <= 2
    )
    if (
        not session.get("onboarding_done")
        and not session.get("conversation_history")
        and not is_intentful_first_turn
        and should_show_onboarding
    ):
        lang_probe = normalize_language_code(language or session.get("language") or _detect_language(cleaned_input), default="en")
        onboarding = "आपको किस तरह की मदद चाहिए?" if lang_probe == "hi" else "What kind of help do you need?"
        session["language"] = lang_probe
        session["onboarding_done"] = True
        _append_history(session, "assistant", onboarding)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=onboarding,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="clarify",
            action="onboarding",
            session=session,
            quick_actions=build_quick_actions(lang_probe, "clarify", "onboarding", session.get("last_scheme"), False),
            voice_text=onboarding,
        )
    session["onboarding_done"] = True

    # Safety-first fallback for noisy first-turn inputs so onboarding does not mask error handling.
    if (
        not session.get("conversation_history")
        and not _has_scheme_signal(cleaned_input)
        and not _is_apply_intent_signal(cleaned_input)
        and is_unclear_input(cleaned_input)
    ):
        fallback_lang = normalize_language_code(language or session.get("language") or _detect_language(cleaned_input), default="en")
        safe_text = "I couldn't find a clear match. Can you clarify your need?" if fallback_lang != "hi" else "मुझे स्पष्ट मैच नहीं मिला। क्या आप अपनी जरूरत स्पष्ट कर सकते हैं?"
        session["last_action"] = "safe_fallback"
        session["_safety_low_confidence"] = True
        session["_safety_ambiguous"] = False
        session["_safety_fallback_triggered"] = True
        _append_history(session, "user", cleaned_input)
        _append_history(session, "assistant", safe_text)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=safe_text,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="clarify",
            action="safe_fallback",
            session=session,
            quick_actions=build_quick_actions(fallback_lang, "clarify", "safe_fallback", session.get("last_scheme"), False),
            voice_text=safe_text,
        )

    _maybe_update_feedback_from_input(session, cleaned_input)
    if language and language.strip():
        session["language"] = normalize_language_code(language, default="en")
    elif session.get("language"):
        session["language"] = normalize_language_code(session.get("language"), default="en")
    else:
        session["language"] = normalize_language_code(_detect_language(cleaned_input), default="en")

    current_field = session.get("next_field") or get_next_field(session)
    session["next_field"] = current_field
    lang = normalize_language_code(session.get("language", "en"), default="en")
    session["language"] = lang
    cleaned_input = _resolve_quick_action_input(cleaned_input, lang, session)

    # First-turn non-domain noise should fail safely, not enter form flow.
    if (
        not session.get("conversation_history")
        and not _has_scheme_signal(cleaned_input)
        and not _is_apply_intent_signal(cleaned_input)
        and not detect_information_input(cleaned_input)
        and not is_generic_help_query(cleaned_input)
    ):
        safe_text = "I couldn't find a clear match. Can you clarify your need?" if lang != "hi" else "मुझे स्पष्ट मैच नहीं मिला। क्या आप अपनी जरूरत स्पष्ट कर सकते हैं?"
        session["last_action"] = "safe_fallback"
        session["_safety_low_confidence"] = True
        session["_safety_ambiguous"] = False
        session["_safety_fallback_triggered"] = True
        _append_history(session, "user", cleaned_input)
        _append_history(session, "assistant", safe_text)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=safe_text,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="clarify",
            action="safe_fallback",
            session=session,
            quick_actions=build_quick_actions(lang, "clarify", "safe_fallback", session.get("last_scheme"), False),
            voice_text=safe_text,
        )

    if session.get("info_detected"):
        # Hard guard: once info path is selected in this turn, do not run model/intent logic.
        response_text = get_field_question(session.get("next_field"), lang)
        _append_history(session, "assistant", response_text)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=response_text,
            field_name=session.get("next_field"),
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="action",
            action="collect_information",
            session=session,
            quick_actions=build_quick_actions(lang, "action", "collect_information", session.get("last_scheme"), False),
            voice_text=response_text,
        )

    if (
        current_field
        and _looks_like_field_value(current_field, cleaned_input)
        and not _is_apply_intent_signal(cleaned_input)
        and not _has_scheme_signal(cleaned_input)
    ):
        validation = validate_field(current_field, cleaned_input, language=lang)
        if validation.get("valid"):
            _append_history(session, "user", cleaned_input)
            session.setdefault("user_profile", {})[current_field] = str(validation.get("normalized") or "")
            session.setdefault("field_completion", {})[current_field] = True
            active_fields = _session_fields(session)
            if current_field in active_fields:
                session["last_completed_field_index"] = active_fields.index(current_field)
            next_field_after = get_next_field(session)
            session["next_field"] = next_field_after
            session["session_complete"] = False
            session["confirmation_done"] = False

            if next_field_after is None:
                session["confirmation_state"] = "pending"
                session["dialogue_state"] = "confirming"
                confirmation_text = _build_confirmation_summary(session, lang)
                confirmation_text = (
                    f"{confirmation_text}\n\nकृपया सब जानकारी देखकर हाँ कहें या बदलाव बताएं।"
                    if lang == "hi"
                    else f"{confirmation_text}\n\nPlease review all details and say yes to submit, or ask to change any field."
                )
                _append_history(session, "assistant", confirmation_text)
                update_session(session_id, session)
                return _build_response(
                    session_id=session_id,
                    response_text=confirmation_text,
                    field_name=None,
                    validation_passed=True,
                    validation_error=None,
                    session_complete=False,
                    mode="action",
                    action="confirm_details",
                    session=session,
                    quick_actions=build_quick_actions(lang, "action", "confirm_details", session.get("last_scheme"), False),
                    voice_text=confirmation_text,
                )

            question_text = get_field_question(next_field_after, lang)
            session["dialogue_state"] = "collecting_info"
            _append_history(session, "assistant", question_text)
            update_session(session_id, session)
            return _build_response(
                session_id=session_id,
                response_text=question_text,
                field_name=next_field_after,
                validation_passed=True,
                validation_error=None,
                session_complete=False,
                mode="action",
                action="collect_information",
                session=session,
                quick_actions=build_quick_actions(lang, "action", "collect_information", session.get("last_scheme"), False),
                voice_text=question_text,
            )

    # Global apply override signal: intent or explicit apply phrase should always force form start.
    lowered_input = str(cleaned_input or "").lower()
    apply_intent_requested = "apply" in lowered_input
    session_context_for_intent = {
        "last_intent": session.get("last_intent"),
        "last_action": session.get("last_action"),
        "last_scheme": session.get("last_scheme") or session.get("selected_scheme") or session.get("current_scheme"),
        "language": lang,
    }
    intent_probe = INTENT_SERVICE.detect(
        cleaned_input,
        debug=True,
        session_context=session_context_for_intent,
    )
    probed_intent = str(intent_probe.get("intent") or "general_query").strip().lower()
    probed_secondary_intents = [str(item or "").strip().lower() for item in intent_probe.get("secondary_intents", [])]
    apply_intent_requested = (
        apply_intent_requested
        or _is_apply_intent_signal(cleaned_input)
        or probed_intent == "apply_scheme"
        or any(item == "apply_scheme" for item in probed_secondary_intents)
    )

    # Returning-user lightweight context nudge on first turn, unless input already names a scheme.
    if (
        not session.get("conversation_history")
        and history_last_scheme
        and not _has_explicit_scheme_reference(cleaned_input)
        and not apply_intent_requested
    ):
        returning_prompt = _build_returning_user_prompt(lang, history_last_scheme)
        _append_history(session, "assistant", returning_prompt)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=returning_prompt,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="clarify",
            action="returning_user_context",
            session=session,
            quick_actions=build_quick_actions(lang, "clarify", "returning_user_context", history_last_scheme, False),
            voice_text=returning_prompt,
        )

    scheme_detection_input = cleaned_input
    if (
        _is_vague_scheme_reference(cleaned_input)
        and history_last_scheme
        and not _has_explicit_scheme_reference(cleaned_input)
        and current_field is None
    ):
        # Resolve pronoun-like references using known last scheme without altering original user text.
        scheme_detection_input = f"{history_last_scheme} {cleaned_input}".strip()

    mentioned_schemes = _detect_scheme_mentions(scheme_detection_input, limit=5)
    session["_scheme_detection_input"] = scheme_detection_input
    session["_scheme_detection_candidates"] = list(mentioned_schemes)
    session["_scheme_detection_decision"] = "none"
    session["_safety_low_confidence"] = False
    session["_safety_ambiguous"] = False
    session["_safety_fallback_triggered"] = False
    session["_apply_flow_forced"] = False

    apply_signal = apply_intent_requested or _is_apply_intent_signal(cleaned_input)

    strong_matches = [
        item
        for item in mentioned_schemes
        if float(item.get("score") or 0.0) > 0.8
    ]

    explicit_match = _prefer_explicit_scheme_match(cleaned_input, mentioned_schemes)
    if explicit_match is not None:
        strong_matches = [explicit_match]

    if apply_intent_requested:
        candidates = list(mentioned_schemes)
        selected_scheme = _resolve_apply_target_scheme(session, cleaned_input, candidates, "general")

        session["selected_scheme"] = selected_scheme
        session["current_scheme"] = selected_scheme
        session["last_scheme"] = selected_scheme

        fields = get_fields_for_scheme(selected_scheme)
        if get_form_type_for_scheme(selected_scheme) == "generic":
            fields = ["full_name", "phone", "aadhaar_number"]
            session["_force_minimal_generic_fields"] = True
        else:
            session["_force_minimal_generic_fields"] = False
        session["field_completion"] = {field: False for field in fields}
        session["next_field"] = fields[0] if fields else None
        session["session_complete"] = False
        session["confirmation_done"] = False
        session["confirmation_state"] = "pending"
        session["dialogue_state"] = "collecting_info"
        session["action_confirmation_pending"] = False
        session["_apply_flow_forced"] = True
        session["_scheme_detection_decision"] = "apply_forced_global"

        intro = (
            f"मैं आपकी {selected_scheme} के लिए आवेदन में मदद करूंगा। चलिए शुरू करते हैं।"
            if lang == "hi"
            else f"I will help you apply for {selected_scheme}. Let's start."
        )
        next_field = session.get("next_field")
        response_text = intro if next_field is None else f"{intro} {get_field_question(next_field, lang)}"
        _append_history(session, "user", cleaned_input)
        _append_history(session, "assistant", response_text)
        update_session(session_id, session)

        return _build_response(
            session_id=session_id,
            response_text=response_text,
            field_name=next_field,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="action",
            action="apply_scheme_forced_start",
            session=session,
            quick_actions=build_quick_actions(lang, "action", "apply_scheme_forced_start", selected_scheme, False),
            voice_text=response_text,
        )

    if len(strong_matches) == 1:
        detected_scheme = str(strong_matches[0].get("scheme") or "").strip()
        session["current_scheme"] = detected_scheme
        session["last_scheme"] = detected_scheme
        session["_scheme_detection_decision"] = "auto_select"
        if not session.get("selected_scheme"):
            session["selected_scheme"] = resolve_scheme_name(detected_scheme)
    elif mentioned_schemes and float(mentioned_schemes[0].get("score") or 0.0) < 0.7 and not apply_signal and not context_applied:
        session["_scheme_detection_decision"] = "low_confidence"
        session["_safety_low_confidence"] = True
        session["_safety_fallback_triggered"] = True
        confirm_text = (
            "मुझे कई संभावित योजनाएँ मिली हैं, क्या आप पुष्टि कर सकते हैं?"
            if lang == "hi"
            else "I found multiple possible schemes, can you confirm?"
        )
        _append_history(session, "user", cleaned_input)
        _append_history(session, "assistant", confirm_text)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=confirm_text,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="clarify",
            action="clarify_scheme",
            session=session,
            quick_actions=build_quick_actions(lang, "clarify", "clarify_scheme", session.get("last_scheme"), False),
            voice_text=confirm_text,
        )
    elif (
        len(mentioned_schemes) >= 2
        and _has_scheme_signal(cleaned_input)
        and not apply_signal
        and not context_applied
        and bool(explicit_match)
    ):
        first = str(mentioned_schemes[0].get("scheme") or "").strip()
        second = str(mentioned_schemes[1].get("scheme") or "").strip()
        session["_scheme_detection_decision"] = "ambiguous"
        session["_safety_ambiguous"] = True
        session["_safety_fallback_triggered"] = True
        ambiguous_text = (
            f"क्या आपका मतलब {first} या {second} है?"
            if lang == "hi"
            else f"Did you mean: {first} or {second}?"
        )
        _append_history(session, "user", cleaned_input)
        _append_history(session, "assistant", ambiguous_text)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=ambiguous_text,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="clarify",
            action="clarify_scheme",
            session=session,
            quick_actions=build_quick_actions(lang, "clarify", "clarify_scheme", session.get("last_scheme"), False),
            voice_text=ambiguous_text,
        )
    elif not mentioned_schemes and _has_scheme_signal(cleaned_input):
        explicit_phrase = _extract_explicit_scheme_phrase(cleaned_input)
        if explicit_phrase:
            session["current_scheme"] = explicit_phrase
            session["last_scheme"] = explicit_phrase
            session["selected_scheme"] = resolve_scheme_name(explicit_phrase)
            session["_scheme_detection_decision"] = "explicit_generic"
        elif not apply_signal and not context_applied:
            session["_scheme_detection_decision"] = "none"
            session["_safety_fallback_triggered"] = True
            no_match_text = "आप किस योजना के बारे में पूछ रहे हैं?" if lang == "hi" else "Which scheme are you asking about?"
            _append_history(session, "user", cleaned_input)
            _append_history(session, "assistant", no_match_text)
            update_session(session_id, session)
            return _build_response(
                session_id=session_id,
                response_text=no_match_text,
                field_name=None,
                validation_passed=True,
                validation_error=None,
                session_complete=False,
                mode="clarify",
                action="clarify_scheme",
                session=session,
                quick_actions=build_quick_actions(lang, "clarify", "clarify_scheme", session.get("last_scheme"), False),
                voice_text=no_match_text,
            )
        else:
            session["_scheme_detection_decision"] = "apply_signal_no_match_override"

    if _is_restart_command(cleaned_input):
        _append_history(session, "user", cleaned_input)
        session = _safe_reset_session(session_id, lang)
        message = get_field_question(session.get("next_field"), lang)
        _append_history(session, "assistant", message)
        update_session(session_id, session)
        return _build_response(
            session_id,
            message,
            session.get("next_field"),
            True,
            False,
            None,
            session=session,
            quick_actions=build_quick_actions(lang, "action", None, session.get("last_scheme"), False),
        )

    if session.get("extraction_conflicts"):
        if not _resolve_extraction_conflicts(session, cleaned_input, lang):
            prompt = str(session.get("pending_conflict_prompt") or "")
            if prompt:
                _append_history(session, "assistant", prompt)
                update_session(session_id, session)
                return _build_response(
                    session_id=session_id,
                    response_text=prompt,
                    field_name=session.get("next_field") or get_next_field(session),
                    validation_passed=False,
                    validation_error="extraction_conflict",
                    session_complete=False,
                    mode="clarify",
                    action="resolve_extraction_conflict",
                    session=session,
                    quick_actions=build_quick_actions(lang, "clarify", "resolve_extraction_conflict", session.get("last_scheme"), False),
                )

    if _is_correction_request(cleaned_input) and _update_dialogue_state(session) in {"collecting_info", "confirming"}:
        previous = _move_to_previous_field(session)
        session["dialogue_state"] = "collecting_info"
        correction_prompt = get_field_question(previous, lang)
        _append_history(session, "assistant", correction_prompt)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=correction_prompt,
            field_name=previous,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="action",
            action="correction",
            session=session,
            quick_actions=build_quick_actions(lang, "action", "correction", session.get("last_scheme"), False),
        )

    normalized_input = normalize_for_intent(scheme_detection_input, language_hint=lang)

    # Clarification resume path: enrich terse follow-up using stack context.
    if session.get("clarification_stack") and len((normalized_input.intent_text or cleaned_input).split()) <= 5:
        previous_context = _pop_clarification(session)
        merged_text = f"{previous_context} {normalized_input.intent_text or cleaned_input}".strip()
        normalized_input = normalize_for_intent(merged_text, language_hint=lang)
    need_signal = detect_user_need(
        normalized_input.intent_text or cleaned_input,
        session_context={
            "user_need_profile": session.get("user_need_profile"),
            "conversation_history": session.get("conversation_history", []),
            "last_intent": session.get("last_intent"),
            "history_summary": session.get("history_summary"),
        },
    )
    need_category = str(need_signal.get("category") or "")
    need_confidence = float(need_signal.get("confidence") or 0.0)
    user_need_profile = _update_user_need_profile(session, cleaned_input, need_category=need_category)
    user_profile_for_rag = _sanitize_user_profile_for_rag(user_need_profile)
    progress_started = any(session.get("field_completion", {}).get(field, False) for field in _session_fields(session))
    session_context_for_intent = {
        "last_intent": session.get("last_intent"),
        "last_action": session.get("last_action"),
        "last_scheme": session.get("last_scheme") or session.get("selected_scheme") or session.get("current_scheme"),
        "language": lang,
    }
    intent_debug = INTENT_SERVICE.detect(
        normalized_input.intent_text or cleaned_input, 
        debug=True, 
        session_context=session_context_for_intent
    )
    
    model_intent = intent_debug.get("intent", "general_query")
    model_confidence = float(intent_debug.get("confidence", 0.0)) / 100.0
    debug_info = intent_debug.get("debug", {})
    model_fallback_used = bool(debug_info.get("fallback_used", False))
    secondary_intents = intent_debug.get("secondary_intents", [])

    intent_decision = {
        "primary_intent": model_intent,
        "confidence": round(model_confidence, 4),
        "fallback_used": model_fallback_used,
        "secondary_intents": secondary_intents,
        "raw_model_output": debug_info.get("raw_model_output"),
        "normalized_intent": debug_info.get("normalized_intent"),
        "context_used": debug_info.get("context_used"),
        "hybrid_debug": debug_info,
    }

    state_transition = f"{session.get('state', STATE_IDLE)}->{session.get('state', STATE_IDLE)}"
    state_result = apply_state_transition(session, cleaned_input, str(model_intent or ""))
    state_transition = str(state_result.get("state_transition") or state_transition)

    intent_decision["state_debug"] = {
        "current_state": state_result.get("current_state", session.get("state", STATE_IDLE)),
        "state_transition": state_transition,
        "current_scheme": state_result.get("current_scheme", session.get("current_scheme")),
        "collected_fields": state_result.get("collected_fields", dict(session.get("collected_fields") or {})),
        "missing_fields": state_result.get("missing_fields", list(session.get("missing_fields") or [])),
    }

    if state_result.get("handled"):
        if apply_signal and str(state_result.get("action") or "") == "state_select_scheme":
            selected_scheme = _resolve_apply_target_scheme(session, cleaned_input, list(mentioned_schemes), "general")

            session["selected_scheme"] = selected_scheme
            session["current_scheme"] = selected_scheme
            session["last_scheme"] = selected_scheme

            fields = get_fields_for_scheme(selected_scheme)
            if get_form_type_for_scheme(selected_scheme) == "generic":
                fields = ["full_name", "phone", "aadhaar_number"]
                session["_force_minimal_generic_fields"] = True
            else:
                session["_force_minimal_generic_fields"] = False
            next_field = fields[0] if fields else None
            session["field_completion"] = {field: False for field in fields}
            session["next_field"] = next_field
            session["session_complete"] = False
            session["confirmation_done"] = False
            session["confirmation_state"] = "pending"
            session["dialogue_state"] = "collecting_info"
            session["action_confirmation_pending"] = False
            session["_apply_flow_forced"] = True

            intro = (
                f"मैं आपकी {selected_scheme} के लिए आवेदन में मदद करूंगा। चलिए शुरू करते हैं।"
                if lang == "hi"
                else f"I will help you apply for {selected_scheme}. Let's start."
            )
            response_text = intro if next_field is None else f"{intro} {get_field_question(next_field, lang)}"
            _append_history(session, "user", cleaned_input)
            _append_history(session, "assistant", response_text)
            update_session(session_id, session)

            return _build_response(
                session_id=session_id,
                response_text=response_text,
                field_name=next_field,
                validation_passed=True,
                validation_error=None,
                session_complete=False,
                mode="action",
                action="apply_scheme_forced_start",
                session=session,
                quick_actions=build_quick_actions(lang, "action", "apply_scheme_forced_start", selected_scheme, False),
                voice_text=response_text,
            )

        _sync_state_machine_fields_to_profile(session)
        if session.get("current_scheme") and not session.get("selected_scheme"):
            session["selected_scheme"] = resolve_scheme_name(session.get("current_scheme"))
            session["last_scheme"] = session.get("current_scheme")
        session["last_intent"] = model_intent
        session["last_secondary_intents"] = secondary_intents
        session["last_action"] = "action"
        if debug:
            session["_intent_debug"] = intent_decision
        else:
            session.pop("_intent_debug", None)

        response_text = str(state_result.get("response_text") or "")
        action = str(state_result.get("action") or "state_machine")
        session_complete = bool(state_result.get("session_complete", False))
        _append_history(session, "user", cleaned_input)
        if response_text:
            _append_history(session, "assistant", response_text)
        update_session(session_id, session)

        return _build_response(
            session_id=session_id,
            response_text=response_text,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=session_complete,
            mode="action",
            action=action,
            session=session,
            quick_actions=build_quick_actions(lang, "action", action, session.get("last_scheme"), session_complete),
            voice_text=response_text,
        )

    detected_intent, detected_mode = detect_intent_and_mode(
        normalized_input.intent_text or cleaned_input,
        predicted_intent=model_intent,
        confidence=model_confidence,
    )

    lowered_input = str(cleaned_input or "").lower()
    apply_intent_requested = (
        str(model_intent or "").strip().lower() in APPLY_INTENTS
        or str(detected_intent or "").strip().lower() in APPLY_INTENTS
        or any(str(intent or "").strip().lower() in APPLY_INTENTS for intent in secondary_intents)
        or apply_signal
    )
    forced_scheme_hint = _forced_scheme_from_query(cleaned_input)
    explicit_apply_request = (
        any(token in lowered_input for token in ("apply", "application", "enroll", "registration"))
        and bool(session.get("selected_scheme") or _extract_explicit_scheme_phrase(cleaned_input))
    )
    if apply_intent_requested and not forced_scheme_hint:
        explicit_apply_request = True
    if explicit_apply_request:
        detected_mode = "action"

    has_action_signal = detected_intent in ACTION_INTENTS or any(intent in ACTION_INTENTS for intent in secondary_intents)
    has_info_signal = detected_intent in INFO_INTENTS or any(intent in INFO_INTENTS for intent in secondary_intents)
    if has_action_signal and has_info_signal and not progress_started:
        detected_mode = "clarify"
    if explicit_apply_request:
        detected_mode = "action"
    if apply_intent_requested and not forced_scheme_hint:
        detected_mode = "action"

    if detected_mode == "info":
        mode = "info"
        # User changed intent to information; do not keep forcing action state.
        session["action_confirmation_pending"] = False
    elif detected_mode == "action":
        mode = "action"
    elif progress_started and _looks_like_field_value(current_field, cleaned_input):
        mode = "action"
    else:
        # Unknown or unrelated inputs should not trap user in form flow.
        mode = "info"
    conversational_intent = str((intent_debug.get("debug") or {}).get("conversation_intent") or "")
    conversational_confidence = float((intent_debug.get("debug") or {}).get("conversation_confidence") or 0.0)
    if conversational_intent == "correction":
        mode = "action"
    if conversational_confidence and conversational_confidence < 0.42 and not explicit_apply_request:
        mode = "clarify"
    if apply_intent_requested and not forced_scheme_hint:
        mode = "action"

    session["last_intent"] = model_intent
    session["last_secondary_intents"] = secondary_intents
    session["last_action"] = mode
    if debug:
        session["_intent_debug"] = intent_decision
    else:
        session.pop("_intent_debug", None)

    raw_input = normalized_input.intent_text or cleaned_input
    log_event(
        "conversation_routing_decision",
        endpoint="conversation_service",
        status="success",
        session_id=session_id,
        user_input_length=len(raw_input or ""),
        user_input_fingerprint=fingerprint_text(raw_input),
        detected_intent=detected_intent,
        model_intent=model_intent,
        confidence=round(model_confidence, 4),
        selected_scheme=session.get("selected_scheme"),
        fallback_used=model_fallback_used,
        secondary_intents=secondary_intents,
        selected_mode=mode,
        current_field=current_field,
        progress_started=progress_started,
    )

    fused_context = build_context_fusion(
        current_intent=model_intent,
        previous_intent=session.get("memory_last_intent"),
        user_profile={**(session.get("user_profile") or {}), **(session.get("user_need_profile") or {})},
        need_category=need_category,
        history_summary=session.get("history_summary"),
    )
    thresholds = adaptive_confidence_thresholds(
        query=normalized_input.intent_text or cleaned_input,
        past_confidence=session.get("past_need_confidence"),
        intent_type=model_intent,
    )
    low_threshold = float(thresholds.get("low", 0.6))
    high_threshold = float(thresholds.get("high", 0.8))
    recommendation_limit = _adaptive_recommendation_limit(need_confidence, low_threshold, high_threshold)
    if mode == "info" and _is_broad_discovery_request(normalized_input.intent_text or cleaned_input):
        recommendation_limit = max(3, recommendation_limit)
    short_mode = _is_short_query(normalized_input.intent_text or cleaned_input)
    session["past_need_confidence"] = need_confidence

    if cleaned_input.lower() in {"auto fill form", "autofill", "auto-fill"}:
        if session.get("session_complete"):
            autofill_result = _run_autofill_with_timeout(session, lang)
            auto_msg = str(autofill_result.get("message") or _autofill_fallback_message(lang))
            autofill_status = str(autofill_result.get("status") or "failed")
            _append_history(session, "assistant", auto_msg)
            update_session(session_id, session)
            return _build_response(
                session_id=session_id,
                response_text=auto_msg,
                field_name=None,
                validation_passed=True,
                validation_error=None,
                session_complete=True,
                mode="action",
                action="auto_fill_form",
                session=session,
                quick_actions=build_quick_actions(lang, "action", "auto_fill_form", session.get("last_scheme"), True),
                voice_text=auto_msg,
                autofill_status=autofill_status,
            )
        auto_msg = (
            "Auto-fill शुरू करने से पहले 1-2 जानकारी और चाहिए। चलिए, उसे पूरा करते हैं।"
            if lang == "hi"
            else "Before auto-fill, I need 1-2 more details. Let us finish those first."
        )
        _append_history(session, "assistant", auto_msg)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=auto_msg,
            field_name=current_field,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="action",
            action="continue_form",
            session=session,
            quick_actions=build_quick_actions(lang, "action", "continue_form", session.get("last_scheme"), False),
            voice_text=auto_msg,
            autofill_status="skipped",
        )

    if cleaned_input.lower() in {"autofill completed", "autofill success", "auto fill done"}:
        success_msg = (
            "शानदार, फॉर्म सफलतापूर्वक भर गया। अब एक बार जानकारी देखकर submit कर दीजिए।"
            if lang == "hi"
            else "Perfect, your form has been filled successfully. Please review once and submit."
        )
        _append_history(session, "assistant", success_msg)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=success_msg,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=bool(session.get("session_complete")),
            mode="action",
            action="autofill_success",
            session=session,
            quick_actions=build_quick_actions(lang, "action", "autofill_success", session.get("last_scheme"), True),
            voice_text=success_msg,
            autofill_status="success",
        )

    if cleaned_input.lower() in {"autofill failed", "auto fill failed", "autofill error"}:
        failure_msg = _autofill_fallback_message(lang)
        recovery = get_field_question(session.get("next_field") or get_next_field(session), lang)
        merged_msg = f"{failure_msg} {recovery}".strip()
        _append_history(session, "assistant", merged_msg)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=merged_msg,
            field_name=session.get("next_field") or get_next_field(session),
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="action",
            action="autofill_recovery",
            session=session,
            quick_actions=build_quick_actions(lang, "action", "autofill_recovery", session.get("last_scheme"), False),
            voice_text=merged_msg,
            autofill_status="failed",
        )

    if apply_intent_requested and not forced_scheme_hint and not progress_started:
        selected_scheme = _resolve_apply_target_scheme(session, cleaned_input, list(mentioned_schemes), need_category)

        session["selected_scheme"] = selected_scheme
        session["current_scheme"] = selected_scheme
        session["last_scheme"] = selected_scheme

        fields = get_fields_for_scheme(selected_scheme)
        if get_form_type_for_scheme(selected_scheme) == "generic":
            fields = ["full_name", "phone", "aadhaar_number"]
            session["_force_minimal_generic_fields"] = True
        else:
            session["_force_minimal_generic_fields"] = False
        next_field = fields[0] if fields else None

        session["field_completion"] = {field: False for field in fields}
        session["next_field"] = next_field
        session["session_complete"] = False
        session["confirmation_done"] = False
        session["confirmation_state"] = "pending"
        session["action_confirmation_pending"] = False
        session["dialogue_state"] = "collecting_info"
        session["_apply_flow_forced"] = True

        intro = (
            f"मैं आपकी {selected_scheme} के लिए आवेदन में मदद करूंगा। चलिए शुरू करते हैं।"
            if lang == "hi"
            else f"I will help you apply for {selected_scheme}. Let's start."
        )
        response_text = intro if next_field is None else f"{intro} {get_field_question(next_field, lang)}"

        _append_history(session, "user", cleaned_input)
        _append_history(session, "assistant", response_text)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=response_text,
            field_name=next_field,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="action",
            action="apply_scheme_forced_start",
            session=session,
            quick_actions=build_quick_actions(lang, "action", "apply_scheme_forced_start", selected_scheme, False),
            voice_text=response_text,
        )

    if (not explicit_apply_request) and (not forced_scheme_hint) and (
        _is_unclear_input(cleaned_input) or _is_generic_help_query(normalized_input.intent_text or cleaned_input)
    ):
        _push_clarification(session, normalized_input.intent_text or cleaned_input)
        record_fallback()
        rag_query = {
            "scheme": session.get("current_scheme"),
            "user_profile": user_profile_for_rag,
        }
        recommendations = recommend_schemes(
            normalized_input.intent_text or cleaned_input,
            lang,
            limit=recommendation_limit,
            need_category=need_category,
            user_profile=user_profile_for_rag,
            scheme_context=rag_query,
            session_feedback=_session_feedback(session),
            context_fusion=fused_context,
        )
        explainable = recommend_schemes_with_reasons(
            normalized_input.intent_text or cleaned_input,
            lang,
            limit=recommendation_limit,
            need_category=need_category,
            user_profile=user_profile_for_rag,
            scheme_context=rag_query,
            session_feedback=_session_feedback(session),
            context_fusion=fused_context,
        )
        recommendations = _apply_recommendation_continuity(session, recommendations)
        need_prefix = (
            "आपकी ज़रूरत के आधार पर, ये योजनाएँ मदद कर सकती हैं।"
            if lang == "hi"
            else "Based on your need, these schemes may help."
        )
        reason_lines = "\n".join(f"- {item['scheme']}: {item['reason']}" for item in explainable)
        confidence_line = (
            "मुझे आपकी जरूरत समझने के लिए थोड़ी और जानकारी चाहिए।"
            if need_confidence < low_threshold and lang == "hi"
            else "I need one more detail to understand your need better."
            if need_confidence < low_threshold
            else ""
        )
        unclear_text = f"{need_prefix}\n{_smart_clarification_message(lang, recommendations, cleaned_input)}"
        if confidence_line:
            unclear_text = f"{unclear_text}\n\n{confidence_line}"
        if reason_lines:
            unclear_text = f"{unclear_text}\n\n{reason_lines}"
        unclear_text = f"{unclear_text}\n\n{_recommendation_confirmation_prompt(lang)}"
        _append_history(session, "user", cleaned_input)
        _append_history(session, "assistant", unclear_text)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=unclear_text,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="clarify",
            action="clarify_intent",
            session=session,
            quick_actions=build_recommendation_quick_actions(recommendations, lang),
            recommended_schemes=recommendations,
            voice_text=unclear_text,
        )

    if mode == "clarify":
        _push_clarification(session, normalized_input.intent_text or cleaned_input)
        response_text = _clarification_message(lang)
        top_category = str(user_history_context.get("top_category") or "").strip()
        if top_category and top_category != "general":
            category_hint = (
                f"पिछली बार आप अक्सर {top_category} से जुड़ी योजनाएँ देखते रहे हैं, चाहें तो मैं उसी तरह की 2-3 योजनाएँ दिखा सकता हूँ।"
                if lang == "hi"
                else f"You often explore {top_category}-related schemes, and I can suggest 2-3 similar options."
            )
            response_text = f"{response_text}\n\n{category_hint}"
        _append_history(session, "user", cleaned_input)
        _append_history(session, "assistant", response_text)
        update_session(session_id, session)

        return _build_response(
            session_id=session_id,
            response_text=response_text,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="clarify",
            action="clarify_intent",
            session=session,
            quick_actions=build_quick_actions(lang, "clarify", "clarify_intent", session.get("last_scheme"), False),
        )

    if mode == "info":
        _append_history(session, "user", cleaned_input)
        query_for_rag = cleaned_input
        if session.get("last_scheme") and is_followup_info_query(cleaned_input) and not _has_explicit_scheme_reference(cleaned_input):
            query_for_rag = f"{session.get('last_scheme')} {cleaned_input}".strip()
        elif (
            _is_vague_scheme_reference(cleaned_input)
            and history_last_scheme
            and not _has_explicit_scheme_reference(cleaned_input)
        ):
            query_for_rag = f"{history_last_scheme} {cleaned_input}".strip()

        rag_scheme_hint = session.get("current_scheme")
        if mentioned_schemes:
            rag_scheme_hint = str(mentioned_schemes[0].get("scheme") or "").strip() or None
        rag_query = {
            "scheme": rag_scheme_hint,
            "user_profile": user_profile_for_rag,
        }

        fast_path_scheme = _forced_scheme_from_query(query_for_rag)
        fast_path_used = bool(fast_path_scheme)
        if fast_path_used:
            rag_response = _fast_scheme_info_response(query_for_rag, lang, fast_path_scheme)
            recommendations = _apply_recommendation_continuity(session, [fast_path_scheme])
            has_match = True
            explainable = [
                {
                    "scheme": fast_path_scheme,
                    "reason": (
                        "Query matched high-signal scheme keywords and was resolved via deterministic fast path."
                        if lang != "hi"
                        else "Query high-signal keyword match se deterministic fast path se resolve hui."
                    ),
                }
            ]
        else:
            rag_response, recommendations, has_match = retrieve_scheme_with_recommendations(
                transcript=query_for_rag,
                language=lang,
                limit=recommendation_limit,
                need_category=need_category,
                user_profile=user_profile_for_rag,
                scheme_context=rag_query,
                session_feedback=_session_feedback(session),
                context_fusion=fused_context,
            )
            rag_safety = dict((rag_response or {}).get("safety") or {})
            if bool(rag_safety.get("fallback_triggered", False)):
                session["_safety_low_confidence"] = bool(rag_safety.get("low_confidence", True))
                session["_safety_ambiguous"] = bool(rag_safety.get("ambiguous", False))
                session["_safety_fallback_triggered"] = True
                session["last_action"] = "safe_fallback"
                fallback_text = str((rag_response or {}).get("confirmation") or "I couldn't find a clear match. Can you clarify your need?")
                _append_history(session, "assistant", fallback_text)
                update_session(session_id, session)
                return _build_response(
                    session_id=session_id,
                    response_text=fallback_text,
                    field_name=None,
                    validation_passed=True,
                    validation_error=None,
                    session_complete=False,
                    mode="clarify",
                    action="safe_fallback",
                    session=session,
                    quick_actions=build_quick_actions(lang, "clarify", "safe_fallback", session.get("last_scheme"), False),
                    voice_text=fallback_text,
                )
            explainable = recommend_schemes_with_reasons(
                query_for_rag,
                language=lang,
                limit=recommendation_limit,
                need_category=need_category,
                user_profile=user_profile_for_rag,
                scheme_context=rag_query,
                session_feedback=_session_feedback(session),
                context_fusion=fused_context,
            )
            recommendations = _apply_recommendation_continuity(session, recommendations)
        top_category = str(user_history_context.get("top_category") or "").strip()
        if (not fast_path_used) and top_category and top_category != "general":
            category_recommendations = recommend_schemes(
                f"{top_category} schemes",
                lang,
                limit=2,
                need_category=top_category,
                user_profile=user_profile_for_rag,
                scheme_context=rag_query,
                session_feedback=_session_feedback(session),
                context_fusion=fused_context,
            )
            merged_recommendations: List[str] = []
            for item in list(recommendations) + list(category_recommendations):
                name = str(item or "").strip()
                if not name or name in merged_recommendations:
                    continue
                merged_recommendations.append(name)
            recommendations = merged_recommendations[: max(recommendation_limit, 3)]

        if (not fast_path_used) and _is_broad_discovery_request(query_for_rag) and len(recommendations) < 2:
            extra_recommendations = recommend_schemes(
                query_for_rag,
                lang,
                limit=max(3, recommendation_limit),
                need_category=need_category,
                user_profile=user_profile_for_rag,
                scheme_context={"scheme": None, "user_profile": user_profile_for_rag},
                session_feedback=_session_feedback(session),
                context_fusion=fused_context,
            )
            merged_recommendations = []
            for item in list(recommendations) + list(extra_recommendations):
                scheme_name = str(item or "").strip()
                if not scheme_name or scheme_name in merged_recommendations:
                    continue
                merged_recommendations.append(scheme_name)
            recommendations = merged_recommendations[:3]

        if rag_response is None:
            current_scheme = str(session.get("current_scheme") or session.get("last_scheme") or "").strip()
            if not current_scheme:
                question = "आप किस योजना के बारे में पूछ रहे हैं?" if lang == "hi" else "Which scheme are you asking about?"
                rag_response = {
                    "confirmation": question,
                    "explanation": question,
                    "next_step": question,
                }
            else:
                rag_response = {
                    "confirmation": (
                        "मैं आपकी बात समझ गया। मैं सही योजना चुनने में मदद कर सकता हूँ।"
                        if lang == "hi"
                        else "I understood your request. I can help you pick the right scheme."
                    ),
                    "explanation": (
                        "कृपया अपनी ज़रूरत बताएं ताकि मैं सही योजना ढूंढ सकूँ।"
                        if lang == "hi"
                        else "Please share your need so I can fetch the right scheme details."
                    ),
                    "next_step": (
                        "आप पात्रता, दस्तावेज़, लाभ या आवेदन प्रक्रिया में से कुछ भी पूछ सकते हैं।"
                        if lang == "hi"
                        else "You can ask for eligibility, documents, benefits, or application process."
                    ),
                }
            fallback_hint = (
                "अगर चाहें तो मैं 2-3 योजनाएँ सीधे सुझाव दूँ, या हम आपकी प्रोफ़ाइल के हिसाब से चुनें।"
                if lang == "hi"
                else "If you want, I can suggest 2-3 schemes directly, or narrow it by your profile."
            )
            rag_response["next_step"] = f"{rag_response['next_step']} {fallback_hint}"

        intent = INTENT_SCHEME_QUERY
        session["last_intent"] = intent
        session["last_action"] = "info"
        response_text = format_info_text(rag_response, lang)
        if need_confidence < low_threshold:
            low_conf = (
                "कृपया बताएं कि आपकी प्राथमिकता क्या है: पैसे की मदद, स्वास्थ्य, या घर?"
                if lang == "hi"
                else "Please clarify your top priority: financial support, health, or housing?"
            )
            response_text = f"{response_text}\n\n{low_conf}"
        elif need_confidence > high_threshold and explainable:
            guidance = "\n".join(f"- {item['scheme']}: {item['reason']}" for item in explainable)
            response_text = f"{response_text}\n\n{guidance}"
        if explainable:
            response_text = f"{response_text}\n\n{_confidence_explanation_line(lang, str(explainable[0].get('reason') or ''))}"
        if short_mode:
            response_text = _short_answer(response_text, lang)
        response_text = f"{response_text}\n\n{_recommendation_confirmation_prompt(lang)}"
        response_text = f"{response_text}{_recommendation_suffix(lang, recommendations)}" if recommendations else response_text
        _append_history(session, "assistant", response_text)

        scheme_details = build_scheme_details(intent, rag_response)
        if has_match and scheme_details and scheme_details.get("title"):
            selected_scheme = resolve_scheme_name(scheme_details.get("title"))
            session["last_scheme"] = scheme_details.get("title")
            session["selected_scheme"] = selected_scheme
            session["field_completion"] = {field: bool(session.get("field_completion", {}).get(field, False)) for field in _session_fields(session)}
            session["next_field"] = get_next_field(session)
            _mark_accepted_scheme(session, str(scheme_details.get("title") or ""))

        if recommendations:
            top_scheme_name = recommendations[0]
            top_scheme = None
            for item in explainable:
                if item.get("scheme") == top_scheme_name:
                    top_scheme = item
                    break
            if top_scheme:
                session["last_recommendation_reason"] = top_scheme.get("reason")

        quick_actions = build_recommendation_quick_actions(recommendations, lang)

        update_session(session_id, session)

        return _build_response(
            session_id=session_id,
            response_text=response_text,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            mode="info",
            action="ask_to_apply_or_more_info",
            session=session,
            scheme_details=scheme_details,
            quick_actions=quick_actions,
            recommended_schemes=recommendations,
        )

    if mode == "action" and not progress_started:
        if not session.get("selected_scheme"):
            derived_scheme = resolve_scheme_name(session.get("last_scheme") or get_default_scheme_for_category(need_category))
            session["selected_scheme"] = derived_scheme
            session["field_completion"] = {field: False for field in _session_fields(session)}
            session["next_field"] = get_next_field(session)

        if session.get("action_confirmation_pending", False):
            if _is_affirmative(cleaned_input):
                session["action_confirmation_pending"] = False
            elif _is_negative(cleaned_input):
                session["action_confirmation_pending"] = False
                response_text = (
                    "ठीक है, पहले जानकारी देखते हैं। आप योजना का नाम बताएं या पात्रता पूछें।"
                    if lang == "hi"
                    else "Sure, let us review information first. Tell me a scheme name or ask for eligibility."
                )
                _append_history(session, "assistant", response_text)
                update_session(session_id, session)
                return _build_response(
                    session_id=session_id,
                    response_text=response_text,
                    field_name=None,
                    validation_passed=True,
                    validation_error=None,
                    session_complete=False,
                    mode="info",
                    action="ask_to_apply_or_more_info",
                    session=session,
                    quick_actions=build_quick_actions(lang, "info", "ask_to_apply_or_more_info", session.get("last_scheme"), False),
                )
            else:
                response_text = _action_start_confirmation_message(lang)
                _append_history(session, "assistant", response_text)
                update_session(session_id, session)
                return _build_response(
                    session_id=session_id,
                    response_text=response_text,
                    field_name=None,
                    validation_passed=False,
                    validation_error="action_confirmation_pending",
                    session_complete=False,
                    mode="clarify",
                    action="confirm_action_start",
                    session=session,
                    quick_actions=build_quick_actions(lang, "clarify", "confirm_action_start", session.get("last_scheme"), False),
                )
        else:
            session["action_confirmation_pending"] = True
            response_text = _action_start_confirmation_message(lang)
            _append_history(session, "assistant", response_text)
            update_session(session_id, session)
            return _build_response(
                session_id=session_id,
                response_text=response_text,
                field_name=None,
                validation_passed=True,
                validation_error=None,
                session_complete=False,
                mode="clarify",
                action="confirm_action_start",
                session=session,
                quick_actions=build_quick_actions(lang, "clarify", "confirm_action_start", session.get("last_scheme"), False),
            )

    # First-turn auto prompt: when session is new and form has not started.
    if not session.get("user_profile") and not session.get("conversation_history"):
        first_question = get_field_question(current_field, lang)
        _append_history(session, "assistant", first_question)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=first_question,
            field_name=current_field,
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            session=session,
            quick_actions=build_quick_actions(lang, "action", None, session.get("last_scheme"), False),
        )

    if not cleaned_input or len(cleaned_input) < 2:
        retry_message = "कृपया स्पष्ट रूप से बताएं।" if lang == "hi" else "Please share your response clearly."
        _append_history(session, "assistant", retry_message)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=retry_message,
            field_name=current_field,
            validation_passed=False,
            validation_error="empty_or_too_short_input",
            session_complete=False,
            session=session,
            quick_actions=build_quick_actions(lang, "action", None, session.get("last_scheme"), False),
        )

    if _is_go_back_command(cleaned_input):
        _append_history(session, "user", cleaned_input)
        previous = _move_to_previous_field(session)
        question = get_field_question(previous, lang)
        session["session_complete"] = False
        _append_history(session, "assistant", question)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=question,
            field_name=session.get("next_field"),
            validation_passed=True,
            validation_error=None,
            session_complete=False,
            session=session,
            quick_actions=build_quick_actions(lang, "action", None, session.get("last_scheme"), False),
        )

    if _is_skip_command(cleaned_input):
        _append_history(session, "user", cleaned_input)
        question = get_field_question(current_field, lang)
        strict_message = (
            f"इस चरण को छोड़ा नहीं जा सकता। {question}"
            if lang == "hi"
            else f"This step cannot be skipped. {question}"
        )
        _append_history(session, "assistant", strict_message)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=strict_message,
            field_name=current_field,
            validation_passed=False,
            validation_error="step_cannot_skip",
            session_complete=False,
            session=session,
            quick_actions=build_quick_actions(lang, "action", None, session.get("last_scheme"), False),
        )

    _append_history(session, "user", cleaned_input)

    if session.get("ocr_confirmation_pending", False):
        return _handle_ocr_confirmation(session_id, session, cleaned_input, lang)

    if session.get("confirmation_state") == "pending" and get_next_field(session) is None:
        return _confirmation_handler(session_id, session, cleaned_input, lang)

    if current_field and cleaned_input:
        extracted = _extract_multi_field_values(cleaned_input)
        if extracted:
            extraction_result = _apply_extracted_fields(session, extracted, lang)
            if extraction_result.get("conflicts"):
                session["extraction_conflicts"] = extraction_result.get("conflicts")
                conflicts = dict(session.get("extraction_conflicts") or {})
                option_field = next(iter(conflicts.keys()))
                options = conflicts.get(option_field, [])
                option_text = "; ".join(f"{idx + 1}. {value}" for idx, value in enumerate(options))
                prompt = (
                    f"मुझे {FIELD_LABELS.get(option_field, {}).get('hi', option_field)} के लिए कई मान मिले। सही विकल्प चुनें: {option_text}"
                    if lang == "hi"
                    else f"I detected multiple values for {FIELD_LABELS.get(option_field, {}).get('en', option_field)}. Please choose: {option_text}"
                )
                _append_history(session, "assistant", prompt)
                update_session(session_id, session)
                return _build_response(
                    session_id=session_id,
                    response_text=prompt,
                    field_name=option_field,
                    validation_passed=False,
                    validation_error="extraction_conflict",
                    session_complete=False,
                    mode="clarify",
                    action="resolve_extraction_conflict",
                    session=session,
                    quick_actions=build_quick_actions(lang, "clarify", "resolve_extraction_conflict", session.get("last_scheme"), False),
                )

            if extraction_result.get("low_confidence"):
                field = next(iter(extraction_result["low_confidence"].keys()))
                prompt = (
                    f"मैंने {FIELD_LABELS.get(field, {}).get('hi', field)} का एक मान पकड़ा है, लेकिन भरोसा कम है। कृपया पुष्टि करें। {get_field_question(field, lang)}"
                    if lang == "hi"
                    else f"I detected a {FIELD_LABELS.get(field, {}).get('en', field)} value with low confidence. Please confirm it. {get_field_question(field, lang)}"
                )
                _append_history(session, "assistant", prompt)
                update_session(session_id, session)
                return _build_response(
                    session_id=session_id,
                    response_text=prompt,
                    field_name=field,
                    validation_passed=False,
                    validation_error="extraction_low_confidence",
                    session_complete=False,
                    mode="clarify",
                    action="confirm_extraction",
                    session=session,
                    quick_actions=build_quick_actions(lang, "clarify", "confirm_extraction", session.get("last_scheme"), False),
                )

            if extraction_result.get("applied"):
                next_after_extract = get_next_field(session)
                session["next_field"] = next_after_extract
                if next_after_extract is None:
                    session["confirmation_state"] = "pending"
                    session["dialogue_state"] = "confirming"
                    summary = _build_confirmation_summary(session, lang)
                    _append_history(session, "assistant", summary)
                    update_session(session_id, session)
                    return _build_response(
                        session_id=session_id,
                        response_text=summary,
                        field_name=None,
                        validation_passed=True,
                        validation_error=None,
                        session_complete=False,
                        mode="action",
                        action="confirm_details",
                        session=session,
                        quick_actions=build_quick_actions(lang, "action", "confirm_details", session.get("last_scheme"), False),
                    )
                question = get_field_question(next_after_extract, lang)
                _append_history(session, "assistant", question)
                update_session(session_id, session)
                return _build_response(
                    session_id=session_id,
                    response_text=question,
                    field_name=next_after_extract,
                    validation_passed=True,
                    validation_error=None,
                    session_complete=False,
                    mode="action",
                    action="collect_next_field",
                    session=session,
                    quick_actions=build_quick_actions(lang, "action", "collect_next_field", session.get("last_scheme"), False),
                )

        if _is_ambiguous_input(cleaned_input):
            message = "कृपया सटीक मान बताएं।" if lang == "hi" else "Please provide the exact value."
            question = get_field_question(current_field, lang)
            merged = f"{message}. {question}"
            _append_history(session, "assistant", merged)
            update_session(session_id, session)
            return _build_response(
                session_id=session_id,
                response_text=merged,
                field_name=current_field,
                validation_passed=False,
                validation_error="ambiguous_input",
                session_complete=False,
                session=session,
                quick_actions=build_quick_actions(lang, "action", None, session.get("last_scheme"), False),
            )

        agent_result = run_agent(session, cleaned_input, store_history=False)
        candidate_value = agent_result.get("field_value") or cleaned_input

        validation = validate_field(current_field, candidate_value, language=lang)
        if not validation.get("valid"):
            attempts = session.setdefault("invalid_attempts", {})
            attempts[current_field] = int(attempts.get(current_field, 0)) + 1
            retry_message = str(validation.get("error_message") or _validation_error_message(current_field, "invalid", lang))
            question = get_field_question(current_field, lang)
            if attempts[current_field] >= MAX_INVALID_ATTEMPTS_PER_FIELD:
                helper = (
                    "लगता है यह चरण कठिन हो रहा है। आप 'restart' बोलकर नई शुरुआत कर सकते हैं, या मैं उदाहरण देकर मदद करूँ।"
                    if lang == "hi"
                    else "This step seems difficult. You can say 'restart' to begin again, or I can guide with examples."
                )
                merged_message = f"{retry_message}. {helper}"
            else:
                merged_message = f"{retry_message}. {question}"
            _append_history(session, "assistant", merged_message)
            update_session(session_id, session)
            return _build_response(
                session_id=session_id,
                response_text=merged_message,
                field_name=current_field,
                validation_passed=False,
                validation_error=str(validation.get("error_code") or "invalid_input"),
                session_complete=False,
                session=session,
                quick_actions=build_quick_actions(lang, "action", "confirm_action_start" if attempts[current_field] >= MAX_INVALID_ATTEMPTS_PER_FIELD else None, session.get("last_scheme"), False),
            )

        session.setdefault("user_profile", {})[current_field] = str(validation.get("normalized") or "")
        session.setdefault("field_completion", {})[current_field] = True
        session.setdefault("invalid_attempts", {})[current_field] = 0
        active_fields = _session_fields(session)
        session["last_completed_field_index"] = active_fields.index(current_field) if current_field in active_fields else -1

    next_field = get_next_field(session)
    session["next_field"] = next_field
    session["session_complete"] = False

    if next_field is None:
        if not session.get("confirmation_done", False):
            session["confirmation_state"] = "pending"
            session["dialogue_state"] = "confirming"
            confirmation_text = _build_confirmation_summary(session, lang)
            confirmation_text = (
                f"{confirmation_text}\n\nकृपया सब जानकारी देखकर हाँ कहें या बदलाव बताएं।"
                if lang == "hi"
                else f"{confirmation_text}\n\nPlease review all details and say yes to submit, or ask to change any field."
            )
            _append_history(session, "assistant", confirmation_text)
            update_session(session_id, session)
            return _build_response(
                session_id=session_id,
                response_text=confirmation_text,
                field_name=None,
                validation_passed=True,
                validation_error=None,
                session_complete=False,
                session=session,
                quick_actions=build_quick_actions(lang, "action", None, session.get("last_scheme"), False),
            )

        session["confirmation_state"] = "confirmed"
        session["session_complete"] = True
        session["dialogue_state"] = "completed"
        completion_message = f"{get_field_question(None, lang)}\n\n{_closing_summary(session, lang)}"
        _append_history(session, "assistant", completion_message)
        update_session(session_id, session)
        return _build_response(
            session_id=session_id,
            response_text=completion_message,
            field_name=None,
            validation_passed=True,
            validation_error=None,
            session_complete=True,
            session=session,
            quick_actions=build_quick_actions(lang, "action", None, session.get("last_scheme"), True),
        )

    prompt_result = run_agent(session, "")
    question = prompt_result.get("next_question_text") or get_field_question(session["next_field"], lang)
    session["dialogue_state"] = "collecting_info"

    update_session(session_id, session)

    return _build_response(
        session_id=session_id,
        response_text=question,
        field_name=session["next_field"],
        validation_passed=True,
        validation_error=None,
        session_complete=False,
        session=session,
        quick_actions=build_quick_actions(lang, "action", None, session.get("last_scheme"), False),
    )


class ConversationService:
    def process(self, session_id: str, user_input: str, language: Optional[str] = None, debug: bool = False) -> Dict[str, Any]:
        log_event(
            "conversation_service_start",
            endpoint="conversation_service",
            status="started",
            session_id=session_id,
            user_input_length=len(user_input or ""),
            user_input_fingerprint=fingerprint_text(user_input),
        )
        try:
            session_for_limit = get_session(session_id) or create_session(session_id)
            lang = normalize_language_code(language or session_for_limit.get("language") or "en", default="en")
            limiter_key = _rate_limit_subject(session_id, session_for_limit)
            if _is_rate_limited(limiter_key):
                limited_result = _build_rate_limit_response(session_id=session_id, language=lang, session=session_for_limit)
                limited_result = _apply_response_length_control(limited_result)
                log_event(
                    "conversation_service_rate_limited",
                    endpoint="conversation_service",
                    status="throttled",
                    session_id=session_id,
                    user_id=str(session_for_limit.get("user_id") or ""),
                    limit_window_seconds=RATE_LIMIT_WINDOW_SECONDS,
                    limit_max_requests=RATE_LIMIT_MAX_REQUESTS,
                )
                return limited_result

            if not MVP_PIPELINE_ENABLED:
                result = handle_conversation(session_id=session_id, user_input=user_input, language=language, debug=debug)
            else:
                normalized = normalize_for_intent(user_input or "", language_hint=lang)
                cleaned_input = str(normalized.intent_text or normalized.normalized_text or normalized.raw_text or "").strip()
                if not cleaned_input.strip():
                    cleaned_input = str(user_input or "").strip()

                safe_session = session_for_limit
                safe_session["language"] = lang

                fallback_used = False
                intent_name = INTENT_SCHEME_QUERY
                confidence_pct = 0.0
                intent_debug: Dict[str, Any] = {
                    "fallback_used": False,
                    "source": "mvp_pipeline",
                }
                rag_match: Optional[Dict[str, Any]] = None
                recommendations: List[str] = []

                try:
                    intent_result = _run_intent_with_timeout(cleaned_input)
                    intent_name = str(intent_result.get("canonical_intent") or intent_result.get("intent") or INTENT_SCHEME_QUERY)
                    confidence_pct = float(intent_result.get("confidence") or 0.0)
                    if confidence_pct <= 1.0:
                        confidence_pct = round(confidence_pct * 100.0, 2)
                    intent_debug.update(
                        {
                            "confidence": confidence_pct,
                            "normalized_intent": intent_name,
                            "raw_model_output": intent_result.get("debug", {}).get("raw_model_output") if isinstance(intent_result.get("debug"), dict) else None,
                        }
                    )

                    intent_confidence_pct = float(intent_result.get("confidence") or 0.0)
                    if intent_confidence_pct <= 1.0:
                        intent_confidence_pct = intent_confidence_pct * 100.0
                    scheme_hint = ""
                    if intent_confidence_pct >= 60.0:
                        scheme_hint = str(intent_result.get("scheme") or safe_session.get("last_scheme") or "").strip()
                    rag_match, recommendations, _ = _run_rag_with_timeout(cleaned_input, lang, scheme_hint)
                except Exception:
                    fallback_used = True
                    record_fallback()

                scheme_name = ""
                response_text = ""
                if rag_match and isinstance(rag_match, dict):
                    scheme_name = str(rag_match.get("confirmation") or "").strip()
                    response_text = str(rag_match.get("explanation") or rag_match.get("next_step") or "").strip()
                    if not recommendations:
                        recommendations = [str(item) for item in (rag_match.get("schemes") or []) if str(item).strip()]

                if not response_text:
                    fallback_text, forced_scheme = _simple_fallback_text(cleaned_input, lang)
                    response_text = fallback_text
                    if forced_scheme and not scheme_name:
                        scheme_name = forced_scheme
                    if forced_scheme and not recommendations:
                        recommendations = [forced_scheme]
                    fallback_used = True
                    intent_debug["fallback_used"] = True
                    intent_debug["fallback_reason"] = "intent_or_rag_unavailable"

                safe_session["last_intent"] = intent_name
                safe_session["last_action"] = "info"
                if scheme_name:
                    safe_session["last_scheme"] = scheme_name
                    safe_session["selected_scheme"] = scheme_name
                safe_session["_safety_fallback_triggered"] = bool(fallback_used)
                safe_session["_safety_low_confidence"] = bool(fallback_used)
                safe_session["_safety_ambiguous"] = False
                update_session(session_id, safe_session)

                result = _build_response(
                    session_id=session_id,
                    response_text=response_text,
                    field_name=None,
                    validation_passed=True,
                    validation_error=None,
                    session_complete=False,
                    mode="info",
                    action="info",
                    session=safe_session,
                    quick_actions=build_quick_actions(lang, "info", "info", scheme_name or None, False),
                    voice_text=response_text,
                    recommended_schemes=recommendations[:3],
                )
                result["primary_intent"] = intent_name
                result["secondary_intents"] = []
                result["intent_debug"] = intent_debug
                result["context_applied"] = False
                result["scheme_details"] = (
                    build_scheme_details(
                        INTENT_SCHEME_QUERY,
                        {
                            "confirmation": scheme_name,
                            "explanation": response_text,
                            "next_step": "",
                        },
                    )
                    if scheme_name
                    else None
                )

            # Enrich session with compact semantic memory for context-aware follow-up replies.
            session = get_session(session_id)
            _update_semantic_memory(session, user_input, result, result.get("primary_intent") or "")
            update_session(session_id, session)
            _persist_user_history_async(session_id, session, user_input, result)
            result = _apply_response_length_control(result)
            fallback_used = bool(((result.get("intent_debug") or {}).get("fallback_used")) or False)
            scheme_detection = result.get("scheme_detection") or (result.get("debug") or {}).get("scheme_detection") or {}
            safety_flags = result.get("safety") or (result.get("debug") or {}).get("safety") or {}
            rag_debug = result.get("rag_debug") or (result.get("debug") or {}).get("rag_debug") or {}

            log_event(
                "conversation_critical_events",
                endpoint="conversation_service",
                status="observed",
                session_id=session_id,
                scheme_decision=scheme_detection.get("decision"),
                selected_scheme=scheme_detection.get("selected_scheme") or session.get("selected_scheme") or session.get("last_scheme"),
                rag_debug=rag_debug,
                safety=safety_flags,
            )
            log_event(
                "conversation_service_success",
                endpoint="conversation_service",
                status="success",
                session_id=session_id,
                user_input_length=len(user_input or ""),
                user_input_fingerprint=fingerprint_text(user_input),
                detected_intent=result.get("primary_intent"),
                confidence=(result.get("intent_debug") or {}).get("confidence"),
                selected_scheme=session.get("selected_scheme") or session.get("last_scheme"),
                fallback_used=fallback_used,
            )
            return result
        except Exception as exc:
            log_event(
                "conversation_service_failure",
                level="error",
                endpoint="conversation_service",
                status="failure",
                session_id=session_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            try:
                safe_session = get_session(session_id) or create_session(session_id)
            except Exception:
                safe_session = create_session(session_id)
            safe_session["_safety_fallback_triggered"] = True
            safe_session["_safety_low_confidence"] = True
            safe_session["_safety_ambiguous"] = False
            safe_session["last_action"] = "safe_fallback"
            safe_message = "Something went wrong. Please try again."
            update_session(session_id, safe_session)
            return _build_response(
                session_id=session_id,
                response_text=safe_message,
                field_name=None,
                validation_passed=True,
                validation_error=None,
                session_complete=False,
                mode="clarify",
                action="safe_fallback",
                session=safe_session,
                quick_actions=build_quick_actions("en", "clarify", "safe_fallback", safe_session.get("last_scheme"), False),
                voice_text=safe_message,
            )

    def merge_ocr(self, session: Dict[str, Any], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        log_event("conversation_service_merge_ocr_start", endpoint="conversation_service", status="success")
        try:
            result = merge_ocr_data(session, extracted_data)
            log_event("conversation_service_merge_ocr_success", endpoint="conversation_service", status="success")
            return result
        except Exception as exc:
            log_event("conversation_service_merge_ocr_failure", level="error", endpoint="conversation_service", status="failure", error_type=type(exc).__name__)
            raise

    def ocr_confirmation(self, session: Dict[str, Any], ocr_data: Dict[str, Any], language: str) -> str:
        log_event("conversation_service_ocr_confirmation_start", endpoint="conversation_service", status="success")
        try:
            result = get_ocr_confirmation_message(session, ocr_data, language)
            log_event("conversation_service_ocr_confirmation_success", endpoint="conversation_service", status="success")
            return result
        except Exception as exc:
            log_event("conversation_service_ocr_confirmation_failure", level="error", endpoint="conversation_service", status="failure", error_type=type(exc).__name__)
            raise
