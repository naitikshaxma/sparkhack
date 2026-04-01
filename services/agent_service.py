import json
from typing import Dict, Any

from backend.shared.language.language import normalize_language_code
from backend.shared.session.form_schema import get_field_question, get_next_field
from backend.shared.security.privacy import redact_sensitive_text

SYSTEM_PROMPT = (
    "You are a government scheme assistant. "
    "Ask ONLY one question at a time. "
    "Always respond ONLY in the user's selected language code provided in session context. "
    "Respond ONLY in {language}. Do not mix languages. "
    "Do NOT mix languages in the same reply. "
    "If language is Hindi, respond fully in Hindi using Devanagari script only. "
    "If language is English, respond fully in English. "
    "Never transliterate or switch script unless that is the selected language. "
    "NEVER use technical field keys like annual_income in user-facing text. "
    "Validate conversationally. If invalid, explain briefly and ask again. "
    "Keep replies short and human-like. Always return JSON only. "
    "Response JSON keys must be exactly: field_name, field_value, validation_passed, "
    "validation_error, next_question_text, session_complete."
)


MAX_HISTORY_MESSAGES = 10


def _trim_history(session: Dict[str, Any]) -> None:
    history = session.setdefault("conversation_history", [])
    if len(history) > MAX_HISTORY_MESSAGES:
        session["conversation_history"] = history[-MAX_HISTORY_MESSAGES:]


def _append_history(session: Dict[str, Any], role: str, content: str) -> None:
    if not content:
        return
    session.setdefault("conversation_history", []).append({"role": role, "content": redact_sensitive_text(content)})
    _trim_history(session)


def _fallback_agent_response(session: Dict[str, Any]) -> Dict[str, Any]:
    next_field = get_next_field(session)
    language = normalize_language_code(session.get("language", "en"), default="en")
    if next_field is None:
        complete_message = get_field_question(None, language)
        return {
            "field_name": None,
            "field_value": None,
            "validation_passed": True,
            "validation_error": None,
            "next_question_text": complete_message,
            "session_complete": True,
        }

    return {
        "field_name": next_field,
        "field_value": None,
        "validation_passed": True,
        "validation_error": None,
        "next_question_text": get_field_question(next_field, language),
        "session_complete": False,
    }


def run_agent(session: Dict[str, Any], user_input: str, store_history: bool = True) -> Dict[str, Any]:
    # Local deterministic form flow; OpenAI is intentionally disabled.
    fallback = _fallback_agent_response(session)
    fallback["mode"] = "local"
    fallback["response"] = "Structured local flow active"
    if store_history:
        if user_input.strip():
            _append_history(session, "user", user_input.strip())
        _append_history(session, "assistant", str(fallback.get("next_question_text") or ""))
    return fallback
