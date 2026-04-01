import re
from typing import Any, Dict, List, Optional

from backend.response_formatter import format_short_voice_text
from backend.shared.language.language import normalize_language_code
from backend.shared.security.privacy import sanitize_profile_for_response
try:
    from backend.data.scheme_data import SCHEME_DATA
except Exception:
    try:
        from src.utils.scheme_data import SCHEME_DATA
    except Exception:
        from backend.src.utils.scheme_data import SCHEME_DATA


CONTROLLED_SCHEME_CLARIFICATION = (
    "I currently provide detailed information for selected schemes. Please ask about a specific scheme like solar, housing, loan, etc."
)


def format_response(text: str, language: str) -> str:
    content = str(text or "").strip()
    if not content:
        return content

    if language == "hi":
        replacements = {
            "कृपया बताएं": "बताइए",
            "कृपया": "ज़रा",
            "क्या आप": "आप",
            "मैं आपकी मदद कर सकता हूँ": "मैं आपकी पूरी मदद करूँगा",
            "क्या यह सही है": "ये ठीक है ना",
        }
        for source, target in replacements.items():
            content = content.replace(source, target)
    else:
        replacements = {
            "Please provide": "Share",
            "Please tell me": "Tell me",
            "I can assist with this.": "I can support this.",
            "Do you want": "Would you like",
        }
        for source, target in replacements.items():
            content = content.replace(source, target)

    # Smooth punctuation keeps replies sounding less robotic.
    content = re.sub(r"\s+", " ", content).strip()
    content = content.replace("..", ".")
    return content


def micro_latency_ack(language: str) -> str:
    return "ठीक है, एक पल..." if language == "hi" else "Got it, just a moment..."


def merge_control_actions(language: str, quick_actions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    controls = [
        {
            "label": "जारी रखें" if language == "hi" else "Continue",
            "value": "continue_flow",
        },
        {
            "label": "सुझाव बदलें" if language == "hi" else "Refine",
            "value": "refine_suggestions",
        },
        {
            "label": "अभी आवेदन करें" if language == "hi" else "Apply",
            "value": "apply_now_direct",
        },
    ]
    seen = set()
    merged: List[Dict[str, str]] = []
    for action in [*quick_actions, *controls]:
        value = str(action.get("value") or "").strip()
        label = str(action.get("label") or "").strip()
        if not value or not label or value in seen:
            continue
        seen.add(value)
        merged.append({"label": label, "value": value})
    return merged


def display_aligned_text(text: str, language: str) -> str:
    words = [token for token in (text or "").replace("\n", " ").split() if token]
    if len(words) <= 34:
        return text
    lead = " ".join(words[:30]).rstrip(".,;: ")
    return f"{lead}..." if language == "en" else f"{lead}..."


def short_answer(text: str, language: str) -> str:
    words = [token for token in (text or "").replace("\n", " ").split() if token]
    if len(words) <= 18:
        return text
    concise = " ".join(words[:16]).rstrip(".,;: ")
    return f"{concise}..." if language == "en" else f"{concise}..."


def build_response_payload(
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
    field_labels: Optional[Dict[str, Dict[str, str]]] = None,
    session_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    steps_done = 0
    active_fields = list(session_fields or [])
    steps_total = len(active_fields)
    if session:
        completion = session.get("field_completion", {})
        steps_done = sum(1 for field in active_fields if completion.get(field))

    language = normalize_language_code((session or {}).get("language", "en") if session else "en", default="en")
    completed_fields = []
    labels = field_labels or {}
    if session:
        completion = session.get("field_completion", {})
        for field in active_fields:
            if completion.get(field):
                completed_fields.append(labels.get(field, {}).get(language, field))

    natural_response = format_response(response_text, language)
    natural_voice = format_response(voice_text or natural_response, language)
    synced_display = display_aligned_text(natural_response, language)
    base_quick_actions = list(quick_actions or [])
    merged_actions = merge_control_actions(language, base_quick_actions)

    return {
        "session_id": session_id,
        "response_text": synced_display,
        "voice_text": natural_voice or format_short_voice_text(natural_response, language),
        "instant_ack": micro_latency_ack(language),
        "primary_intent": (session or {}).get("last_intent"),
        "secondary_intents": (session or {}).get("last_secondary_intents", []),
        "field_name": field_name,
        "validation_passed": validation_passed,
        "validation_error": validation_error,
        "session_complete": session_complete,
        "mode": mode,
        "action": action,
        "steps_done": steps_done,
        "steps_total": steps_total,
        "completed_fields": completed_fields,
        "scheme_details": scheme_details,
        "quick_actions": merged_actions,
        "recommended_schemes": recommended_schemes or [],
        "user_profile": sanitize_profile_for_response((session or {}).get("user_profile", {})),
        "intent_debug": (session or {}).get("_intent_debug"),
    }


def build_hackathon_response(
    *,
    success: bool,
    response_type: str,
    message: str,
    summary: str,
    reason: str,
    next_step: str,
    data: Optional[Dict[str, Any]] = None,
    confidence: float = 0.0,
    language: str = "en",
) -> Dict[str, Any]:
    try:
        normalized_language = normalize_language_code(language, default="en")
        payload = dict(data or {})
        payload.setdefault("summary", str(summary or "").strip())
        payload.setdefault("eligibility", "")
        payload.setdefault("steps", "")

        # Backward-compatible metadata while preserving the new structured shape.
        payload.setdefault("reason", str(reason or "").strip())
        payload.setdefault("next_step", str(next_step or "").strip())

        normalized_type = str(response_type or "general").strip() or "general"
        scheme_name = str(payload.get("scheme") or "").strip()
        normalized_message = str(message or "").strip()

        if normalized_type in {"scheme_info", "eligibility", "application_help"}:
            scheme_message = _build_scheme_data_message(normalized_type, scheme_name)
            if scheme_message is not None:
                normalized_message = scheme_message
            else:
                normalized_message = CONTROLLED_SCHEME_CLARIFICATION
                payload.setdefault("mode", "clarification")
        generic_placeholders = {
            "here is the information",
            "here is the information.",
            "here are the details",
            "here are the details.",
        }
        is_generic_scheme_placeholder = (
            normalized_type == "scheme_info"
            and normalized_message.lower() in generic_placeholders
        )

        if normalized_type in {"scheme_info", "eligibility", "application_help"} and (not normalized_message or is_generic_scheme_placeholder):
            normalized_message = CONTROLLED_SCHEME_CLARIFICATION
        elif not normalized_message or is_generic_scheme_placeholder:
            normalized_message = generate_default_message(normalized_type, scheme_name, normalized_language)

        if not normalized_message:
            normalized_message = "Here is the information you requested."

        return {
            "success": bool(success),
            "type": normalized_type,
            "message": normalized_message,
            "data": payload,
            "confidence": float(confidence or 0.0),
        }
    except Exception:
        return {
            "success": True,
            "type": "general",
            "message": "यह रही आपकी मांगी गई जानकारी।" if normalize_language_code(language, default="en") == "hi" else "Here is the information you requested.",
            "data": {
                "summary": "",
                "reason": "",
                "next_step": "",
            },
            "confidence": 0.0,
        }


def generate_default_message(intent: str, scheme_name: str, language: str = "en") -> str:
    normalized_intent = str(intent or "").strip().lower()
    normalized_scheme = str(scheme_name or "").strip()
    normalized_language = normalize_language_code(language, default="en")

    if normalized_intent in {"scheme_info", "eligibility", "application_help"}:
        return CONTROLLED_SCHEME_CLARIFICATION

    if normalized_language == "hi":
        return "यह रही आपकी मांगी गई जानकारी।"

    return "Here is the information you requested."


def _lookup_scheme_data(scheme_name: str) -> Optional[Dict[str, str]]:
    key = str(scheme_name or "").strip().lower()
    if not key:
        return None
    value = SCHEME_DATA.get(key)
    if isinstance(value, dict):
        return value
    return None


def _build_scheme_data_message(intent: str, scheme_name: str) -> Optional[str]:
    scheme_info = _lookup_scheme_data(scheme_name)
    if not scheme_info:
        return None

    summary = str(scheme_info.get("summary") or "").strip()
    eligibility = str(scheme_info.get("eligibility") or "").strip()
    steps = str(scheme_info.get("steps") or "").strip()

    if intent == "scheme_info":
        if not summary and not eligibility and not steps:
            return None
        summary_parts = [segment.strip().rstrip(".") for segment in summary.replace("\n", ". ").split(". ") if segment.strip()]
        eligibility_parts = [segment.strip().rstrip(".") for segment in eligibility.replace("\n", ". ").split(". ") if segment.strip()]
        line_1 = summary_parts[0] if summary_parts else f"{scheme_name.title()} is a government support scheme"
        benefit_line = ""
        if len(summary_parts) > 1:
            benefit_line = summary_parts[1]
        elif eligibility_parts:
            benefit_line = eligibility_parts[0]
        elif steps:
            benefit_line = steps.split(".")[0].strip()
        if not benefit_line:
            benefit_line = "It provides benefits to eligible applicants"
        return f"{line_1}.\nBenefit: {benefit_line}."
    if intent == "eligibility":
        if eligibility:
            bullets = [segment.strip().rstrip(".") for segment in eligibility.replace("\n", ". ").split(". ") if segment.strip()]
            while len(bullets) < 3:
                bullets.append("Provide valid supporting documents as per official guidelines")
            return "Eligibility Criteria:\n- " + "\n- ".join(bullets)
        return None
    if intent == "application_help":
        if steps:
            normalized_steps = re.sub(r"(\d+)\)\s*", r"\1. ", steps)
            if not re.search(r"\b1\.\s", normalized_steps):
                parts = [segment.strip() for segment in normalized_steps.replace("\n", " ").split(". ") if segment.strip()]
                normalized_steps = " ".join(f"{idx + 1}. {part}." for idx, part in enumerate(parts))
            return "Application Process:\n" + normalized_steps
        return None
    return None
