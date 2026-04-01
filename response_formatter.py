from typing import Any, Dict, List, Optional

from .intents import INTENT_SCHEME_QUERY


def _qa(label: str, value: str) -> Dict[str, str]:
    return {"label": label, "value": value}


def format_info_text(response: Dict[str, Any], language: str) -> str:
    parts = [
        response.get("confirmation", ""),
        response.get("explanation", ""),
        response.get("next_step", ""),
    ]
    content = "\n".join(part for part in parts if part).strip()
    action_hint = (
        "Ask me to start application when you are ready."
        if language == "en"
        else "जब तैयार हों, आवेदन शुरू करने के लिए कहें।"
    )
    return f"{content}\n\n{action_hint}" if content else action_hint


def build_scheme_details(intent: str, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if intent != INTENT_SCHEME_QUERY:
        return None

    return {
        "title": response.get("confirmation", ""),
        "description": response.get("explanation", ""),
        "next_step": response.get("next_step", ""),
    }


def format_short_voice_text(response: str, language: str, max_words: int = 18) -> str:
    text = (response or "").replace("\n", " ").strip()
    if not text:
        return ""

    words = text.split()
    if len(words) <= max_words:
        return text

    concise = " ".join(words[:max_words]).rstrip(".,;: ")
    if language == "hi":
        return f"{concise}... आगे बताऊँ?"
    return f"{concise}... want more details?"


def build_quick_actions(
    language: str,
    mode: str,
    action: Optional[str],
    last_scheme: Optional[str],
    session_complete: bool,
) -> List[Dict[str, str]]:
    lang = "hi" if language == "hi" else "en"

    if action == "confirm_action_start":
        return [
            _qa("हाँ" if lang == "hi" else "Yes", "confirm_yes"),
            _qa("नहीं" if lang == "hi" else "No", "confirm_no"),
        ]

    if mode == "clarify":
        return [
            _qa("जानकारी चाहिए" if lang == "hi" else "Need information", "need_information"),
            _qa("आवेदन शुरू करें" if lang == "hi" else "Start application", "start_application"),
        ]

    if mode == "info":
        scheme_hint = (
            f"{last_scheme} पात्रता" if lang == "hi" and last_scheme else
            f"{last_scheme} eligibility" if last_scheme else
            "पात्रता बताएं" if lang == "hi" else
            "Show eligibility"
        )
        return [
            _qa(scheme_hint, "show_eligibility"),
            _qa("और जानकारी" if lang == "hi" else "More info", "more_info"),
            _qa("आवेदन शुरू करें" if lang == "hi" else "Apply now", "start_application"),
        ]

    if session_complete:
        return [
            _qa("ऑटो फिल करें" if lang == "hi" else "Auto fill form", "auto_fill_form"),
            _qa("फिर से शुरू करें" if lang == "hi" else "Restart", "restart_session"),
        ]

    if action == "ask_to_apply":
        return [
            _qa("आवेदन शुरू करें" if lang == "hi" else "Apply now", "start_application"),
            _qa("और जानकारी" if lang == "hi" else "More info", "more_info"),
        ]

    return [
        _qa("अगला चरण" if lang == "hi" else "Next step", "next_step"),
        _qa("आवेदन स्थिति" if lang == "hi" else "Application status", "application_status"),
    ]


def build_recommendation_quick_actions(recommendations: List[str], language: str) -> List[Dict[str, str]]:
    lang = "hi" if language == "hi" else "en"
    actions: List[Dict[str, str]] = []
    for recommendation in recommendations[:2]:
        label = recommendation.strip()
        if label:
            actions.append(_qa(label, f"recommend_scheme:{label}"))

    actions.append(_qa("और जानकारी" if lang == "hi" else "More info", "more_info"))
    return actions
