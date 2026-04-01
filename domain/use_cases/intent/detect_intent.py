from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ....infrastructure.session.session_store import get_session, update_session


async def _detect_intent_core(
    *,
    text: str,
    normalized_text: str,
    session_id: Optional[str],
    debug: bool,
    intent_service: Any,
    timings: Optional[Dict[str, Any]] = None,
    get_session_fn: Callable[[str], Dict[str, Any]] = get_session,
    update_session_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]] = update_session,
) -> Dict[str, Any]:
    resolved_session_id = (session_id or "").strip()
    session: Optional[Dict[str, Any]] = None
    if resolved_session_id:
        session = get_session_fn(resolved_session_id)

    session_context = {
        "last_intent": str((session or {}).get("last_intent") or ""),
        "last_action": str((session or {}).get("last_action") or ""),
        "last_scheme": str((session or {}).get("last_scheme") or (session or {}).get("selected_scheme") or ""),
    }

    result = await intent_service.detect_async(
        text=text,
        debug=debug,
        timings=timings or {},
        session_context=session_context,
    )

    lowered = normalized_text.lower()
    if lowered in {"haan", "yes", "continue"} and session is not None:
        previous_intent = str(session.get("last_intent") or "").strip()
        if previous_intent:
            result["intent"] = previous_intent
            result["canonical_intent"] = previous_intent
            if debug:
                result["debug"] = {
                    **(result.get("debug") or {}),
                    "context_used": True,
                    "inferred_from_previous_intent": True,
                }

    resolved_intent = str(result.get("canonical_intent") or result.get("intent") or "").strip()
    if resolved_session_id and session is not None and resolved_intent:
        previous_value = str(session.get("last_intent") or "").strip()
        detected_scheme = str(result.get("scheme") or "").strip()
        scheme_changed = detected_scheme and str(session.get("last_scheme") or "").strip() != detected_scheme
        if previous_value != resolved_intent or scheme_changed:
            session["last_intent"] = resolved_intent
            if detected_scheme:
                session["last_scheme"] = detected_scheme
            update_session_fn(resolved_session_id, session)

    return result


async def detect_intent(
    *,
    text: str,
    normalized_text: str,
    session_id: Optional[str],
    debug: bool,
    intent_service: Any,
    timings: Optional[Dict[str, Any]] = None,
    get_session_fn: Callable[[str], Dict[str, Any]] = get_session,
    update_session_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]] = update_session,
) -> Dict[str, Any]:
    return await _detect_intent_core(
        text=text,
        normalized_text=normalized_text,
        session_id=session_id,
        debug=debug,
        intent_service=intent_service,
        timings=timings,
        get_session_fn=get_session_fn,
        update_session_fn=update_session_fn,
    )
