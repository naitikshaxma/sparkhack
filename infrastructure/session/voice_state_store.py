from __future__ import annotations

from typing import Any, Dict

from ... import voice_state as legacy_voice_state


def set_voice_state(session_id: str, state: str) -> None:
    return legacy_voice_state.set_voice_state(session_id, state)


def interrupt_voice(session_id: str) -> None:
    return legacy_voice_state.interrupt_voice(session_id)


def clear_interrupt(session_id: str) -> None:
    return legacy_voice_state.clear_interrupt(session_id)


def is_interrupted(session_id: str) -> bool:
    return legacy_voice_state.is_interrupted(session_id)


def begin_stream(session_id: str) -> int:
    return legacy_voice_state.begin_stream(session_id)


def is_stream_active(session_id: str, generation: int) -> bool:
    return legacy_voice_state.is_stream_active(session_id, generation)


def end_stream(session_id: str, generation: int) -> None:
    return legacy_voice_state.end_stream(session_id, generation)


def get_voice_state(session_id: str) -> Dict[str, Any]:
    return legacy_voice_state.get_voice_state(session_id)
