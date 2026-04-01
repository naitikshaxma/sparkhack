from __future__ import annotations

import threading
from dataclasses import dataclass
import os
import time


@dataclass
class SessionVoiceState:
    state: str = "idle"
    interrupted: bool = False
    stream_generation: int = 0
    last_seen: float = 0.0


_LOCK = threading.RLock()
_STATE: dict[str, SessionVoiceState] = {}
_MAX_VOICE_STATE_ENTRIES = max(100, int((os.getenv("MAX_VOICE_STATE_ENTRIES") or "2000").strip() or "2000"))


def _prune_state_locked() -> None:
    if len(_STATE) <= _MAX_VOICE_STATE_ENTRIES:
        return
    # Drop oldest entries first.
    sorted_items = sorted(_STATE.items(), key=lambda item: item[1].last_seen)
    excess = len(_STATE) - _MAX_VOICE_STATE_ENTRIES
    for idx in range(excess):
        key, _ = sorted_items[idx]
        _STATE.pop(key, None)


def _get_or_create(session_id: str) -> SessionVoiceState:
    key = (session_id or "").strip() or "anonymous"
    with _LOCK:
        value = _STATE.get(key)
        if value is None:
            value = SessionVoiceState()
            _STATE[key] = value
        value.last_seen = time.time()
        _prune_state_locked()
        return value


def set_voice_state(session_id: str, state: str) -> None:
    with _LOCK:
        current = _get_or_create(session_id)
        current.state = state
        if state != "interrupted":
            current.interrupted = False


def interrupt_voice(session_id: str) -> None:
    with _LOCK:
        current = _get_or_create(session_id)
        current.interrupted = True
        current.state = "interrupted"
        current.stream_generation += 1


def clear_interrupt(session_id: str) -> None:
    with _LOCK:
        current = _get_or_create(session_id)
        current.interrupted = False
        if current.state == "interrupted":
            current.state = "idle"


def is_interrupted(session_id: str) -> bool:
    with _LOCK:
        current = _get_or_create(session_id)
        return bool(current.interrupted)


def begin_stream(session_id: str) -> int:
    with _LOCK:
        current = _get_or_create(session_id)
        current.stream_generation += 1
        current.interrupted = False
        return current.stream_generation


def is_stream_active(session_id: str, generation: int) -> bool:
    with _LOCK:
        current = _get_or_create(session_id)
        return current.stream_generation == generation


def end_stream(session_id: str, generation: int) -> None:
    with _LOCK:
        current = _get_or_create(session_id)
        if current.stream_generation != generation:
            return
        if current.state != "interrupted":
            current.state = "idle"


def get_voice_state(session_id: str) -> dict:
    with _LOCK:
        current = _get_or_create(session_id)
        return {
            "state": current.state,
            "interrupted": current.interrupted,
        }
