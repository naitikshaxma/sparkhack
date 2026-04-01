from __future__ import annotations

import re
from contextvars import ContextVar
from fastapi import HTTPException
from backend.shared.security.privacy import mask_aadhaar

try:
    import bcrypt  # type: ignore
except Exception:  # pragma: no cover - optional dependency until installed
    bcrypt = None

_CURRENT_USER_ID: ContextVar[str] = ContextVar("current_user_id", default="")
_AADHAAR_DIGITS_RE = re.compile(r"\D")


def set_current_user_id(user_id: str) -> None:
    _CURRENT_USER_ID.set((user_id or "").strip())


def get_current_user_id() -> str:
    return (_CURRENT_USER_ID.get() or "").strip()


def clear_current_user_id() -> None:
    _CURRENT_USER_ID.set("")


def hash_password(password: str) -> str:
    raw = str(password or "")
    if len(raw) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if bcrypt is None:
        raise HTTPException(status_code=500, detail="bcrypt is not installed")
    return bcrypt.hashpw(raw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    if bcrypt is None:
        return False
    raw = str(password or "")
    stored = str(password_hash or "")
    if not raw or not stored:
        return False
    try:
        return bool(bcrypt.checkpw(raw.encode("utf-8"), stored.encode("utf-8")))
    except Exception:
        return False


def protect_aadhaar(aadhaar_value: str) -> str:
    digits = _AADHAAR_DIGITS_RE.sub("", str(aadhaar_value or ""))
    if not digits:
        return ""
    if len(digits) != 12:
        raise HTTPException(status_code=400, detail="Aadhaar must be exactly 12 digits")
    # Store a masked representation to avoid persisting raw Aadhaar.
    return mask_aadhaar(digits)
