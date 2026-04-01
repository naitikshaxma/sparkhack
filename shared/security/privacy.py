import hashlib
import re
from typing import Any, Dict, Iterable, List


AADHAAR_RE = re.compile(r"(?<!\d)(\d{12})(?!\d)")
AADHAAR_LOOSE_RE = re.compile(r"(?<!\d)[\s\-().]*\+?(?:\d[\s\-().]*?){12}(?!\d)")
PHONE_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")
PHONE_LOOSE_RE = re.compile(r"(?<!\d)[\s\-().]*\+?(?:\d[\s\-().]*?){10}(?!\d)")
EMAIL_RE = re.compile(r"(?i)([A-Z0-9._%+-])([A-Z0-9._%+-]*)(@[A-Z0-9.-]+\.[A-Z]{2,})")
ACCOUNT_RE = re.compile(r"(?<!\d)(\d{13,18})(?!\d)")
ACCOUNT_LOOSE_RE = re.compile(r"(?<!\d)[\s\-().]*\+?(?:\d[\s\-().]*?){13,18}(?!\d)")
ALLOWED_PROFILE_FIELDS = {"full_name", "phone", "aadhaar_number", "annual_income"}


def digits_only(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def fingerprint_text(value: str, length: int = 12) -> str:
    text = str(value or "")
    if not text:
        return ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[: max(6, int(length))]


def _is_masked(value: str) -> bool:
    text = str(value or "")
    return "X" in text or "*" in text


def mask_aadhaar(value: str) -> str:
    if _is_masked(value):
        return str(value)
    digits = digits_only(value)
    if len(digits) == 12:
        return f"XXXX XXXX {digits[-4:]}"
    if len(digits) >= 8:
        return f"{'X' * (len(digits) - 4)}{digits[-4:]}"
    return digits


def mask_phone(value: str) -> str:
    if _is_masked(value):
        return str(value)
    digits = digits_only(value)
    if len(digits) == 10:
        return f"{'X' * 7}{digits[-3:]}"
    return digits


def _mask_email_match(match: re.Match[str]) -> str:
    first_char = match.group(1)
    domain = match.group(3)
    return f"{first_char}***{domain}"


def _mask_aadhaar_match(match: re.Match[str]) -> str:
    digits = match.group(1)
    return mask_aadhaar(digits)


def _mask_aadhaar_loose_match(match: re.Match[str]) -> str:
    digits = digits_only(match.group(0))
    if len(digits) == 12:
        return mask_aadhaar(digits)
    return match.group(0)


def _mask_phone_match(match: re.Match[str]) -> str:
    digits = match.group(1)
    return mask_phone(digits)


def _mask_phone_loose_match(match: re.Match[str]) -> str:
    digits = digits_only(match.group(0))
    if len(digits) == 10:
        return mask_phone(digits)
    return match.group(0)


def _mask_account_match(match: re.Match[str]) -> str:
    digits = match.group(1)
    if len(digits) <= 4:
        return "X" * len(digits)
    return f"{'X' * (len(digits) - 4)}{digits[-4:]}"


def _mask_account_loose_match(match: re.Match[str]) -> str:
    digits = digits_only(match.group(0))
    if 13 <= len(digits) <= 18:
        if len(digits) <= 4:
            return "X" * len(digits)
        return f"{'X' * (len(digits) - 4)}{digits[-4:]}"
    return match.group(0)


def redact_sensitive_data(text: str) -> str:
    if not text:
        return text
    content = str(text)
    content = EMAIL_RE.sub(_mask_email_match, content)
    content = ACCOUNT_LOOSE_RE.sub(_mask_account_loose_match, content)
    content = AADHAAR_LOOSE_RE.sub(_mask_aadhaar_loose_match, content)
    content = PHONE_LOOSE_RE.sub(_mask_phone_loose_match, content)
    content = AADHAAR_RE.sub(_mask_aadhaar_match, content)
    content = PHONE_RE.sub(_mask_phone_match, content)
    content = ACCOUNT_RE.sub(_mask_account_match, content)
    return content


def redact_sensitive_text(text: str) -> str:
    return redact_sensitive_data(text)


def redact_sensitive_payload(value: Any, *, skip_keys: Iterable[str] | None = None) -> Any:
    if skip_keys is None:
        skip_keys = set()
    if isinstance(value, dict):
        cleaned: Dict[Any, Any] = {}
        for key, val in value.items():
            key_text = str(key)
            redacted_key = redact_sensitive_data(key_text) if key_text else key_text
            safe_key = redacted_key if redacted_key != key_text else key
            if key_text in skip_keys:
                cleaned[safe_key] = val
            else:
                cleaned[safe_key] = redact_sensitive_payload(val, skip_keys=skip_keys)
        return cleaned
    if isinstance(value, list):
        return [redact_sensitive_payload(item, skip_keys=skip_keys) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_payload(item, skip_keys=skip_keys) for item in value)
    if isinstance(value, str):
        return redact_sensitive_data(value)
    return value


def sanitize_profile_for_storage(profile: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key in ALLOWED_PROFILE_FIELDS:
        value = profile.get(key)
        if value is None:
            cleaned[key] = None
            continue
        text_value = str(value).strip()
        if key == "aadhaar_number":
            cleaned[key] = mask_aadhaar(text_value)
        elif key == "phone":
            cleaned[key] = mask_phone(text_value)
        else:
            cleaned[key] = redact_sensitive_data(text_value)
    return cleaned


def sanitize_profile_for_response(profile: Dict[str, Any]) -> Dict[str, Any]:
    return sanitize_profile_for_storage(profile)


def sanitize_history_for_storage(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned_history: List[Dict[str, str]] = []
    for item in history or []:
        role = str(item.get("role") or "assistant")
        content = redact_sensitive_text(str(item.get("content") or ""))
        cleaned_history.append({"role": role, "content": content})
    return cleaned_history[-10:]


def _sanitize_semantic_memory(memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for item in memory or []:
        if not isinstance(item, dict):
            continue

        entities = item.get("entities")
        if not isinstance(entities, dict):
            entities = {}

        schemes = [str(s).strip() for s in (entities.get("schemes") or []) if str(s).strip()]
        numbers: List[str] = []
        for raw in entities.get("numbers") or []:
            digits = digits_only(str(raw))
            if not digits:
                continue
            if len(digits) >= 8:
                numbers.append(mask_aadhaar(digits))
            else:
                numbers.append(digits)

        cleaned.append(
            {
                "ts": item.get("ts"),
                "intent": str(item.get("intent") or "").strip(),
                "entities": {"schemes": schemes, "numbers": numbers},
                "user_input": redact_sensitive_text(str(item.get("user_input") or "")),
                "assistant_summary": redact_sensitive_text(str(item.get("assistant_summary") or "")),
            }
        )
    return cleaned


def sanitize_session_payload(session_data: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(session_data or {})

    user_profile = payload.get("user_profile")
    if isinstance(user_profile, dict):
        payload["user_profile"] = sanitize_profile_for_storage(user_profile)
    else:
        payload["user_profile"] = {}

    history = payload.get("conversation_history")
    if isinstance(history, list):
        payload["conversation_history"] = sanitize_history_for_storage(history)
    else:
        payload["conversation_history"] = []

    semantic_memory = payload.get("semantic_memory")
    if isinstance(semantic_memory, list):
        payload["semantic_memory"] = _sanitize_semantic_memory(semantic_memory)
    elif "semantic_memory" in payload:
        payload["semantic_memory"] = []

    # Do not persist raw OCR payloads.
    payload.pop("ocr_text", None)
    payload.pop("raw_ocr_text", None)

    return payload
