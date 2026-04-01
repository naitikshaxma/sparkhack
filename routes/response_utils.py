from typing import Any, Dict

from backend.shared.security.privacy import redact_sensitive_payload


RESPONSE_REDACTION_SKIP_KEYS = {
    "access_token",
    "audio_base64",
    "audio_chunk",
    "audio_bytes",
    "job_id",
    "session_id",
    "user_id",
    "request_id",
}


def standardized_success(payload: Any) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {
        "success": True,
        "data": payload,
        "error": None,
    }
    if isinstance(payload, dict):
        envelope.update(payload)
    return redact_sensitive_payload(envelope, skip_keys=RESPONSE_REDACTION_SKIP_KEYS)


def standardized_error(message: str, *, data: Any = None) -> Dict[str, Any]:
    if data is None:
        data = {}
    envelope = {
        "success": False,
        "data": data,
        "error": message,
        "detail": message,
    }
    return redact_sensitive_payload(envelope, skip_keys=RESPONSE_REDACTION_SKIP_KEYS)
