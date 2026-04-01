from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional


SCRIPT_INJECTION_RE = re.compile(r"(<script|</script|javascript:|onerror\s*=|onload\s*=)", re.IGNORECASE)
SQL_INJECTION_RE = re.compile(r"(drop\s+table|union\s+select|or\s+1\s*=\s*1|insert\s+into|delete\s+from)", re.IGNORECASE)
PATH_TRAVERSAL_RE = re.compile(r"(\.\./|\.\.\\\\|%2e%2e%2f|%2e%2e\\\\|%00)", re.IGNORECASE)
PROMPT_INJECTION_RE = re.compile(r"(ignore\s+previous\s+instructions|system\s+prompt|developer\s+mode|jailbreak|do\s+anything\s+now)", re.IGNORECASE)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")

REJECTABLE_THREAT_TYPES = {"script injection", "SQL injection", "path traversal"}


@dataclass(frozen=True)
class InputValidationResult:
    is_valid: bool
    normalized_text: str
    sanitized_text: str
    threat_types: tuple[str, ...]
    rejected_reason: Optional[str] = None

    @property
    def is_suspicious(self) -> bool:
        return bool(self.threat_types)


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _strip_dangerous_chars(text: str) -> str:
    return CONTROL_CHARS_RE.sub("", text)


def sanitize_input(text: str) -> str:
    raw = text or ""
    stripped = _strip_dangerous_chars(raw)
    normalized = _normalize_whitespace(stripped)
    # Escape script/html-oriented payloads before model processing.
    return html.escape(normalized, quote=False)


def _classify_threats(normalized_text: str) -> tuple[str, ...]:
    lowered = normalized_text.lower()
    threats: list[str] = []

    if SCRIPT_INJECTION_RE.search(lowered):
        threats.append("script injection")
    if SQL_INJECTION_RE.search(lowered):
        threats.append("SQL injection")
    if PATH_TRAVERSAL_RE.search(lowered):
        threats.append("path traversal")
    if PROMPT_INJECTION_RE.search(lowered):
        threats.append("prompt injection")

    return tuple(threats)


def validate_input(text: str, max_chars: int = 500) -> InputValidationResult:
    normalized_text = _normalize_whitespace(_strip_dangerous_chars(text or ""))
    if not normalized_text:
        return InputValidationResult(
            is_valid=False,
            normalized_text="",
            sanitized_text="",
            threat_types=tuple(),
            rejected_reason="text payload is empty.",
        )

    if len(normalized_text) > max_chars:
        return InputValidationResult(
            is_valid=False,
            normalized_text=normalized_text,
            sanitized_text="",
            threat_types=tuple(),
            rejected_reason="text exceeds allowed length.",
        )

    threat_types = _classify_threats(normalized_text)
    sanitized_text = sanitize_input(normalized_text)
    if any(threat in REJECTABLE_THREAT_TYPES for threat in threat_types):
        return InputValidationResult(
            is_valid=False,
            normalized_text=normalized_text,
            sanitized_text=sanitized_text,
            threat_types=threat_types,
            rejected_reason="text contains suspicious content.",
        )

    return InputValidationResult(
        is_valid=True,
        normalized_text=normalized_text,
        sanitized_text=sanitized_text,
        threat_types=threat_types,
    )


class InputValidator:
    def __init__(self, max_chars: int, logger: Optional[logging.Logger] = None) -> None:
        self.max_chars = max_chars
        self.logger = logger or logging.getLogger("security.input_validator")

    def sanitize_input(self, text: str) -> str:
        return sanitize_input(text)

    def validate_input(self, text: str, *, client_ip: str = "", endpoint: str = "") -> InputValidationResult:
        result = validate_input(text, max_chars=self.max_chars)
        if result.is_suspicious or not result.is_valid:
            self.logger.warning(
                json.dumps(
                    {
                        "event": "security_input_analysis",
                        "client_ip": client_ip or "unknown",
                        "endpoint": endpoint or "unknown",
                        "is_valid": result.is_valid,
                        "threat_types": list(result.threat_types),
                        "rejected_reason": result.rejected_reason,
                    }
                )
            )
        return result
