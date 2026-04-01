from __future__ import annotations

import json
import logging
import logging.handlers
import queue
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from backend.shared.security.privacy import redact_sensitive_data, redact_sensitive_payload

_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")
_ENDPOINT: ContextVar[str] = ContextVar("endpoint", default="")
_METHOD: ContextVar[str] = ContextVar("method", default="")
_USER_ID: ContextVar[str] = ContextVar("user_id", default="")


_LOG_QUEUE: "queue.Queue[logging.LogRecord]" = queue.Queue(maxsize=20000)
_QUEUE_LISTENER: logging.handlers.QueueListener | None = None
_REDACTION_FACTORY_INSTALLED = False
_ORIGINAL_LOG_RECORD_FACTORY = logging.getLogRecordFactory()
_STANDARD_LOG_RECORD_ATTRS = set(_ORIGINAL_LOG_RECORD_FACTORY("", 0, "", 0, "", (), None).__dict__.keys())


def _redacting_log_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    record = _ORIGINAL_LOG_RECORD_FACTORY(*args, **kwargs)
    try:
        message = record.getMessage()
    except Exception:
        message = str(record.msg)
    record.msg = redact_sensitive_data(message)
    record.args = ()

    if record.exc_info:
        try:
            exc_text = "".join(traceback.format_exception(*record.exc_info))
        except Exception:
            exc_text = ""
        record.exc_text = redact_sensitive_data(exc_text)
        record.exc_info = None
    if record.stack_info:
        record.stack_info = redact_sensitive_data(str(record.stack_info))

    for key, value in list(record.__dict__.items()):
        if key in _STANDARD_LOG_RECORD_ATTRS or key in {"msg", "args", "message"}:
            continue
        record.__dict__[key] = redact_sensitive_payload(value)
    return record


def install_log_redaction() -> None:
    global _REDACTION_FACTORY_INSTALLED
    if _REDACTION_FACTORY_INSTALLED:
        return
    logging.setLogRecordFactory(_redacting_log_record_factory)
    _REDACTION_FACTORY_INSTALLED = True


def configure_logging() -> None:
    global _QUEUE_LISTENER

    logger = logging.getLogger("voice_os")
    if _QUEUE_LISTENER is not None:
        return

    install_log_redaction()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    queue_handler = logging.handlers.QueueHandler(_LOG_QUEUE)

    logger.setLevel(logging.INFO)
    logger.addHandler(queue_handler)
    logger.propagate = False

    _QUEUE_LISTENER = logging.handlers.QueueListener(_LOG_QUEUE, stream_handler, respect_handler_level=True)
    _QUEUE_LISTENER.start()


def set_request_context(request_id: str, endpoint: str, method: str = "", user_id: str = "") -> None:
    _REQUEST_ID.set(request_id or "")
    _ENDPOINT.set(endpoint or "")
    _METHOD.set(method or "")
    _USER_ID.set(user_id or "")


def clear_request_context() -> None:
    _REQUEST_ID.set("")
    _ENDPOINT.set("")
    _METHOD.set("")
    _USER_ID.set("")


def get_request_context() -> dict[str, str]:
    return {
        "request_id": _REQUEST_ID.get(),
        "endpoint": _ENDPOINT.get(),
        "method": _METHOD.get(),
        "user_id": _USER_ID.get(),
    }


def _base_payload(**fields: Any) -> dict[str, Any]:
    context = get_request_context()

    error_value = fields.pop("error", None)
    if error_value is None:
        error_value = fields.pop("error_type", None)

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": fields.pop("request_id", context.get("request_id", "")),
        "endpoint": fields.pop("endpoint", context.get("endpoint", "")),
        "method": fields.pop("method", context.get("method", "")),
        "status": fields.pop("status", None),
        "response_time_ms": fields.pop("response_time_ms", None),
        "user_id": fields.pop("user_id", context.get("user_id", "")),
        "error": error_value,
    }
    payload.update(fields)
    return payload


def log_event(event: str, *, level: str = "info", **fields: Any) -> None:
    configure_logging()
    logger = logging.getLogger("voice_os")
    payload = _base_payload(event=event, **fields)
    payload = redact_sensitive_payload(payload)
    message = json.dumps(payload, ensure_ascii=False)

    level_name = (level or "info").lower()
    if level_name == "debug":
        logger.debug(message)
    elif level_name == "warning":
        logger.warning(message)
    elif level_name == "error":
        logger.error(message)
    else:
        logger.info(message)


def safe_log(message: str, *, level: str = "info", **fields: Any) -> None:
    log_event("safe_log", level=level, status="success", message=message, **fields)


def log_exception(error: Exception, *, safe_context: dict[str, Any] | None = None, **fields: Any) -> None:
    payload = {
        "stack_trace": traceback.format_exc(),
        "safe_context": safe_context or {},
    }
    payload.update(fields)
    log_event(
        "exception",
        level="error",
        status="failure",
        error_type=type(error).__name__,
        **payload,
    )


install_log_redaction()
