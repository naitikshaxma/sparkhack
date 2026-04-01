from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from backend.core.logger import log_event


# Load .env once for all backend modules.
load_dotenv(override=False)

# Force local-only runtime mode to avoid environment-dependent startup failures.
ENV = "development"


_PLACEHOLDER_VALUES = {
    "",
    "replace-with-your-openai-key",
    "replace-with-your-real-openai-key",
    "replace-with-strong-random-token",
    "change-me",
    "changeme",
    "default",
    "your-api-key",
}

_ALLOWED_ENVS = {
    "development",
    "production",
    "staging",
    "test",
    "qa",
    "uat",
}

_ENV_ALIASES = {
    "dev": "development",
    "local": "development",
    "prod": "production",
    "stage": "staging",
}

_WEAK_JWT_SECRETS = {
    "secret",
    "password",
    "jwtsecret",
    "jwt-secret",
    "token",
    "dev",
    "test",
    "default",
}


def has_valid_openai_key(key: str | None = None) -> bool:
    # OpenAI is intentionally disabled; keep helper for backward compatibility.
    return False


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _as_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a valid integer.") from exc


def _is_placeholder(secret_value: str) -> bool:
    value = (secret_value or "").strip().lower()
    if value in _PLACEHOLDER_VALUES:
        return True

    if re.search(r"x{6,}", value):
        return True

    return (
        "replace" in value
        or "example" in value
        or "dummy" in value
        or "test-key" in value
    )


def _is_weak_jwt_secret(secret_value: str) -> bool:
    value = (secret_value or "").strip().lower()
    if not value:
        return True
    if value in _WEAK_JWT_SECRETS:
        return True
    if len(set(value)) == 1:
        return True
    if re.fullmatch(r"(secret|password|jwtsecret|jwt-secret|changeme|default|token)+", value):
        return True
    return False


def _normalize_env(value: str | None) -> str:
    raw = (value or "development").strip().lower()
    return _ENV_ALIASES.get(raw, raw)


@dataclass(frozen=True)
class Settings:
    env: str
    openai_api_key: str
    openai_chat_model: str
    redis_url: str
    session_ttl_seconds: int
    model_path: Path
    hf_intent_model_id: str
    whisper_model_size: str
    response_tone: str
    max_text_input_chars: int
    max_session_id_chars: int
    api_rate_limit_window_seconds: int
    api_rate_limit_max_requests: int
    api_key_rate_limit_max_requests: int
    max_request_size_bytes: int
    trust_proxy_headers: bool
    enable_api_key_auth: bool
    api_auth_key: str
    database_url: str
    jwt_secret_key: str
    jwt_algorithm: str
    jwt_expiration_minutes: int
    jwt_required_for_protected_routes: bool
    jwt_protected_prefixes: tuple[str, ...]
    frontend_dev_origin: str
    frontend_production_origin: str
    cors_allow_origins: tuple[str, ...]

    def has_usable_openai_key(self) -> bool:
        return False

    def __getitem__(self, key: str):
        mapping = {
            "ENV": self.env,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def validate_runtime(self) -> None:
        env = (self.env or "development").strip().lower()
        if env not in _ALLOWED_ENVS:
            allowed = ", ".join(sorted(_ALLOWED_ENVS))
            raise RuntimeError(f"ENV must be one of: {allowed}.")
        log_event(
            "openai_disabled_local_runtime",
            level="info",
            endpoint="startup",
            status="success",
        )

        parsed = urlparse(self.redis_url)
        if parsed.scheme not in {"redis", "rediss"}:
            raise RuntimeError("REDIS_URL must use redis:// or rediss:// scheme.")

        if self.session_ttl_seconds <= 0:
            raise RuntimeError("SESSION_TTL_SECONDS must be > 0.")
        if self.max_text_input_chars < 64:
            raise RuntimeError("MAX_TEXT_INPUT_CHARS must be >= 64.")
        if self.max_session_id_chars < 8:
            raise RuntimeError("MAX_SESSION_ID_CHARS must be >= 8.")
        if self.api_rate_limit_window_seconds <= 0:
            raise RuntimeError("API_RATE_LIMIT_WINDOW_SECONDS must be > 0.")
        if self.api_rate_limit_max_requests <= 0:
            raise RuntimeError("API_RATE_LIMIT_MAX_REQUESTS must be > 0.")
        if self.api_key_rate_limit_max_requests <= 0:
            raise RuntimeError("API_KEY_RATE_LIMIT_MAX_REQUESTS must be > 0.")
        if self.max_request_size_bytes <= 0:
            raise RuntimeError("MAX_REQUEST_SIZE_BYTES must be > 0.")

        if self.enable_api_key_auth:
            if not self.api_auth_key:
                raise RuntimeError("API_AUTH_KEY is required when ENABLE_API_KEY_AUTH=true.")
            if _is_placeholder(self.api_auth_key):
                raise RuntimeError("API_AUTH_KEY uses a placeholder/default value. Refusing startup.")
        # JWT is disabled for local-only runtime.


def get_settings() -> Settings:
    raw_prefixes = (os.getenv("JWT_PROTECTED_PREFIXES") or "/api/private,/api/v1/private").strip()
    jwt_protected_prefixes = tuple(prefix.strip() for prefix in raw_prefixes.split(",") if prefix.strip())
    frontend_dev_origin = (os.getenv("FRONTEND_DEV_ORIGIN") or "http://127.0.0.1:5173").strip()
    frontend_production_origin = (os.getenv("FRONTEND_PRODUCTION_ORIGIN") or "https://voice-os-bharat.com").strip()
    raw_cors_origins = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
    if raw_cors_origins:
        cors_allow_origins = tuple(origin.strip() for origin in raw_cors_origins.split(",") if origin.strip())
    else:
        cors_allow_origins = tuple(
            origin
            for origin in (frontend_dev_origin, frontend_production_origin)
            if origin
        )

    return Settings(
        env=ENV,
        openai_api_key="",
        openai_chat_model="local",
        redis_url=(os.getenv("REDIS_URL") or "redis://localhost:6379/0").strip(),
        session_ttl_seconds=_as_int("SESSION_TTL_SECONDS", 86400),
        model_path=Path(os.getenv("MODEL_PATH", "./models/intent_model_distilbert")).resolve(),
        hf_intent_model_id=(os.getenv("HF_INTENT_MODEL_ID") or "distilbert-base-uncased").strip(),
        whisper_model_size=(os.getenv("WHISPER_MODEL_SIZE") or "medium").strip(),
        response_tone=(os.getenv("RESPONSE_TONE") or "assistant-like").strip().lower(),
        max_text_input_chars=_as_int("MAX_TEXT_INPUT_CHARS", 500),
        max_session_id_chars=_as_int("MAX_SESSION_ID_CHARS", 64),
        api_rate_limit_window_seconds=_as_int("API_RATE_LIMIT_WINDOW_SECONDS", 60),
        api_rate_limit_max_requests=_as_int("API_RATE_LIMIT_MAX_REQUESTS", 100),
        api_key_rate_limit_max_requests=_as_int("API_KEY_RATE_LIMIT_MAX_REQUESTS", 120),
        max_request_size_bytes=_as_int("MAX_REQUEST_SIZE_BYTES", 1048576),
        trust_proxy_headers=_as_bool(os.getenv("TRUST_PROXY_HEADERS"), False),
        enable_api_key_auth=_as_bool(os.getenv("ENABLE_API_KEY_AUTH"), False),
        api_auth_key=(os.getenv("API_AUTH_KEY") or "").strip(),
        database_url=(os.getenv("DATABASE_URL") or "").strip(),
        jwt_secret_key="",
        jwt_algorithm=(os.getenv("JWT_ALGORITHM") or "HS256").strip(),
        jwt_expiration_minutes=_as_int("JWT_EXPIRATION_MINUTES", 60),
        jwt_required_for_protected_routes=False,
        jwt_protected_prefixes=tuple(),
        frontend_dev_origin=frontend_dev_origin,
        frontend_production_origin=frontend_production_origin,
        cors_allow_origins=cors_allow_origins,
    )


if ENV == "production":
    get_settings = lru_cache(maxsize=1)(get_settings)


def reload_settings() -> None:
    try:
        get_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
