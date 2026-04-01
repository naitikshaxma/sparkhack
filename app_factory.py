import importlib
import hashlib
import json
import os
import time
import uuid
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.auth import clear_current_user_id, set_current_user_id
from backend.core.config import get_settings, reload_settings
from backend.infrastructure.database.connection import init_db
from backend.core.logger import clear_request_context, configure_logging, log_event, log_exception, set_request_context
from backend.core.metrics import record_error, record_request
from backend.api.v1.routes.intent import router as intent_router
from backend.api.v1.routes.health import router as health_router
from backend.routes.response_utils import RESPONSE_REDACTION_SKIP_KEYS, standardized_error
from backend.shared.security.privacy import redact_sensitive_payload
from backend.api.v1.routes.system_routes import router as system_router
from backend.api.v1.routes.voice_routes import router as voice_router
from backend.api.v1.routes.voice_ws import router as voice_ws_router
from backend.services.rag_service import get_scheme_registry_snapshot, warmup_rag_resources, warmup_scheme_registry_cache
from backend.services.intent_service import get_intent_dataset_status, warmup_intent_dataset_cache
from backend.shared.security.rate_limit import allow_request
from backend.infrastructure.ml.whisper_service import warmup_whisper


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Voice OS Bharat")
    configure_logging()
    startup_logger = logging.getLogger(__name__)

    def _env_flag(name: str, default: str = "1") -> bool:
        raw = (os.getenv(name) or default).strip().lower()
        return raw not in {"0", "false", "no", "off"}

    max_concurrent = max(1, int((os.getenv("MAX_CONCURRENT_REQUESTS") or "50").strip() or "50"))
    request_timeout_seconds = max(1.0, float((os.getenv("REQUEST_TIMEOUT_SECONDS") or "30").strip() or "30"))
    concurrency_timeout_seconds = max(0.1, float((os.getenv("CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS") or "1.5").strip() or "1.5"))
    app.state.concurrency_semaphore = asyncio.Semaphore(max_concurrent)
    app.state.request_timeout_seconds = request_timeout_seconds
    app.state.concurrency_timeout_seconds = concurrency_timeout_seconds
    streaming_paths = {
        "/api/process-text-stream",
        "/api/tts-stream",
        "/api/v1/process-text-stream",
        "/api/v1/tts-stream",
    }

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def startup() -> None:
        reload_settings()
        current_settings = get_settings()
        log_event(
            "startup_env",
            endpoint="startup",
            status="success",
            env=current_settings["ENV"],
        )
        startup_logger.info("Local-only mode active: auth/OpenAI disabled")
        current_settings.validate_runtime()
        try:
            init_db()
        except Exception as exc:
            log_event(
                "database_init_failed",
                level="warning",
                endpoint="startup",
                status="failure",
                error_type=type(exc).__name__,
            )
        warmup_rag = _env_flag("RAG_WARMUP_ON_STARTUP", "1")
        if warmup_rag:
            try:
                warmup_rag_resources(precompute_embeddings=True)
            except Exception as exc:
                log_event(
                    "rag_warmup_failed",
                    level="warning",
                    endpoint="startup",
                    status="failure",
                    error_type=type(exc).__name__,
                )
        if _env_flag("WHISPER_WARMUP_ON_STARTUP", "0"):
            try:
                warmup_whisper()
            except Exception as exc:
                log_event(
                    "whisper_warmup_failed",
                    level="warning",
                    endpoint="startup",
                    status="failure",
                    error_type=type(exc).__name__,
                )

        if _env_flag("SCHEME_REGISTRY_WARMUP_ON_STARTUP", "1"):
            try:
                registry_snapshot = warmup_scheme_registry_cache() or {}
                startup_logger.info("Chunks loaded: %s", int(registry_snapshot.get("chunk_rows", 0)))
                startup_logger.info("Scheme rows loaded: %s", int(registry_snapshot.get("scheme_rows", 0)))
            except Exception as exc:
                log_event(
                    "scheme_registry_warmup_failed",
                    level="warning",
                    endpoint="startup",
                    status="failure",
                    error_type=type(exc).__name__,
                )

        if _env_flag("INTENT_DATASET_WARMUP_ON_STARTUP", "1"):
            try:
                warmup_intent_dataset_cache(force=False)
                intent_status = get_intent_dataset_status()
                startup_logger.info("Intent rows loaded: %s", int(intent_status.get("row_count", 0)))
            except Exception as exc:
                log_event(
                    "intent_dataset_warmup_failed",
                    level="warning",
                    endpoint="startup",
                    status="failure",
                    error_type=type(exc).__name__,
                )

        # Optional monitoring sweeper for non-MVP deployments.
        if _env_flag("ENABLE_MONITORING_SWEEPER", "0"):
            try:
                sweeper_module = importlib.import_module("backend.infrastructure.monitoring.sweeper")
                run_sweeper = getattr(sweeper_module, "run_sweeper")
                asyncio.ensure_future(run_sweeper())
                log_event("sweeper_started", level="info", endpoint="startup", status="success")
            except Exception as exc:
                log_event(
                    "sweeper_start_failed",
                    level="warning",
                    endpoint="startup",
                    status="failure",
                    error_type=type(exc).__name__,
                )

    def _extract_client_ip(request: Request) -> str:
        if settings.trust_proxy_headers:
            forwarded_for = (request.headers.get("x-forwarded-for") or "").strip()
            if forwarded_for:
                first_ip = forwarded_for.split(",", 1)[0].strip()
                if first_ip:
                    return first_ip

            real_ip = (request.headers.get("x-real-ip") or "").strip()
            if real_ip:
                return real_ip

        return request.client.host if request.client else "unknown"

    def _enforce_request_size_limit(request: Request, client_ip: str, request_id: str) -> None:
        if not request.url.path.startswith("/api"):
            return

        content_length = request.headers.get("content-length")
        if not content_length:
            return

        try:
            size_bytes = int(content_length)
        except ValueError:
            return

        if size_bytes > settings.max_request_size_bytes:
            log_event(
                "security_request_rejected",
                level="warning",
                request_id=request_id,
                endpoint=request.url.path,
                status="failure",
                error_type="request_size_limit_exceeded",
                client_ip=client_ip,
                size_bytes=size_bytes,
                max_request_size_bytes=settings.max_request_size_bytes,
            )
            raise HTTPException(status_code=413, detail="Request payload too large.")

    def _require_api_key_if_enabled(request: Request) -> None:
        # Auth is intentionally disabled in local-only runtime.
        return

    def _check_rate_limit(request: Request, request_id: str) -> None:
        if not request.url.path.startswith("/api"):
            return

        client_ip = _extract_client_ip(request)
        ip_allowed = allow_request(
            f"ip:{client_ip}",
            max_requests=settings.api_rate_limit_max_requests,
            window_seconds=settings.api_rate_limit_window_seconds,
        )
        if not ip_allowed:
            log_event(
                "security_request_rejected",
                level="warning",
                request_id=request_id,
                endpoint=request.url.path,
                status="failure",
                error_type="ip_rate_limit_exceeded",
                client_ip=client_ip,
            )
            raise HTTPException(status_code=429, detail="Too many requests. Please retry later.")

    def _set_security_headers(response) -> None:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'; base-uri 'none';"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    @app.middleware("http")
    async def api_safety_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request_start = time.perf_counter()
        client_ip = _extract_client_ip(request)
        request.state.request_id = request_id
        request.state.timings = {}
        set_request_context(request_id, request.url.path, request.method, "")
        acquired = False
        semaphore_released = False
        is_api = request.url.path.startswith("/api")
        is_streaming = request.url.path in streaming_paths

        authenticated_user_id = (request.headers.get("x-user-id") or "").strip()

        request.state.user_id = authenticated_user_id
        set_request_context(request_id, request.url.path, request.method, authenticated_user_id)
        set_current_user_id(authenticated_user_id)

        log_event(
            "request_start",
            request_id=request_id,
            endpoint=request.url.path,
            status="success",
            client_ip=client_ip,
            method=request.method,
            user_id=authenticated_user_id,
        )

        try:
            _enforce_request_size_limit(request, client_ip, request_id)
            _require_api_key_if_enabled(request)
            _check_rate_limit(request, request_id)
            if is_api and not is_streaming:
                try:
                    await asyncio.wait_for(app.state.concurrency_semaphore.acquire(), timeout=app.state.concurrency_timeout_seconds)
                    acquired = True
                except asyncio.TimeoutError:
                    log_event(
                        "concurrency_limit_reached",
                        level="warning",
                        request_id=request_id,
                        endpoint=request.url.path,
                        status="failure",
                        error_type="concurrency_limit_reached",
                        client_ip=client_ip,
                    )
                    clear_request_context()
                    clear_current_user_id()
                    response = JSONResponse(status_code=429, content=standardized_error("Server busy. Please retry shortly."), headers={"x-request-id": request_id})
                    _set_security_headers(response)
                    return response
        except HTTPException as exc:
            response_time_ms = round((time.perf_counter() - request_start) * 1000.0, 2)
            record_request(response_time_ms=response_time_ms, success=False)
            record_error(type(exc).__name__)
            log_event(
                "middleware_policy_rejection",
                level="warning",
                request_id=request_id,
                endpoint=request.url.path,
                status="failure",
                error_type=type(exc).__name__,
                response_time_ms=response_time_ms,
                client_ip=client_ip,
                status_code=exc.status_code,
            )
            clear_request_context()
            clear_current_user_id()
            response = JSONResponse(status_code=exc.status_code, content=standardized_error(str(exc.detail)), headers={"x-request-id": request_id})
            _set_security_headers(response)
            return response

        try:
            if is_api and not is_streaming:
                response = await asyncio.wait_for(call_next(request), timeout=app.state.request_timeout_seconds)
            else:
                response = await call_next(request)
        except asyncio.TimeoutError:
            response_time_ms = round((time.perf_counter() - request_start) * 1000.0, 2)
            record_request(response_time_ms=response_time_ms, success=False)
            record_error("request_timeout")
            log_event(
                "request_timeout",
                level="warning",
                request_id=request_id,
                endpoint=request.url.path,
                status="failure",
                error_type="request_timeout",
                response_time_ms=response_time_ms,
                client_ip=client_ip,
            )
            clear_request_context()
            clear_current_user_id()
            if acquired:
                app.state.concurrency_semaphore.release()
                semaphore_released = True
            response = JSONResponse(status_code=504, content=standardized_error("Request timed out."), headers={"x-request-id": request_id})
            _set_security_headers(response)
            return response
        except Exception:
            if acquired and not semaphore_released:
                app.state.concurrency_semaphore.release()
                semaphore_released = True
            raise
        try:
            response_time_ms = round((time.perf_counter() - request_start) * 1000.0, 2)
            status = "success" if response.status_code < 400 else "failure"
            record_request(response_time_ms=response_time_ms, success=(status == "success"))
            if status == "failure":
                record_error(f"http_{response.status_code}")
            log_event(
                "request_complete",
                request_id=request_id,
                endpoint=request.url.path,
                status=status,
                error_type=None if status == "success" else f"http_{response.status_code}",
                intent=getattr(request.state, "intent", None),
                confidence=getattr(request.state, "confidence", None),
                user_input_length=getattr(request.state, "user_input_length", None),
                response_time_ms=response_time_ms,
                timings=getattr(request.state, "timings", {}),
                status_code=response.status_code,
                method=request.method,
                user_id=authenticated_user_id,
            )
            if isinstance(response, JSONResponse):
                try:
                    if response.body:
                        body_payload = json.loads(response.body.decode(response.charset or "utf-8"))
                        redacted = redact_sensitive_payload(body_payload, skip_keys=RESPONSE_REDACTION_SKIP_KEYS)
                        response.body = response.render(redacted)
                        response.headers["content-length"] = str(len(response.body))
                except Exception:
                    pass
            response.headers["x-request-id"] = request_id
            _set_security_headers(response)
            clear_request_context()
            clear_current_user_id()
            return response
        finally:
            if acquired and not semaphore_released:
                app.state.concurrency_semaphore.release()
                semaphore_released = True

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        record_error(type(exc).__name__)
        log_event(
            "request_validation_error",
            level="warning",
            request_id=getattr(request.state, "request_id", ""),
            endpoint=request.url.path,
            status="failure",
            error_type=type(exc).__name__,
            user_input_length=getattr(request.state, "user_input_length", None),
        )
        response = JSONResponse(status_code=422, content=standardized_error("Invalid request payload.", data={"error_count": len(exc.errors())}))
        _set_security_headers(response)
        clear_request_context()
        clear_current_user_id()
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        record_error(type(exc).__name__)
        log_event(
            "http_exception",
            level="warning",
            request_id=getattr(request.state, "request_id", ""),
            endpoint=request.url.path,
            status="failure",
            error_type=type(exc).__name__,
            status_code=exc.status_code,
        )
        response = JSONResponse(status_code=exc.status_code, content=standardized_error(str(exc.detail)))
        _set_security_headers(response)
        clear_request_context()
        clear_current_user_id()
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        record_error(type(exc).__name__)
        log_exception(
            exc,
            request_id=getattr(request.state, "request_id", ""),
            endpoint=request.url.path,
            safe_context={
                "method": request.method,
                "client_ip": _extract_client_ip(request),
                "user_input_length": getattr(request.state, "user_input_length", None),
            },
        )
        response = JSONResponse(status_code=500, content=standardized_error("Internal server error."))
        _set_security_headers(response)
        clear_request_context()
        clear_current_user_id()
        return response

    if settings.env != "production":
        @app.get("/debug/schemes")
        async def debug_schemes() -> dict:
            return get_scheme_registry_snapshot()

    # Versioned + legacy compatibility mounts.
    app.include_router(intent_router, prefix="/api")
    app.include_router(voice_router, prefix="/api")

    app.include_router(intent_router, prefix="/api/v1")
    app.include_router(voice_router, prefix="/api/v1")

    app.include_router(system_router)
    app.include_router(system_router, prefix="/api")
    app.include_router(system_router, prefix="/api/v1")

    # Phase 7: Async WebSocket pipeline (Redis-backed with sync fallback)
    app.include_router(voice_ws_router, prefix="/api/v1")
    app.include_router(voice_ws_router)  # also available at /ws/voice/{session_id}

    # Phase 9: Observability endpoints
    app.include_router(health_router)
    app.include_router(health_router, prefix="/api")
    app.include_router(health_router, prefix="/api/v1")

    return app
