"""FastAPI application — entry point, middleware, and health endpoint.

Creates the Makaronas backend API with:
- API versioning via router prefix (/api/v1/)
- CORS middleware (origins from settings)
- Request logging middleware (raw ASGI — no response body buffering)
- CSP middleware (Content-Security-Policy on every response)
- Rate limiting middleware (per-session on AI endpoints)
- Static file serving (static/ directory at root URL)
- Global exception handlers (HTTPException, validation, catch-all)
- Health endpoint

Run with: uvicorn backend.main:app --reload

Tier 3 orchestration module: imports from config (Tier 2), deps (Tier 2),
schemas (Tier 1).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.config import get_settings
from backend.schemas import ApiError, ApiResponse

logger = logging.getLogger("makaronas")


# ---------------------------------------------------------------------------
# Request logging middleware (raw ASGI — streaming-safe)
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware:
    """Logs method, path, status code, and duration for every request.

    Uses raw ASGI to avoid response body buffering (safe for SSE streaming
    in Phase 3b). Does NOT log request/response bodies, query params,
    auth headers, or client IPs (Framework Principle 3 — GDPR).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Wraps the ASGI call to measure timing and capture status code."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "?")
        path = scope.get("path", "?")
        start = time.monotonic()
        status_code = 0

        async def logging_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, logging_send)
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            logger.info(
                "%s %s %d %.1fms", method, path, status_code, duration_ms
            )


# ---------------------------------------------------------------------------
# CSP middleware (raw ASGI — streaming-safe)
# ---------------------------------------------------------------------------

# Framework Principle 13: Security by Design.
# Restricts resource loading to same-origin. 'unsafe-inline' for styles only
# (needed for dynamic style changes in Phase 2a section switching).
CSP_HEADER_VALUE = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' blob: data:; "
    "media-src 'self'; "
    "connect-src 'self'"
)


class CSPMiddleware:
    """Adds Content-Security-Policy header to every HTTP response.

    Uses raw ASGI to avoid response body buffering (safe for SSE streaming).
    Intercepts http.response.start to inject the header.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Wraps send to inject CSP header into response start."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def csp_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append(
                    (b"content-security-policy", CSP_HEADER_VALUE.encode())
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, csp_send)


# ---------------------------------------------------------------------------
# Rate limiting middleware (raw ASGI — streaming-safe)
# ---------------------------------------------------------------------------

# Per-session rate limit on AI-consuming endpoints only.
# Framework Principle 13: a single student's runaway session must not
# exhaust the school's token budget or the platform's API quota.
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

# Paths that trigger rate limiting (suffixes after /session/{id}/)
_RATE_LIMITED_SUFFIXES = ("/respond", "/generate")

# The path prefix that rate-limited endpoints share
_SESSION_PATH_PREFIX = "/api/v1/student/session/"


class RateLimitMiddleware:
    """Per-session rate limiter on AI-consuming endpoints.

    Tracks request counts in a fixed window per session ID. Only applies
    to POST requests on /respond and /generate endpoints. Returns HTTP 429
    with ApiResponse envelope body when the limit is exceeded.

    Uses raw ASGI to avoid response body buffering (safe for SSE streaming).
    MVP: in-memory dict — production replaces with Redis or equivalent.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        # session_id -> (request_count, window_start_monotonic)
        self._counters: dict[str, tuple[int, float]] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Checks rate limit for matching requests, passes through otherwise."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Only rate-limit POST requests (G7: skip OPTIONS/GET)
        method = scope.get("method", "")
        if method != "POST":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        session_id = self._extract_session_id(path)
        if session_id is None:
            await self.app(scope, receive, send)
            return

        # Check and update counter
        now = time.monotonic()
        count, window_start = self._counters.get(session_id, (0, now))

        # Reset window if expired
        if now - window_start >= RATE_LIMIT_WINDOW_SECONDS:
            count = 0
            window_start = now

        count += 1
        self._counters[session_id] = (count, window_start)

        if count > RATE_LIMIT_MAX_REQUESTS:
            await self._send_429(send)
            return

        await self.app(scope, receive, send)

    def _extract_session_id(self, path: str) -> str | None:
        """Extracts session ID from rate-limited paths, or returns None.

        Matches: /api/v1/student/session/{id}/respond
                 /api/v1/student/session/{id}/generate
        """
        if not path.startswith(_SESSION_PATH_PREFIX):
            return None

        # Check if path ends with a rate-limited suffix
        suffix_match = False
        for suffix in _RATE_LIMITED_SUFFIXES:
            if path.endswith(suffix):
                suffix_match = True
                break

        if not suffix_match:
            return None

        # Extract session ID: everything between /session/ and the last /
        rest = path[len(_SESSION_PATH_PREFIX):]
        # rest looks like "{session_id}/respond" or "{session_id}/generate"
        parts = rest.rsplit("/", 1)
        if len(parts) != 2 or not parts[0]:
            return None

        return parts[0]

    async def _send_429(self, send: Send) -> None:
        """Sends HTTP 429 response with ApiResponse envelope body."""
        body = json.dumps({
            "ok": False,
            "data": None,
            "error": {
                "code": "RATE_LIMITED",
                "message": "Too many requests. Please wait a moment.",
            },
        }).encode()

        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def _http_exception_response(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Wraps HTTPException in ApiResponse envelope.

    If the detail is already an ApiResponse dict (from deps.py auth),
    returns it directly. Otherwise wraps in a generic error.
    """
    if isinstance(exc.detail, dict) and "ok" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse(
            ok=False,
            error=ApiError(code="HTTP_ERROR", message=str(exc.detail)),
        ).model_dump(),
    )


def _validation_error_response(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Wraps Pydantic validation errors in ApiResponse envelope.

    Returns a human-readable summary without leaking internal details.
    """
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = " -> ".join(str(part) for part in first.get("loc", []))
        msg = first.get("msg", "Validation error")
        detail = f"{loc}: {msg}" if loc else msg
    else:
        detail = "Request validation failed."

    return JSONResponse(
        status_code=422,
        content=ApiResponse(
            ok=False,
            error=ApiError(code="VALIDATION_ERROR", message=detail),
        ).model_dump(),
    )


def _unhandled_exception_response(request: Request, exc: Exception) -> JSONResponse:
    """Catches all unhandled exceptions — never leaks internals to client.

    Logs the full traceback server-side. Returns a generic 500 response.
    Framework Principle 13: Error Opacity.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)

    return JSONResponse(
        status_code=500,
        content=ApiResponse(
            ok=False,
            error=ApiError(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred.",
            ),
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------


def _init_task_registry() -> None:
    """Initializes the task registry singleton during app startup.

    Uses PROJECT_ROOT from config (Framework P17 — single source of truth
    for derived paths). Loads all cartridges, sets the singleton in deps.py.
    Logs but does not crash on empty or missing content directories.
    """
    from backend.api import deps
    from backend.config import PROJECT_ROOT
    from backend.tasks.registry import TaskRegistry

    content_dir = PROJECT_ROOT / "content"
    taxonomy_path = content_dir / "taxonomy.json"

    registry = TaskRegistry(content_dir, taxonomy_path)
    registry.load()
    deps._task_registry = registry


def _init_ai_services() -> None:
    """Initializes AI service singletons during app startup.

    Creates the prompt loader, provider, context manager, and trickster
    engine. Runs startup checks for API keys and prompt existence.
    Logs warnings/errors but never prevents startup — static tasks
    still work without AI.

    Must be called AFTER _init_task_registry() (prompt enforcement
    needs loaded cartridges).

    Uses local imports to match _init_task_registry() pattern and
    avoid circular imports during module loading.
    """
    from backend.api import deps
    from backend.config import PROJECT_ROOT
    from backend.models import TIER_MAP

    settings = get_settings()

    # 1. Create PromptLoader
    from backend.ai.prompts import PromptLoader

    prompt_loader = PromptLoader(PROJECT_ROOT / "prompts")
    deps._prompt_loader = prompt_loader

    # Wire prompt cache invalidation to registry reload (Vision SS4.4).
    # Single entry point for any future reload trigger.
    def reload_all() -> None:
        deps._task_registry.reload()
        prompt_loader.invalidate()

    deps._reload_all = reload_all

    # 2. Resolve "standard" tier and create the default provider
    from backend.ai.context import ContextManager

    standard_config = TIER_MAP["standard"]
    try:
        provider = deps.create_provider(standard_config, settings)
    except Exception:
        logger.warning(
            "Failed to create AI provider for 'standard' tier (%s). "
            "AI features will be unavailable.",
            standard_config.provider,
        )
        # Still set prompt_loader (useful for validation), but engine stays None
        _run_startup_checks(settings, deps)
        return

    # 3. Create ContextManager and TricksterEngine
    from backend.ai.intensity import load_intensity_indicators
    from backend.ai.trickster import TricksterEngine

    content_dir = PROJECT_ROOT / "content"
    context_manager = ContextManager(prompt_loader, content_dir=content_dir)
    deps._context_manager = context_manager

    # Load intensity indicators (graceful degradation if absent)
    intensity_indicators: dict | None = None
    try:
        intensity_indicators = load_intensity_indicators(
            content_dir / "intensity_indicators.json",
        )
    except Exception:
        logger.warning(
            "Failed to load intensity indicators. "
            "Intensity enforcement will be disabled.",
        )

    engine = TricksterEngine(
        provider, context_manager,
        intensity_indicators=intensity_indicators,
    )
    deps._trickster_engine = engine

    # 4. Run startup checks
    _run_startup_checks(settings, deps)

    logger.info(
        "AI services initialized: provider=%s, model=%s",
        standard_config.provider,
        standard_config.model_id,
    )


def _run_startup_checks(settings: object, deps: object) -> None:
    """Runs API key and prompt enforcement checks at startup.

    Logs warnings/errors but never raises — the system starts regardless.
    Static tasks still work without AI services.

    Args:
        settings: The application Settings instance.
        deps: The deps module (for accessing _task_registry and _prompt_loader).
    """
    from backend.models import TIER_MAP

    _check_api_keys(settings, TIER_MAP)
    _check_prompt_enforcement(deps)


def _check_api_keys(settings: object, tier_map: dict) -> None:
    """Verifies API keys are configured for all providers in TIER_MAP.

    Args:
        settings: The application Settings instance.
        tier_map: The TIER_MAP dict mapping tiers to ModelConfig.
    """
    # Collect unique providers referenced by TIER_MAP
    providers_seen: set[str] = set()
    for config in tier_map.values():
        providers_seen.add(config.provider)

    key_map = {
        "gemini": ("GOOGLE_API_KEY", getattr(settings, "google_api_key", "")),
        "anthropic": ("ANTHROPIC_API_KEY", getattr(settings, "anthropic_api_key", "")),
    }

    for provider_name in sorted(providers_seen):
        if provider_name in key_map:
            env_var, key_value = key_map[provider_name]
            if not key_value:
                logger.warning(
                    "Missing %s for provider '%s'. "
                    "AI features using this provider will fail at runtime.",
                    env_var,
                    provider_name,
                )


def _check_prompt_enforcement(deps: object) -> None:
    """Validates prompt files exist for all active cartridges with AI phases.

    Args:
        deps: The deps module (for accessing _task_registry and _prompt_loader).
    """
    registry = getattr(deps, "_task_registry", None)
    prompt_loader = getattr(deps, "_prompt_loader", None)

    if registry is None or prompt_loader is None:
        return

    task_ids = registry.get_all_task_ids(status="active")
    for task_id in task_ids:
        cartridge = registry.get_task(task_id)
        if cartridge is None:
            logger.debug("Task %s disappeared during startup check, skipping.", task_id)
            continue

        errors = prompt_loader.validate_task_prompts(cartridge)
        for error in errors:
            logger.error("Prompt enforcement [%s]: %s", task_id, error)


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    settings = get_settings()

    # Configure logging level
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    application = FastAPI(
        title="Makaronas",
        description="Educational AI platform for media literacy",
        version="0.1.0",
    )

    # -- Middleware (order matters: last added = first executed) --
    # Execution order: CSP → RateLimit → Logging → CORS → App

    # CORS — innermost, handles preflight at framework level
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging — raw ASGI, streaming-safe
    application.add_middleware(RequestLoggingMiddleware)

    # Rate limiting — per-session on AI endpoints (429s appear in logs)
    application.add_middleware(RateLimitMiddleware)

    # CSP — outermost, ensures ALL responses get the header (including 429s)
    application.add_middleware(CSPMiddleware)

    # -- Exception handlers --
    application.add_exception_handler(StarletteHTTPException, _http_exception_response)
    application.add_exception_handler(RequestValidationError, _validation_error_response)
    application.add_exception_handler(Exception, _unhandled_exception_response)

    # -- Routers --
    _register_routes(application)

    # -- Static files (AFTER routes — catch-all, must not intercept API) --
    from starlette.staticfiles import StaticFiles

    from backend.config import PROJECT_ROOT

    application.mount(
        "/", StaticFiles(directory=str(PROJECT_ROOT / "static"), html=True),
        name="static",
    )

    # -- Task registry --
    _init_task_registry()

    # -- AI services (must come after task registry) --
    _init_ai_services()

    return application


def _register_routes(application: FastAPI) -> None:
    """Registers all API routers on the application."""
    from fastapi import APIRouter

    v1 = APIRouter(prefix="/api/v1")

    # Health endpoint — the simplest proof the building breathes
    @v1.get("/health")
    async def health() -> dict[str, Any]:
        return ApiResponse(ok=True, data={"status": "healthy"}).model_dump()

    # Sub-routers (BEFORE including v1 into the app):
    from backend.api.student import router as student_router

    v1.include_router(student_router, prefix="/student", tags=["student"])

    from backend.api.teacher import router as teacher_router

    v1.include_router(teacher_router, prefix="/teacher", tags=["teacher"])

    from backend.api.composer import asset_router, router as composer_router

    v1.include_router(composer_router, prefix="/composer", tags=["composer"])
    v1.include_router(asset_router, prefix="/assets", tags=["assets"])

    application.include_router(v1)


app = create_app()
