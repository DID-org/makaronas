"""FastAPI application — entry point, middleware, and health endpoint.

Creates the Makaronas backend API with:
- API versioning via router prefix (/api/v1/)
- CORS middleware (origins from settings)
- Request logging middleware (raw ASGI — no response body buffering)
- Global exception handlers (HTTPException, validation, catch-all)
- Health endpoint

Run with: uvicorn backend.main:app --reload

Tier 3 orchestration module: imports from config (Tier 2), deps (Tier 2),
schemas (Tier 1).
"""

from __future__ import annotations

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
    # TODO: Wire prompt_loader.invalidate() to registry reload so prompt
    # changes take effect on hot-reload (Vision §4.4).

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
    from backend.ai.trickster import TricksterEngine

    context_manager = ContextManager(prompt_loader)
    engine = TricksterEngine(provider, context_manager)
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

    # CORS — must be outermost to handle preflight before auth
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging — raw ASGI, streaming-safe
    application.add_middleware(RequestLoggingMiddleware)

    # -- Exception handlers --
    application.add_exception_handler(StarletteHTTPException, _http_exception_response)
    application.add_exception_handler(RequestValidationError, _validation_error_response)
    application.add_exception_handler(Exception, _unhandled_exception_response)

    # -- Routers --
    _register_routes(application)

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
