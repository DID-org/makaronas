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
from pathlib import Path
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

    Derives content_dir and taxonomy_path from the project structure,
    loads all cartridges, and sets the singleton in deps.py. Logs but
    does not crash on empty or missing content directories.
    """
    from backend.api import deps
    from backend.tasks.registry import TaskRegistry

    project_root = Path(__file__).resolve().parent.parent
    content_dir = project_root / "content"
    taxonomy_path = content_dir / "taxonomy.json"

    registry = TaskRegistry(content_dir, taxonomy_path)
    registry.load()
    deps._task_registry = registry


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
