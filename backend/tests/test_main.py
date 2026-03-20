"""Tests for FastAPI app, middleware, and dependency injection.

Covers: health endpoint, CORS headers, auth dependency (happy path + failures),
exception handlers (HTTPException, validation, unhandled), request logging,
CSP headers, rate limiting, static file serving.

Uses httpx.AsyncClient with ASGITransport (async test client). All tests use
explicit @pytest.mark.asyncio per strict mode (Python 3.13.5, Phase 1a note).
"""

import logging
import time

import httpx
import pytest
from fastapi import APIRouter, Depends
from httpx import ASGITransport
from pydantic import BaseModel

from backend.api.deps import get_current_user
from backend.main import (
    CSP_HEADER_VALUE,
    RATE_LIMIT_MAX_REQUESTS,
    app,
    create_app,
)
from backend.schemas import User


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> httpx.AsyncClient:
    """Async test client wired to the app."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# Helper: mount a tiny test route for auth testing
# ---------------------------------------------------------------------------

_test_router = APIRouter(prefix="/api/v1/test")


@_test_router.get("/protected")
async def protected_route(user: User = Depends(get_current_user)) -> dict:
    return {"user_id": user.id, "role": user.role}


class _BodyModel(BaseModel):
    name: str
    age: int


@_test_router.post("/validated")
async def validated_route(body: _BodyModel) -> dict:
    return {"name": body.name}


@_test_router.get("/explode")
async def exploding_route() -> dict:
    raise RuntimeError("Something went terribly wrong")


app.include_router(_test_router)

# The StaticFiles mount at "/" is a catch-all — routes added AFTER it
# are unreachable. Move the mount to the end so test routes match first.
for _i, _route in enumerate(app.routes):
    if getattr(_route, "name", None) == "static":
        app.routes.append(app.routes.pop(_i))
        break


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """GET /api/v1/health — the proof the building breathes."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_api_response(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/health")
        body = resp.json()
        assert body["ok"] is True
        assert body["data"]["status"] == "healthy"
        assert body["error"] is None


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    """CORS middleware — allows configured origins, blocks others."""

    @pytest.mark.asyncio
    async def test_allowed_origin_gets_cors_header(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.options(
                "/api/v1/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_disallowed_origin_no_cors_header(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.options(
                "/api/v1/health",
                headers={
                    "Origin": "http://evil.example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert "access-control-allow-origin" not in resp.headers


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


class TestAuthDependency:
    """get_current_user — Bearer token extraction and validation."""

    @pytest.mark.asyncio
    async def test_valid_bearer_token(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/test/protected",
                headers={"Authorization": "Bearer test-token-123"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "fake-user-1"
        assert body["role"] == "student"

    @pytest.mark.asyncio
    async def test_missing_header_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/test/protected")
        assert resp.status_code == 401
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "UNAUTHORIZED"

    @pytest.mark.asyncio
    async def test_empty_bearer_token_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/test/protected",
                headers={"Authorization": "Bearer "},
            )
        assert resp.status_code == 401
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "UNAUTHORIZED"

    @pytest.mark.asyncio
    async def test_malformed_header_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/test/protected",
                headers={"Authorization": "Basic abc123"},
            )
        assert resp.status_code == 401
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "UNAUTHORIZED"

    @pytest.mark.asyncio
    async def test_no_scheme_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/test/protected",
                headers={"Authorization": "just-a-token"},
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


class TestExceptionHandlers:
    """Global exception handling — consistent ApiResponse envelopes."""

    @pytest.mark.asyncio
    async def test_unhandled_exception_returns_500(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/test/explode")
        assert resp.status_code == 500
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert body["error"]["message"] == "An unexpected error occurred."

    @pytest.mark.asyncio
    async def test_unhandled_exception_no_traceback_in_body(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get("/api/v1/test/explode")
        body_text = resp.text
        assert "RuntimeError" not in body_text
        assert "traceback" not in body_text.lower()
        assert "went terribly wrong" not in body_text

    @pytest.mark.asyncio
    async def test_validation_error_returns_422(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/test/validated",
                json={"name": "test"},  # missing 'age' field
            )
        assert resp.status_code == 422
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_404_returns_api_response(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/nonexistent")
        assert resp.status_code == 404
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "HTTP_ERROR"


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------


class TestRequestLogging:
    """Request logging middleware — captures method, path, status, duration."""

    @pytest.mark.asyncio
    async def test_request_is_logged(
        self, client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="makaronas"):
            async with client:
                await client.get("/api/v1/health")

        log_messages = [r.message for r in caplog.records if r.name == "makaronas"]
        assert any("GET" in msg and "/api/v1/health" in msg and "200" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_log_includes_duration(
        self, client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="makaronas"):
            async with client:
                await client.get("/api/v1/health")

        log_messages = [r.message for r in caplog.records if r.name == "makaronas"]
        assert any("ms" in msg for msg in log_messages)


# ---------------------------------------------------------------------------
# CSP middleware
# ---------------------------------------------------------------------------


class TestCSPMiddleware:
    """Content-Security-Policy header — appears on every HTTP response."""

    @pytest.mark.asyncio
    async def test_csp_on_api_response(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/health")
        assert resp.headers.get("content-security-policy") == CSP_HEADER_VALUE

    @pytest.mark.asyncio
    async def test_csp_on_static_file(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/")
        assert resp.headers.get("content-security-policy") == CSP_HEADER_VALUE

    @pytest.mark.asyncio
    async def test_csp_on_error_response(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/nonexistent")
        assert resp.status_code == 404
        assert resp.headers.get("content-security-policy") == CSP_HEADER_VALUE

    @pytest.mark.asyncio
    async def test_csp_header_appears_exactly_once(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/health")
        csp_values = resp.headers.get_list("content-security-policy")
        assert len(csp_values) == 1


# ---------------------------------------------------------------------------
# Rate limiting middleware
# ---------------------------------------------------------------------------


class TestRateLimitMiddleware:
    """Per-session rate limiter on AI-consuming endpoints."""

    @pytest.fixture
    def fresh_client(self) -> httpx.AsyncClient:
        """Creates a fresh app instance so rate limiter state is clean."""
        fresh_app = create_app()
        transport = ASGITransport(app=fresh_app, raise_app_exceptions=False)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    @pytest.mark.asyncio
    async def test_requests_within_limit_pass(self, fresh_client: httpx.AsyncClient) -> None:
        async with fresh_client:
            for _ in range(3):
                resp = await fresh_client.post(
                    "/api/v1/student/session/test-sess/respond",
                    headers={"Authorization": "Bearer test-token"},
                    json={"action": "chat", "payload": "hello"},
                )
                # May get 422 (validation) or other errors but NOT 429
                assert resp.status_code != 429

    @pytest.mark.asyncio
    async def test_exceeding_limit_returns_429(self, fresh_client: httpx.AsyncClient) -> None:
        async with fresh_client:
            for _ in range(RATE_LIMIT_MAX_REQUESTS):
                await fresh_client.post(
                    "/api/v1/student/session/test-sess/respond",
                    headers={"Authorization": "Bearer test-token"},
                    json={"action": "chat", "payload": "hello"},
                )
            # Next request should be rate-limited
            resp = await fresh_client.post(
                "/api/v1/student/session/test-sess/respond",
                headers={"Authorization": "Bearer test-token"},
                json={"action": "chat", "payload": "hello"},
            )
        assert resp.status_code == 429
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_different_sessions_independent(self, fresh_client: httpx.AsyncClient) -> None:
        async with fresh_client:
            # Exhaust limit for session-a
            for _ in range(RATE_LIMIT_MAX_REQUESTS + 1):
                await fresh_client.post(
                    "/api/v1/student/session/session-a/respond",
                    headers={"Authorization": "Bearer test-token"},
                    json={"action": "chat", "payload": "hello"},
                )
            # session-b should still work
            resp = await fresh_client.post(
                "/api/v1/student/session/session-b/respond",
                headers={"Authorization": "Bearer test-token"},
                json={"action": "chat", "payload": "hello"},
            )
        assert resp.status_code != 429

    @pytest.mark.asyncio
    async def test_non_limited_path_not_blocked(self, fresh_client: httpx.AsyncClient) -> None:
        async with fresh_client:
            # Exhaust limit for a session on /respond
            for _ in range(RATE_LIMIT_MAX_REQUESTS + 1):
                await fresh_client.post(
                    "/api/v1/student/session/test-sess/respond",
                    headers={"Authorization": "Bearer test-token"},
                    json={"action": "chat", "payload": "hello"},
                )
            # /health should not be affected
            resp = await fresh_client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_requests_not_rate_limited(self, fresh_client: httpx.AsyncClient) -> None:
        async with fresh_client:
            for _ in range(RATE_LIMIT_MAX_REQUESTS + 5):
                resp = await fresh_client.get(
                    "/api/v1/student/session/test-sess/respond",
                    headers={"Authorization": "Bearer test-token"},
                )
            # GET should never trigger rate limit (status might be 405, that's fine)
            assert resp.status_code != 429

    @pytest.mark.asyncio
    async def test_generate_endpoint_also_limited(self, fresh_client: httpx.AsyncClient) -> None:
        async with fresh_client:
            for _ in range(RATE_LIMIT_MAX_REQUESTS):
                await fresh_client.post(
                    "/api/v1/student/session/test-sess/generate",
                    headers={"Authorization": "Bearer test-token"},
                    json={"source_content": "x", "student_prompt": "y"},
                )
            resp = await fresh_client.post(
                "/api/v1/student/session/test-sess/generate",
                headers={"Authorization": "Bearer test-token"},
                json={"source_content": "x", "student_prompt": "y"},
            )
        assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_respond_and_generate_share_counter(
        self, fresh_client: httpx.AsyncClient
    ) -> None:
        async with fresh_client:
            # Send half on /respond, half on /generate
            for _ in range(RATE_LIMIT_MAX_REQUESTS // 2):
                await fresh_client.post(
                    "/api/v1/student/session/test-sess/respond",
                    headers={"Authorization": "Bearer test-token"},
                    json={"action": "chat", "payload": "hello"},
                )
            for _ in range(RATE_LIMIT_MAX_REQUESTS // 2):
                await fresh_client.post(
                    "/api/v1/student/session/test-sess/generate",
                    headers={"Authorization": "Bearer test-token"},
                    json={"source_content": "x", "student_prompt": "y"},
                )
            # Next request on either should be rate-limited
            resp = await fresh_client.post(
                "/api/v1/student/session/test-sess/respond",
                headers={"Authorization": "Bearer test-token"},
                json={"action": "chat", "payload": "hello"},
            )
        assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_window_reset_allows_requests(self) -> None:
        """After the rate limit window expires, requests succeed again."""
        from unittest.mock import patch

        fresh_app = create_app()
        transport = ASGITransport(app=fresh_app, raise_app_exceptions=False)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")

        async with client:
            # Exhaust the limit
            for _ in range(RATE_LIMIT_MAX_REQUESTS + 1):
                await client.post(
                    "/api/v1/student/session/test-sess/respond",
                    headers={"Authorization": "Bearer test-token"},
                    json={"action": "chat", "payload": "hello"},
                )

            # Simulate time passing by patching time.monotonic
            original_monotonic = time.monotonic
            offset = 61  # past the 60s window

            def shifted_monotonic() -> float:
                return original_monotonic() + offset

            with patch("backend.main.time.monotonic", shifted_monotonic):
                resp = await client.post(
                    "/api/v1/student/session/test-sess/respond",
                    headers={"Authorization": "Bearer test-token"},
                    json={"action": "chat", "payload": "hello"},
                )
            assert resp.status_code != 429

    @pytest.mark.asyncio
    async def test_429_includes_csp_header(self, fresh_client: httpx.AsyncClient) -> None:
        """Rate-limited responses also get the CSP header (CSP is outermost)."""
        async with fresh_client:
            for _ in range(RATE_LIMIT_MAX_REQUESTS + 1):
                await fresh_client.post(
                    "/api/v1/student/session/test-sess/respond",
                    headers={"Authorization": "Bearer test-token"},
                    json={"action": "chat", "payload": "hello"},
                )
            resp = await fresh_client.post(
                "/api/v1/student/session/test-sess/respond",
                headers={"Authorization": "Bearer test-token"},
                json={"action": "chat", "payload": "hello"},
            )
        assert resp.status_code == 429
        assert resp.headers.get("content-security-policy") == CSP_HEADER_VALUE


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------


class TestStaticFiles:
    """StaticFiles mount — serves static/ directory at root URL."""

    @pytest.mark.asyncio
    async def test_root_serves_index_html(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert "Makaronas" in resp.text

    @pytest.mark.asyncio
    async def test_nonexistent_static_file_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get("/nonexistent.js")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_api_routes_take_priority(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["data"]["status"] == "healthy"
