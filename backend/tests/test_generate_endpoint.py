"""Tests for POST /api/v1/student/session/{session_id}/generate endpoint (Phase 7b).

Tests the context-isolated Tool AI generation path: HTTP request through
provider.complete() to JSON response. Uses MockProvider with direct DI
overrides — no real AI API calls.

Test categories:
1. Happy path — valid generation returns text + artifact_index
2. Safety violation — check_output catches boundary, returns fallback
3. Auth enforcement — missing/invalid token returns 401
4. Ownership enforcement — wrong user returns 403
5. Session not found — 404 SESSION_NOT_FOUND
6. No current task — 422 NO_TASK_ASSIGNED
7. Empty input — 422 INVALID_REQUEST
8. AI unavailable — missing API key returns 503
9. Artifact accumulation — multiple calls append to generated_artifacts
10. Full integration — HTTP through provider through JSON response
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from backend.ai.context import AssembledContext, ContextManager
from backend.ai.providers.mock import MockProvider
from backend.api import deps
from backend.api.deps import (
    get_context_manager,
    get_task_registry,
    get_trickster_engine,
)
from backend.main import app
from backend.tasks.registry import TaskRegistry

AUTH_HEADER = {"Authorization": "Bearer test-token-123"}
FAKE_USER_ID = "fake-user-1"
FAKE_SCHOOL_ID = "school-test-1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _use_registry_with(cartridges):
    """Injects a pre-loaded registry into app dependency overrides."""
    registry = TaskRegistry(Path("/tmp"), Path("/tmp"))
    for c in cartridges:
        registry._by_id[c.task_id] = c
        registry._by_status.setdefault(c.status, set()).add(c.task_id)
        registry._by_trigger[c.trigger].add(c.task_id)
        registry._by_technique[c.technique].add(c.task_id)
        registry._by_medium[c.medium].add(c.task_id)
        for tag in c.tags:
            registry._by_tag[tag].add(c.task_id)
    app.dependency_overrides[get_task_registry] = lambda: registry


def _mock_context_manager():
    """Creates a mock ContextManager that returns a known AssembledContext."""
    cm = AsyncMock(spec=ContextManager)
    cm.assemble_generation_call = lambda source_content, student_prompt: AssembledContext(
        system_prompt="Test generation system prompt",
        messages=[
            {"role": "user", "content": source_content},
            {"role": "user", "content": student_prompt},
        ],
        tools=None,
    )
    return cm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> httpx.AsyncClient:
    """Async test client wired to the app."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture(autouse=True)
def _cleanup_overrides():
    """Ensures dependency overrides are cleaned up after each test."""
    yield
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def session_with_task(make_session, make_cartridge):
    """Creates a session with a current task and injects the cartridge."""
    cartridge = make_cartridge()
    session = make_session(
        student_id=FAKE_USER_ID,
        school_id=FAKE_SCHOOL_ID,
        current_task=cartridge.task_id,
        current_phase="phase_ai",
    )
    await deps._session_store.save_session(session)
    _use_registry_with([cartridge])
    return session, cartridge


# ---------------------------------------------------------------------------
# Stub engine for DI (endpoints that don't use it still resolve it)
# ---------------------------------------------------------------------------


def _stub_engine():
    """Stubs out the trickster engine dependency."""
    from backend.ai.trickster import TricksterEngine

    engine = AsyncMock(spec=TricksterEngine)
    app.dependency_overrides[get_trickster_engine] = lambda: engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateHappyPath:
    """Test 1 & 10: Happy path generation with full HTTP chain."""

    @pytest.mark.asyncio
    @patch("backend.api.student._check_generation_readiness", return_value=[])
    async def test_generate_returns_text_and_artifact_index(
        self, _mock_readiness, client, session_with_task,
    ):
        """Valid generation returns generated_text and artifact_index=0."""
        session, cartridge = session_with_task
        _stub_engine()

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        with patch(
            "backend.api.student.create_provider",
        ) as mock_create:
            provider = MockProvider(responses=["Generated social post"])
            mock_create.return_value = provider

            resp = await client.post(
                f"/api/v1/student/session/{session.session_id}/generate",
                json={
                    "source_content": "Some source article text",
                    "student_prompt": "Make it misleading",
                },
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["data"]["generated_text"] == "Generated social post"
        assert data["data"]["artifact_index"] == 0
        assert "safety_redacted" not in data["data"]

        # Verify artifact stored in session
        stored = await deps._session_store.get_session(session.session_id)
        assert len(stored.generated_artifacts) == 1
        artifact = stored.generated_artifacts[0]
        assert artifact["student_prompt"] == "Make it misleading"
        assert artifact["generated_text"] == "Generated social post"
        assert artifact["safety_redacted"] is False
        assert "timestamp" in artifact


class TestGenerateSafety:
    """Test 2: Safety violation returns fallback text."""

    @pytest.mark.asyncio
    @patch("backend.api.student._check_generation_readiness", return_value=[])
    async def test_safety_violation_returns_fallback(
        self, _mock_readiness, client, session_with_task,
    ):
        """When check_output detects violation, returns fallback_text."""
        session, cartridge = session_with_task
        _stub_engine()

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        # The cartridge has safety boundary "self_harm"
        # MockProvider returns text containing a self_harm keyword
        with patch(
            "backend.api.student.create_provider",
        ) as mock_create, patch(
            "backend.api.student.check_output",
        ) as mock_safety:
            provider = MockProvider(responses=["unsafe content about self harm"])
            mock_create.return_value = provider

            from backend.ai.safety import SafetyResult, SafetyViolation

            mock_safety.return_value = SafetyResult(
                is_safe=False,
                violation=SafetyViolation(
                    boundary="self_harm",
                    fallback_text="Turinys pa\u0161alintas d\u0117l saugumo.",
                ),
            )

            resp = await client.post(
                f"/api/v1/student/session/{session.session_id}/generate",
                json={
                    "source_content": "Source text",
                    "student_prompt": "Generate something",
                },
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["data"]["generated_text"] == "Turinys pa\u0161alintas d\u0117l saugumo."
        assert data["data"]["safety_redacted"] is True

        # Verify fallback stored, not original
        stored = await deps._session_store.get_session(session.session_id)
        assert stored.generated_artifacts[0]["safety_redacted"] is True
        assert stored.generated_artifacts[0]["generated_text"] == "Turinys pa\u0161alintas d\u0117l saugumo."


class TestGenerateAuth:
    """Tests 3 & 4: Auth and ownership enforcement."""

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, client, session_with_task):
        """Missing Authorization header returns 401."""
        session, _ = session_with_task
        _stub_engine()

        resp = await client.post(
            f"/api/v1/student/session/{session.session_id}/generate",
            json={"source_content": "x", "student_prompt": "y"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_owner_returns_403(
        self, client, make_session, make_cartridge,
    ):
        """Session owned by another user returns 403."""
        cartridge = make_cartridge()
        session = make_session(
            student_id="other-user",
            school_id=FAKE_SCHOOL_ID,
            current_task=cartridge.task_id,
            current_phase="phase_ai",
        )
        await deps._session_store.save_session(session)
        _use_registry_with([cartridge])
        _stub_engine()

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        resp = await client.post(
            f"/api/v1/student/session/{session.session_id}/generate",
            json={"source_content": "x", "student_prompt": "y"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 403


class TestGenerateNotFound:
    """Test 5: Session not found."""

    @pytest.mark.asyncio
    async def test_invalid_session_returns_404(self, client, make_cartridge):
        """Non-existent session_id returns 404."""
        _stub_engine()
        cartridge = make_cartridge()
        _use_registry_with([cartridge])

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        resp = await client.post(
            "/api/v1/student/session/nonexistent-session/generate",
            json={"source_content": "x", "student_prompt": "y"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"


class TestGenerateNoTask:
    """Test 6: No current task assigned."""

    @pytest.mark.asyncio
    async def test_no_current_task_returns_422(self, client, make_session):
        """Session without current_task returns 422."""
        session = make_session(
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
        )
        await deps._session_store.save_session(session)
        _stub_engine()

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        resp = await client.post(
            f"/api/v1/student/session/{session.session_id}/generate",
            json={"source_content": "x", "student_prompt": "y"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "NO_TASK_ASSIGNED"


class TestGenerateEmptyInput:
    """Test 7: Empty source_content or student_prompt."""

    @pytest.mark.asyncio
    async def test_empty_source_content_returns_422(
        self, client, session_with_task,
    ):
        """Empty source_content returns 422 INVALID_REQUEST."""
        session, _ = session_with_task
        _stub_engine()

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        resp = await client.post(
            f"/api/v1/student/session/{session.session_id}/generate",
            json={"source_content": "  ", "student_prompt": "y"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "INVALID_REQUEST"

    @pytest.mark.asyncio
    async def test_empty_student_prompt_returns_422(
        self, client, session_with_task,
    ):
        """Empty student_prompt returns 422 INVALID_REQUEST."""
        session, _ = session_with_task
        _stub_engine()

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        resp = await client.post(
            f"/api/v1/student/session/{session.session_id}/generate",
            json={"source_content": "some content", "student_prompt": ""},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "INVALID_REQUEST"


class TestGenerateAiUnavailable:
    """Test 8: Missing API key for fast tier returns 503."""

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_503(
        self, client, session_with_task,
    ):
        """When fast tier API key is missing, returns 503."""
        session, _ = session_with_task
        _stub_engine()

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        with patch(
            "backend.api.student._check_generation_readiness",
            return_value=["Missing API key for provider 'gemini' (tier: 'fast')"],
        ):
            resp = await client.post(
                f"/api/v1/student/session/{session.session_id}/generate",
                json={"source_content": "x", "student_prompt": "y"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "AI_UNAVAILABLE"


class TestGenerateArtifactAccumulation:
    """Test 9: Multiple generate calls append to generated_artifacts."""

    @pytest.mark.asyncio
    @patch("backend.api.student._check_generation_readiness", return_value=[])
    async def test_multiple_calls_accumulate_artifacts(
        self, _mock_readiness, client, session_with_task,
    ):
        """Each generate call appends a new artifact."""
        session, cartridge = session_with_task
        _stub_engine()

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        for i in range(3):
            with patch(
                "backend.api.student.create_provider",
            ) as mock_create:
                provider = MockProvider(responses=[f"Generated post {i}"])
                mock_create.return_value = provider

                resp = await client.post(
                    f"/api/v1/student/session/{session.session_id}/generate",
                    json={
                        "source_content": "Source text",
                        "student_prompt": f"Prompt {i}",
                    },
                    headers=AUTH_HEADER,
                )

            assert resp.status_code == 200
            assert resp.json()["data"]["artifact_index"] == i

        # Verify all 3 artifacts stored
        stored = await deps._session_store.get_session(session.session_id)
        assert len(stored.generated_artifacts) == 3
        assert stored.generated_artifacts[0]["generated_text"] == "Generated post 0"
        assert stored.generated_artifacts[2]["generated_text"] == "Generated post 2"


class TestGenerateTaskNotFound:
    """Task not in registry returns 404."""

    @pytest.mark.asyncio
    async def test_task_not_in_registry_returns_404(
        self, client, make_session,
    ):
        """Session references a task not in the registry."""
        session = make_session(
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="nonexistent-task",
        )
        await deps._session_store.save_session(session)
        _stub_engine()

        # Empty registry
        registry = TaskRegistry(Path("/tmp"), Path("/tmp"))
        app.dependency_overrides[get_task_registry] = lambda: registry

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        resp = await client.post(
            f"/api/v1/student/session/{session.session_id}/generate",
            json={"source_content": "x", "student_prompt": "y"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "TASK_NOT_FOUND"


class TestGenerateNoSafetyConfig:
    """When cartridge has no ai_config, safety check is skipped gracefully."""

    @pytest.mark.asyncio
    @patch("backend.api.student._check_generation_readiness", return_value=[])
    async def test_no_safety_config_still_generates(
        self, _mock_readiness, client, make_session, make_cartridge,
    ):
        """Static-only cartridge (no ai_config) still allows generation."""
        cartridge = make_cartridge(ai_config=None, task_type="static")
        session = make_session(
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task=cartridge.task_id,
            current_phase="phase_intro",
        )
        await deps._session_store.save_session(session)
        _use_registry_with([cartridge])
        _stub_engine()

        cm = _mock_context_manager()
        app.dependency_overrides[get_context_manager] = lambda: cm

        with patch(
            "backend.api.student.create_provider",
        ) as mock_create:
            provider = MockProvider(responses=["Generated without safety"])
            mock_create.return_value = provider

            resp = await client.post(
                f"/api/v1/student/session/{session.session_id}/generate",
                json={
                    "source_content": "Source text",
                    "student_prompt": "Generate something",
                },
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        assert resp.json()["data"]["generated_text"] == "Generated without safety"
