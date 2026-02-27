"""Tests for Student-facing API endpoints.

Covers: session creation, next task (registry-backed), session lookup (404),
ownership enforcement (403), SSE streaming (respond + debrief), radar profile
(student + teacher access), GDPR deletion + export, auth enforcement on all
endpoints.

Uses httpx.AsyncClient with ASGITransport (async test client). All tests use
explicit @pytest.mark.asyncio per strict mode (Python 3.13.5, Phase 1a note).

Updated: Phase 4b — rewrote TestNextTask for registry-backed endpoint,
    updated TestRespond for open action type.
Updated: Phase 6b — updated TestRespond and TestDebrief for real AI integration,
    injecting mock engine instead of relying on removed stubs.
"""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from backend.ai.trickster import DebriefResult, TricksterResult
from backend.api import deps
from backend.api.deps import get_current_user, get_task_registry, get_trickster_engine
from backend.main import app
from backend.schemas import GameSession, StudentProfile, User
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import TaskCartridge

AUTH_HEADER = {"Authorization": "Bearer test-token-123"}

# The user returned by FakeAuthService for any valid token
FAKE_USER_ID = "fake-user-1"
FAKE_SCHOOL_ID = "school-test-1"

TEACHER_USER = User(
    id="teacher-1", role="teacher", name="Test Teacher", school_id=FAKE_SCHOOL_ID
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> httpx.AsyncClient:
    """Async test client wired to the app."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest_asyncio.fixture
async def session_id() -> str:
    """Creates a session in the store and returns its ID.

    Uses the fake auth user's ID so ownership checks pass.
    """
    session = GameSession(
        session_id="test-session-aaa",
        student_id=FAKE_USER_ID,
        school_id=FAKE_SCHOOL_ID,
    )
    await deps._session_store.save_session(session)
    return session.session_id


@pytest_asyncio.fixture
async def other_session_id() -> str:
    """Creates a session owned by a different student."""
    session = GameSession(
        session_id="test-session-other",
        student_id="other-user",
        school_id=FAKE_SCHOOL_ID,
    )
    await deps._session_store.save_session(session)
    return session.session_id


@pytest.fixture(autouse=True)
def _cleanup_overrides():
    """Ensures dependency overrides are cleaned up after each test."""
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Registry test helpers
# ---------------------------------------------------------------------------


def _minimal_cartridge_data(task_id: str, **overrides: object) -> dict:
    """Returns a minimal valid cartridge dict with presentation blocks.

    Includes two blocks (TextBlock + SocialPostBlock) and an initial phase
    referencing both via visible_blocks, plus a terminal phase.
    """
    data: dict = {
        "task_id": task_id,
        "task_type": "static",
        "title": "Testas",
        "description": "Testo aprašymas",
        "version": "1.0",
        "trigger": "urgency",
        "technique": "headline_manipulation",
        "medium": "article",
        "learning_objectives": ["Atpažinti manipuliaciją"],
        "difficulty": 3,
        "time_minutes": 15,
        "is_evergreen": True,
        "is_clean": False,
        "initial_phase": "phase_intro",
        "presentation_blocks": [
            {
                "id": "block-headline",
                "type": "text",
                "text": "Mokslininkai patvirtino: kava suteikia nemirtingumą",
            },
            {
                "id": "block-post",
                "type": "social_post",
                "author": "user123",
                "text": "Negaliu patikėti! Turite tai perskaityti!",
                "platform": "generic",
            },
        ],
        "phases": [
            {
                "id": "phase_intro",
                "title": "Įvadas",
                "visible_blocks": ["block-headline", "block-post"],
                "is_ai_phase": False,
                "interaction": {
                    "type": "button",
                    "choices": [
                        {"label": "Tęsti", "target_phase": "phase_reveal"},
                    ],
                },
            },
            {
                "id": "phase_reveal",
                "title": "Atskleidimas",
                "is_terminal": True,
                "evaluation_outcome": "trickster_loses",
            },
        ],
        "evaluation": {
            "patterns_embedded": [
                {
                    "id": "p1",
                    "description": "Urgency pattern",
                    "technique": "headline_manipulation",
                    "real_world_connection": "Common in news",
                },
            ],
            "checklist": [
                {
                    "id": "c1",
                    "description": "Identified urgency",
                    "pattern_refs": ["p1"],
                    "is_mandatory": True,
                },
            ],
            "pass_conditions": {
                "trickster_wins": "Mokinys pasidalino",
                "partial": "Mokinys perskaitė, bet praleido",
                "trickster_loses": "Mokinys atpažino technikas",
            },
        },
        "reveal": {"key_lesson": "Antraštė buvo sukurta skubos jausmui sukelti"},
        "safety": {
            "content_boundaries": ["no_real_harm"],
            "intensity_ceiling": 3,
            "cold_start_safe": True,
        },
    }
    data.update(overrides)
    return data


def _build_cartridge(task_id: str, **overrides: object) -> TaskCartridge:
    """Builds a validated TaskCartridge from minimal data with overrides."""
    data = _minimal_cartridge_data(task_id, **overrides)
    return TaskCartridge.model_validate(data)


def _use_registry_with(cartridges: list[TaskCartridge]) -> None:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse_events(body: str) -> list[dict]:
    """Parses raw SSE body into a list of {type, data} dicts."""
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = None
        data_json = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_json = line[6:]
        if event_type and data_json:
            events.append({"type": event_type, "data": json.loads(data_json)})
    return events


# ---------------------------------------------------------------------------
# Mock engine for respond/debrief tests (replaces stubs)
# ---------------------------------------------------------------------------


class _StubEngine:
    """Minimal mock TricksterEngine for existing endpoint tests.

    Returns canned tokens and post-completion data. Keeps existing tests
    focused on auth, ownership, and SSE format verification.
    """

    async def respond(self, session, cartridge, phase, student_input):
        """Returns a TricksterResult with canned tokens."""
        async def _tokens():
            for t in ["Mock ", "response. "]:
                yield t
            result.done_data = {
                "phase_transition": None,
                "next_phase": None,
                "exchanges_count": 1,
            }
        result = TricksterResult(token_iterator=_tokens())
        return result

    async def debrief(self, session, cartridge):
        """Returns a DebriefResult with canned tokens."""
        async def _tokens():
            for t in ["Mock ", "debrief. "]:
                yield t
            result.done_data = {"debrief_complete": True}
        result = DebriefResult(token_iterator=_tokens())
        return result


def _ai_cartridge_data(task_id: str) -> dict:
    """Returns a minimal AI-capable cartridge dict for respond/debrief tests."""
    return {
        "task_id": task_id,
        "task_type": "hybrid",
        "title": "AI testas",
        "description": "Testo aprasymas",
        "version": "1.0",
        "trigger": "urgency",
        "technique": "headline_manipulation",
        "medium": "article",
        "learning_objectives": ["Atpazinti"],
        "difficulty": 3,
        "time_minutes": 15,
        "is_evergreen": True,
        "is_clean": False,
        "initial_phase": "phase_intro",
        "ai_config": {
            "model_preference": "standard",
            "prompt_directory": "tasks/" + task_id,
            "persona_mode": "presenting",
            "has_static_fallback": False,
            "context_requirements": "session_only",
        },
        "phases": [
            {
                "id": "phase_intro",
                "title": "Ivadas",
                "is_ai_phase": False,
                "interaction": {
                    "type": "button",
                    "choices": [
                        {"label": "Testi", "target_phase": "phase_ai"},
                    ],
                },
            },
            {
                "id": "phase_ai",
                "title": "Pokalbis",
                "is_ai_phase": True,
                "interaction": {
                    "type": "freeform",
                    "trickster_opening": "Sveiki!",
                    "min_exchanges": 1,
                    "max_exchanges": 5,
                },
                "ai_transitions": {
                    "on_success": "phase_reveal",
                    "on_max_exchanges": "phase_reveal",
                    "on_partial": "phase_reveal",
                },
            },
            {
                "id": "phase_reveal",
                "title": "Atskleidimas",
                "is_terminal": True,
                "evaluation_outcome": "trickster_loses",
            },
        ],
        "evaluation": {
            "patterns_embedded": [
                {
                    "id": "p1",
                    "description": "Pattern",
                    "technique": "headline_manipulation",
                    "real_world_connection": "Common",
                },
            ],
            "checklist": [
                {
                    "id": "c1",
                    "description": "Found it",
                    "pattern_refs": ["p1"],
                    "is_mandatory": True,
                },
            ],
            "pass_conditions": {
                "trickster_wins": "Lost",
                "partial": "Partial",
                "trickster_loses": "Won",
            },
        },
        "reveal": {"key_lesson": "Test lesson"},
        "safety": {
            "content_boundaries": ["self_harm"],
            "intensity_ceiling": 3,
            "cold_start_safe": True,
        },
    }


async def _setup_ai_session(task_id: str = "task-ai-test-001") -> str:
    """Creates an AI-ready session with matching cartridge in registry.

    Returns the session_id. Sets up:
    - AI cartridge in registry override
    - Session with current_task and current_phase pointing to AI phase
    - Stub engine in DI overrides
    - Patches check_ai_readiness to return no issues
    """
    cartridge = TaskCartridge.model_validate(_ai_cartridge_data(task_id))

    # Registry
    registry = TaskRegistry(Path("/tmp"), Path("/tmp"))
    registry._by_id[task_id] = cartridge
    registry._by_status.setdefault(cartridge.status, set()).add(task_id)
    registry._by_trigger[cartridge.trigger].add(task_id)
    registry._by_technique[cartridge.technique].add(task_id)
    registry._by_medium[cartridge.medium].add(task_id)
    app.dependency_overrides[get_task_registry] = lambda: registry

    # Engine
    app.dependency_overrides[get_trickster_engine] = lambda: _StubEngine()

    # Session
    session = GameSession(
        session_id="test-ai-session",
        student_id=FAKE_USER_ID,
        school_id=FAKE_SCHOOL_ID,
        current_task=task_id,
        current_phase="phase_ai",
    )
    await deps._session_store.save_session(session)
    return session.session_id


# ---------------------------------------------------------------------------
# POST /session
# ---------------------------------------------------------------------------


class TestCreateSession:
    """POST /api/v1/student/session — creates a new game session."""

    @pytest.mark.asyncio
    async def test_creates_session_returns_200(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session",
                json={},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "session_id" in body["data"]
        assert body["data"]["language"] == "lt"

    @pytest.mark.asyncio
    async def test_custom_language(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session",
                json={"language": "en"},
                headers=AUTH_HEADER,
            )
        body = resp.json()
        assert body["data"]["language"] == "en"

    @pytest.mark.asyncio
    async def test_custom_roadmap_id(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session",
                json={"roadmap_id": "intro-sequence"},
                headers=AUTH_HEADER,
            )
        body = resp.json()
        assert body["ok"] is True

    @pytest.mark.asyncio
    async def test_session_persisted_in_store(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session",
                json={},
                headers=AUTH_HEADER,
            )
        sid = resp.json()["data"]["session_id"]
        session = await deps._session_store.get_session(sid)
        assert session is not None
        assert session.student_id == FAKE_USER_ID
        assert session.school_id == FAKE_SCHOOL_ID

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post("/api/v1/student/session", json={})
        assert resp.status_code == 401
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# GET /session/{session_id}/next
# ---------------------------------------------------------------------------


class TestNextTask:
    """GET /api/v1/student/session/{id}/next — registry-backed task content."""

    # --- Core response tests ---

    @pytest.mark.asyncio
    async def test_returns_real_task_data(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        cartridge = _build_cartridge("task-test-001")
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-test-001",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert data["task_id"] == "task-test-001"
        assert data["task_type"] == "static"
        assert data["medium"] == "article"
        assert data["title"] == "Testas"
        assert data["current_phase"] == "phase_intro"
        assert isinstance(data["content"], list)
        assert len(data["content"]) > 0
        assert isinstance(data["available_actions"], list)

    @pytest.mark.asyncio
    async def test_content_blocks_resolved_correctly(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        cartridge = _build_cartridge("task-blocks-001")
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-blocks-001",
                headers=AUTH_HEADER,
            )
        data = resp.json()["data"]
        blocks = data["content"]
        assert len(blocks) == 2
        # First block: TextBlock
        assert blocks[0]["id"] == "block-headline"
        assert blocks[0]["type"] == "text"
        assert "text" in blocks[0]
        # Second block: SocialPostBlock
        assert blocks[1]["id"] == "block-post"
        assert blocks[1]["type"] == "social_post"
        assert "author" in blocks[1]

    # --- Available actions tests ---

    @pytest.mark.asyncio
    async def test_available_actions_button(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Button interaction → ["button_click"]."""
        cartridge = _build_cartridge("task-btn-001")
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-btn-001",
                headers=AUTH_HEADER,
            )
        assert resp.json()["data"]["available_actions"] == ["button_click"]

    @pytest.mark.asyncio
    async def test_available_actions_freeform(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Freeform interaction → ["freeform"]."""
        cartridge = _build_cartridge(
            "task-ff-001",
            task_type="hybrid",
            phases=[
                {
                    "id": "phase_intro",
                    "title": "Pokalbis",
                    "is_ai_phase": True,
                    "interaction": {
                        "type": "freeform",
                        "trickster_opening": "Na, ką manai?",
                        "min_exchanges": 1,
                        "max_exchanges": 3,
                    },
                    "ai_transitions": {
                        "on_success": "phase_reveal",
                        "on_max_exchanges": "phase_reveal",
                        "on_partial": "phase_reveal",
                    },
                },
                {
                    "id": "phase_reveal",
                    "title": "Atskleidimas",
                    "is_terminal": True,
                    "evaluation_outcome": "trickster_loses",
                },
            ],
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "tasks/task-ff-001",
                "persona_mode": "presenting",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-ff-001",
                headers=AUTH_HEADER,
            )
        assert resp.json()["data"]["available_actions"] == ["freeform"]

    @pytest.mark.asyncio
    async def test_available_actions_investigation(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Investigation interaction → ["investigate"]."""
        cartridge = _build_cartridge(
            "task-inv-001",
            phases=[
                {
                    "id": "phase_intro",
                    "title": "Tyrimas",
                    "interaction": {
                        "type": "investigation",
                        "starting_queries": ["kas nutiko?"],
                        "submit_target": "phase_reveal",
                    },
                },
                {
                    "id": "phase_reveal",
                    "title": "Atskleidimas",
                    "is_terminal": True,
                    "evaluation_outcome": "trickster_loses",
                },
            ],
        )
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-inv-001",
                headers=AUTH_HEADER,
            )
        assert resp.json()["data"]["available_actions"] == ["investigate"]

    @pytest.mark.asyncio
    async def test_available_actions_no_interaction(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Phase with no interaction → []."""
        cartridge = _build_cartridge(
            "task-noint-001",
            phases=[
                {
                    "id": "phase_intro",
                    "title": "Įvadas",
                    "is_terminal": True,
                    "evaluation_outcome": "trickster_wins",
                },
            ],
        )
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-noint-001",
                headers=AUTH_HEADER,
            )
        assert resp.json()["data"]["available_actions"] == []

    # --- Trickster intro tests ---

    @pytest.mark.asyncio
    async def test_trickster_intro_from_trickster_content(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Static phase with trickster_content → uses it as intro."""
        cartridge = _build_cartridge(
            "task-tc-001",
            phases=[
                {
                    "id": "phase_intro",
                    "title": "Įvadas",
                    "trickster_content": "Sveiki, aš esu Triukšmadarys!",
                    "interaction": {
                        "type": "button",
                        "choices": [
                            {"label": "Tęsti", "target_phase": "phase_reveal"},
                        ],
                    },
                },
                {
                    "id": "phase_reveal",
                    "title": "Atskleidimas",
                    "is_terminal": True,
                    "evaluation_outcome": "trickster_loses",
                },
            ],
        )
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-tc-001",
                headers=AUTH_HEADER,
            )
        assert resp.json()["data"]["trickster_intro"] == "Sveiki, aš esu Triukšmadarys!"

    @pytest.mark.asyncio
    async def test_trickster_intro_from_freeform_opening(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Freeform phase → trickster_opening as intro."""
        cartridge = _build_cartridge(
            "task-fo-001",
            task_type="hybrid",
            phases=[
                {
                    "id": "phase_intro",
                    "title": "Pokalbis",
                    "is_ai_phase": True,
                    "interaction": {
                        "type": "freeform",
                        "trickster_opening": "Na, ką manai apie šį straipsnį?",
                        "min_exchanges": 1,
                        "max_exchanges": 3,
                    },
                    "ai_transitions": {
                        "on_success": "phase_reveal",
                        "on_max_exchanges": "phase_reveal",
                        "on_partial": "phase_reveal",
                    },
                },
                {
                    "id": "phase_reveal",
                    "title": "Atskleidimas",
                    "is_terminal": True,
                    "evaluation_outcome": "trickster_loses",
                },
            ],
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "tasks/task-fo-001",
                "persona_mode": "presenting",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-fo-001",
                headers=AUTH_HEADER,
            )
        assert resp.json()["data"]["trickster_intro"] == "Na, ką manai apie šį straipsnį?"

    @pytest.mark.asyncio
    async def test_trickster_intro_absent(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Button-only phase with no trickster_content → null."""
        cartridge = _build_cartridge("task-noti-001")
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-noti-001",
                headers=AUTH_HEADER,
            )
        assert resp.json()["data"]["trickster_intro"] is None

    # --- Error cases ---

    @pytest.mark.asyncio
    async def test_task_not_found_returns_404(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        _use_registry_with([])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=nonexistent",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "TASK_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_draft_task_hidden_returns_404(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Draft tasks return 404 — no leaking draft existence."""
        cartridge = _build_cartridge("task-draft-001", status="draft")
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-draft-001",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "TASK_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_no_task_assigned_returns_422(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """No task_id param and session.current_task is None → 422."""
        _use_registry_with([])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "NO_TASK_ASSIGNED"

    # --- Session state tests ---

    @pytest.mark.asyncio
    async def test_session_current_task_used_when_no_query_param(
        self, client: httpx.AsyncClient
    ) -> None:
        """Pre-populated session.current_task serves the correct task."""
        cartridge = _build_cartridge("task-session-001")
        _use_registry_with([cartridge])

        # Create session with current_task pre-set
        session = GameSession(
            session_id="test-session-pre",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-session-001",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.get(
                "/api/v1/student/session/test-session-pre/next",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["task_id"] == "task-session-001"

    @pytest.mark.asyncio
    async def test_session_updated_after_serving(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Session current_task and current_phase are persisted."""
        cartridge = _build_cartridge("task-upd-001")
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-upd-001",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

        session = await deps._session_store.get_session(session_id)
        assert session.current_task == "task-upd-001"
        assert session.current_phase == "phase_intro"

    # --- Stale phase detection (Framework P21) ---

    @pytest.mark.asyncio
    async def test_stale_phase_returns_409(self, client: httpx.AsyncClient) -> None:
        """Stale phase → TASK_CONTENT_UPDATED (409)."""
        cartridge = _build_cartridge("task-stale-001")
        _use_registry_with([cartridge])

        # Session with a phase that doesn't exist in the cartridge
        session = GameSession(
            session_id="test-session-stale",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-stale-001",
            current_phase="phase_that_was_removed",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.get(
                "/api/v1/student/session/test-session-stale/next?task_id=task-stale-001",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"]["code"] == "TASK_CONTENT_UPDATED"
        assert body["data"]["initial_phase"] == "phase_intro"

    # --- Existing auth/session/ownership tests ---

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/student/session/nonexistent-id/next",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "SESSION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_other_students_session_returns_403(
        self, client: httpx.AsyncClient, other_session_id: str
    ) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{other_session_id}/next",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/student/session/any-id/next")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /session/{session_id}/respond
# ---------------------------------------------------------------------------


class TestRespond:
    """POST /api/v1/student/session/{id}/respond — SSE stream."""

    @pytest.fixture(autouse=True)
    def _inject_ai_deps(self):
        """Injects AI deps so endpoint Depends() resolve without 503."""
        app.dependency_overrides[get_trickster_engine] = lambda: _StubEngine()
        if get_task_registry not in app.dependency_overrides:
            app.dependency_overrides[get_task_registry] = lambda: TaskRegistry(
                Path("/tmp"), Path("/tmp")
            )

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_returns_sse_stream(
        self, _mock_readiness, client: httpx.AsyncClient
    ) -> None:
        session_id = await _setup_ai_session()
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{session_id}/respond",
                json={"action": "freeform", "payload": "I think this is fake"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse_events(resp.text)
        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(token_events) >= 1
        assert len(done_events) == 1
        # Engine returns mock done_data, not the old "action_received" stub
        assert done_events[0]["data"]["data"]["phase_transition"] is None

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session/bad-id/respond",
                json={"action": "freeform", "payload": "hello"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_open_action_type_accepted(
        self, _mock_readiness, client: httpx.AsyncClient
    ) -> None:
        """Open action type — non-standard action strings pass validation."""
        session_id = await _setup_ai_session()
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{session_id}/respond",
                json={"action": "timeline_scrub", "payload": "test"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_wrong_action_type_returns_422(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Non-string action type → 422 from Pydantic validation."""
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{session_id}/respond",
                json={"action": 123, "payload": "test"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_payload_returns_422(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{session_id}/respond",
                json={"action": "freeform"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_other_students_session_returns_403(
        self, client: httpx.AsyncClient, other_session_id: str
    ) -> None:
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{other_session_id}/respond",
                json={"action": "freeform", "payload": "test"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session/any-id/respond",
                json={"action": "freeform", "payload": "test"},
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /session/{session_id}/debrief
# ---------------------------------------------------------------------------


class TestDebrief:
    """GET /api/v1/student/session/{id}/debrief — SSE stream."""

    @pytest.fixture(autouse=True)
    def _inject_ai_deps(self):
        """Injects AI deps so endpoint Depends() resolve without 503."""
        app.dependency_overrides[get_trickster_engine] = lambda: _StubEngine()
        if get_task_registry not in app.dependency_overrides:
            app.dependency_overrides[get_task_registry] = lambda: TaskRegistry(
                Path("/tmp"), Path("/tmp")
            )

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_returns_sse_stream(
        self, _mock_readiness, client: httpx.AsyncClient
    ) -> None:
        session_id = await _setup_ai_session()
        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/debrief",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse_events(resp.text)
        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(token_events) >= 1
        assert len(done_events) == 1
        assert done_events[0]["data"]["full_text"] == "Mock debrief. "
        assert done_events[0]["data"]["data"]["debrief_complete"] is True

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/student/session/bad-id/debrief",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_other_students_session_returns_403(
        self, client: httpx.AsyncClient, other_session_id: str
    ) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{other_session_id}/debrief",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/student/session/any-id/debrief")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /profile/{student_id}/radar
# ---------------------------------------------------------------------------


class TestRadarProfile:
    """GET /api/v1/student/profile/{id}/radar — radar profile data."""

    @pytest.mark.asyncio
    async def test_student_own_profile(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/radar",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["data"]["student_id"] == FAKE_USER_ID

    @pytest.mark.asyncio
    async def test_student_other_profile_returns_403(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/student/profile/someone-else/radar",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_teacher_can_view_any_student(
        self, client: httpx.AsyncClient
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: TEACHER_USER

        async with client:
            resp = await client.get(
                "/api/v1/student/profile/any-student-id/radar",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True

    @pytest.mark.asyncio
    async def test_returns_real_profile_when_exists(
        self, client: httpx.AsyncClient
    ) -> None:
        profile = StudentProfile(
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            sessions_completed=3,
        )
        await deps._database.save_student_profile(profile)

        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/radar",
                headers=AUTH_HEADER,
            )
        body = resp.json()
        assert body["data"]["sessions_completed"] == 3

        # Cleanup
        await deps._database.delete_student_profile(FAKE_USER_ID, FAKE_SCHOOL_ID)

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/radar"
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /profile/{student_id}
# ---------------------------------------------------------------------------


class TestDeleteProfile:
    """DELETE /api/v1/student/profile/{id} — GDPR deletion."""

    @pytest.mark.asyncio
    async def test_student_deletes_own_profile(
        self, client: httpx.AsyncClient
    ) -> None:
        # Seed a profile
        profile = StudentProfile(
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            sessions_completed=5,
        )
        await deps._database.save_student_profile(profile)

        async with client:
            resp = await client.delete(
                f"/api/v1/student/profile/{FAKE_USER_ID}",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["data"]["deleted"] is True

        # Verify actually deleted
        result = await deps._database.get_student_profile(
            FAKE_USER_ID, FAKE_SCHOOL_ID
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_student_cannot_delete_other_profile(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.delete(
                "/api/v1/student/profile/someone-else",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_teacher_can_delete_student_profile(
        self, client: httpx.AsyncClient
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: TEACHER_USER

        # Seed a profile
        profile = StudentProfile(
            student_id="student-to-delete",
            school_id=FAKE_SCHOOL_ID,
        )
        await deps._database.save_student_profile(profile)

        async with client:
            resp = await client.delete(
                "/api/v1/student/profile/student-to-delete",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_profile_still_200(
        self, client: httpx.AsyncClient
    ) -> None:
        """Deletion is idempotent — deleting nothing is fine."""
        async with client:
            resp = await client.delete(
                f"/api/v1/student/profile/{FAKE_USER_ID}",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.delete(
                f"/api/v1/student/profile/{FAKE_USER_ID}"
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /profile/{student_id}/export
# ---------------------------------------------------------------------------


class TestExportProfile:
    """GET /api/v1/student/profile/{id}/export — GDPR data export."""

    @pytest.mark.asyncio
    async def test_export_own_data(self, client: httpx.AsyncClient) -> None:
        # Seed a profile
        profile = StudentProfile(
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            sessions_completed=7,
        )
        await deps._database.save_student_profile(profile)

        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/export",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "profile" in body["data"]

        # Cleanup
        await deps._database.delete_student_profile(FAKE_USER_ID, FAKE_SCHOOL_ID)

    @pytest.mark.asyncio
    async def test_export_empty_returns_empty_dict(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/export",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["data"] == {}

    @pytest.mark.asyncio
    async def test_student_cannot_export_other_data(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/student/profile/someone-else/export",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_teacher_can_export_student_data(
        self, client: httpx.AsyncClient
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: TEACHER_USER

        async with client:
            resp = await client.get(
                "/api/v1/student/profile/any-student/export",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/export"
            )
        assert resp.status_code == 401
