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
        # interaction field present (button type for default cartridge)
        assert data["interaction"] is not None
        assert data["interaction"]["type"] == "button"

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
        data = resp.json()["data"]
        assert data["available_actions"] == ["button_click"]
        interaction = data["interaction"]
        assert interaction is not None
        assert interaction["type"] == "button"
        assert isinstance(interaction["choices"], list)
        assert len(interaction["choices"]) == 1
        assert interaction["choices"][0]["label"] == "T\u0119sti"
        assert interaction["choices"][0]["target_phase"] == "phase_reveal"

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
        data = resp.json()["data"]
        assert data["available_actions"] == ["freeform"]
        interaction = data["interaction"]
        assert interaction is not None
        assert interaction["type"] == "freeform"
        assert interaction["trickster_opening"] == "Na, k\u0105 manai?"

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
        data = resp.json()["data"]
        assert data["available_actions"] == ["investigate"]
        interaction = data["interaction"]
        assert interaction is not None
        assert interaction["type"] == "investigation"
        assert interaction["submit_target"] == "phase_reveal"

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
        data = resp.json()["data"]
        assert data["available_actions"] == []
        assert data["interaction"] is None

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

    # --- Terminal phase field tests (Phase 1a) ---

    @pytest.mark.asyncio
    async def test_next_includes_terminal_fields_non_terminal(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Non-terminal initial phase → is_terminal=False, evaluation_outcome=None, reveal=None."""
        cartridge = _build_cartridge("task-term-001")
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-term-001",
                headers=AUTH_HEADER,
            )
        data = resp.json()["data"]
        assert data["is_terminal"] is False
        assert data["evaluation_outcome"] is None
        assert data["reveal"] is None

    @pytest.mark.asyncio
    async def test_next_terminal_phase_includes_reveal(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Terminal initial phase → is_terminal=True with reveal data."""
        cartridge = _build_cartridge(
            "task-termrev-001",
            phases=[
                {
                    "id": "phase_intro",
                    "title": "Atskleidimas",
                    "is_terminal": True,
                    "evaluation_outcome": "trickster_wins",
                },
            ],
            reveal={
                "key_lesson": "Test lesson",
                "additional_resources": ["https://example.com"],
            },
        )
        _use_registry_with([cartridge])

        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next?task_id=task-termrev-001",
                headers=AUTH_HEADER,
            )
        data = resp.json()["data"]
        assert data["is_terminal"] is True
        assert data["evaluation_outcome"] == "trickster_wins"
        assert data["reveal"]["key_lesson"] == "Test lesson"
        assert data["reveal"]["additional_resources"] == ["https://example.com"]

    # --- Task-switch state reset tests (Phase 1c) ---

    @pytest.mark.asyncio
    async def test_task_switch_resets_per_task_state(
        self, client: httpx.AsyncClient
    ) -> None:
        """Switching to a different task resets per-task fields, preserves task_history."""
        cartridge_a = _build_cartridge("task-sw-001")
        cartridge_b = _build_cartridge("task-sw-002")
        _use_registry_with([cartridge_a, cartridge_b])

        # Create session and load first task
        session = GameSession(
            session_id="test-switch-session",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
        )
        await deps._session_store.save_session(session)

        async with client:
            await client.get(
                "/api/v1/student/session/test-switch-session/next?task_id=task-sw-001",
                headers=AUTH_HEADER,
            )

            # Populate per-task state to simulate mid-task activity
            session = await deps._session_store.get_session("test-switch-session")
            session.exchanges = [{"role": "student", "content": "test", "timestamp": "t1"}]
            session.choices = [{"target_phase": "p2", "context_label": "clicked"}]
            session.turn_intensities = [0.5, 0.7]
            session.generated_artifacts = [{"text": "artifact"}]
            session.prompt_snapshots = {"persona": "snapshot"}
            session.checklist_progress = {"c1": True}
            session.investigation_paths = ["/path/1"]
            session.raw_performance = {"score": 42}
            session.last_redaction_reason = "test reason"
            session.task_history = [{"task_id": "task-sw-001", "evaluation_outcome": "trickster_loses"}]
            await deps._session_store.save_session(session)

            # Switch to a different task
            resp = await client.get(
                "/api/v1/student/session/test-switch-session/next?task_id=task-sw-002",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

        # Verify per-task state was reset
        session = await deps._session_store.get_session("test-switch-session")
        assert session.exchanges == []
        assert session.choices == []
        assert session.turn_intensities == []
        assert session.generated_artifacts == []
        assert session.prompt_snapshots is None
        assert session.checklist_progress == {}
        assert session.investigation_paths == []
        assert session.raw_performance == {}
        assert session.last_redaction_reason is None
        # task_history persists across tasks
        assert len(session.task_history) == 1
        assert session.task_history[0]["task_id"] == "task-sw-001"

    @pytest.mark.asyncio
    async def test_same_task_reload_does_not_reset_state(
        self, client: httpx.AsyncClient
    ) -> None:
        """Reloading the same task preserves per-task state."""
        cartridge = _build_cartridge("task-reload-001")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-reload-session",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
        )
        await deps._session_store.save_session(session)

        async with client:
            # Load task first time
            await client.get(
                "/api/v1/student/session/test-reload-session/next?task_id=task-reload-001",
                headers=AUTH_HEADER,
            )

            # Populate per-task state
            session = await deps._session_store.get_session("test-reload-session")
            session.exchanges = [{"role": "student", "content": "test", "timestamp": "t1"}]
            session.turn_intensities = [0.5]
            await deps._session_store.save_session(session)

            # Reload same task
            resp = await client.get(
                "/api/v1/student/session/test-reload-session/next?task_id=task-reload-001",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

        # Per-task state preserved (not reset)
        session = await deps._session_store.get_session("test-reload-session")
        assert len(session.exchanges) == 1
        assert len(session.turn_intensities) == 1

    @pytest.mark.asyncio
    async def test_first_task_load_no_prior_task(
        self, client: httpx.AsyncClient
    ) -> None:
        """First task (current_task=None) works without triggering reset."""
        cartridge = _build_cartridge("task-first-001")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-first-session",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.get(
                "/api/v1/student/session/test-first-session/next?task_id=task-first-001",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["task_id"] == "task-first-001"


# ---------------------------------------------------------------------------
# GET /session/{session_id}/current
# ---------------------------------------------------------------------------


class TestCurrentSession:
    """GET /api/v1/student/session/{id}/current — read-only recovery endpoint."""

    @pytest.mark.asyncio
    async def test_returns_current_phase_content(
        self, client: httpx.AsyncClient
    ) -> None:
        """Session with active task returns full phase content."""
        cartridge = _build_cartridge("task-cur-001")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-cur",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-cur-001",
            current_phase="phase_intro",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.get(
                "/api/v1/student/session/test-session-cur/current",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert data["task_id"] == "task-cur-001"
        assert data["current_phase"] == "phase_intro"
        assert data["is_terminal"] is False
        assert data["evaluation_outcome"] is None
        assert data["reveal"] is None
        assert isinstance(data["content"], list)
        assert isinstance(data["available_actions"], list)
        assert isinstance(data["dialogue_history"], list)
        assert len(data["dialogue_history"]) == 0

    @pytest.mark.asyncio
    async def test_returns_dialogue_history(
        self, client: httpx.AsyncClient
    ) -> None:
        """Session with exchanges includes dialogue_history."""
        from backend.schemas import Exchange

        cartridge = _build_cartridge("task-hist-001")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-hist",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-hist-001",
            current_phase="phase_intro",
            exchanges=[
                Exchange(role="trickster", content="Sveiki!"),
                Exchange(role="student", content="Tai yra manipuliacija."),
            ],
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.get(
                "/api/v1/student/session/test-session-hist/current",
                headers=AUTH_HEADER,
            )
        data = resp.json()["data"]
        assert len(data["dialogue_history"]) == 2
        assert data["dialogue_history"][0]["role"] == "trickster"
        assert data["dialogue_history"][0]["content"] == "Sveiki!"
        assert data["dialogue_history"][1]["role"] == "student"
        assert data["dialogue_history"][1]["content"] == "Tai yra manipuliacija."
        # Timestamps should be present
        assert "timestamp" in data["dialogue_history"][0]

    @pytest.mark.asyncio
    async def test_no_current_task_returns_null(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Session with no current_task returns {current_task: null}."""
        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/current",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["current_task"] is None

    @pytest.mark.asyncio
    async def test_session_not_found_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/student/session/nonexistent/current",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_ownership_check_returns_403(
        self, client: httpx.AsyncClient, other_session_id: str
    ) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{other_session_id}/current",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_stale_phase_returns_409(
        self, client: httpx.AsyncClient
    ) -> None:
        """Stale phase after reload → TASK_CONTENT_UPDATED (409)."""
        cartridge = _build_cartridge("task-stale-cur-001")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-stale-cur",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-stale-cur-001",
            current_phase="phase_that_was_removed",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.get(
                "/api/v1/student/session/test-session-stale-cur/current",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"]["code"] == "TASK_CONTENT_UPDATED"
        assert body["data"]["initial_phase"] == "phase_intro"

    @pytest.mark.asyncio
    async def test_read_only_no_mutation(
        self, client: httpx.AsyncClient
    ) -> None:
        """Calling /current twice returns identical results — no state mutation."""
        cartridge = _build_cartridge("task-ro-001")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-ro",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-ro-001",
            current_phase="phase_intro",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp1 = await client.get(
                "/api/v1/student/session/test-session-ro/current",
                headers=AUTH_HEADER,
            )
            resp2 = await client.get(
                "/api/v1/student/session/test-session-ro/current",
                headers=AUTH_HEADER,
            )
        assert resp1.json() == resp2.json()

        # Verify session wasn't mutated
        stored = await deps._session_store.get_session("test-session-ro")
        assert stored.current_task == "task-ro-001"
        assert stored.current_phase == "phase_intro"

    @pytest.mark.asyncio
    async def test_terminal_phase_includes_reveal(
        self, client: httpx.AsyncClient
    ) -> None:
        """Terminal phase in /current returns reveal data."""
        cartridge = _build_cartridge(
            "task-cur-term-001",
            phases=[
                {
                    "id": "phase_intro",
                    "title": "Atskleidimas",
                    "is_terminal": True,
                    "evaluation_outcome": "trickster_loses",
                },
            ],
        )
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-cur-term",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-cur-term-001",
            current_phase="phase_intro",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.get(
                "/api/v1/student/session/test-session-cur-term/current",
                headers=AUTH_HEADER,
            )
        data = resp.json()["data"]
        assert data["is_terminal"] is True
        assert data["evaluation_outcome"] == "trickster_loses"
        assert data["reveal"]["key_lesson"] is not None
        assert isinstance(data["reveal"]["additional_resources"], list)

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/student/session/any-id/current")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /session/{session_id}/choice
# ---------------------------------------------------------------------------


class TestChoice:
    """POST /api/v1/student/session/{id}/choice — phase transition endpoint."""

    @pytest.mark.asyncio
    async def test_happy_path_button_choice(
        self, client: httpx.AsyncClient
    ) -> None:
        """Valid button choice transitions to new phase and returns content."""
        cartridge = _build_cartridge("task-choice-001")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-choice",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-choice-001",
            current_phase="phase_intro",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/test-session-choice/choice",
                headers=AUTH_HEADER,
                json={"target_phase": "phase_reveal"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert data["current_phase"] == "phase_reveal"
        assert data["task_id"] == "task-choice-001"
        assert data["is_terminal"] is True
        assert data["evaluation_outcome"] == "trickster_loses"
        assert data["reveal"] is not None

        # Verify session was updated
        stored = await deps._session_store.get_session("test-session-choice")
        assert stored.current_phase == "phase_reveal"

    @pytest.mark.asyncio
    async def test_context_label_recorded(
        self, client: httpx.AsyncClient
    ) -> None:
        """Choice with context_label records it in session.choices."""
        cartridge = _build_cartridge("task-choice-label")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-label",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-choice-label",
            current_phase="phase_intro",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/test-session-label/choice",
                headers=AUTH_HEADER,
                json={
                    "target_phase": "phase_reveal",
                    "context_label": "Mokinys pasirinko t\u0119sti",
                },
            )
        assert resp.status_code == 200

        stored = await deps._session_store.get_session("test-session-label")
        assert len(stored.choices) == 1
        choice = stored.choices[0]
        assert choice["phase"] == "phase_intro"
        assert choice["target_phase"] == "phase_reveal"
        assert choice["context_label"] == "Mokinys pasirinko t\u0119sti"
        assert "timestamp" in choice

    @pytest.mark.asyncio
    async def test_null_context_label_still_records_choice(
        self, client: httpx.AsyncClient
    ) -> None:
        """Choice with null context_label still records the transition."""
        cartridge = _build_cartridge("task-choice-null")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-null-label",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-choice-null",
            current_phase="phase_intro",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/test-session-null-label/choice",
                headers=AUTH_HEADER,
                json={"target_phase": "phase_reveal"},
            )
        assert resp.status_code == 200

        stored = await deps._session_store.get_session("test-session-null-label")
        assert len(stored.choices) == 1
        assert stored.choices[0]["context_label"] is None

    @pytest.mark.asyncio
    async def test_invalid_transition_wrong_target(
        self, client: httpx.AsyncClient
    ) -> None:
        """Target phase exists in cartridge but is not a legal edge → 422."""
        # Build cartridge with intro → reveal only, and another non-connected phase
        cartridge = _build_cartridge(
            "task-choice-inv",
            phases=[
                {
                    "id": "phase_intro",
                    "title": "\u012evadas",
                    "visible_blocks": ["block-headline"],
                    "is_ai_phase": False,
                    "interaction": {
                        "type": "button",
                        "choices": [
                            {"label": "T\u0119sti", "target_phase": "phase_reveal"},
                        ],
                    },
                },
                {
                    "id": "phase_other",
                    "title": "Kitas",
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

        session = GameSession(
            session_id="test-session-inv",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-choice-inv",
            current_phase="phase_intro",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/test-session-inv/choice",
                headers=AUTH_HEADER,
                json={"target_phase": "phase_other"},
            )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "INVALID_PHASE_TRANSITION"

    @pytest.mark.asyncio
    async def test_invalid_transition_freeform_phase(
        self, client: httpx.AsyncClient
    ) -> None:
        """Freeform phase has no choice targets → any /choice returns 422."""
        cartridge = _build_cartridge(
            "task-choice-freeform",
            phases=[
                {
                    "id": "phase_freeform",
                    "title": "Dialogas",
                    "is_ai_phase": True,
                    "interaction": {
                        "type": "freeform",
                        "trickster_opening": "Sveiki!",
                        "min_exchanges": 1,
                        "max_exchanges": 3,
                    },
                },
                {
                    "id": "phase_reveal",
                    "title": "Atskleidimas",
                    "is_terminal": True,
                    "evaluation_outcome": "trickster_loses",
                },
            ],
            initial_phase="phase_freeform",
        )
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-freeform",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-choice-freeform",
            current_phase="phase_freeform",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/test-session-freeform/choice",
                headers=AUTH_HEADER,
                json={"target_phase": "phase_reveal"},
            )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "INVALID_PHASE_TRANSITION"

    @pytest.mark.asyncio
    async def test_invalid_transition_nonexistent_phase(
        self, client: httpx.AsyncClient
    ) -> None:
        """Target phase doesn't exist anywhere → 422 INVALID_PHASE_TRANSITION."""
        cartridge = _build_cartridge("task-choice-noexist")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-noexist",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-choice-noexist",
            current_phase="phase_intro",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/test-session-noexist/choice",
                headers=AUTH_HEADER,
                json={"target_phase": "does_not_exist"},
            )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "INVALID_PHASE_TRANSITION"

    @pytest.mark.asyncio
    async def test_terminal_phase_content(
        self, client: httpx.AsyncClient
    ) -> None:
        """Transitioning to a terminal phase returns terminal fields."""
        cartridge = _build_cartridge("task-choice-term")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-term",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-choice-term",
            current_phase="phase_intro",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/test-session-term/choice",
                headers=AUTH_HEADER,
                json={"target_phase": "phase_reveal"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["is_terminal"] is True
        assert data["evaluation_outcome"] == "trickster_loses"
        assert data["reveal"] is not None
        assert data["reveal"]["key_lesson"] is not None

    @pytest.mark.asyncio
    async def test_session_not_found(self, client: httpx.AsyncClient) -> None:
        """Nonexistent session → 404."""
        async with client:
            resp = await client.post(
                "/api/v1/student/session/nonexistent/choice",
                headers=AUTH_HEADER,
                json={"target_phase": "anywhere"},
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_ownership_check(
        self, client: httpx.AsyncClient, other_session_id: str
    ) -> None:
        """Session owned by another user → 403."""
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{other_session_id}/choice",
                headers=AUTH_HEADER,
                json={"target_phase": "anywhere"},
            )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_no_active_task(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Session with no current_task → 422."""
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{session_id}/choice",
                headers=AUTH_HEADER,
                json={"target_phase": "anywhere"},
            )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "NO_TASK_ASSIGNED"

    @pytest.mark.asyncio
    async def test_no_active_phase(self, client: httpx.AsyncClient) -> None:
        """Session with task but no current_phase → 422."""
        cartridge = _build_cartridge("task-choice-nophase")
        _use_registry_with([cartridge])

        session = GameSession(
            session_id="test-session-nophase",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task="task-choice-nophase",
            current_phase=None,
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/test-session-nophase/choice",
                headers=AUTH_HEADER,
                json={"target_phase": "phase_intro"},
            )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "NO_ACTIVE_PHASE"


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
# Done event enrichment tests (Phase 1c)
# ---------------------------------------------------------------------------


class _TransitionStubEngine:
    """Stub engine that simulates a phase transition in done_data.

    Used to test done event enrichment — sets next_phase to a target
    phase ID so the SSE generator can enrich the done event.
    """

    def __init__(self, next_phase: str, transition: str = "trickster_loses"):
        self._next_phase = next_phase
        self._transition = transition

    async def respond(self, session, cartridge, phase, student_input):
        """Returns a TricksterResult with a phase transition."""
        async def _tokens():
            for t in ["Transition ", "response. "]:
                yield t
            result.done_data = {
                "phase_transition": self._transition,
                "next_phase": self._next_phase,
                "exchanges_count": 2,
            }
        result = TricksterResult(token_iterator=_tokens())
        return result

    async def debrief(self, session, cartridge):
        """Returns a DebriefResult (no transition)."""
        async def _tokens():
            for t in ["Mock ", "debrief. "]:
                yield t
            result.done_data = {"debrief_complete": True}
        result = DebriefResult(token_iterator=_tokens())
        return result


class _InvalidPhaseStubEngine:
    """Stub engine that returns a next_phase not in the cartridge."""

    async def respond(self, session, cartridge, phase, student_input):
        async def _tokens():
            for t in ["Bad ", "phase. "]:
                yield t
            result.done_data = {
                "phase_transition": "trickster_loses",
                "next_phase": "nonexistent_phase",
                "exchanges_count": 1,
            }
        result = TricksterResult(token_iterator=_tokens())
        return result


class TestDoneEventEnrichment:
    """Tests for SSE done event enrichment with next_phase_content (Phase 1c)."""

    @pytest.fixture(autouse=True)
    def _inject_ai_deps(self):
        """Injects AI deps so endpoint Depends() resolve without 503."""
        # Default engine — overridden per test as needed
        app.dependency_overrides[get_trickster_engine] = lambda: _StubEngine()
        if get_task_registry not in app.dependency_overrides:
            app.dependency_overrides[get_task_registry] = lambda: TaskRegistry(
                Path("/tmp"), Path("/tmp")
            )

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_done_event_includes_next_phase_content_on_transition(
        self, _mock_readiness, client: httpx.AsyncClient
    ) -> None:
        """When a phase transition occurs, done event includes next_phase_content."""
        task_id = "task-enrich-001"
        cartridge = TaskCartridge.model_validate(_ai_cartridge_data(task_id))

        registry = TaskRegistry(Path("/tmp"), Path("/tmp"))
        registry._by_id[task_id] = cartridge
        registry._by_status.setdefault(cartridge.status, set()).add(task_id)
        registry._by_trigger[cartridge.trigger].add(task_id)
        registry._by_technique[cartridge.technique].add(task_id)
        registry._by_medium[cartridge.medium].add(task_id)
        app.dependency_overrides[get_task_registry] = lambda: registry

        # Engine that transitions to phase_reveal (terminal phase)
        app.dependency_overrides[get_trickster_engine] = lambda: _TransitionStubEngine(
            next_phase="phase_reveal"
        )

        session = GameSession(
            session_id="test-enrich-session",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task=task_id,
            current_phase="phase_ai",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/test-enrich-session/respond",
                json={"action": "freeform", "payload": "I see the trick"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        done_data = done_events[0]["data"]["data"]
        assert "next_phase_content" in done_data
        npc = done_data["next_phase_content"]
        assert npc["task_id"] == task_id
        assert npc["current_phase"] == "phase_reveal"
        assert npc["is_terminal"] is True
        assert npc["evaluation_outcome"] == "trickster_loses"
        assert npc["reveal"] is not None
        assert npc["reveal"]["key_lesson"] == "Test lesson"

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_done_event_no_enrichment_without_transition(
        self, _mock_readiness, client: httpx.AsyncClient
    ) -> None:
        """When no phase transition occurs, done event has no next_phase_content."""
        session_id = await _setup_ai_session()

        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{session_id}/respond",
                json={"action": "freeform", "payload": "test"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        done_data = done_events[0]["data"]["data"]
        assert "next_phase_content" not in done_data

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_done_event_degrades_gracefully_on_invalid_phase(
        self, _mock_readiness, client: httpx.AsyncClient
    ) -> None:
        """When next_phase references a nonexistent phase, done event emits without crash."""
        task_id = "task-badphase-001"
        cartridge = TaskCartridge.model_validate(_ai_cartridge_data(task_id))

        registry = TaskRegistry(Path("/tmp"), Path("/tmp"))
        registry._by_id[task_id] = cartridge
        registry._by_status.setdefault(cartridge.status, set()).add(task_id)
        registry._by_trigger[cartridge.trigger].add(task_id)
        registry._by_technique[cartridge.technique].add(task_id)
        registry._by_medium[cartridge.medium].add(task_id)
        app.dependency_overrides[get_task_registry] = lambda: registry

        app.dependency_overrides[get_trickster_engine] = lambda: _InvalidPhaseStubEngine()

        session = GameSession(
            session_id="test-badphase-session",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            current_task=task_id,
            current_phase="phase_ai",
        )
        await deps._session_store.save_session(session)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/test-badphase-session/respond",
                json={"action": "freeform", "payload": "test"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        done_data = done_events[0]["data"]["data"]
        # next_phase is set but no next_phase_content (graceful degradation)
        assert done_data["next_phase"] == "nonexistent_phase"
        assert "next_phase_content" not in done_data


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
