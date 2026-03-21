"""API-level journey tests — validates endpoint sequences the frontend calls.

Tests the *deterministic* path: session lifecycle, phase transitions via
/choice, recovery via /current, and response shape contracts. Uses static
cartridges with no AI, so tests are fast and reliable.

AI streaming journeys (respond, debrief, redaction) are covered by
test_student_ai.py. Generation flow cannot be journey-tested until
a cartridge with available_actions including "generate" exists.

Phase 7d — v-makaronas-student-experience-20260320.
"""

from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from backend.api import deps
from backend.api.deps import get_task_registry
from backend.main import app
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import TaskCartridge

AUTH_HEADER = {"Authorization": "Bearer test-token-123"}

# ---------------------------------------------------------------------------
# Response shape contracts — the fields the frontend reads
# ---------------------------------------------------------------------------

PHASE_RESPONSE_KEYS = {
    "task_id", "task_type", "medium", "title", "content",
    "available_actions", "trickster_intro", "current_phase",
    "is_terminal", "evaluation_outcome", "reveal", "interaction",
}

SESSION_KEYS = {"session_id", "language", "intro"}

CURRENT_EXTRA_KEYS = {"dialogue_history"}


# ---------------------------------------------------------------------------
# Cartridge fixtures
# ---------------------------------------------------------------------------


def _button_cartridge_data(task_id: str = "test-journey-buttons") -> dict:
    """Static task with 3 phases: intro → mid → reveal."""
    return {
        "task_id": task_id,
        "task_type": "static",
        "title": "Kelion\u0117s testas",
        "description": "Kelion\u0117s testo apra\u0161ymas",
        "version": "1.0",
        "trigger": "urgency",
        "technique": "headline_manipulation",
        "medium": "article",
        "learning_objectives": ["Atpa\u017Einti manipuliacij\u0105"],
        "difficulty": 3,
        "time_minutes": 10,
        "is_evergreen": True,
        "is_clean": False,
        "initial_phase": "phase_intro",
        "presentation_blocks": [
            {
                "id": "block-text",
                "type": "text",
                "text": "Antrašt\u0117: Mokslininkai patvirtino stebukling\u0105 atradim\u0105",
            },
            {
                "id": "block-post",
                "type": "social_post",
                "author": "user123",
                "text": "Negaliu patik\u0117ti!",
                "platform": "generic",
            },
            {
                "id": "block-search",
                "type": "search_result",
                "query": "stebuklingas atradimas",
                "title": "Tyrimo rezultatai",
                "snippet": "Joki\u0173 \u012Frodym\u0173 nerasta.",
            },
        ],
        "phases": [
            {
                "id": "phase_intro",
                "title": "\u012Evadas",
                "visible_blocks": ["block-text", "block-post"],
                "is_ai_phase": False,
                "interaction": {
                    "type": "button",
                    "choices": [
                        {"label": "Toliau", "target_phase": "phase_mid"},
                        {"label": "Pereiti prie atskleidimo", "target_phase": "phase_reveal"},
                    ],
                },
            },
            {
                "id": "phase_mid",
                "title": "Vidurin\u0117 dalis",
                "visible_blocks": ["block-text"],
                "is_ai_phase": False,
                "interaction": {
                    "type": "button",
                    "choices": [
                        {"label": "Atskleidimas", "target_phase": "phase_reveal"},
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
                "partial": "Mokinys praleido",
                "trickster_loses": "Mokinys atpa\u017Eino",
            },
        },
        "reveal": {"key_lesson": "Antra\u0161t\u0117 buvo sukurta skubos jausmui sukelti"},
        "safety": {
            "content_boundaries": ["no_real_harm"],
            "intensity_ceiling": 3,
            "cold_start_safe": True,
        },
    }


def _investigation_cartridge_data(
    task_id: str = "test-journey-investigation",
) -> dict:
    """Static task with investigation phase + reveal."""
    return {
        "task_id": task_id,
        "task_type": "static",
        "title": "Tyrimo testas",
        "description": "Tyrimo testo apra\u0161ymas",
        "version": "1.0",
        "trigger": "authority",
        "technique": "cherry_picking",
        "medium": "social_media",
        "learning_objectives": ["I\u0161tirti \u0161altinius"],
        "difficulty": 4,
        "time_minutes": 20,
        "is_evergreen": True,
        "is_clean": False,
        "initial_phase": "phase_search",
        "presentation_blocks": [
            {
                "id": "sr-main",
                "type": "search_result",
                "query": "pagrindinis klausimas",
                "title": "Pagrindinis rezultatas",
                "snippet": "\u012Edomus tyrimo rezultatas.",
                "is_key_finding": True,
            },
            {
                "id": "sr-secondary",
                "type": "search_result",
                "query": "antrinis klausimas",
                "title": "Antrinis rezultatas",
                "snippet": "Papildoma informacija.",
            },
        ],
        "phases": [
            {
                "id": "phase_search",
                "title": "Tyrimas",
                "visible_blocks": ["sr-main", "sr-secondary"],
                "is_ai_phase": False,
                "interaction": {
                    "type": "investigation",
                    "starting_queries": ["pagrindinis klausimas"],
                    "submit_target": "phase_reveal",
                    "min_key_findings": 1,
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
                    "description": "Cherry picking",
                    "technique": "cherry_picking",
                    "real_world_connection": "Common in debates",
                },
            ],
            "checklist": [
                {
                    "id": "c1",
                    "description": "Found key sources",
                    "pattern_refs": ["p1"],
                    "is_mandatory": True,
                },
            ],
            "pass_conditions": {
                "trickster_wins": "Mokinys ne\u012Fsigilino",
                "partial": "Mokinys rado dal\u012F",
                "trickster_loses": "Mokinys rado visus",
            },
        },
        "reveal": {"key_lesson": "\u0160altini\u0173 patikrinimas yra esminis"},
        "safety": {
            "content_boundaries": ["no_real_harm"],
            "intensity_ceiling": 3,
            "cold_start_safe": True,
        },
    }


def _build_cartridge(data: dict) -> TaskCartridge:
    """Validates cartridge data into a TaskCartridge model."""
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> httpx.AsyncClient:
    """Async test client wired to the app."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture(autouse=True)
def _cleanup_overrides():
    """Clears dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_sessions():
    """Clears session store after each test to prevent cross-test leakage."""
    yield
    deps._session_store._sessions.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_session(client: httpx.AsyncClient) -> str:
    """Creates a session and returns the session_id."""
    resp = await client.post(
        "/api/v1/student/session",
        json={},
        headers=AUTH_HEADER,
    )
    assert resp.status_code == 200, f"Session creation failed: {resp.text}"
    body = resp.json()
    assert body["ok"] is True
    return body["data"]["session_id"]


async def _load_task(
    client: httpx.AsyncClient, session_id: str, task_id: str
) -> dict:
    """Loads a task and returns the phase data."""
    resp = await client.get(
        f"/api/v1/student/session/{session_id}/next",
        params={"task_id": task_id},
        headers=AUTH_HEADER,
    )
    assert resp.status_code == 200, f"Load task failed: {resp.text}"
    body = resp.json()
    assert body["ok"] is True
    return body["data"]


async def _submit_choice(
    client: httpx.AsyncClient,
    session_id: str,
    target_phase: str,
    context_label: str | None = None,
) -> dict:
    """Submits a choice and returns the new phase data."""
    payload: dict = {"target_phase": target_phase}
    if context_label is not None:
        payload["context_label"] = context_label
    resp = await client.post(
        f"/api/v1/student/session/{session_id}/choice",
        json=payload,
        headers=AUTH_HEADER,
    )
    assert resp.status_code == 200, f"Choice failed: {resp.text}"
    body = resp.json()
    assert body["ok"] is True
    return body["data"]


def _assert_phase_shape(data: dict) -> None:
    """Asserts a phase response has the expected shape."""
    assert PHASE_RESPONSE_KEYS.issubset(
        data.keys()
    ), f"Missing keys: {PHASE_RESPONSE_KEYS - data.keys()}"


def _assert_content_blocks(data: dict) -> None:
    """Asserts content blocks have the expected structure."""
    assert isinstance(data["content"], list)
    for block in data["content"]:
        assert "type" in block, f"Block missing 'type': {block}"


# ---------------------------------------------------------------------------
# Journey 1: Button Flow (Happy Path)
# ---------------------------------------------------------------------------


class TestButtonJourney:
    """Full button journey: session → intro → mid → reveal."""

    @pytest.mark.asyncio
    async def test_full_button_flow(self, client: httpx.AsyncClient) -> None:
        """Walks the full button path: create → load → choose → terminal."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        # 1. Create session
        session_id = await _create_session(client)
        assert session_id

        # 2. Load task → phase_intro
        data = await _load_task(client, session_id, "test-journey-buttons")
        _assert_phase_shape(data)
        _assert_content_blocks(data)

        assert data["task_id"] == "test-journey-buttons"
        assert data["task_type"] == "static"
        assert data["medium"] == "article"
        assert data["current_phase"] == "phase_intro"
        assert data["is_terminal"] is False
        assert "button_click" in data["available_actions"]
        assert data["interaction"]["type"] == "button"
        assert len(data["interaction"]["choices"]) == 2
        # Verify choices have target_phase
        for choice in data["interaction"]["choices"]:
            assert "target_phase" in choice
            assert "label" in choice

        # 3. Choose → phase_mid
        data = await _submit_choice(
            client, session_id, "phase_mid", "Student chose Toliau"
        )
        _assert_phase_shape(data)
        assert data["current_phase"] == "phase_mid"
        assert data["is_terminal"] is False

        # 4. Choose → phase_reveal (terminal)
        data = await _submit_choice(client, session_id, "phase_reveal")
        _assert_phase_shape(data)
        assert data["current_phase"] == "phase_reveal"
        assert data["is_terminal"] is True
        assert data["evaluation_outcome"] is not None
        assert data["reveal"] is not None
        assert "key_lesson" in data["reveal"]

    @pytest.mark.asyncio
    async def test_direct_to_reveal(self, client: httpx.AsyncClient) -> None:
        """Tests skipping mid phase — intro → reveal directly."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)
        await _load_task(client, session_id, "test-journey-buttons")
        data = await _submit_choice(client, session_id, "phase_reveal")

        assert data["current_phase"] == "phase_reveal"
        assert data["is_terminal"] is True


# ---------------------------------------------------------------------------
# Journey 2: Task Sequence (Multi-Task Flow)
# ---------------------------------------------------------------------------


class TestTaskSequenceJourney:
    """Tests switching between tasks within a session."""

    @pytest.mark.asyncio
    async def test_task_switch(self, client: httpx.AsyncClient) -> None:
        """After completing task A, loading task B resets to B's initial phase."""
        cart_a = _build_cartridge(_button_cartridge_data("test-task-a"))
        cart_b = _build_cartridge(_button_cartridge_data("test-task-b"))
        _use_registry_with([cart_a, cart_b])

        session_id = await _create_session(client)

        # Complete task A
        await _load_task(client, session_id, "test-task-a")
        await _submit_choice(client, session_id, "phase_reveal")

        # Load task B
        data = await _load_task(client, session_id, "test-task-b")
        assert data["task_id"] == "test-task-b"
        assert data["current_phase"] == "phase_intro"

        # /current should also show task B
        resp = await client.get(
            f"/api/v1/student/session/{session_id}/current",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        current = resp.json()["data"]
        assert current["task_id"] == "test-task-b"


# ---------------------------------------------------------------------------
# Journey 3: Investigation Flow
# ---------------------------------------------------------------------------


class TestInvestigationJourney:
    """Investigation phase: load → submit via submit_target."""

    @pytest.mark.asyncio
    async def test_investigation_flow(self, client: httpx.AsyncClient) -> None:
        """Investigation phase with starting_queries and submit_target."""
        cart = _build_cartridge(_investigation_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)

        # Load investigation task
        data = await _load_task(client, session_id, "test-journey-investigation")
        _assert_phase_shape(data)

        assert data["current_phase"] == "phase_search"
        assert "investigate" in data["available_actions"]
        assert data["interaction"]["type"] == "investigation"
        assert isinstance(data["interaction"]["starting_queries"], list)
        assert len(data["interaction"]["starting_queries"]) > 0
        assert isinstance(data["interaction"]["submit_target"], str)
        assert isinstance(data["interaction"]["min_key_findings"], int)

        # Submit via submit_target
        submit_target = data["interaction"]["submit_target"]
        data = await _submit_choice(client, session_id, submit_target)
        assert data["current_phase"] == "phase_reveal"
        assert data["is_terminal"] is True


# ---------------------------------------------------------------------------
# Journey 4: Session Recovery
# ---------------------------------------------------------------------------


class TestSessionRecovery:
    """Tests GET /current for session recovery after page refresh."""

    @pytest.mark.asyncio
    async def test_recovery_after_choice(
        self, client: httpx.AsyncClient
    ) -> None:
        """After navigating to a non-terminal phase, /current returns it."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)
        await _load_task(client, session_id, "test-journey-buttons")
        await _submit_choice(client, session_id, "phase_mid")

        # Simulate page refresh — call /current
        resp = await client.get(
            f"/api/v1/student/session/{session_id}/current",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]

        _assert_phase_shape(data)
        assert data["current_phase"] == "phase_mid"
        assert "dialogue_history" in data
        assert isinstance(data["dialogue_history"], list)

    @pytest.mark.asyncio
    async def test_recovery_at_terminal(
        self, client: httpx.AsyncClient
    ) -> None:
        """Recovery at a terminal phase includes reveal data."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)
        await _load_task(client, session_id, "test-journey-buttons")
        await _submit_choice(client, session_id, "phase_reveal")

        resp = await client.get(
            f"/api/v1/student/session/{session_id}/current",
            headers=AUTH_HEADER,
        )
        data = resp.json()["data"]
        assert data["is_terminal"] is True
        assert data["reveal"] is not None


# ---------------------------------------------------------------------------
# Journey 5: Error Cases
# ---------------------------------------------------------------------------


class TestErrorJourneys:
    """Tests error responses match frontend expectations."""

    @pytest.mark.asyncio
    async def test_session_not_found(self, client: httpx.AsyncClient) -> None:
        """Nonexistent session returns 404 with SESSION_NOT_FOUND."""
        resp = await client.get(
            "/api/v1/student/session/nonexistent/current",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "SESSION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_invalid_phase_transition(
        self, client: httpx.AsyncClient
    ) -> None:
        """Invalid target_phase returns 422 with INVALID_PHASE_TRANSITION."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)
        await _load_task(client, session_id, "test-journey-buttons")

        resp = await client.post(
            f"/api/v1/student/session/{session_id}/choice",
            json={"target_phase": "nonexistent_phase"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "INVALID_PHASE_TRANSITION"

    @pytest.mark.asyncio
    async def test_no_task_assigned(self, client: httpx.AsyncClient) -> None:
        """Session without a task returns error on /next without task_id."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)

        resp = await client.get(
            f"/api/v1/student/session/{session_id}/next",
            headers=AUTH_HEADER,
        )
        # Should fail — no task_id provided and no current task
        assert resp.status_code in (400, 422)
        body = resp.json()
        assert body["ok"] is False

    @pytest.mark.asyncio
    async def test_no_auth_header(self, client: httpx.AsyncClient) -> None:
        """Missing auth header returns 401."""
        resp = await client.post("/api/v1/student/session")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Journey 6: Response Shape Contracts
# ---------------------------------------------------------------------------


class TestShapeContracts:
    """Verifies response shapes match what the frontend explicitly reads."""

    @pytest.mark.asyncio
    async def test_session_creation_shape(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /session returns session_id, language, intro."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        resp = await client.post(
            "/api/v1/student/session",
            json={},
            headers=AUTH_HEADER,
        )
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert SESSION_KEYS.issubset(data.keys()), (
            f"Missing session keys: {SESSION_KEYS - data.keys()}"
        )

    @pytest.mark.asyncio
    async def test_phase_response_shape(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /next returns all phase response keys."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)
        data = await _load_task(client, session_id, "test-journey-buttons")
        _assert_phase_shape(data)

    @pytest.mark.asyncio
    async def test_choice_response_shape(
        self, client: httpx.AsyncClient
    ) -> None:
        """POST /choice returns all phase response keys."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)
        await _load_task(client, session_id, "test-journey-buttons")
        data = await _submit_choice(client, session_id, "phase_mid")
        _assert_phase_shape(data)

    @pytest.mark.asyncio
    async def test_current_response_shape(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /current returns phase keys + dialogue_history."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)
        await _load_task(client, session_id, "test-journey-buttons")

        resp = await client.get(
            f"/api/v1/student/session/{session_id}/current",
            headers=AUTH_HEADER,
        )
        data = resp.json()["data"]
        expected = PHASE_RESPONSE_KEYS | CURRENT_EXTRA_KEYS
        assert expected.issubset(data.keys()), (
            f"Missing current keys: {expected - data.keys()}"
        )

    @pytest.mark.asyncio
    async def test_content_block_shapes(
        self, client: httpx.AsyncClient
    ) -> None:
        """Content blocks all have 'type' key — the renderer dispatches on it."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)
        data = await _load_task(client, session_id, "test-journey-buttons")

        assert isinstance(data["content"], list)
        assert len(data["content"]) > 0
        for block in data["content"]:
            assert isinstance(block, dict)
            assert "type" in block, f"Block missing 'type': {block}"
            assert "id" in block, f"Block missing 'id': {block}"

    @pytest.mark.asyncio
    async def test_terminal_phase_includes_reveal(
        self, client: httpx.AsyncClient
    ) -> None:
        """Terminal phase has reveal with key_lesson, non-terminal has None."""
        cart = _build_cartridge(_button_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)

        # Non-terminal: reveal is None
        data = await _load_task(client, session_id, "test-journey-buttons")
        assert data["reveal"] is None
        assert data["is_terminal"] is False

        # Terminal: reveal has key_lesson
        data = await _submit_choice(client, session_id, "phase_reveal")
        assert data["reveal"] is not None
        assert "key_lesson" in data["reveal"]
        assert data["is_terminal"] is True
        assert data["evaluation_outcome"] is not None

    @pytest.mark.asyncio
    async def test_investigation_interaction_shape(
        self, client: httpx.AsyncClient
    ) -> None:
        """Investigation interaction has starting_queries, submit_target, min_key_findings."""
        cart = _build_cartridge(_investigation_cartridge_data())
        _use_registry_with([cart])

        session_id = await _create_session(client)
        data = await _load_task(client, session_id, "test-journey-investigation")

        interaction = data["interaction"]
        assert interaction["type"] == "investigation"
        assert "starting_queries" in interaction
        assert isinstance(interaction["starting_queries"], list)
        assert "submit_target" in interaction
        assert isinstance(interaction["submit_target"], str)
        assert "min_key_findings" in interaction
        assert isinstance(interaction["min_key_findings"], int)

    @pytest.mark.asyncio
    async def test_error_response_shape(
        self, client: httpx.AsyncClient
    ) -> None:
        """Error responses have ok=False with error.code and error.message."""
        resp = await client.get(
            "/api/v1/student/session/nonexistent/current",
            headers=AUTH_HEADER,
        )
        body = resp.json()
        assert body["ok"] is False
        assert "error" in body
        assert "code" in body["error"]
        assert "message" in body["error"]
        assert isinstance(body["error"]["code"], str)
        assert isinstance(body["error"]["message"], str)
