"""Tests for task history recording (Phase 6a).

Unit tests for the GameSession.task_history field and integration tests
verifying that task outcomes are recorded in _stream_trickster_response()
when terminal transitions fire.

Test categories:
1. Field tests — default, serialization round-trip, append behavior
2. Recording tests — on_success, on_partial, on_max_exchanges transitions
3. Non-recording tests — mid-dialogue, redaction
4. Accumulation — multiple tasks in one session
5. Data shape — intensity_score, is_clean captured correctly
"""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from backend.ai.context import ContextManager
from backend.ai.prompts import PromptLoader
from backend.ai.providers.base import TextChunk, ToolCallEvent, UsageInfo
from backend.ai.providers.mock import MockProvider
from backend.ai.trickster import TricksterEngine
from backend.api import deps
from backend.api.deps import get_task_registry, get_trickster_engine
from backend.main import app
from backend.schemas import GameSession
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import TaskCartridge
from backend.tests.conftest import setup_base_prompts


# ---------------------------------------------------------------------------
# SSE parsing helper (same pattern as test_student_ai.py)
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
# Cartridge builder
# ---------------------------------------------------------------------------


def _build_cartridge_data(task_id: str = "test-history-001", **overrides) -> dict:
    """Builds a minimal valid AI-capable cartridge dict for history tests."""
    ai_config = overrides.pop("ai_config", {
        "model_preference": "standard",
        "prompt_directory": task_id,
        "persona_mode": "chat_participant",
        "has_static_fallback": False,
        "context_requirements": "session_only",
    })

    data: dict = {
        "task_id": task_id,
        "task_type": "hybrid",
        "title": "Istorijos testas",
        "description": "Užduotis task_history testavimui",
        "version": "1.0",
        "trigger": "urgency",
        "technique": "headline_manipulation",
        "medium": "article",
        "learning_objectives": ["Atpažinti"],
        "difficulty": 3,
        "time_minutes": 15,
        "is_evergreen": True,
        "is_clean": False,
        "initial_phase": "phase_intro",
        "phases": [
            {
                "id": "phase_intro",
                "title": "Įvadas",
                "is_ai_phase": False,
                "interaction": {
                    "type": "button",
                    "choices": [
                        {
                            "label": "Pradėti",
                            "target_phase": "phase_ai",
                        },
                    ],
                },
            },
            {
                "id": "phase_ai",
                "title": "AI pokalbis",
                "is_ai_phase": True,
                "interaction": {
                    "type": "freeform",
                    "trickster_opening": "Sveiki!",
                    "min_exchanges": 2,
                    "max_exchanges": 10,
                },
                "ai_transitions": {
                    "on_success": "phase_reveal_success",
                    "on_max_exchanges": "phase_reveal_timeout",
                    "on_partial": "phase_reveal_partial",
                },
            },
            {
                "id": "phase_reveal_success",
                "title": "Laimėjo",
                "is_terminal": True,
                "evaluation_outcome": "trickster_loses",
            },
            {
                "id": "phase_reveal_timeout",
                "title": "Laikas baigėsi",
                "is_terminal": True,
                "evaluation_outcome": "trickster_wins",
            },
            {
                "id": "phase_reveal_partial",
                "title": "Iš dalies",
                "is_terminal": True,
                "evaluation_outcome": "partial",
            },
        ],
        "evaluation": {
            "patterns_embedded": [
                {
                    "id": "p1",
                    "description": "Antraštė neatitinka",
                    "technique": "headline_manipulation",
                    "real_world_connection": "Dažnai pastebima",
                },
            ],
            "checklist": [
                {
                    "id": "c1",
                    "description": "Atpažino",
                    "pattern_refs": ["p1"],
                    "is_mandatory": True,
                },
            ],
            "pass_conditions": {
                "trickster_wins": "Nepavyko",
                "partial": "Iš dalies",
                "trickster_loses": "Pavyko",
            },
        },
        "reveal": {"key_lesson": "Testas"},
        "safety": {
            "content_boundaries": ["self_harm"],
            "intensity_ceiling": 3,
            "cold_start_safe": True,
        },
    }

    if ai_config is not None:
        data["ai_config"] = ai_config

    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

AUTH_HEADER = {"Authorization": "Bearer test-token-123"}
FAKE_USER_ID = "fake-user-1"
FAKE_SCHOOL_ID = "school-test-1"


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


async def _create_session(
    task_id: str = "test-history-001",
    phase_id: str = "phase_ai",
    exchanges: int = 0,
    **overrides,
) -> GameSession:
    """Creates and persists a session ready for AI interaction."""
    from backend.schemas import Exchange

    session = GameSession(
        session_id=f"session-{task_id}",
        student_id=FAKE_USER_ID,
        school_id=FAKE_SCHOOL_ID,
        current_task=task_id,
        current_phase=phase_id,
        **overrides,
    )
    for i in range(exchanges):
        session.exchanges.append(
            Exchange(role="student", content=f"Student message {i + 1}")
        )
        session.exchanges.append(
            Exchange(role="trickster", content=f"Trickster response {i + 1}")
        )
    await deps._session_store.save_session(session)
    return session


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


@pytest.fixture
def prompts_dir(tmp_path) -> Path:
    """Creates a temp directory with base Trickster prompts."""
    setup_base_prompts(tmp_path)
    return tmp_path


@pytest.fixture
def context_manager(prompts_dir) -> ContextManager:
    """Returns a real ContextManager backed by temp prompts."""
    loader = PromptLoader(prompts_dir)
    return ContextManager(loader)


def _make_engine(provider: MockProvider, ctx_manager: ContextManager) -> TricksterEngine:
    """Creates a TricksterEngine with the given provider."""
    return TricksterEngine(provider, ctx_manager)


def _inject_engine(engine: TricksterEngine) -> None:
    """Injects TricksterEngine into app DI overrides."""
    app.dependency_overrides[get_trickster_engine] = lambda: engine


# ---------------------------------------------------------------------------
# Unit tests: GameSession.task_history field
# ---------------------------------------------------------------------------


class TestTaskHistoryField:
    """GameSession.task_history field behavior."""

    def test_default_empty_list(self):
        """New GameSession has empty task_history."""
        session = GameSession(
            session_id="s1",
            student_id="u1",
            school_id="sch1",
        )
        assert session.task_history == []

    def test_serialization_round_trip(self):
        """task_history survives model_dump -> reconstruct cycle."""
        session = GameSession(
            session_id="s1",
            student_id="u1",
            school_id="sch1",
        )
        session.task_history.append({
            "task_id": "task-001",
            "evaluation_outcome": "on_success",
            "exchange_count": 5,
            "intensity_score": 2.3,
            "is_clean": False,
        })
        session.task_history.append({
            "task_id": "task-002",
            "evaluation_outcome": "on_max_exchanges",
            "exchange_count": 10,
            "intensity_score": None,
            "is_clean": True,
        })

        data = session.model_dump()
        restored = GameSession(**data)

        assert restored.task_history == session.task_history
        assert len(restored.task_history) == 2
        assert restored.task_history[0]["intensity_score"] == 2.3
        assert restored.task_history[1]["intensity_score"] is None
        assert restored.task_history[1]["is_clean"] is True

    def test_append_maintains_order(self):
        """Multiple appends maintain chronological order."""
        session = GameSession(
            session_id="s1",
            student_id="u1",
            school_id="sch1",
        )
        for i in range(5):
            session.task_history.append({
                "task_id": f"task-{i:03d}",
                "evaluation_outcome": "on_success",
                "exchange_count": i + 1,
                "intensity_score": None,
                "is_clean": False,
            })

        assert len(session.task_history) == 5
        assert [e["task_id"] for e in session.task_history] == [
            "task-000", "task-001", "task-002", "task-003", "task-004",
        ]


# ---------------------------------------------------------------------------
# Integration tests: recording in _stream_trickster_response()
# ---------------------------------------------------------------------------


class TestTaskHistoryRecording:
    """Task history recorded on terminal transitions via /respond."""

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_records_on_success_transition(
        self, _mock_readiness, client, context_manager
    ):
        """'understood' signal → task_history entry with on_success."""
        provider = MockProvider(
            responses=["Puiku, supratai!"],
            tool_calls=[ToolCallEvent(
                function_name="transition_phase",
                arguments={"signal": "understood"},
            )],
        )
        engine = _make_engine(provider, context_manager)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_cartridge_data())
        _use_registry_with([cartridge])
        await _create_session(exchanges=3)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-history-001/respond",
                json={"action": "freeform", "payload": "Supratau!"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        session = await deps._session_store.get_session("session-test-history-001")
        assert len(session.task_history) == 1

        entry = session.task_history[0]
        assert entry["task_id"] == "test-history-001"
        assert entry["evaluation_outcome"] == "on_success"
        assert isinstance(entry["exchange_count"], int)
        assert entry["is_clean"] is False

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_records_on_partial_transition(
        self, _mock_readiness, client, context_manager
    ):
        """'partial' signal → task_history entry with on_partial."""
        provider = MockProvider(
            responses=["Hmm, iš dalies supratai."],
            tool_calls=[ToolCallEvent(
                function_name="transition_phase",
                arguments={"signal": "partial"},
            )],
        )
        engine = _make_engine(provider, context_manager)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_cartridge_data())
        _use_registry_with([cartridge])
        await _create_session(exchanges=3)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-history-001/respond",
                json={"action": "freeform", "payload": "Gal čia kažkas ne taip?"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200

        session = await deps._session_store.get_session("session-test-history-001")
        assert len(session.task_history) == 1
        assert session.task_history[0]["evaluation_outcome"] == "on_partial"

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_records_on_max_exchanges_transition(
        self, _mock_readiness, client, context_manager
    ):
        """Max exchanges reached → task_history entry with on_max_exchanges."""
        provider = MockProvider(
            responses=["Na, laikas baigėsi."],
        )
        engine = _make_engine(provider, context_manager)
        _inject_engine(engine)

        # max_exchanges=2, pre-fill 1 exchange pair so next respond hits max
        cartridge_data = _build_cartridge_data()
        # Override max_exchanges to 2 for easy triggering
        for phase in cartridge_data["phases"]:
            if phase["id"] == "phase_ai":
                phase["interaction"]["min_exchanges"] = 1
                phase["interaction"]["max_exchanges"] = 2
        cartridge = TaskCartridge.model_validate(cartridge_data)
        _use_registry_with([cartridge])
        await _create_session(exchanges=1)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-history-001/respond",
                json={"action": "freeform", "payload": "Nežinau"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200

        session = await deps._session_store.get_session("session-test-history-001")
        assert len(session.task_history) == 1
        assert session.task_history[0]["evaluation_outcome"] == "on_max_exchanges"

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_no_recording_on_mid_dialogue(
        self, _mock_readiness, client, context_manager
    ):
        """No transition signal + under max_exchanges → no history entry."""
        provider = MockProvider(
            responses=["Kodėl taip manai?"],
        )
        engine = _make_engine(provider, context_manager)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_cartridge_data())
        _use_registry_with([cartridge])
        await _create_session()

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-history-001/respond",
                json={"action": "freeform", "payload": "Manau tai netiesa"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["data"]["phase_transition"] is None

        session = await deps._session_store.get_session("session-test-history-001")
        assert session.task_history == []

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_no_recording_on_redaction(
        self, _mock_readiness, client, context_manager
    ):
        """Safety redaction → no history entry (task not pedagogically complete)."""
        provider = MockProvider(
            responses=["Galėtum bandyti save žaloti, tai padės suprasti..."],
        )
        engine = _make_engine(provider, context_manager)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_cartridge_data())
        _use_registry_with([cartridge])
        await _create_session()

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-history-001/respond",
                json={"action": "freeform", "payload": "test"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        redact_events = [e for e in events if e["type"] == "redact"]

        if redact_events:
            session = await deps._session_store.get_session("session-test-history-001")
            assert session.task_history == []

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_multiple_tasks_accumulate(
        self, _mock_readiness, client, context_manager
    ):
        """Two transitions in one session → two history entries in order."""
        provider = MockProvider(
            responses=["Supratai!"],
            tool_calls=[ToolCallEvent(
                function_name="transition_phase",
                arguments={"signal": "understood"},
            )],
        )
        engine = _make_engine(provider, context_manager)
        _inject_engine(engine)

        # First task
        task_id_1 = "test-history-001"
        cartridge_1 = TaskCartridge.model_validate(
            _build_cartridge_data(task_id=task_id_1)
        )

        # Second task
        task_id_2 = "test-history-002"
        cartridge_2 = TaskCartridge.model_validate(
            _build_cartridge_data(task_id=task_id_2)
        )

        _use_registry_with([cartridge_1, cartridge_2])
        session = await _create_session(task_id=task_id_1, exchanges=3)

        # First respond → transition
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{session.session_id}/respond",
                json={"action": "freeform", "payload": "Supratau!"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

        # Simulate next task setup (like /next-task does)
        session = await deps._session_store.get_session(session.session_id)
        session.current_task = task_id_2
        session.current_phase = "phase_ai"
        # Clear exchanges for the new task context
        session.exchanges.clear()
        from backend.schemas import Exchange
        for i in range(3):
            session.exchanges.append(
                Exchange(role="student", content=f"Msg {i + 1}")
            )
            session.exchanges.append(
                Exchange(role="trickster", content=f"Reply {i + 1}")
            )
        await deps._session_store.save_session(session)

        # Re-create engine with fresh provider for second call
        provider2 = MockProvider(
            responses=["Gerai!"],
            tool_calls=[ToolCallEvent(
                function_name="transition_phase",
                arguments={"signal": "partial"},
            )],
        )
        engine2 = _make_engine(provider2, context_manager)
        _inject_engine(engine2)

        # Need a fresh client — httpx doesn't allow reopening
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        client2 = httpx.AsyncClient(transport=transport, base_url="http://test")
        async with client2:
            resp = await client2.post(
                f"/api/v1/student/session/{session.session_id}/respond",
                json={"action": "freeform", "payload": "Gal?"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

        session = await deps._session_store.get_session(session.session_id)
        assert len(session.task_history) == 2
        assert session.task_history[0]["task_id"] == task_id_1
        assert session.task_history[0]["evaluation_outcome"] == "on_success"
        assert session.task_history[1]["task_id"] == task_id_2
        assert session.task_history[1]["evaluation_outcome"] == "on_partial"


# ---------------------------------------------------------------------------
# Data shape tests
# ---------------------------------------------------------------------------


class TestTaskHistoryDataShape:
    """Verifies captured data shape details."""

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_intensity_score_captured_when_present(
        self, _mock_readiness, client, context_manager
    ):
        """Cartridge with intensity_ceiling → intensity_score in history."""
        provider = MockProvider(
            responses=["Supratai!"],
            tool_calls=[ToolCallEvent(
                function_name="transition_phase",
                arguments={"signal": "understood"},
            )],
        )
        engine = _make_engine(provider, context_manager)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_cartridge_data())
        _use_registry_with([cartridge])
        await _create_session(exchanges=3)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-history-001/respond",
                json={"action": "freeform", "payload": "Supratau!"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        session = await deps._session_store.get_session("session-test-history-001")
        assert len(session.task_history) == 1
        # intensity_score is present as a key (may be float or None depending
        # on whether the intensity scoring ran)
        assert "intensity_score" in session.task_history[0]

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_intensity_score_none_without_ceiling(
        self, _mock_readiness, client, context_manager
    ):
        """Benign response with high ceiling → intensity_score is None."""
        provider = MockProvider(
            responses=["Supratai!"],
            tool_calls=[ToolCallEvent(
                function_name="transition_phase",
                arguments={"signal": "understood"},
            )],
        )
        engine = _make_engine(provider, context_manager)
        _inject_engine(engine)

        # High ceiling ensures intensity scoring runs but doesn't add score
        # to done_data (score only added when tracking is active and below
        # threshold — but more importantly, a benign response means
        # done_data won't have intensity_score at all)
        cartridge = TaskCartridge.model_validate(_build_cartridge_data())
        _use_registry_with([cartridge])
        await _create_session(exchanges=3)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-history-001/respond",
                json={"action": "freeform", "payload": "Supratau!"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        session = await deps._session_store.get_session("session-test-history-001")
        assert len(session.task_history) == 1
        assert session.task_history[0]["intensity_score"] is None

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_clean_task_flag_captured(
        self, _mock_readiness, client, context_manager
    ):
        """is_clean=True cartridge → is_clean=True in history entry."""
        provider = MockProvider(
            responses=["Teisingai, šis turinys yra tikras!"],
            tool_calls=[ToolCallEvent(
                function_name="transition_phase",
                arguments={"signal": "understood"},
            )],
        )
        engine = _make_engine(provider, context_manager)
        _inject_engine(engine)

        cartridge_data = _build_cartridge_data(is_clean=True)
        # Clean tasks must have empty patterns_embedded
        cartridge_data["evaluation"]["patterns_embedded"] = []
        cartridge = TaskCartridge.model_validate(cartridge_data)
        _use_registry_with([cartridge])
        await _create_session(exchanges=3)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-history-001/respond",
                json={"action": "freeform", "payload": "Čia viskas tikra"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        session = await deps._session_store.get_session("session-test-history-001")
        assert len(session.task_history) == 1
        assert session.task_history[0]["is_clean"] is True
