"""API-level integration tests for intensity tracking flow (Phase 4c).

Exercises the full chain: HTTP POST /respond -> TricksterEngine ->
MockProvider -> SSE response, verifying intensity scoring, de-escalation
flag in done_data, and de-escalation context injection on subsequent calls.

Constraint #15: API-level integration test for intensity tracking.
"""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from backend.ai.context import ContextManager
from backend.ai.prompts import PromptLoader
from backend.ai.providers.base import UsageInfo
from backend.ai.providers.mock import MockProvider
from backend.ai.trickster import TricksterEngine
from backend.api import deps
from backend.api.deps import (
    get_task_registry,
    get_trickster_engine,
)
from backend.main import app
from backend.schemas import Exchange, GameSession
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import TaskCartridge
from backend.tests.conftest import setup_base_prompts
from backend.tests.test_intensity_engine import _make_indicators_moderate_score

AUTH_HEADER = {"Authorization": "Bearer test-token-123"}
FAKE_USER_ID = "fake-user-1"
FAKE_SCHOOL_ID = "school-test-1"


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


def _build_ai_cartridge_data(task_id: str = "test-ai-task-001") -> dict:
    """Builds a minimal valid AI-capable cartridge dict."""
    return {
        "task_id": task_id,
        "task_type": "hybrid",
        "title": "AI testo uzduotis",
        "description": "Minimali AI uzduotis testavimui",
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
        "phases": [
            {
                "id": "phase_intro",
                "title": "Ivadas",
                "is_ai_phase": False,
                "interaction": {
                    "type": "button",
                    "choices": [
                        {
                            "label": "Pradeti pokalbi",
                            "target_phase": "phase_ai",
                        },
                    ],
                },
            },
            {
                "id": "phase_ai",
                "title": "Pokalbis su Triksteriu",
                "is_ai_phase": True,
                "interaction": {
                    "type": "freeform",
                    "trickster_opening": "Sveiki! Paziurekime...",
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
                "title": "Atskleidimas - laimejo",
                "is_terminal": True,
                "evaluation_outcome": "trickster_loses",
            },
            {
                "id": "phase_reveal_timeout",
                "title": "Atskleidimas - laikas",
                "is_terminal": True,
                "evaluation_outcome": "trickster_wins",
            },
            {
                "id": "phase_reveal_partial",
                "title": "Atskleidimas - dalinis",
                "is_terminal": True,
                "evaluation_outcome": "partial",
            },
        ],
        "evaluation": {
            "patterns_embedded": [
                {
                    "id": "p1",
                    "description": "Antraste neatitinka turinio",
                    "technique": "headline_manipulation",
                    "real_world_connection": "Daznai pastebima",
                },
            ],
            "checklist": [
                {
                    "id": "c1",
                    "description": "Atpazino neatitikima",
                    "pattern_refs": ["p1"],
                    "is_mandatory": True,
                },
            ],
            "pass_conditions": {
                "trickster_wins": "Nepavyko",
                "partial": "Is dalies",
                "trickster_loses": "Pavyko",
            },
        },
        "reveal": {"key_lesson": "Antraste buvo sukurta skubos jausmui"},
        "safety": {
            "content_boundaries": ["self_harm"],
            "intensity_ceiling": 3,
            "cold_start_safe": True,
        },
        "ai_config": {
            "model_preference": "standard",
            "prompt_directory": "test-ai-task-001",
            "persona_mode": "chat_participant",
            "has_static_fallback": False,
            "context_requirements": "session_only",
        },
    }


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


def _make_engine(
    provider: MockProvider,
    ctx_manager: ContextManager,
    indicators: dict | None = None,
) -> TricksterEngine:
    """Creates a TricksterEngine with the given provider and indicators."""
    return TricksterEngine(
        provider, ctx_manager,
        intensity_indicators=indicators,
    )


def _inject_engine(engine: TricksterEngine) -> None:
    """Injects TricksterEngine into app DI overrides."""
    app.dependency_overrides[get_trickster_engine] = lambda: engine


async def _create_ai_session(
    task_id: str = "test-ai-task-001",
    phase_id: str = "phase_ai",
    exchanges: int = 0,
    turn_intensities: list[float] | None = None,
) -> GameSession:
    """Creates and persists a session ready for AI interaction."""
    session = GameSession(
        session_id=f"session-{task_id}",
        student_id=FAKE_USER_ID,
        school_id=FAKE_SCHOOL_ID,
        current_task=task_id,
        current_phase=phase_id,
    )
    for i in range(exchanges):
        session.exchanges.append(
            Exchange(role="student", content=f"Student message {i + 1}")
        )
        session.exchanges.append(
            Exchange(role="trickster", content=f"Trickster response {i + 1}")
        )
    if turn_intensities:
        session.turn_intensities = turn_intensities
    await deps._session_store.save_session(session)
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntensityIntegration:
    """Full-chain intensity tracking: HTTP -> engine -> SSE response."""

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_intensity_score_in_done_data(
        self, _mock_readiness, client, context_manager,
    ):
        """Respond with intensity keywords produces score in done_data."""
        indicators = _make_indicators_moderate_score()
        # Response text contains intensity keywords that trigger scoring
        provider = MockProvider(
            responses=["Tu klysti, tai absurdas!"],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=20),
        )
        engine = _make_engine(provider, context_manager, indicators)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_ai_cartridge_data())
        _use_registry_with([cartridge])
        await _create_ai_session()

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-ai-task-001/respond",
                json={"action": "freeform", "payload": "Manau tai netiesa"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        done_data = done_events[0]["data"]["data"]
        assert "intensity_score" in done_data
        assert done_data["intensity_score"] > 0
        assert "intensity_deescalation" in done_data

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_deescalation_flag_when_above_ceiling(
        self, _mock_readiness, client, context_manager,
    ):
        """Intensity above ceiling sets deescalation flag in done_data."""
        indicators = _make_indicators_moderate_score()
        # "tu klysti" + "tai absurdas" should score above ceiling of 3
        provider = MockProvider(
            responses=["Tu klysti, tai absurdas!"],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=20),
        )
        engine = _make_engine(provider, context_manager, indicators)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_ai_cartridge_data())
        _use_registry_with([cartridge])
        await _create_ai_session()

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-ai-task-001/respond",
                json={"action": "freeform", "payload": "Manau tai netiesa"},
                headers=AUTH_HEADER,
            )

        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        done_data = done_events[0]["data"]["data"]
        assert done_data["intensity_deescalation"] is True

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_deescalation_context_on_second_call(
        self, _mock_readiness, client, context_manager,
    ):
        """After a hot turn, the next call's system prompt has de-escalation."""
        indicators = _make_indicators_moderate_score()
        provider = MockProvider(
            responses=["Gerai, pamastykim..."],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=10),
        )
        engine = _make_engine(provider, context_manager, indicators)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_ai_cartridge_data())
        _use_registry_with([cartridge])
        # Pre-populate with a hot turn — last score above ceiling
        await _create_ai_session(
            exchanges=1,
            turn_intensities=[3.8],
        )

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-ai-task-001/respond",
                json={"action": "freeform", "payload": "Del ko?"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200

        # Verify the system prompt sent to the provider contained de-escalation
        assert provider.last_system_prompt is not None
        assert "De-eskalacijos instrukcija" in provider.last_system_prompt

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_no_deescalation_when_below_ceiling(
        self, _mock_readiness, client, context_manager,
    ):
        """When prior turn was cool, no de-escalation in system prompt."""
        indicators = _make_indicators_moderate_score()
        provider = MockProvider(
            responses=["Gerai, pamastykim..."],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=10),
        )
        engine = _make_engine(provider, context_manager, indicators)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_ai_cartridge_data())
        _use_registry_with([cartridge])
        # Prior turn was below ceiling
        await _create_ai_session(
            exchanges=1,
            turn_intensities=[2.0],
        )

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-ai-task-001/respond",
                json={"action": "freeform", "payload": "Del ko?"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        assert provider.last_system_prompt is not None
        assert "De-eskalacijos instrukcija" not in provider.last_system_prompt
