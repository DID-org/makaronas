"""Cross-capability integration tests for V5 Trickster Engine (Phase 10a).

Exercises multiple V5 capabilities simultaneously through the full HTTP
chain (HTTP request -> FastAPI -> TricksterEngine -> MockProvider ->
SSE/JSON response). Each test class verifies a cross-capability interaction
that no prior phase tested individually.

Test scenarios:
1. Mode + Intensity + Task History — single session exercises all three
2. Generation -> Evaluation Flow — /generate then /respond with artifacts
3. Clean Task + Mode-Specific Prompt — coexistence in system prompt
4. Debrief with Fourth Wall + Task History — both in debrief system prompt
5. Multi-Task Session — task history accumulates across task switches
"""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from backend.ai.context import ContextManager
from backend.ai.prompts import PromptLoader
from backend.ai.providers.base import ToolCallEvent, UsageInfo
from backend.ai.providers.mock import MockProvider
from backend.ai.trickster import TricksterEngine
from backend.api import deps
from backend.api.deps import (
    get_context_manager,
    get_task_registry,
    get_trickster_engine,
)
from backend.main import app
from backend.schemas import Exchange, GameSession
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import TaskCartridge
from backend.tests.conftest import setup_base_prompts, write_prompt_file
from backend.tests.test_intensity_engine import _make_indicators_moderate_score

AUTH_HEADER = {"Authorization": "Bearer test-token-123"}
FAKE_USER_ID = "fake-user-1"
FAKE_SCHOOL_ID = "school-test-1"


# ---------------------------------------------------------------------------
# Helpers (copied from test_student_ai.py per plan — test modules shouldn't
# cross-import helpers)
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


def _build_ai_cartridge_data(task_id: str = "test-cross-001", **overrides) -> dict:
    """Builds a minimal valid AI-capable cartridge dict with overrides."""
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
        "title": "Cross-capability testo uzduotis",
        "description": "Integracinio testo uzduotis",
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
    }

    if ai_config is not None:
        data["ai_config"] = ai_config

    data.update(overrides)
    return data


_CLEAN_EVAL_DATA = {
    "patterns_embedded": [],
    "checklist": [],
    "pass_conditions": {
        "trickster_wins": "Mokinys neteisingai apkaltino",
        "partial": "Mokinys abejojo",
        "trickster_loses": "Mokinys teisingai atpazino",
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


async def _create_ai_session(
    task_id: str = "test-cross-001",
    phase_id: str = "phase_ai",
    exchanges: int = 0,
    turn_intensities: list[float] | None = None,
    task_history: list[dict] | None = None,
    generated_artifacts: list[dict] | None = None,
    session_id: str | None = None,
) -> GameSession:
    """Creates and persists a session ready for AI interaction."""
    sid = session_id or f"session-{task_id}"
    session = GameSession(
        session_id=sid,
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
    if task_history:
        session.task_history = task_history
    if generated_artifacts:
        session.generated_artifacts = generated_artifacts
    await deps._session_store.save_session(session)
    return session


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


# ---------------------------------------------------------------------------
# Scenario 1: Mode + Intensity + Task History in One Session
# ---------------------------------------------------------------------------


class TestModeIntensityTaskHistory:
    """Single session exercises mode prompt, intensity scoring, and task
    history recording across multiple sequential respond calls."""

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_mode_and_intensity_coexist(
        self, _mock_readiness, client, prompts_dir,
    ):
        """First respond: mode prompt in system prompt + intensity in done_data."""
        write_prompt_file(
            prompts_dir / "trickster" / "persona_chat_participant_base.md",
            "CHAT_PARTICIPANT_MODE_MARKER",
        )
        loader = PromptLoader(prompts_dir)
        cm = ContextManager(loader)

        indicators = _make_indicators_moderate_score()
        # Response with intensity keywords to trigger scoring above ceiling
        provider = MockProvider(
            responses=["Tu klysti, tai absurdas!"],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=20),
        )
        engine = _make_engine(provider, cm, indicators)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_ai_cartridge_data())
        _use_registry_with([cartridge])
        await _create_ai_session(exchanges=2)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-cross-001/respond",
                json={"action": "freeform", "payload": "Manau tai netiesa"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        done_data = done_events[0]["data"]["data"]
        # Intensity active
        assert "intensity_score" in done_data
        assert done_data["intensity_score"] > 0
        assert "intensity_deescalation" in done_data

        # Mode content in system prompt
        assert provider.last_system_prompt is not None
        assert "CHAT_PARTICIPANT_MODE_MARKER" in provider.last_system_prompt

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_deescalation_after_hot_turn_with_mode(
        self, _mock_readiness, client, prompts_dir,
    ):
        """Second respond after a hot turn: de-escalation in system prompt
        alongside mode content."""
        write_prompt_file(
            prompts_dir / "trickster" / "persona_chat_participant_base.md",
            "CHAT_PARTICIPANT_MODE_MARKER",
        )
        loader = PromptLoader(prompts_dir)
        cm = ContextManager(loader)

        indicators = _make_indicators_moderate_score()
        provider = MockProvider(
            responses=["Gerai, pamastykim..."],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=10),
        )
        engine = _make_engine(provider, cm, indicators)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_ai_cartridge_data())
        _use_registry_with([cartridge])
        # Pre-populate with a hot prior turn
        await _create_ai_session(exchanges=2, turn_intensities=[3.8])

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-cross-001/respond",
                json={"action": "freeform", "payload": "Del ko?"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        assert provider.last_system_prompt is not None
        # Both mode and de-escalation coexist
        assert "CHAT_PARTICIPANT_MODE_MARKER" in provider.last_system_prompt
        assert "De-eskalacijos instrukcija" in provider.last_system_prompt

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_transition_records_task_history_with_intensity(
        self, _mock_readiness, client, prompts_dir,
    ):
        """Transition via tool call records task_history with intensity_score."""
        write_prompt_file(
            prompts_dir / "trickster" / "persona_chat_participant_base.md",
            "CHAT_PARTICIPANT_MODE_MARKER",
        )
        loader = PromptLoader(prompts_dir)
        cm = ContextManager(loader)

        indicators = _make_indicators_moderate_score()
        provider = MockProvider(
            responses=["Tu klysti, tai absurdas!"],
            tool_calls=[ToolCallEvent(
                function_name="transition_phase",
                arguments={"signal": "understood"},
            )],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=20),
        )
        engine = _make_engine(provider, cm, indicators)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_ai_cartridge_data())
        _use_registry_with([cartridge])
        await _create_ai_session(exchanges=3)

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-cross-001/respond",
                json={"action": "freeform", "payload": "Supratau ka darote!"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        done_data = done_events[0]["data"]["data"]
        assert done_data["phase_transition"] == "on_success"

        # Verify task_history recorded with intensity
        session = await deps._session_store.get_session(
            "session-test-cross-001"
        )
        assert len(session.task_history) == 1
        entry = session.task_history[0]
        assert entry["evaluation_outcome"] == "on_success"
        assert entry["exchange_count"] > 0
        assert entry["intensity_score"] is not None
        assert entry["task_id"] == "test-cross-001"

        # Verify turn_intensities accumulated
        assert len(session.turn_intensities) >= 1


# ---------------------------------------------------------------------------
# Scenario 2: Generation -> Evaluation Flow
# ---------------------------------------------------------------------------


class TestGenerationEvaluationFlow:
    """Student generates content via /generate, then Trickster evaluates it
    via /respond — verifying artifacts appear in Trickster context."""

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    @patch("backend.api.student._check_generation_readiness", return_value=[])
    async def test_generation_artifacts_in_trickster_context(
        self, _mock_gen_readiness, _mock_ai_readiness, prompts_dir,
    ):
        """Generated artifacts flow into Trickster's system prompt on /respond."""
        # Write creation eval prompt
        write_prompt_file(
            prompts_dir / "trickster" / "creation_eval_base.md",
            "CREATION_EVAL_COACHING_MARKER",
        )
        write_prompt_file(
            prompts_dir / "trickster" / "persona_presenting_base.md",
            "PRESENTING_MODE_MARKER",
        )
        loader = PromptLoader(prompts_dir)
        cm = ContextManager(loader)

        # Provider for /respond (separate from /generate's provider)
        respond_provider = MockProvider(
            responses=["Pabandykim dar karta..."],
            usage=UsageInfo(prompt_tokens=200, completion_tokens=20),
        )
        engine = _make_engine(respond_provider, cm)
        _inject_engine(engine)

        # Override context manager for /generate endpoint
        app.dependency_overrides[get_context_manager] = lambda: cm

        cartridge_data = _build_ai_cartridge_data(
            task_id="test-gen-eval-001",
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "test-gen-eval-001",
                "persona_mode": "presenting",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )
        cartridge = TaskCartridge.model_validate(cartridge_data)
        _use_registry_with([cartridge])
        await _create_ai_session(
            task_id="test-gen-eval-001",
            exchanges=2,
        )

        # Use a single client for both sequential calls
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as c:
            # Step 1: POST /generate — create an artifact
            gen_provider = MockProvider(
                responses=["Sugeneruotas klaidinantis postas"],
            )
            with patch(
                "backend.api.student.create_provider",
                return_value=gen_provider,
            ):
                gen_resp = await c.post(
                    "/api/v1/student/session/session-test-gen-eval-001/generate",
                    json={
                        "source_content": "Saltinio tekstas apie klimata",
                        "student_prompt": "Padaryk klaidinanti",
                    },
                    headers=AUTH_HEADER,
                )

            assert gen_resp.status_code == 200
            assert gen_resp.json()["data"]["artifact_index"] == 0

            # Verify artifact stored in session
            session = await deps._session_store.get_session(
                "session-test-gen-eval-001"
            )
            assert len(session.generated_artifacts) == 1

            # Step 2: POST /respond — Trickster evaluates the student's creation
            resp = await c.post(
                "/api/v1/student/session/session-test-gen-eval-001/respond",
                json={
                    "action": "freeform",
                    "payload": "Ar mano postas geras?",
                },
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        # Verify system prompt contains artifact and creation eval coaching
        assert respond_provider.last_system_prompt is not None
        sys_prompt = respond_provider.last_system_prompt
        assert "Sugeneruotas klaidinantis postas" in sys_prompt
        assert "CREATION_EVAL_COACHING_MARKER" in sys_prompt
        assert "Mokinio sukurtas turinys" in sys_prompt


# ---------------------------------------------------------------------------
# Scenario 3: Clean Task with Mode-Specific Prompt
# ---------------------------------------------------------------------------


class TestCleanTaskWithMode:
    """Clean task (is_clean=True) with non-default persona mode — verifying
    both coexist in the system prompt."""

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_clean_and_narrator_mode_coexist(
        self, _mock_readiness, client, prompts_dir,
    ):
        """System prompt contains both clean task context and narrator mode."""
        write_prompt_file(
            prompts_dir / "trickster" / "persona_narrator_base.md",
            "NARRATOR_MODE_MARKER",
        )
        write_prompt_file(
            prompts_dir / "trickster" / "clean_task_base.md",
            "CLEAN_TASK_CONTENT_MARKER",
        )
        loader = PromptLoader(prompts_dir)
        cm = ContextManager(loader)

        provider = MockProvider(
            responses=["Turinys yra patikimas."],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=10),
        )
        engine = _make_engine(provider, cm)
        _inject_engine(engine)

        cartridge_data = _build_ai_cartridge_data(
            task_id="test-clean-mode-001",
            is_clean=True,
            evaluation=_CLEAN_EVAL_DATA,
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "test-clean-mode-001",
                "persona_mode": "narrator",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )
        cartridge = TaskCartridge.model_validate(cartridge_data)
        _use_registry_with([cartridge])
        await _create_ai_session(task_id="test-clean-mode-001")

        async with client:
            resp = await client.post(
                "/api/v1/student/session/session-test-clean-mode-001/respond",
                json={"action": "freeform", "payload": "Ar cia tiesa?"},
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        # Verify both clean task and narrator mode in system prompt
        assert provider.last_system_prompt is not None
        sys_prompt = provider.last_system_prompt
        assert "NARRATOR_MODE_MARKER" in sys_prompt
        assert "CLEAN_TASK_CONTENT_MARKER" in sys_prompt
        # Clean task context header should be present
        assert "Svaraus turinio kontekstas" in sys_prompt
        # Adversarial task context header should NOT be present
        assert "Uzduoties kontekstas" not in sys_prompt


# ---------------------------------------------------------------------------
# Scenario 4: Debrief with Fourth Wall + Task History
# ---------------------------------------------------------------------------


class TestDebriefFourthWallTaskHistory:
    """Debrief call verifying fourth wall content appears in system prompt.
    Task history is NOT injected in the debrief path (discovered during
    implementation — see IMPLEMENTATION_NOTES)."""

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_debrief_fourth_wall_with_persona_override(
        self, _mock_readiness, client, prompts_dir,
    ):
        """Debrief system prompt contains fourth wall + persona override."""
        write_prompt_file(
            prompts_dir / "trickster" / "fourth_wall_base.md",
            "FOURTH_WALL_AI_LITERACY_MARKER",
        )
        loader = PromptLoader(prompts_dir)
        cm = ContextManager(loader)

        provider = MockProvider(
            responses=["Atskleidimas ir AI pamoka."],
            usage=UsageInfo(prompt_tokens=200, completion_tokens=20),
        )
        engine = _make_engine(provider, cm)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_ai_cartridge_data())
        _use_registry_with([cartridge])
        await _create_ai_session(
            exchanges=4,
            task_history=[{
                "task_id": "prior-task",
                "evaluation_outcome": "on_success",
                "exchange_count": 5,
                "is_clean": False,
            }],
        )

        async with client:
            resp = await client.get(
                "/api/v1/student/session/session-test-cross-001/debrief",
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["data"]["debrief_complete"] is True

        # Fourth wall and persona override present
        assert provider.last_system_prompt is not None
        sys_prompt = provider.last_system_prompt
        assert "FOURTH_WALL_AI_LITERACY_MARKER" in sys_prompt
        assert "Persona per\u0117jimas" in sys_prompt

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_debrief_task_history_not_in_prompt(
        self, _mock_readiness, client, prompts_dir,
    ):
        """Documents that task history is NOT injected in debrief system
        prompt (only in dialogue). This is an architectural observation,
        not a bug — the debrief focuses on the current task's reveal."""
        write_prompt_file(
            prompts_dir / "trickster" / "fourth_wall_base.md",
            "FOURTH_WALL_CONTENT",
        )
        loader = PromptLoader(prompts_dir)
        cm = ContextManager(loader)

        provider = MockProvider(
            responses=["Atskleidimas."],
            usage=UsageInfo(prompt_tokens=200, completion_tokens=10),
        )
        engine = _make_engine(provider, cm)
        _inject_engine(engine)

        cartridge = TaskCartridge.model_validate(_build_ai_cartridge_data())
        _use_registry_with([cartridge])
        await _create_ai_session(
            exchanges=4,
            task_history=[{
                "task_id": "prior-task",
                "evaluation_outcome": "on_success",
                "exchange_count": 5,
                "is_clean": False,
            }],
        )

        async with client:
            resp = await client.get(
                "/api/v1/student/session/session-test-cross-001/debrief",
                headers=AUTH_HEADER,
            )

        assert resp.status_code == 200
        # Task history context header should NOT appear in debrief
        assert provider.last_system_prompt is not None
        assert "Ankstesni\u0173 u\u017eduo\u010di\u0173" not in provider.last_system_prompt


# ---------------------------------------------------------------------------
# Scenario 5: Multi-Task Session with Task History Accumulation
# ---------------------------------------------------------------------------


class TestMultiTaskSessionHistory:
    """Session completes two tasks sequentially — task history accumulates
    and the second task's context includes history from the first."""

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_task_history_flows_across_tasks(
        self, _mock_readiness, prompts_dir,
    ):
        """Task A completes with transition, task B's system prompt includes
        task history referencing task A."""
        write_prompt_file(
            prompts_dir / "trickster" / "persona_chat_participant_base.md",
            "CHAT_PARTICIPANT_MARKER",
        )
        write_prompt_file(
            prompts_dir / "trickster" / "persona_presenting_base.md",
            "PRESENTING_MARKER",
        )
        loader = PromptLoader(prompts_dir)
        cm = ContextManager(loader)

        # --- Task A cartridge ---
        cartridge_a_data = _build_ai_cartridge_data(
            task_id="task-a-multi",
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "task-a-multi",
                "persona_mode": "chat_participant",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )
        cartridge_a = TaskCartridge.model_validate(cartridge_a_data)

        # --- Task B cartridge ---
        cartridge_b_data = _build_ai_cartridge_data(
            task_id="task-b-multi",
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "task-b-multi",
                "persona_mode": "presenting",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )
        cartridge_b = TaskCartridge.model_validate(cartridge_b_data)
        _use_registry_with([cartridge_a, cartridge_b])

        session_id = "session-multi-task"

        # --- Step 1: Create session on task A, pre-fill exchanges ---
        await _create_ai_session(
            task_id="task-a-multi",
            exchanges=3,
            session_id=session_id,
        )

        # Use a single client for both sequential HTTP calls
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as c:
            # --- Step 2: Complete task A via transition tool call ---
            provider_a = MockProvider(
                responses=["Puiku, supratai!"],
                tool_calls=[ToolCallEvent(
                    function_name="transition_phase",
                    arguments={"signal": "understood"},
                )],
                usage=UsageInfo(prompt_tokens=100, completion_tokens=10),
            )
            engine_a = _make_engine(provider_a, cm)
            _inject_engine(engine_a)

            resp_a = await c.post(
                f"/api/v1/student/session/{session_id}/respond",
                json={"action": "freeform", "payload": "Supratau!"},
                headers=AUTH_HEADER,
            )

            assert resp_a.status_code == 200
            events_a = _parse_sse_events(resp_a.text)
            done_a = [e for e in events_a if e["type"] == "done"]
            assert done_a[0]["data"]["data"]["phase_transition"] == "on_success"

            # Verify task_history has 1 entry
            session = await deps._session_store.get_session(session_id)
            assert len(session.task_history) == 1
            assert session.task_history[0]["task_id"] == "task-a-multi"
            assert session.task_history[0]["evaluation_outcome"] == "on_success"

            # --- Step 3: Switch to task B ---
            # Manually reset session state for task B (next_task endpoint only
            # sets current_task and current_phase — doesn't clear exchanges,
            # prompt_snapshots, turn_intensities, generated_artifacts)
            session.current_task = "task-b-multi"
            session.current_phase = "phase_ai"
            session.exchanges = []
            session.prompt_snapshots = None
            session.turn_intensities = []
            session.generated_artifacts = []
            # Pre-fill exchanges for task B past min_exchanges gate
            for i in range(3):
                session.exchanges.append(
                    Exchange(role="student", content=f"Task B msg {i + 1}")
                )
                session.exchanges.append(
                    Exchange(
                        role="trickster", content=f"Task B resp {i + 1}",
                    )
                )
            await deps._session_store.save_session(session)

            # --- Step 4: Respond on task B, verify task history in context ---
            provider_b = MockProvider(
                responses=["Dabar pabandykim kita..."],
                usage=UsageInfo(prompt_tokens=200, completion_tokens=15),
            )
            engine_b = _make_engine(provider_b, cm)
            _inject_engine(engine_b)

            resp_b = await c.post(
                f"/api/v1/student/session/{session_id}/respond",
                json={
                    "action": "freeform",
                    "payload": "Manau cia kita problema",
                },
                headers=AUTH_HEADER,
            )

        assert resp_b.status_code == 200
        events_b = _parse_sse_events(resp_b.text)
        done_b = [e for e in events_b if e["type"] == "done"]
        assert len(done_b) == 1

        # Verify task B's system prompt contains task history from task A
        assert provider_b.last_system_prompt is not None
        sys_prompt = provider_b.last_system_prompt
        assert "PRESENTING_MARKER" in sys_prompt
        # Task history context header (Lithuanian)
        assert "Ankstesni\u0173 u\u017eduo\u010di\u0173" in sys_prompt

        # task_history still has 1 entry (task B not yet completed)
        session = await deps._session_store.get_session(session_id)
        assert len(session.task_history) == 1
