"""Tests for task history context injection (Phase 6b).

Unit tests for _build_task_history_context() and integration tests
verifying the layer appears correctly in assembled system prompts.

Test categories:
T1: Zero history -> None return
T2: Single entry -> summary produced
T3: Exactly 3 entries -> all included
T4: 4+ entries -> oldest dropped (3-task cap)
T5: Context fencing verification
T6: Clean task annotation
T7: Intensity score inclusion
T8: Integration with assemble_trickster_call
T9: Outcome label mapping
T10: API-level integration test with multi-task session
"""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from backend.ai.context import ContextManager, _MAX_HISTORY_TASKS, _OUTCOME_LABELS
from backend.ai.prompts import PromptLoader
from backend.ai.providers.base import TextChunk, UsageInfo
from backend.ai.providers.mock import MockProvider
from backend.ai.trickster import TricksterEngine
from backend.api import deps
from backend.api.deps import get_task_registry, get_trickster_engine
from backend.main import app
from backend.schemas import Exchange, GameSession
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import TaskCartridge
from backend.tests.conftest import setup_base_prompts


# ---------------------------------------------------------------------------
# SSE parsing helper
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
# Test data helpers
# ---------------------------------------------------------------------------


def _history_entry(
    task_id: str = "task-test-001",
    outcome: str = "on_success",
    exchanges: int = 5,
    intensity: float | None = None,
    is_clean: bool = False,
) -> dict:
    """Creates a task_history entry dict."""
    return {
        "task_id": task_id,
        "evaluation_outcome": outcome,
        "exchange_count": exchanges,
        "intensity_score": intensity,
        "is_clean": is_clean,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


AUTH_HEADER = {"Authorization": "Bearer test-token-123"}
FAKE_USER_ID = "fake-user-1"
FAKE_SCHOOL_ID = "school-test-1"


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


# ---------------------------------------------------------------------------
# T1: Zero history -> None return
# ---------------------------------------------------------------------------


class TestZeroHistory:
    """Empty task_history produces None."""

    def test_empty_history_returns_none(self, make_session):
        session = make_session(task_history=[])
        result = ContextManager._build_task_history_context(session)
        assert result is None

    def test_no_history_layer_in_system_prompt(
        self, make_session, make_cartridge, context_manager
    ):
        """assemble_trickster_call with empty history has no history section."""
        session = make_session(
            task_history=[],
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )
        cartridge = make_cartridge()
        ctx = context_manager.assemble_trickster_call(
            session, cartridge, "gemini", exchange_count=0, min_exchanges=2
        )
        assert "Ankstesni\u0173 u\u017eduo\u010di\u0173 kontekstas" not in ctx.system_prompt


# ---------------------------------------------------------------------------
# T2: Single entry -> summary produced
# ---------------------------------------------------------------------------


class TestSingleEntry:
    """One task_history entry produces a valid summary."""

    def test_single_entry_returns_string(self, make_session):
        session = make_session(task_history=[_history_entry()])
        result = ContextManager._build_task_history_context(session)
        assert result is not None
        assert isinstance(result, str)

    def test_section_header_present(self, make_session):
        session = make_session(task_history=[_history_entry()])
        result = ContextManager._build_task_history_context(session)
        assert "Ankstesni\u0173 u\u017eduo\u010di\u0173 kontekstas" in result

    def test_outcome_label_present(self, make_session):
        session = make_session(
            task_history=[_history_entry(outcome="on_success")]
        )
        result = ContextManager._build_task_history_context(session)
        assert "Mokinys suprato" in result

    def test_exchange_count_present(self, make_session):
        session = make_session(
            task_history=[_history_entry(exchanges=7)]
        )
        result = ContextManager._build_task_history_context(session)
        assert "7 apsikeitimai" in result

    def test_fencing_instruction_present(self, make_session):
        session = make_session(task_history=[_history_entry()])
        result = ContextManager._build_task_history_context(session)
        assert "NIEKADA" in result


# ---------------------------------------------------------------------------
# T3: Exactly 3 entries -> all included
# ---------------------------------------------------------------------------


class TestThreeEntries:
    """Three entries all appear in the output."""

    def test_all_three_present(self, make_session):
        entries = [
            _history_entry(task_id=f"task-{i}", exchanges=i + 3)
            for i in range(3)
        ]
        session = make_session(task_history=entries)
        result = ContextManager._build_task_history_context(session)
        for i in range(3):
            assert f"{i + 3} apsikeitimai" in result

    def test_chronological_order_preserved(self, make_session):
        entries = [
            _history_entry(task_id="first", outcome="on_success"),
            _history_entry(task_id="second", outcome="on_partial"),
            _history_entry(task_id="third", outcome="on_max_exchanges"),
        ]
        session = make_session(task_history=entries)
        result = ContextManager._build_task_history_context(session)
        pos_first = result.index("Mokinys suprato")
        pos_second = result.index("Dalinis supratimas")
        pos_third = result.index("Nepavyko suprasti")
        assert pos_first < pos_second < pos_third


# ---------------------------------------------------------------------------
# T4: 4+ entries -> oldest dropped (3-task cap)
# ---------------------------------------------------------------------------


class TestCapEnforcement:
    """More than 3 entries are capped to the most recent 3."""

    def test_four_entries_caps_to_three(self, make_session):
        entries = [
            _history_entry(task_id="oldest", exchanges=1),
            _history_entry(task_id="second", exchanges=2),
            _history_entry(task_id="third", exchanges=3),
            _history_entry(task_id="newest", exchanges=4),
        ]
        session = make_session(task_history=entries)
        result = ContextManager._build_task_history_context(session)
        # Oldest should be dropped
        assert "1 apsikeitimai" not in result
        # Recent 3 present
        assert "2 apsikeitimai" in result
        assert "3 apsikeitimai" in result
        assert "4 apsikeitimai" in result

    def test_five_entries_keeps_last_three(self, make_session):
        entries = [_history_entry(exchanges=i) for i in range(5)]
        session = make_session(task_history=entries)
        result = ContextManager._build_task_history_context(session)
        # Only exchanges 2, 3, 4 should be present
        assert "2 apsikeitimai" in result
        assert "3 apsikeitimai" in result
        assert "4 apsikeitimai" in result
        assert "0 apsikeitimai" not in result
        assert "1 apsikeitimai" not in result

    def test_max_history_tasks_constant_is_three(self):
        assert _MAX_HISTORY_TASKS == 3


# ---------------------------------------------------------------------------
# T5: Context fencing verification
# ---------------------------------------------------------------------------


class TestContextFencing:
    """Fencing instruction prohibits topic references."""

    def test_prohibition_keywords_present(self, make_session):
        session = make_session(task_history=[_history_entry()])
        result = ContextManager._build_task_history_context(session)
        assert "NIEKADA" in result
        assert "pavadinim\u0173" in result or "pavadinimu" in result.lower()

    def test_pedagogical_pattern_guidance(self, make_session):
        session = make_session(task_history=[_history_entry()])
        result = ContextManager._build_task_history_context(session)
        # Should mention pedagogical patterns as acceptable references
        assert "pedagogini" in result.lower() or "d\u0117sniais" in result


# ---------------------------------------------------------------------------
# T6: Clean task annotation
# ---------------------------------------------------------------------------


class TestCleanTaskAnnotation:
    """Clean tasks get explicit annotation."""

    def test_clean_true_annotated(self, make_session):
        session = make_session(
            task_history=[_history_entry(is_clean=True)]
        )
        result = ContextManager._build_task_history_context(session)
        assert "\u0161varus turinys" in result

    def test_clean_false_no_annotation(self, make_session):
        session = make_session(
            task_history=[_history_entry(is_clean=False)]
        )
        result = ContextManager._build_task_history_context(session)
        assert "\u0161varus turinys" not in result


# ---------------------------------------------------------------------------
# T7: Intensity score inclusion
# ---------------------------------------------------------------------------


class TestIntensityScore:
    """Intensity score included when present, omitted when None."""

    def test_intensity_present_when_set(self, make_session):
        session = make_session(
            task_history=[_history_entry(intensity=3.8)]
        )
        result = ContextManager._build_task_history_context(session)
        assert "3.8" in result
        assert "intensyvumas" in result

    def test_intensity_absent_when_none(self, make_session):
        session = make_session(
            task_history=[_history_entry(intensity=None)]
        )
        result = ContextManager._build_task_history_context(session)
        assert "intensyvumas" not in result


# ---------------------------------------------------------------------------
# T8: Integration with assemble_trickster_call
# ---------------------------------------------------------------------------


class TestAssemblyIntegration:
    """Task history layer appears in assembled system prompt."""

    def test_history_in_system_prompt(
        self, make_session, make_cartridge, context_manager
    ):
        entries = [
            _history_entry(outcome="on_success", exchanges=4),
            _history_entry(outcome="on_partial", exchanges=6),
        ]
        session = make_session(
            task_history=entries,
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )
        cartridge = make_cartridge()
        ctx = context_manager.assemble_trickster_call(
            session, cartridge, "gemini", exchange_count=0, min_exchanges=2
        )
        # History section present
        assert "Ankstesni\u0173 u\u017eduo\u010di\u0173 kontekstas" in ctx.system_prompt
        assert "Mokinys suprato" in ctx.system_prompt
        assert "Dalinis supratimas" in ctx.system_prompt

    def test_history_after_task_context_before_safety(
        self, make_session, make_cartridge, context_manager
    ):
        """Verify layer ordering: task context < history < safety config."""
        session = make_session(
            task_history=[_history_entry()],
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )
        cartridge = make_cartridge()
        ctx = context_manager.assemble_trickster_call(
            session, cartridge, "gemini", exchange_count=0, min_exchanges=2
        )
        prompt = ctx.system_prompt
        # Task history should appear after task context markers
        history_pos = prompt.index("Ankstesni\u0173 u\u017eduo\u010di\u0173 kontekstas")
        # Safety config section should appear after history
        safety_pos = prompt.index("Saugumo nustatymai")
        assert history_pos < safety_pos


# ---------------------------------------------------------------------------
# T9: Outcome label mapping
# ---------------------------------------------------------------------------


class TestOutcomeLabels:
    """Each outcome maps to the correct Lithuanian label."""

    @pytest.mark.parametrize(
        "outcome, expected_label",
        [
            ("on_success", "Mokinys suprato"),
            ("on_partial", "Dalinis supratimas"),
            ("on_max_exchanges", "Nepavyko suprasti"),
        ],
    )
    def test_known_outcomes(self, make_session, outcome, expected_label):
        session = make_session(
            task_history=[_history_entry(outcome=outcome)]
        )
        result = ContextManager._build_task_history_context(session)
        assert expected_label in result

    def test_unknown_outcome_uses_raw_string(self, make_session):
        """Unknown outcome doesn't crash — uses raw string."""
        session = make_session(
            task_history=[_history_entry(outcome="on_custom_thing")]
        )
        result = ContextManager._build_task_history_context(session)
        assert result is not None
        assert "on_custom_thing" in result

    def test_outcome_labels_constant_has_three_keys(self):
        assert len(_OUTCOME_LABELS) == 3
        assert set(_OUTCOME_LABELS.keys()) == {
            "on_success",
            "on_partial",
            "on_max_exchanges",
        }


# ---------------------------------------------------------------------------
# T10: API-level integration test with multi-task session
# ---------------------------------------------------------------------------


def _build_cartridge_data(task_id: str = "test-ctx-001", **overrides) -> dict:
    """Builds a minimal valid AI-capable cartridge dict."""
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
        "title": "Konteksto testas",
        "description": "U\u017eduotis konteksto testavimui",
        "version": "1.0",
        "trigger": "urgency",
        "technique": "headline_manipulation",
        "medium": "article",
        "learning_objectives": ["Atpa\u017einti"],
        "difficulty": 3,
        "time_minutes": 15,
        "is_evergreen": True,
        "is_clean": False,
        "initial_phase": "phase_intro",
        "phases": [
            {
                "id": "phase_intro",
                "title": "\u012evadas",
                "is_ai_phase": False,
                "interaction": {
                    "type": "button",
                    "choices": [
                        {
                            "label": "Prad\u0117ti",
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
                "title": "Laim\u0117jo",
                "is_terminal": True,
                "evaluation_outcome": "trickster_loses",
            },
            {
                "id": "phase_reveal_timeout",
                "title": "Laikas baig\u0117si",
                "is_terminal": True,
                "evaluation_outcome": "trickster_wins",
            },
            {
                "id": "phase_reveal_partial",
                "title": "I\u0161 dalies",
                "is_terminal": True,
                "evaluation_outcome": "partial",
            },
        ],
        "evaluation": {
            "patterns_embedded": [
                {
                    "id": "p1",
                    "description": "Testas",
                    "technique": "headline_manipulation",
                    "real_world_connection": "Da\u017enai",
                },
            ],
            "checklist": [
                {
                    "id": "c1",
                    "description": "Atpa\u017eino",
                    "pattern_refs": ["p1"],
                    "is_mandatory": True,
                },
            ],
            "pass_conditions": {
                "trickster_wins": "Nepavyko",
                "partial": "I\u0161 dalies",
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
    task_id: str = "test-ctx-001",
    phase_id: str = "phase_ai",
    exchanges: int = 0,
    **overrides,
) -> GameSession:
    """Creates and persists a session ready for AI interaction."""
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


def _make_engine(provider: MockProvider, ctx_manager: ContextManager) -> TricksterEngine:
    """Creates a TricksterEngine with the given provider."""
    return TricksterEngine(provider, ctx_manager)


def _inject_engine(engine: TricksterEngine) -> None:
    """Injects TricksterEngine into app DI overrides."""
    app.dependency_overrides[get_trickster_engine] = lambda: engine


@pytest.fixture
def api_prompts_dir(tmp_path) -> Path:
    """Creates a temp directory with base Trickster prompts."""
    setup_base_prompts(tmp_path)
    return tmp_path


@pytest.fixture
def api_context_manager(api_prompts_dir) -> ContextManager:
    """Returns a real ContextManager backed by temp prompts."""
    loader = PromptLoader(api_prompts_dir)
    return ContextManager(loader)


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


class TestAPILevelHistoryInjection:
    """API-level integration: task history appears in system prompt."""

    @pytest.mark.asyncio
    @patch("backend.api.student.check_ai_readiness", return_value=[])
    async def test_system_prompt_contains_history(
        self, _mock_readiness, client, api_context_manager
    ):
        """POST /respond with pre-existing task_history -> system prompt has history."""
        task_id = "test-ctx-001"
        cartridge = TaskCartridge(**_build_cartridge_data(task_id))
        _use_registry_with([cartridge])

        provider = MockProvider(
            responses=["AI atsakymas mokiniui."],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=20),
        )
        engine = _make_engine(provider, api_context_manager)
        _inject_engine(engine)

        prior_history = [
            _history_entry(
                task_id="task-prior-001",
                outcome="on_success",
                exchanges=4,
            ),
        ]

        await _create_session(
            task_id=task_id,
            phase_id="phase_ai",
            exchanges=1,
            task_history=prior_history,
        )

        resp = await client.post(
            f"/api/v1/student/session/session-{task_id}/respond",
            json={"action": "freeform", "payload": "Ar tai manipuliacija?"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200

        # Verify system prompt captured by MockProvider
        assert provider.last_system_prompt is not None
        prompt = provider.last_system_prompt
        assert "Ankstesni\u0173 u\u017eduo\u010di\u0173 kontekstas" in prompt
        assert "Mokinys suprato" in prompt
        assert "NIEKADA" in prompt
