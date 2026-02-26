"""Tests that conftest.py fixture factories produce valid objects.

Verifies that make_session, make_cartridge, mock_provider, and mock_registry
produce correctly shaped instances with sensible defaults and support overrides.
"""

import pytest

from backend.ai.providers.base import AIProvider, ToolCallEvent, UsageInfo
from backend.ai.providers.mock import MockProvider
from backend.schemas import Exchange, GameSession
from backend.tasks.schemas import (
    AiConfig,
    AiTransitions,
    EvaluationContract,
    FreeformInteraction,
    SafetyConfig,
    TaskCartridge,
)


# ---------------------------------------------------------------------------
# mock_provider factory
# ---------------------------------------------------------------------------


class TestMockProviderFactory:
    """mock_provider fixture returns a factory for MockProvider instances."""

    def test_default(self, mock_provider) -> None:
        provider = mock_provider()
        assert isinstance(provider, MockProvider)
        assert isinstance(provider, AIProvider)

    def test_with_custom_responses(self, mock_provider) -> None:
        provider = mock_provider(responses=["custom"])
        assert provider.responses == ["custom"]

    def test_with_error(self, mock_provider) -> None:
        err = ValueError("test error")
        provider = mock_provider(error=err)
        assert provider.error is err

    def test_with_tool_calls(self, mock_provider) -> None:
        tool = ToolCallEvent(function_name="test", arguments={})
        provider = mock_provider(tool_calls=[tool])
        assert len(provider.tool_calls) == 1

    def test_with_custom_usage(self, mock_provider) -> None:
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50)
        provider = mock_provider(usage=usage)
        assert provider.usage == usage


# ---------------------------------------------------------------------------
# make_session factory
# ---------------------------------------------------------------------------


class TestMakeSession:
    """make_session fixture produces valid GameSession instances."""

    def test_default_session(self, make_session) -> None:
        session = make_session()
        assert isinstance(session, GameSession)
        assert session.session_id.startswith("session-")
        assert session.student_id.startswith("student-")
        assert session.school_id == "school-test-001"
        assert session.language == "lt"
        assert session.exchanges == []
        assert session.choices == []

    def test_unique_ids(self, make_session) -> None:
        s1 = make_session()
        s2 = make_session()
        assert s1.session_id != s2.session_id
        assert s1.student_id != s2.student_id

    def test_override_fields(self, make_session) -> None:
        session = make_session(
            school_id="school-custom",
            language="en",
            session_id="fixed-session",
        )
        assert session.school_id == "school-custom"
        assert session.language == "en"
        assert session.session_id == "fixed-session"

    def test_with_exchanges(self, make_session) -> None:
        exchange = Exchange(role="student", content="Ar tai tikra?")
        session = make_session(exchanges=[exchange])
        assert len(session.exchanges) == 1
        assert session.exchanges[0].role == "student"
        assert session.exchanges[0].content == "Ar tai tikra?"

    def test_with_current_task(self, make_session) -> None:
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )
        assert session.current_task == "test-ai-task-001"
        assert session.current_phase == "phase_ai"


# ---------------------------------------------------------------------------
# make_cartridge factory
# ---------------------------------------------------------------------------


class TestMakeCartridge:
    """make_cartridge fixture produces valid AI-capable TaskCartridge instances."""

    def test_default_cartridge_is_valid(self, make_cartridge) -> None:
        cartridge = make_cartridge()
        assert isinstance(cartridge, TaskCartridge)
        assert cartridge.task_id == "test-ai-task-001"
        assert cartridge.task_type == "hybrid"

    def test_has_ai_config(self, make_cartridge) -> None:
        cartridge = make_cartridge()
        assert cartridge.ai_config is not None
        assert isinstance(cartridge.ai_config, AiConfig)
        assert cartridge.ai_config.model_preference == "standard"

    def test_has_ai_phase(self, make_cartridge) -> None:
        """At least one phase has is_ai_phase=True with FreeformInteraction."""
        cartridge = make_cartridge()
        ai_phases = [p for p in cartridge.phases if p.is_ai_phase]
        assert len(ai_phases) >= 1
        ai_phase = ai_phases[0]
        assert isinstance(ai_phase.interaction, FreeformInteraction)
        assert ai_phase.interaction.type == "freeform"
        assert ai_phase.interaction.min_exchanges == 2
        assert ai_phase.interaction.max_exchanges == 10

    def test_ai_transitions_present(self, make_cartridge) -> None:
        """AI phase has ai_transitions mapping all three signals."""
        cartridge = make_cartridge()
        ai_phase = next(p for p in cartridge.phases if p.is_ai_phase)
        assert ai_phase.ai_transitions is not None
        assert isinstance(ai_phase.ai_transitions, AiTransitions)

        # All three transition targets should be valid phase IDs
        phase_ids = {p.id for p in cartridge.phases}
        assert ai_phase.ai_transitions.on_success in phase_ids
        assert ai_phase.ai_transitions.on_max_exchanges in phase_ids
        assert ai_phase.ai_transitions.on_partial in phase_ids

    def test_has_safety_config(self, make_cartridge) -> None:
        cartridge = make_cartridge()
        assert isinstance(cartridge.safety, SafetyConfig)
        assert "self_harm" in cartridge.safety.content_boundaries
        assert cartridge.safety.intensity_ceiling == 3

    def test_has_evaluation_contract(self, make_cartridge) -> None:
        cartridge = make_cartridge()
        assert isinstance(cartridge.evaluation, EvaluationContract)
        assert len(cartridge.evaluation.patterns_embedded) >= 1
        assert len(cartridge.evaluation.checklist) >= 1
        assert cartridge.evaluation.pass_conditions is not None

    def test_has_terminal_phases(self, make_cartridge) -> None:
        cartridge = make_cartridge()
        terminal = [p for p in cartridge.phases if p.is_terminal]
        assert len(terminal) >= 1

    def test_static_cartridge_override(self, make_cartridge) -> None:
        """Can create a static-only cartridge by removing ai_config."""
        cartridge = make_cartridge(ai_config=None, task_type="static")
        assert cartridge.task_type == "static"
        assert cartridge.ai_config is None

    def test_override_fields(self, make_cartridge) -> None:
        cartridge = make_cartridge(difficulty=5, time_minutes=30)
        assert cartridge.difficulty == 5
        assert cartridge.time_minutes == 30

    def test_override_task_id(self, make_cartridge) -> None:
        cartridge = make_cartridge(task_id="custom-task-id")
        assert cartridge.task_id == "custom-task-id"

    def test_frozen(self, make_cartridge) -> None:
        """Cartridge is immutable."""
        cartridge = make_cartridge()
        with pytest.raises(Exception):
            cartridge.task_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# mock_registry fixture
# ---------------------------------------------------------------------------


class TestMockRegistry:
    """mock_registry provides a pre-populated TaskRegistry."""

    def test_contains_cartridge(self, mock_registry) -> None:
        cartridge = mock_registry.get_task("test-ai-task-001")
        assert cartridge is not None
        assert isinstance(cartridge, TaskCartridge)

    def test_queryable_by_status(self, mock_registry) -> None:
        """Active task IDs include the default cartridge."""
        task_ids = mock_registry.get_all_task_ids(status="active")
        assert "test-ai-task-001" in task_ids

    def test_queryable_by_trigger(self, mock_registry) -> None:
        results = mock_registry.query(trigger="urgency")
        assert len(results) >= 1
        assert any(c.task_id == "test-ai-task-001" for c in results)

    def test_queryable_by_technique(self, mock_registry) -> None:
        results = mock_registry.query(technique="headline_manipulation")
        assert len(results) >= 1

    def test_queryable_by_medium(self, mock_registry) -> None:
        results = mock_registry.query(medium="article")
        assert len(results) >= 1

    def test_count(self, mock_registry) -> None:
        assert mock_registry.count(status="active") >= 1

    def test_nonexistent_task(self, mock_registry) -> None:
        assert mock_registry.get_task("nonexistent") is None
