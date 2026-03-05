"""Tests for the context manager — assembly, budgeting, snapshotting."""

import base64
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.ai.context import (
    TRANSITION_TOOL,
    AssembledContext,
    ContextManager,
    _TOKENS_PER_IMAGE,
)
from backend.ai.prompts import PromptLoader, TricksterPrompts
from backend.schemas import Exchange, GameSession
from backend.tests.conftest import setup_base_prompts, write_prompt_file


def _make_exchange(role: str, content: str) -> Exchange:
    """Creates an Exchange with a fixed timestamp."""
    return Exchange(
        role=role,
        content=content,
        timestamp=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_exchange_pair(n: int) -> list[Exchange]:
    """Creates a student+trickster exchange pair for testing."""
    return [
        _make_exchange("student", f"Student message {n}"),
        _make_exchange("trickster", f"Trickster response {n}"),
    ]


# ---------------------------------------------------------------------------
# AssembledContext dataclass
# ---------------------------------------------------------------------------


class TestAssembledContext:
    def test_frozen(self) -> None:
        """AssembledContext is immutable."""
        ctx = AssembledContext(
            system_prompt="test",
            messages=[],
            tools=None,
        )
        with pytest.raises(AttributeError):
            ctx.system_prompt = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        """All fields accessible."""
        ctx = AssembledContext(
            system_prompt="prompt",
            messages=[{"role": "user", "content": "hi"}],
            tools=[TRANSITION_TOOL],
        )
        assert ctx.system_prompt == "prompt"
        assert len(ctx.messages) == 1
        assert ctx.tools is not None
        assert len(ctx.tools) == 1


# ---------------------------------------------------------------------------
# Full assembly with all 8 layers
# ---------------------------------------------------------------------------


class TestFullAssembly:
    def test_all_8_layers_present(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """System prompt contains all 8 layers when all data is present."""
        setup_base_prompts(tmp_path)
        write_prompt_file(
            tmp_path / "tasks" / "test-ai-task-001" / "trickster_base.md",
            "Task override content",
        )
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(
            current_phase="phase_ai",
            choices=[{"context_label": "Mokinys pasirinko pradeti"}],
            exchanges=[
                _make_exchange("student", "Ar tai tikra?"),
                _make_exchange("trickster", "Taip, tikrai."),
            ],
        )
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=2, min_exchanges=2,
        )

        # Layer 1: Persona
        assert "Test persona content." in result.system_prompt
        # Layer 2: Behaviour
        assert "Test behaviour content." in result.system_prompt
        # Layer 3: Safety
        assert "Test safety content." in result.system_prompt
        # Layer 4: Task override
        assert "Task override content" in result.system_prompt
        # Layer 5: Task context
        assert "Uzduoties kontekstas" in result.system_prompt
        assert "chat_participant" in result.system_prompt
        assert "phase_ai" in result.system_prompt
        # Layer 6: Safety config
        assert "Saugumo nustatymai" in result.system_prompt
        assert "self_harm" in result.system_prompt
        # Layer 7: Language
        assert "lietuviškai" in result.system_prompt.lower() or \
            "lietuviskai" in result.system_prompt.lower() or \
            "lietuviškai" in result.system_prompt
        # Layer 8: Context labels
        assert "Mokinio pasirinkimai" in result.system_prompt
        assert "Mokinys pasirinko pradeti" in result.system_prompt

    def test_messages_in_correct_order_with_role_mapping(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Messages are chronological with correct role mapping."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(
            exchanges=[
                _make_exchange("student", "Message 1"),
                _make_exchange("trickster", "Response 1"),
                _make_exchange("student", "Message 2"),
                _make_exchange("trickster", "Response 2"),
            ],
        )
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=4, min_exchanges=2,
        )

        assert len(result.messages) == 4
        assert result.messages[0] == {"role": "user", "content": "Message 1"}
        assert result.messages[1] == {"role": "assistant", "content": "Response 1"}
        assert result.messages[2] == {"role": "user", "content": "Message 2"}
        assert result.messages[3] == {"role": "assistant", "content": "Response 2"}


# ---------------------------------------------------------------------------
# Missing optional layers
# ---------------------------------------------------------------------------


class TestMissingLayers:
    def test_no_task_override(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Assembly works without task override (no None text in prompt)."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "None" not in result.system_prompt
        assert "Test persona content." in result.system_prompt

    def test_no_context_labels(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Assembly works when no choices have context_label."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(choices=[{"some_key": "value"}])
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "Mokinio pasirinkimai" not in result.system_prompt

    def test_empty_choices(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """No context label section when choices list is empty."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(choices=[])
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "Mokinio pasirinkimai" not in result.system_prompt

    def test_no_redaction(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """No redaction section when last_redaction_reason is None."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(last_redaction_reason=None)
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "Sistemos pastaba" not in result.system_prompt

    def test_no_exchanges(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Assembly works with empty exchange history."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(exchanges=[])
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=0, min_exchanges=2,
        )

        assert result.messages == []
        assert result.system_prompt  # System prompt still present


# ---------------------------------------------------------------------------
# Token budgeting and trimming
# ---------------------------------------------------------------------------


class TestTokenBudgeting:
    def test_5_exchanges_no_trimming(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """5 exchanges stay intact with default budget."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        exchanges = []
        for i in range(5):
            exchanges.extend(_make_exchange_pair(i))

        session = make_session(exchanges=exchanges)
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=10, min_exchanges=2,
        )

        assert len(result.messages) == 10  # 5 pairs

    def test_15_exchanges_no_trimming(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """15 exchanges stay intact with default budget."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        exchanges = []
        for i in range(15):
            exchanges.extend(_make_exchange_pair(i))

        session = make_session(exchanges=exchanges)
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=30, min_exchanges=2,
        )

        assert len(result.messages) == 30

    def test_trimming_with_small_budget(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Oldest exchanges are trimmed when budget is exceeded."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        # Very small budget to force trimming.
        cm = ContextManager(loader, token_budget=200)

        exchanges = []
        for i in range(10):
            exchanges.extend(_make_exchange_pair(i))

        session = make_session(exchanges=exchanges)
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=20, min_exchanges=2,
        )

        # Some messages should have been trimmed.
        assert len(result.messages) < 20
        # Trimmed in pairs — even count remaining.
        assert len(result.messages) % 2 == 0

    def test_trimming_preserves_newest(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Trimming removes oldest pairs, preserving newest."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader, token_budget=200)

        exchanges = []
        for i in range(10):
            exchanges.extend(_make_exchange_pair(i))

        session = make_session(exchanges=exchanges)
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=20, min_exchanges=2,
        )

        if result.messages:
            # The last message should be the newest trickster response.
            last = result.messages[-1]
            assert last["role"] == "assistant"
            assert "Trickster response 9" in last["content"]

    def test_system_prompt_survives_trimming(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """System prompt is never trimmed, only messages."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader, token_budget=200)

        exchanges = []
        for i in range(10):
            exchanges.extend(_make_exchange_pair(i))

        session = make_session(exchanges=exchanges)
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=20, min_exchanges=2,
        )

        # System prompt always present with all layers.
        assert "Test persona content." in result.system_prompt
        assert "Test behaviour content." in result.system_prompt
        assert "Test safety content." in result.system_prompt


# ---------------------------------------------------------------------------
# Prompt snapshotting
# ---------------------------------------------------------------------------


class TestPromptSnapshotting:
    def test_snapshot_stores_non_none_fields(
        self, tmp_path: Path, make_session,
    ) -> None:
        """snapshot_prompts() stores all non-None prompt fields."""
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        prompts = TricksterPrompts(
            persona="persona text",
            behaviour="behaviour text",
            safety="safety text",
            task_override="override text",
        )
        session = make_session()
        cm.snapshot_prompts(session, prompts)

        assert session.prompt_snapshots is not None
        assert session.prompt_snapshots["persona"] == "persona text"
        assert session.prompt_snapshots["behaviour"] == "behaviour text"
        assert session.prompt_snapshots["safety"] == "safety text"
        assert session.prompt_snapshots["task_override"] == "override text"

    def test_snapshot_skips_none_fields(
        self, tmp_path: Path, make_session,
    ) -> None:
        """snapshot_prompts() does not store None fields."""
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        prompts = TricksterPrompts(
            persona="persona text",
            behaviour=None,
            safety="safety text",
            task_override=None,
        )
        session = make_session()
        cm.snapshot_prompts(session, prompts)

        assert session.prompt_snapshots is not None
        assert "persona" in session.prompt_snapshots
        assert "behaviour" not in session.prompt_snapshots
        assert "safety" in session.prompt_snapshots
        assert "task_override" not in session.prompt_snapshots

    def test_get_snapshot_reconstructs_correctly(
        self, tmp_path: Path, make_session,
    ) -> None:
        """get_prompt_snapshot() reconstructs TricksterPrompts from snapshot."""
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        original = TricksterPrompts(
            persona="persona",
            behaviour="behaviour",
            safety="safety",
            task_override="override",
        )
        session = make_session()
        cm.snapshot_prompts(session, original)

        recovered = cm.get_prompt_snapshot(session)
        assert recovered is not None
        assert recovered.persona == "persona"
        assert recovered.behaviour == "behaviour"
        assert recovered.safety == "safety"
        assert recovered.task_override == "override"

    def test_get_snapshot_returns_none_when_absent(
        self, tmp_path: Path, make_session,
    ) -> None:
        """get_prompt_snapshot() returns None when no snapshot exists."""
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        assert cm.get_prompt_snapshot(session) is None

    def test_assembly_uses_snapshot_over_loader(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Assembly uses snapshotted prompts, not fresh loader content."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        # Take a snapshot with distinct content.
        snapshot_prompts = TricksterPrompts(
            persona="SNAPSHOT persona",
            behaviour="SNAPSHOT behaviour",
            safety="SNAPSHOT safety",
            task_override=None,
        )
        cm.snapshot_prompts(session, snapshot_prompts)

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Should see snapshot content, not loader content.
        assert "SNAPSHOT persona" in result.system_prompt
        assert "SNAPSHOT behaviour" in result.system_prompt
        assert "SNAPSHOT safety" in result.system_prompt
        assert "Test persona content." not in result.system_prompt

    def test_assembly_falls_back_to_loader(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Assembly uses loader when no snapshot exists."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "Test persona content." in result.system_prompt


# ---------------------------------------------------------------------------
# Context labels
# ---------------------------------------------------------------------------


class TestContextLabels:
    def test_labels_appear_in_system_prompt(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Choices with context_label appear in system prompt."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(
            choices=[
                {"context_label": "Pasirinko A"},
                {"context_label": "Pasirinko B"},
            ],
        )
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "Mokinio pasirinkimai" in result.system_prompt
        assert "Pasirinko A" in result.system_prompt
        assert "Pasirinko B" in result.system_prompt

    def test_choices_without_context_label_skipped(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Choices without context_label are not included."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(
            choices=[
                {"context_label": "Visible label"},
                {"target_phase": "next"},
            ],
        )
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "Visible label" in result.system_prompt
        assert "next" not in result.system_prompt


# ---------------------------------------------------------------------------
# Redaction context
# ---------------------------------------------------------------------------


class TestRedactionContext:
    def test_redaction_appears_in_system_prompt(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Redaction note appears when last_redaction_reason is set."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(last_redaction_reason="content_boundary_violation")
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "Sistemos pastaba" in result.system_prompt
        assert "content_boundary_violation" in result.system_prompt

    def test_redaction_flag_cleared_after_assembly(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Flag is cleared after being injected into the system prompt."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(last_redaction_reason="test_reason")
        cartridge = make_cartridge()

        cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert session.last_redaction_reason is None

    def test_no_redaction_when_flag_none(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """No redaction section when flag is None."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "Sistemos pastaba" not in result.system_prompt


# ---------------------------------------------------------------------------
# Debrief assembly
# ---------------------------------------------------------------------------


class TestDebriefAssembly:
    def test_debrief_includes_evaluation_data(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Debrief system prompt includes EvaluationContract data."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(
            exchanges=[
                _make_exchange("student", "Ar tai tikra?"),
                _make_exchange("trickster", "Taip."),
            ],
        )
        cartridge = make_cartridge()

        result = cm.assemble_debrief_call(session, cartridge, "gemini")

        # Patterns
        assert "Atskleidimo kontekstas" in result.system_prompt
        assert "headline_manipulation" in result.system_prompt
        # Checklist
        assert "Ko mokinys turejo pastebeti" in result.system_prompt
        # Pass conditions
        assert "Triksteris laimi" in result.system_prompt
        assert "Triksteris pralaimi" in result.system_prompt
        # Debrief instruction
        assert "Instrukcija" in result.system_prompt

    def test_debrief_full_exchange_history(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Debrief includes all exchanges (no trimming)."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        exchanges = []
        for i in range(15):
            exchanges.extend(_make_exchange_pair(i))

        session = make_session(exchanges=exchanges)
        cartridge = make_cartridge()

        result = cm.assemble_debrief_call(session, cartridge, "gemini")

        assert len(result.messages) == 30  # All 15 pairs

    def test_debrief_tools_none(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Debrief has no tools."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_debrief_call(session, cartridge, "gemini")

        assert result.tools is None

    def test_debrief_uses_snapshot(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Debrief uses prompt snapshot when available."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        snapshot = TricksterPrompts(
            persona="DEBRIEF SNAPSHOT persona",
            behaviour="DEBRIEF SNAPSHOT behaviour",
            safety="DEBRIEF SNAPSHOT safety",
            task_override=None,
        )
        cm.snapshot_prompts(session, snapshot)

        result = cm.assemble_debrief_call(session, cartridge, "gemini")

        assert "DEBRIEF SNAPSHOT persona" in result.system_prompt
        assert "Test persona content." not in result.system_prompt


# ---------------------------------------------------------------------------
# Transition tool
# ---------------------------------------------------------------------------


class TestTransitionTool:
    def test_tool_included_when_at_min_exchanges(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Tools include transition tool when exchange_count >= min_exchanges."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=2, min_exchanges=2,
        )

        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "transition_phase"

    def test_tool_included_when_above_min_exchanges(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Tools include transition tool when exchange_count > min_exchanges."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=5, min_exchanges=2,
        )

        assert result.tools is not None

    def test_tool_absent_when_below_min_exchanges(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Tools is None when exchange_count < min_exchanges."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert result.tools is None

    def test_transition_tool_schema(self) -> None:
        """TRANSITION_TOOL has correct schema structure."""
        assert TRANSITION_TOOL["name"] == "transition_phase"
        params = TRANSITION_TOOL["parameters"]
        assert params["type"] == "object"
        assert "signal" in params["properties"]
        assert set(params["properties"]["signal"]["enum"]) == {
            "understood", "partial", "max_reached",
        }
        assert params["required"] == ["signal"]


# ---------------------------------------------------------------------------
# Context levels (MVP stub)
# ---------------------------------------------------------------------------


class TestContextLevels:
    def test_session_only_works(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """session_only context requirement works normally."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert result.system_prompt  # No error

    def test_learning_profile_degrades_gracefully(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """learning_profile context resolves without error."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge(
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "test-ai-task-001",
                "persona_mode": "chat_participant",
                "has_static_fallback": False,
                "context_requirements": "learning_profile",
            },
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert result.system_prompt  # No error, works normally

    def test_full_history_degrades_gracefully(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """full_history context resolves without error."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge(
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "test-ai-task-001",
                "persona_mode": "chat_participant",
                "has_static_fallback": False,
                "context_requirements": "full_history",
            },
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert result.system_prompt  # No error, works normally


# ---------------------------------------------------------------------------
# Layer 5 content verification
# ---------------------------------------------------------------------------


class TestLayer5Content:
    def test_patterns_serialized(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """EmbeddedPattern fields appear in the system prompt."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Default cartridge has one pattern with these fields.
        assert "headline_manipulation" in result.system_prompt
        assert "naujien" in result.system_prompt.lower()  # real_world_connection

    def test_checklist_serialized(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """ChecklistItem fields appear, with mandatory marker."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Default cartridge checklist has is_mandatory=True.
        assert "PRIVALOMA" in result.system_prompt

    def test_pass_conditions_serialized(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """PassConditions appear in the system prompt."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "Triksteris laimi" in result.system_prompt
        assert "Triksteris pralaimi" in result.system_prompt


# ---------------------------------------------------------------------------
# Exchange pair coherence
# ---------------------------------------------------------------------------


class TestExchangePairCoherence:
    def test_trimming_removes_complete_pairs(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Trimming always removes complete student+trickster pairs."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader, token_budget=100)

        exchanges = []
        for i in range(20):
            exchanges.extend(_make_exchange_pair(i))

        session = make_session(exchanges=exchanges)
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=40, min_exchanges=2,
        )

        # Remaining messages should be in complete pairs.
        assert len(result.messages) % 2 == 0
        # First remaining message should be a user message.
        if result.messages:
            assert result.messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Mode layer assembly (Phase 1b)
# ---------------------------------------------------------------------------


class TestModeLayerAssembly:
    def test_mode_layer_between_persona_and_behaviour(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Mode behaviour content appears between persona and behaviour."""
        setup_base_prompts(tmp_path)
        write_prompt_file(
            tmp_path / "trickster" / "persona_presenting_base.md",
            "PRESENTING MODE CONTENT",
        )
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge(
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "test-ai-task-001",
                "persona_mode": "presenting",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        prompt = result.system_prompt
        assert "PRESENTING MODE CONTENT" in prompt
        # Verify ordering: persona < mode < behaviour
        persona_pos = prompt.index("Test persona content.")
        mode_pos = prompt.index("PRESENTING MODE CONTENT")
        behaviour_pos = prompt.index("Test behaviour content.")
        assert persona_pos < mode_pos < behaviour_pos

    def test_each_mode_produces_distinct_output(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Each of the 4 modes produces its own content in the prompt."""
        setup_base_prompts(tmp_path)
        modes = ["presenting", "chat_participant", "narrator", "commenter"]
        for mode in modes:
            write_prompt_file(
                tmp_path / "trickster" / f"persona_{mode}_base.md",
                f"MODE_{mode.upper()}_CONTENT",
            )
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        for mode in modes:
            session = make_session()
            cartridge = make_cartridge(
                ai_config={
                    "model_preference": "standard",
                    "prompt_directory": "test-ai-task-001",
                    "persona_mode": mode,
                    "has_static_fallback": False,
                    "context_requirements": "session_only",
                },
            )
            result = cm.assemble_trickster_call(
                session, cartridge, "gemini",
                exchange_count=1, min_exchanges=2,
            )
            assert f"MODE_{mode.upper()}_CONTENT" in result.system_prompt

    def test_mode_layer_absent_when_no_file(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """No mode layer when mode file does not exist on disk."""
        setup_base_prompts(tmp_path)
        # No mode file written
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge(
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "test-ai-task-001",
                "persona_mode": "presenting",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Persona and behaviour present, no error
        assert "Test persona content." in result.system_prompt
        assert "Test behaviour content." in result.system_prompt

    def test_mode_layer_absent_when_no_ai_config(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """No mode layer and no error for static cartridges (ai_config=None)."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge(ai_config=None)

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "Test persona content." in result.system_prompt

    def test_debrief_includes_mode_layer(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Mode behaviour also appears in the debrief system prompt."""
        setup_base_prompts(tmp_path)
        write_prompt_file(
            tmp_path / "trickster" / "persona_presenting_base.md",
            "DEBRIEF MODE CONTENT",
        )
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge(
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "test-ai-task-001",
                "persona_mode": "presenting",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )

        result = cm.assemble_debrief_call(session, cartridge, "gemini")

        assert "DEBRIEF MODE CONTENT" in result.system_prompt


# ---------------------------------------------------------------------------
# Mode snapshotting (Phase 1b)
# ---------------------------------------------------------------------------


class TestModeSnapshotting:
    def test_snapshot_includes_mode_behaviour(
        self, tmp_path: Path, make_session,
    ) -> None:
        """snapshot_prompts() stores mode_behaviour when non-None."""
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        prompts = TricksterPrompts(
            persona="persona",
            behaviour="behaviour",
            safety="safety",
            task_override=None,
            mode_behaviour="mode content",
        )
        session = make_session()
        cm.snapshot_prompts(session, prompts)

        assert session.prompt_snapshots is not None
        assert session.prompt_snapshots["mode_behaviour"] == "mode content"

    def test_snapshot_skips_mode_behaviour_when_none(
        self, tmp_path: Path, make_session,
    ) -> None:
        """snapshot_prompts() does not store mode_behaviour when None."""
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        prompts = TricksterPrompts(
            persona="persona",
            behaviour="behaviour",
            safety="safety",
            task_override=None,
        )
        session = make_session()
        cm.snapshot_prompts(session, prompts)

        assert "mode_behaviour" not in session.prompt_snapshots

    def test_snapshot_roundtrip_preserves_mode(
        self, tmp_path: Path, make_session,
    ) -> None:
        """Snapshot + get_prompt_snapshot roundtrip preserves mode_behaviour."""
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        original = TricksterPrompts(
            persona="persona",
            behaviour="behaviour",
            safety="safety",
            task_override=None,
            mode_behaviour="roundtrip mode",
        )
        session = make_session()
        cm.snapshot_prompts(session, original)

        recovered = cm.get_prompt_snapshot(session)
        assert recovered is not None
        assert recovered.mode_behaviour == "roundtrip mode"

    def test_old_snapshot_without_mode_reconstructs_none(
        self, tmp_path: Path, make_session,
    ) -> None:
        """Old snapshots (no mode_behaviour key) reconstruct with None."""
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        # Simulate an old snapshot dict without mode_behaviour
        session.prompt_snapshots = {
            "persona": "old persona",
            "behaviour": "old behaviour",
            "safety": "old safety",
        }

        recovered = cm.get_prompt_snapshot(session)
        assert recovered is not None
        assert recovered.mode_behaviour is None
        assert recovered.persona == "old persona"

    def test_assembly_uses_snapshotted_mode(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Assembly uses mode_behaviour from snapshot, not loader."""
        setup_base_prompts(tmp_path)
        # Write a mode file that should NOT be used (snapshot takes priority)
        write_prompt_file(
            tmp_path / "trickster" / "persona_chat_participant_base.md",
            "DISK MODE CONTENT",
        )
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge()

        snapshot = TricksterPrompts(
            persona="SNAPSHOT persona",
            behaviour="SNAPSHOT behaviour",
            safety="SNAPSHOT safety",
            task_override=None,
            mode_behaviour="SNAPSHOT MODE",
        )
        cm.snapshot_prompts(session, snapshot)

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "SNAPSHOT MODE" in result.system_prompt
        assert "DISK MODE CONTENT" not in result.system_prompt


# ---------------------------------------------------------------------------
# Mode resolution (Phase 1b)
# ---------------------------------------------------------------------------


class TestModeResolution:
    def test_resolve_prompts_passes_mode_to_loader(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """_resolve_prompts passes persona_mode to loader (indirect test)."""
        setup_base_prompts(tmp_path)
        write_prompt_file(
            tmp_path / "trickster" / "persona_narrator_base.md",
            "NARRATOR LOADED",
        )
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge(
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "test-ai-task-001",
                "persona_mode": "narrator",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "NARRATOR LOADED" in result.system_prompt

    def test_resolve_uses_snapshot_mode_over_loader(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Snapshot mode takes priority over loader mode."""
        setup_base_prompts(tmp_path)
        write_prompt_file(
            tmp_path / "trickster" / "persona_narrator_base.md",
            "LOADER NARRATOR",
        )
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session()
        cartridge = make_cartridge(
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "test-ai-task-001",
                "persona_mode": "narrator",
                "has_static_fallback": False,
                "context_requirements": "session_only",
            },
        )

        snapshot = TricksterPrompts(
            persona="SNAP persona",
            behaviour="SNAP behaviour",
            safety="SNAP safety",
            task_override=None,
            mode_behaviour="SNAP NARRATOR",
        )
        cm.snapshot_prompts(session, snapshot)

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        assert "SNAP NARRATOR" in result.system_prompt
        assert "LOADER NARRATOR" not in result.system_prompt


# ---------------------------------------------------------------------------
# Multimodal image assembly (Phase 2c)
# ---------------------------------------------------------------------------


def _setup_image_content(
    content_dir: Path,
    task_id: str,
    filename: str,
    data: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
) -> Path:
    """Creates a dummy image file in the content/tasks/{task_id}/assets/ dir."""
    assets_dir = content_dir / "tasks" / task_id / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    img_path = assets_dir / filename
    img_path.write_bytes(data)
    return img_path


def _make_image_cartridge(make_cartridge, **overrides):
    """Creates a cartridge with ImageBlock in visible_blocks on the AI phase."""
    defaults = {
        "presentation_blocks": [
            {
                "id": "img1",
                "type": "image",
                "src": "photo1.png",
                "alt_text": "Nuotrauka 1",
            },
            {
                "id": "img2",
                "type": "image",
                "src": "photo2.png",
                "alt_text": "Nuotrauka 2",
            },
        ],
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
                            "context_label": "Mokinys pasirinko prad\u0117ti",
                        },
                    ],
                },
            },
            {
                "id": "phase_ai",
                "title": "Pokalbis",
                "is_ai_phase": True,
                "visible_blocks": ["img1", "img2"],
                "interaction": {
                    "type": "freeform",
                    "trickster_opening": "Pažiūrėkime į nuotraukas...",
                    "min_exchanges": 2,
                    "max_exchanges": 10,
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
    }
    defaults.update(overrides)
    return make_cartridge(**defaults)


class TestMultimodalAssembly:
    """Tests for ContextManager multimodal image extraction and assembly."""

    def test_image_blocks_produce_multimodal_message(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Cartridge with ImageBlocks in visible_blocks produces multimodal first message."""
        setup_base_prompts(tmp_path / "prompts")
        content_dir = tmp_path / "content"
        img_data = b"\x89PNG_test_image_data"
        _setup_image_content(content_dir, "test-ai-task-001", "photo1.png", img_data)
        _setup_image_content(content_dir, "test-ai-task-001", "photo2.png", img_data)

        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader, content_dir=content_dir)

        cartridge = _make_image_cartridge(make_cartridge)
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # First message should be multimodal (image context).
        first_msg = result.messages[0]
        assert first_msg["role"] == "user"
        assert isinstance(first_msg["content"], list)

        # Should have: 1 text label + 2 image parts = 3 parts.
        parts = first_msg["content"]
        assert len(parts) == 3

        # First part is the label.
        assert parts[0]["type"] == "text"
        assert "vaizdin" in parts[0]["text"].lower()

        # Image parts contain correct base64 data.
        expected_b64 = base64.b64encode(img_data).decode()
        assert parts[1]["type"] == "image"
        assert parts[1]["media_type"] == "image/png"
        assert parts[1]["data"] == expected_b64
        assert parts[2]["type"] == "image"
        assert parts[2]["data"] == expected_b64

    def test_media_type_derived_from_extension(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Media type is correctly derived from file extension."""
        setup_base_prompts(tmp_path / "prompts")
        content_dir = tmp_path / "content"
        _setup_image_content(content_dir, "test-ai-task-001", "photo1.jpg")

        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader, content_dir=content_dir)

        cartridge = _make_image_cartridge(
            make_cartridge,
            presentation_blocks=[
                {"id": "img1", "type": "image", "src": "photo1.jpg", "alt_text": "JPG"},
            ],
            phases=[
                {
                    "id": "phase_intro", "title": "I", "is_ai_phase": False,
                    "interaction": {"type": "button", "choices": [
                        {"label": "Go", "target_phase": "phase_ai"},
                    ]},
                },
                {
                    "id": "phase_ai", "title": "AI", "is_ai_phase": True,
                    "visible_blocks": ["img1"],
                    "interaction": {
                        "type": "freeform", "trickster_opening": "...",
                        "min_exchanges": 2, "max_exchanges": 10,
                    },
                    "ai_transitions": {
                        "on_success": "phase_reveal",
                        "on_max_exchanges": "phase_reveal",
                        "on_partial": "phase_reveal",
                    },
                },
                {"id": "phase_reveal", "title": "R", "is_terminal": True,
                 "evaluation_outcome": "trickster_loses"},
            ],
        )
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )
        img_part = result.messages[0]["content"][1]
        assert img_part["media_type"] == "image/jpeg"

    def test_multiple_images_in_single_message(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Multiple images are all packed into one multimodal message."""
        setup_base_prompts(tmp_path / "prompts")
        content_dir = tmp_path / "content"
        _setup_image_content(content_dir, "test-ai-task-001", "photo1.png", b"img1")
        _setup_image_content(content_dir, "test-ai-task-001", "photo2.png", b"img2")

        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader, content_dir=content_dir)

        cartridge = _make_image_cartridge(make_cartridge)
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Only one multimodal message, not one per image.
        multimodal_msgs = [
            m for m in result.messages
            if isinstance(m.get("content"), list)
        ]
        assert len(multimodal_msgs) == 1

        # Two distinct base64 values.
        image_parts = [
            p for p in multimodal_msgs[0]["content"]
            if p.get("type") == "image"
        ]
        assert len(image_parts) == 2
        assert image_parts[0]["data"] != image_parts[1]["data"]


class TestMultimodalMemeBlock:
    """Tests for MemeBlock image extraction with text overlay."""

    def test_meme_block_produces_image_and_text(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """MemeBlock produces image part + text overlay part."""
        setup_base_prompts(tmp_path / "prompts")
        content_dir = tmp_path / "content"
        _setup_image_content(content_dir, "test-ai-task-001", "meme.png")

        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader, content_dir=content_dir)

        cartridge = _make_image_cartridge(
            make_cartridge,
            presentation_blocks=[
                {
                    "id": "meme1",
                    "type": "meme",
                    "image_src": "meme.png",
                    "top_text": "WHEN YOU",
                    "bottom_text": "SEE IT",
                    "alt_text": "Meme",
                },
            ],
            phases=[
                {
                    "id": "phase_intro", "title": "I", "is_ai_phase": False,
                    "interaction": {"type": "button", "choices": [
                        {"label": "Go", "target_phase": "phase_ai"},
                    ]},
                },
                {
                    "id": "phase_ai", "title": "AI", "is_ai_phase": True,
                    "visible_blocks": ["meme1"],
                    "interaction": {
                        "type": "freeform", "trickster_opening": "...",
                        "min_exchanges": 2, "max_exchanges": 10,
                    },
                    "ai_transitions": {
                        "on_success": "phase_reveal",
                        "on_max_exchanges": "phase_reveal",
                        "on_partial": "phase_reveal",
                    },
                },
                {"id": "phase_reveal", "title": "R", "is_terminal": True,
                 "evaluation_outcome": "trickster_loses"},
            ],
        )
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        first_msg = result.messages[0]
        parts = first_msg["content"]

        # Label + image + text overlay = 3 parts.
        assert len(parts) == 3
        assert parts[1]["type"] == "image"
        assert "WHEN YOU" in parts[2]["text"]
        assert "SEE IT" in parts[2]["text"]


class TestMultimodalBackwardCompat:
    """Backward compatibility: text-only cartridges produce identical output."""

    def test_no_image_blocks_no_multimodal(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Cartridge with no image blocks produces text-only messages."""
        setup_base_prompts(tmp_path / "prompts")
        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader, content_dir=tmp_path / "content")

        cartridge = make_cartridge()
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
            exchanges=[
                _make_exchange("student", "Hello"),
                _make_exchange("trickster", "Hi"),
            ],
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=2, min_exchanges=2,
        )

        # All messages should have string content.
        for msg in result.messages:
            assert isinstance(msg["content"], str)

    def test_content_dir_none_skips_images(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """content_dir=None skips image resolution even with image blocks."""
        setup_base_prompts(tmp_path / "prompts")
        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader)  # No content_dir.

        cartridge = _make_image_cartridge(make_cartridge)
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # No multimodal messages.
        for msg in result.messages:
            assert isinstance(msg["content"], str)


class TestMultimodalGracefulFallback:
    """Graceful degradation when images are missing or unreadable."""

    def test_missing_image_skipped_with_warning(
        self, tmp_path: Path, make_session, make_cartridge, caplog,
    ) -> None:
        """Missing image file logs warning and skips that image."""
        setup_base_prompts(tmp_path / "prompts")
        content_dir = tmp_path / "content"
        # Create only photo1.png, NOT photo2.png.
        _setup_image_content(content_dir, "test-ai-task-001", "photo1.png")

        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader, content_dir=content_dir)

        cartridge = _make_image_cartridge(make_cartridge)
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )

        import logging
        with caplog.at_level(logging.WARNING):
            result = cm.assemble_trickster_call(
                session, cartridge, "gemini",
                exchange_count=1, min_exchanges=2,
            )

        # Warning logged for missing file.
        assert any("photo2.png" in r.message for r in caplog.records)

        # Still has multimodal message with the one image that exists.
        first_msg = result.messages[0]
        assert isinstance(first_msg["content"], list)
        image_parts = [
            p for p in first_msg["content"] if p.get("type") == "image"
        ]
        assert len(image_parts) == 1

    def test_all_images_missing_falls_back_to_text_only(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """When all images are missing, no multimodal message is prepended."""
        setup_base_prompts(tmp_path / "prompts")
        content_dir = tmp_path / "content"
        # Create task dir but no image files.
        (content_dir / "tasks" / "test-ai-task-001" / "assets").mkdir(
            parents=True, exist_ok=True,
        )

        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader, content_dir=content_dir)

        cartridge = _make_image_cartridge(make_cartridge)
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # No multimodal messages — all text-only.
        for msg in result.messages:
            assert isinstance(msg["content"], str)


class TestMultimodalPhaseResolution:
    """Phase resolution edge cases for image extraction."""

    def test_current_phase_none_no_images(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """session.current_phase=None produces no images."""
        setup_base_prompts(tmp_path / "prompts")
        content_dir = tmp_path / "content"
        _setup_image_content(content_dir, "test-ai-task-001", "photo1.png")

        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader, content_dir=content_dir)

        cartridge = _make_image_cartridge(make_cartridge)
        session = make_session(
            current_task="test-ai-task-001",
            current_phase=None,
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        for msg in result.messages:
            assert isinstance(msg["content"], str)

    def test_phase_without_visible_blocks_no_images(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Phase with empty visible_blocks produces no images."""
        setup_base_prompts(tmp_path / "prompts")
        content_dir = tmp_path / "content"
        _setup_image_content(content_dir, "test-ai-task-001", "photo1.png")

        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader, content_dir=content_dir)

        # Default cartridge has no visible_blocks on the AI phase.
        cartridge = make_cartridge()
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        for msg in result.messages:
            assert isinstance(msg["content"], str)

    def test_visible_blocks_with_only_text_blocks(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Phase with only text blocks in visible_blocks produces no images."""
        setup_base_prompts(tmp_path / "prompts")
        loader = PromptLoader(tmp_path / "prompts")
        cm = ContextManager(loader, content_dir=tmp_path / "content")

        cartridge = _make_image_cartridge(
            make_cartridge,
            presentation_blocks=[
                {"id": "txt1", "type": "text", "text": "Some article text"},
            ],
            phases=[
                {
                    "id": "phase_intro", "title": "I", "is_ai_phase": False,
                    "interaction": {"type": "button", "choices": [
                        {"label": "Go", "target_phase": "phase_ai"},
                    ]},
                },
                {
                    "id": "phase_ai", "title": "AI", "is_ai_phase": True,
                    "visible_blocks": ["txt1"],
                    "interaction": {
                        "type": "freeform", "trickster_opening": "...",
                        "min_exchanges": 2, "max_exchanges": 10,
                    },
                    "ai_transitions": {
                        "on_success": "phase_reveal",
                        "on_max_exchanges": "phase_reveal",
                        "on_partial": "phase_reveal",
                    },
                },
                {"id": "phase_reveal", "title": "R", "is_terminal": True,
                 "evaluation_outcome": "trickster_loses"},
            ],
        )
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        for msg in result.messages:
            assert isinstance(msg["content"], str)


class TestMultimodalTokenBudgeting:
    """Token budgeting with multimodal image messages."""

    def test_image_token_cost_accounted(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Exchange trimming triggers earlier when images are present."""
        setup_base_prompts(tmp_path / "prompts")
        content_dir = tmp_path / "content"
        _setup_image_content(content_dir, "test-ai-task-001", "photo1.png")
        _setup_image_content(content_dir, "test-ai-task-001", "photo2.png")

        loader = PromptLoader(tmp_path / "prompts")

        # Use a tight budget that barely fits system prompt + images + a few exchanges.
        # Each image = 258 tokens. Two images = 516 tokens.
        # Text label ~15 chars / 3 = ~5 tokens. Total image msg ~521 tokens.
        # System prompt is ~200 chars / 3 = ~67 tokens.
        # Budget of 700 leaves ~112 tokens for exchanges.
        cm = ContextManager(loader, token_budget=700, content_dir=content_dir)

        # Create many exchanges that exceed the remaining budget.
        exchanges = []
        for i in range(10):
            exchanges.extend(_make_exchange_pair(i))

        cartridge = _make_image_cartridge(make_cartridge)
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
            exchanges=exchanges,
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=20, min_exchanges=2,
        )

        # First message should be multimodal (image context, never trimmed).
        assert isinstance(result.messages[0]["content"], list)

        # Exchanges should be trimmed — fewer than original 20.
        exchange_msgs = [
            m for m in result.messages
            if isinstance(m.get("content"), str)
        ]
        assert len(exchange_msgs) < 20

    def test_image_context_message_never_trimmed(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Image context message survives trimming even with tiny budget."""
        setup_base_prompts(tmp_path / "prompts")
        content_dir = tmp_path / "content"
        _setup_image_content(content_dir, "test-ai-task-001", "photo1.png")

        loader = PromptLoader(tmp_path / "prompts")
        # Very small budget — forces aggressive trimming.
        cm = ContextManager(loader, token_budget=50, content_dir=content_dir)

        cartridge = _make_image_cartridge(
            make_cartridge,
            presentation_blocks=[
                {"id": "img1", "type": "image", "src": "photo1.png",
                 "alt_text": "Photo"},
            ],
            phases=[
                {
                    "id": "phase_intro", "title": "I", "is_ai_phase": False,
                    "interaction": {"type": "button", "choices": [
                        {"label": "Go", "target_phase": "phase_ai"},
                    ]},
                },
                {
                    "id": "phase_ai", "title": "AI", "is_ai_phase": True,
                    "visible_blocks": ["img1"],
                    "interaction": {
                        "type": "freeform", "trickster_opening": "...",
                        "min_exchanges": 2, "max_exchanges": 10,
                    },
                    "ai_transitions": {
                        "on_success": "phase_reveal",
                        "on_max_exchanges": "phase_reveal",
                        "on_partial": "phase_reveal",
                    },
                },
                {"id": "phase_reveal", "title": "R", "is_terminal": True,
                 "evaluation_outcome": "trickster_loses"},
            ],
        )

        exchanges = []
        for i in range(5):
            exchanges.extend(_make_exchange_pair(i))

        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
            exchanges=exchanges,
        )

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=10, min_exchanges=2,
        )

        # Image context message survives — always first.
        assert isinstance(result.messages[0]["content"], list)
        image_parts = [
            p for p in result.messages[0]["content"]
            if p.get("type") == "image"
        ]
        assert len(image_parts) == 1

    def test_estimate_message_tokens_text_only(self) -> None:
        """Token estimation for text-only message."""
        tokens = ContextManager._estimate_message_tokens(
            {"role": "user", "content": "aaa"}
        )
        assert tokens == 1.0  # 3 chars / 3 chars_per_token

    def test_estimate_message_tokens_multimodal(self) -> None:
        """Token estimation for multimodal message with text + images."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "aaaaaa"},   # 6 chars / 3 = 2 tokens
                {"type": "image", "media_type": "image/png", "data": "..."},
                {"type": "image", "media_type": "image/png", "data": "..."},
            ],
        }
        tokens = ContextManager._estimate_message_tokens(msg)
        assert tokens == 2.0 + 2 * _TOKENS_PER_IMAGE


# ---------------------------------------------------------------------------
# De-escalation context injection
# ---------------------------------------------------------------------------


class TestDeescalationContext:
    """Tests for _build_deescalation_context and its wiring."""

    def test_no_intensities_returns_none(
        self, make_session, make_cartridge,
    ) -> None:
        """Fresh session with empty turn_intensities produces no de-escalation."""
        session = make_session()
        cartridge = make_cartridge()
        assert session.turn_intensities == []
        result = ContextManager._build_deescalation_context(session, cartridge)
        assert result is None

    def test_last_score_below_ceiling_returns_none(
        self, make_session, make_cartridge,
    ) -> None:
        """Last intensity below ceiling produces no de-escalation."""
        session = make_session()
        session.turn_intensities = [2.5]
        cartridge = make_cartridge()  # intensity_ceiling=3
        result = ContextManager._build_deescalation_context(session, cartridge)
        assert result is None

    def test_last_score_equal_ceiling_returns_none(
        self, make_session, make_cartridge,
    ) -> None:
        """Last intensity exactly at ceiling produces no de-escalation."""
        session = make_session()
        session.turn_intensities = [3.0]
        cartridge = make_cartridge()  # intensity_ceiling=3
        result = ContextManager._build_deescalation_context(session, cartridge)
        assert result is None

    def test_last_score_above_ceiling_returns_text(
        self, make_session, make_cartridge,
    ) -> None:
        """Last intensity above ceiling produces de-escalation instructions."""
        session = make_session()
        session.turn_intensities = [3.5]
        cartridge = make_cartridge()  # intensity_ceiling=3
        result = ContextManager._build_deescalation_context(session, cartridge)
        assert result is not None
        assert "De-eskalacijos instrukcija" in result

    def test_only_last_score_matters(
        self, make_session, make_cartridge,
    ) -> None:
        """Earlier hot turns don't trigger de-escalation if last is cool."""
        session = make_session()
        session.turn_intensities = [3.5, 2.0]
        cartridge = make_cartridge()  # intensity_ceiling=3
        result = ContextManager._build_deescalation_context(session, cartridge)
        assert result is None

    def test_ceiling_value_in_text(
        self, make_session, make_cartridge,
    ) -> None:
        """De-escalation text includes the ceiling value."""
        session = make_session()
        session.turn_intensities = [4.0]
        cartridge = make_cartridge()  # intensity_ceiling=3
        result = ContextManager._build_deescalation_context(session, cartridge)
        assert result is not None
        assert "3/5" in result

    def test_deescalation_in_assembled_prompt(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """De-escalation appears in assembled system prompt when active."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(
            current_phase="phase_ai",
            exchanges=[
                _make_exchange("student", "Test"),
                _make_exchange("trickster", "Response"),
            ],
        )
        session.turn_intensities = [4.0]
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=2, min_exchanges=2,
        )
        assert "De-eskalacijos instrukcija" in result.system_prompt

    def test_no_deescalation_when_below_ceiling(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """No de-escalation in assembled prompt when score is within bounds."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(
            current_phase="phase_ai",
            exchanges=[
                _make_exchange("student", "Test"),
                _make_exchange("trickster", "Response"),
            ],
        )
        session.turn_intensities = [2.0]
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=2, min_exchanges=2,
        )
        assert "De-eskalacijos instrukcija" not in result.system_prompt

    def test_deescalation_position_between_task_and_safety(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """De-escalation text appears after task context, before safety config."""
        setup_base_prompts(tmp_path)
        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        session = make_session(
            current_phase="phase_ai",
            exchanges=[
                _make_exchange("student", "Test"),
                _make_exchange("trickster", "Response"),
            ],
        )
        session.turn_intensities = [4.5]
        cartridge = make_cartridge()

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=2, min_exchanges=2,
        )
        prompt = result.system_prompt
        task_pos = prompt.index("Uzduoties kontekstas")
        deesc_pos = prompt.index("De-eskalacijos instrukcija")
        safety_pos = prompt.index("Saugumo nustatymai")
        assert task_pos < deesc_pos < safety_pos
