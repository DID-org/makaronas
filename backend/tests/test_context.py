"""Tests for the context manager — assembly, budgeting, snapshotting."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.ai.context import (
    TRANSITION_TOOL,
    AssembledContext,
    ContextManager,
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
