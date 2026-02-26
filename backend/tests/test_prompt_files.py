"""Integration tests for Phase 3b prompt files.

Verifies that all Trickster prompt files load correctly via PromptLoader,
contain meaningful Lithuanian content, and pass validation for reference
cartridges. These tests read real files on disk — no mocking.
"""

from pathlib import Path

import pytest

from backend.ai.prompts import PromptLoader
from backend.config import PROJECT_ROOT


@pytest.fixture
def loader() -> PromptLoader:
    """Creates a PromptLoader pointed at the real prompts directory."""
    return PromptLoader(PROJECT_ROOT / "prompts")


# ---------------------------------------------------------------------------
# Base prompts load
# ---------------------------------------------------------------------------


class TestBasePromptsLoad:
    """All three base Trickster prompts load as non-None strings."""

    def test_persona_loads(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.persona is not None
        assert isinstance(prompts.persona, str)

    def test_behaviour_loads(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.behaviour is not None
        assert isinstance(prompts.behaviour, str)

    def test_safety_loads(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.safety is not None
        assert isinstance(prompts.safety, str)

    def test_no_task_override_for_base(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.task_override is None


# ---------------------------------------------------------------------------
# Task overrides load
# ---------------------------------------------------------------------------


class TestTaskOverridesLoad:
    """Task-specific overrides load for both reference tasks."""

    def test_clickbait_trap_override(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts(
            "gemini", "task-clickbait-trap-001"
        )
        assert prompts.task_override is not None
        assert isinstance(prompts.task_override, str)

    def test_follow_money_override(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts(
            "gemini", "task-follow-money-001"
        )
        assert prompts.task_override is not None
        assert isinstance(prompts.task_override, str)

    def test_clickbait_still_has_base(self, loader: PromptLoader) -> None:
        """Task override doesn't break base prompt loading."""
        prompts = loader.load_trickster_prompts(
            "gemini", "task-clickbait-trap-001"
        )
        assert prompts.persona is not None
        assert prompts.behaviour is not None
        assert prompts.safety is not None

    def test_follow_money_still_has_base(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts(
            "gemini", "task-follow-money-001"
        )
        assert prompts.persona is not None
        assert prompts.behaviour is not None
        assert prompts.safety is not None


# ---------------------------------------------------------------------------
# Lithuanian character integrity
# ---------------------------------------------------------------------------


class TestLithuanianCharacters:
    """Lithuanian characters survive the load cycle."""

    # Lithuanian-specific characters to check for
    _LT_CHARS = [
        "\u0105",  # ą
        "\u0161",  # š
        "\u017e",  # ž
        "\u0173",  # ų
        "\u0117",  # ė
        "\u012f",  # į
        "\u016b",  # ū
        "\u010d",  # č
    ]

    _LT_QUOTES = [
        "\u201e",  # „ (opening)
        "\u201c",  # " (closing)
    ]

    def test_persona_contains_lt_chars(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.persona is not None
        for char in self._LT_CHARS:
            assert char in prompts.persona, (
                f"Lithuanian char U+{ord(char):04X} missing from persona"
            )

    def test_persona_contains_lt_quotes(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.persona is not None
        for char in self._LT_QUOTES:
            assert char in prompts.persona, (
                f"Lithuanian quote U+{ord(char):04X} missing from persona"
            )

    def test_behaviour_contains_lt_chars(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.behaviour is not None
        # Check a subset — behaviour has Lithuanian examples
        assert "\u0105" in prompts.behaviour  # ą
        assert "\u0161" in prompts.behaviour  # š

    def test_safety_contains_lt_chars(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.safety is not None
        assert "\u0105" in prompts.safety  # ą
        assert "\u0117" in prompts.safety  # ė

    def test_clickbait_override_contains_lt_chars(
        self, loader: PromptLoader
    ) -> None:
        prompts = loader.load_trickster_prompts(
            "gemini", "task-clickbait-trap-001"
        )
        assert prompts.task_override is not None
        assert "\u0161" in prompts.task_override  # š
        assert "\u201e" in prompts.task_override  # „

    def test_follow_money_override_contains_lt_chars(
        self, loader: PromptLoader
    ) -> None:
        prompts = loader.load_trickster_prompts(
            "gemini", "task-follow-money-001"
        )
        assert prompts.task_override is not None
        assert "\u017e" in prompts.task_override  # ž
        assert "\u201e" in prompts.task_override  # „


# ---------------------------------------------------------------------------
# Non-empty content
# ---------------------------------------------------------------------------


class TestNonEmptyContent:
    """Each prompt has meaningful content (>50 characters)."""

    _MIN_LENGTH = 50

    def test_persona_meaningful(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.persona is not None
        assert len(prompts.persona) > self._MIN_LENGTH

    def test_behaviour_meaningful(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.behaviour is not None
        assert len(prompts.behaviour) > self._MIN_LENGTH

    def test_safety_meaningful(self, loader: PromptLoader) -> None:
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.safety is not None
        assert len(prompts.safety) > self._MIN_LENGTH

    def test_clickbait_override_meaningful(
        self, loader: PromptLoader
    ) -> None:
        prompts = loader.load_trickster_prompts(
            "gemini", "task-clickbait-trap-001"
        )
        assert prompts.task_override is not None
        assert len(prompts.task_override) > self._MIN_LENGTH

    def test_follow_money_override_meaningful(
        self, loader: PromptLoader
    ) -> None:
        prompts = loader.load_trickster_prompts(
            "gemini", "task-follow-money-001"
        )
        assert prompts.task_override is not None
        assert len(prompts.task_override) > self._MIN_LENGTH


# ---------------------------------------------------------------------------
# Content checks — key elements present
# ---------------------------------------------------------------------------


class TestContentElements:
    """Prompt files contain expected key elements."""

    def test_persona_has_makaronas(self, loader: PromptLoader) -> None:
        """Persona defines the Trickster's name."""
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.persona is not None
        assert "Makaronas" in prompts.persona

    def test_behaviour_has_transition_phase(
        self, loader: PromptLoader
    ) -> None:
        """Behaviour references the transition_phase tool."""
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.behaviour is not None
        assert "transition_phase" in prompts.behaviour

    def test_behaviour_has_rubric_review(self, loader: PromptLoader) -> None:
        """Behaviour includes rubric/checklist review instruction."""
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.behaviour is not None
        # Check for the Lithuanian term for checklist
        assert "checklist" in prompts.behaviour or "kontrolin" in prompts.behaviour

    def test_safety_has_boundaries(self, loader: PromptLoader) -> None:
        """Safety file defines content boundaries."""
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.safety is not None
        # Should mention forbidden content categories
        assert "savi\u017eal" in prompts.safety  # savižal (self-harm)

    def test_clickbait_references_patterns(
        self, loader: PromptLoader
    ) -> None:
        """Clickbait override references the 4 embedded patterns."""
        prompts = loader.load_trickster_prompts(
            "gemini", "task-clickbait-trap-001"
        )
        assert prompts.task_override is not None
        assert "p-emotional-lang" in prompts.task_override
        assert "p-urgency" in prompts.task_override
        assert "p-headline-contradiction" in prompts.task_override
        assert "p-snippet-framing" in prompts.task_override

    def test_follow_money_references_patterns(
        self, loader: PromptLoader
    ) -> None:
        """Follow the Money override references the 4 embedded patterns."""
        prompts = loader.load_trickster_prompts(
            "gemini", "task-follow-money-001"
        )
        assert prompts.task_override is not None
        assert "p-selective-framing" in prompts.task_override
        assert "p-omission" in prompts.task_override
        assert "p-financial-incentive-a" in prompts.task_override
        assert "p-financial-incentive-b" in prompts.task_override

    def test_follow_money_references_chains(
        self, loader: PromptLoader
    ) -> None:
        """Follow the Money override describes both financial chains."""
        prompts = loader.load_trickster_prompts(
            "gemini", "task-follow-money-001"
        )
        assert prompts.task_override is not None
        assert "TrailBound" in prompts.task_override
        assert "Harland Ventures" in prompts.task_override
        assert "NovaTech" in prompts.task_override


# ---------------------------------------------------------------------------
# Validation for reference cartridges
# ---------------------------------------------------------------------------


class TestValidation:
    """validate_task_prompts() returns no errors for reference cartridges."""

    def test_clickbait_cartridge_validates(
        self, loader: PromptLoader, make_cartridge
    ) -> None:
        """No validation errors for Clickbait Trap cartridge."""
        cartridge = make_cartridge(task_id="task-clickbait-trap-001")
        errors = loader.validate_task_prompts(cartridge)
        assert errors == []

    def test_follow_money_cartridge_validates(
        self, loader: PromptLoader, make_cartridge
    ) -> None:
        """No validation errors for Follow the Money cartridge."""
        cartridge = make_cartridge(task_id="task-follow-money-001")
        errors = loader.validate_task_prompts(cartridge)
        assert errors == []
