"""Tests for the prompt loader — fallback chain, caching, validation, UTF-8."""

from pathlib import Path

import pytest

from backend.ai.prompts import PromptLoader, TricksterPrompts
from backend.tests.conftest import setup_base_prompts, write_prompt_file


# ---------------------------------------------------------------------------
# TricksterPrompts dataclass
# ---------------------------------------------------------------------------


class TestTricksterPrompts:
    def test_frozen(self) -> None:
        """TricksterPrompts is immutable."""
        tp = TricksterPrompts(
            persona="a", behaviour="b", safety="c", task_override=None
        )
        with pytest.raises(AttributeError):
            tp.persona = "changed"  # type: ignore[misc]

    def test_all_none(self) -> None:
        """All fields can be None."""
        tp = TricksterPrompts(
            persona=None, behaviour=None, safety=None, task_override=None
        )
        assert tp.persona is None
        assert tp.behaviour is None
        assert tp.safety is None
        assert tp.task_override is None


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_model_specific_preferred_over_base(self, tmp_path: Path) -> None:
        """When model-specific file exists, it's returned instead of base."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_base.md", "base persona")
        write_prompt_file(trickster / "persona_gemini.md", "gemini persona")
        write_prompt_file(trickster / "behaviour_base.md", "base behaviour")
        write_prompt_file(trickster / "safety_base.md", "base safety")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.persona == "gemini persona"
        assert result.behaviour == "base behaviour"
        assert result.safety == "base safety"

    def test_base_used_when_no_model_specific(self, tmp_path: Path) -> None:
        """Falls back to base when model-specific file is absent."""
        setup_base_prompts(tmp_path)

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.persona == "Test persona content."
        assert result.behaviour == "Test behaviour content."
        assert result.safety == "Test safety content."

    def test_none_when_neither_exists(self, tmp_path: Path) -> None:
        """Returns None for a prompt type when no file exists at all."""
        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.persona is None
        assert result.behaviour is None
        assert result.safety is None
        assert result.task_override is None

    def test_empty_model_specific_falls_back_to_base(self, tmp_path: Path) -> None:
        """Empty model-specific file is treated as absent — falls back to base."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_gemini.md", "   ")  # whitespace only
        write_prompt_file(trickster / "persona_base.md", "base persona")
        write_prompt_file(trickster / "behaviour_base.md", "base behaviour")
        write_prompt_file(trickster / "safety_base.md", "base safety")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.persona == "base persona"

    def test_empty_base_returns_none(self, tmp_path: Path) -> None:
        """Empty base file (no model-specific) returns None."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_base.md", "")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.persona is None


# ---------------------------------------------------------------------------
# Provider suffix mapping
# ---------------------------------------------------------------------------


class TestProviderSuffix:
    def test_anthropic_uses_claude_suffix(self, tmp_path: Path) -> None:
        """Anthropic provider looks for _claude.md files."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_claude.md", "claude persona")
        write_prompt_file(trickster / "behaviour_base.md", "base behaviour")
        write_prompt_file(trickster / "safety_base.md", "base safety")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("anthropic")

        assert result.persona == "claude persona"

    def test_gemini_uses_gemini_suffix(self, tmp_path: Path) -> None:
        """Gemini provider looks for _gemini.md files."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_gemini.md", "gemini persona")
        write_prompt_file(trickster / "behaviour_base.md", "base behaviour")
        write_prompt_file(trickster / "safety_base.md", "base safety")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.persona == "gemini persona"

    def test_unknown_provider_uses_base_only(self, tmp_path: Path) -> None:
        """Unknown provider skips model-specific, loads base only."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_base.md", "base persona")
        write_prompt_file(trickster / "persona_gemini.md", "gemini persona")
        write_prompt_file(trickster / "behaviour_base.md", "base behaviour")
        write_prompt_file(trickster / "safety_base.md", "base safety")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("openai")

        assert result.persona == "base persona"


# ---------------------------------------------------------------------------
# Task override loading
# ---------------------------------------------------------------------------


class TestTaskOverride:
    def test_task_override_loaded(self, tmp_path: Path) -> None:
        """Task-specific trickster override is loaded from tasks/{task_id}/."""
        setup_base_prompts(tmp_path)
        task_dir = tmp_path / "tasks" / "task-test-001"
        write_prompt_file(task_dir / "trickster_base.md", "task override content")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini", task_id="task-test-001")

        assert result.task_override == "task override content"

    def test_task_override_model_specific(self, tmp_path: Path) -> None:
        """Model-specific task override preferred over base."""
        setup_base_prompts(tmp_path)
        task_dir = tmp_path / "tasks" / "task-test-001"
        write_prompt_file(task_dir / "trickster_base.md", "base override")
        write_prompt_file(task_dir / "trickster_gemini.md", "gemini override")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini", task_id="task-test-001")

        assert result.task_override == "gemini override"

    def test_no_task_override_returns_none(self, tmp_path: Path) -> None:
        """Missing task override directory results in task_override=None."""
        setup_base_prompts(tmp_path)

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini", task_id="task-nonexistent")

        assert result.task_override is None

    def test_no_task_id_skips_override(self, tmp_path: Path) -> None:
        """When task_id is None, task_override is always None."""
        setup_base_prompts(tmp_path)
        task_dir = tmp_path / "tasks" / "task-test-001"
        write_prompt_file(task_dir / "trickster_base.md", "should not load")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.task_override is None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_cache_hit_returns_same_object(self, tmp_path: Path) -> None:
        """Second call returns cached result (same object identity)."""
        setup_base_prompts(tmp_path)

        loader = PromptLoader(tmp_path)
        first = loader.load_trickster_prompts("gemini")
        second = loader.load_trickster_prompts("gemini")

        assert first is second

    def test_cache_returns_stale_after_file_change(self, tmp_path: Path) -> None:
        """Cache returns stale content after file modification (no TTL)."""
        setup_base_prompts(tmp_path)

        loader = PromptLoader(tmp_path)
        first = loader.load_trickster_prompts("gemini")
        assert first.persona == "Test persona content."

        # Modify the file — cached result should still be stale
        write_prompt_file(tmp_path / "trickster" / "persona_base.md", "Updated persona")
        second = loader.load_trickster_prompts("gemini")

        assert second.persona == "Test persona content."  # stale

    def test_invalidate_clears_cache(self, tmp_path: Path) -> None:
        """After invalidate(), next load reads from disk."""
        setup_base_prompts(tmp_path)

        loader = PromptLoader(tmp_path)
        first = loader.load_trickster_prompts("gemini")
        assert first.persona == "Test persona content."

        write_prompt_file(tmp_path / "trickster" / "persona_base.md", "Updated persona")
        loader.invalidate()
        fresh = loader.load_trickster_prompts("gemini")

        assert fresh.persona == "Updated persona"

    def test_different_providers_are_separate_cache_entries(
        self, tmp_path: Path
    ) -> None:
        """Different providers have separate cache entries."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_base.md", "base persona")
        write_prompt_file(trickster / "persona_gemini.md", "gemini persona")
        write_prompt_file(trickster / "persona_claude.md", "claude persona")
        write_prompt_file(trickster / "behaviour_base.md", "base behaviour")
        write_prompt_file(trickster / "safety_base.md", "base safety")

        loader = PromptLoader(tmp_path)
        gemini_result = loader.load_trickster_prompts("gemini")
        anthropic_result = loader.load_trickster_prompts("anthropic")

        assert gemini_result.persona == "gemini persona"
        assert anthropic_result.persona == "claude persona"
        assert gemini_result is not anthropic_result

    def test_with_and_without_task_id_are_separate(self, tmp_path: Path) -> None:
        """(provider, None) and (provider, task_id) are distinct cache keys."""
        setup_base_prompts(tmp_path)
        task_dir = tmp_path / "tasks" / "task-001"
        write_prompt_file(task_dir / "trickster_base.md", "task override")

        loader = PromptLoader(tmp_path)
        without_task = loader.load_trickster_prompts("gemini")
        with_task = loader.load_trickster_prompts("gemini", task_id="task-001")

        assert without_task.task_override is None
        assert with_task.task_override == "task override"
        assert without_task is not with_task


# ---------------------------------------------------------------------------
# UTF-8 / Lithuanian characters
# ---------------------------------------------------------------------------


class TestUtf8Lithuanian:
    def test_lithuanian_characters_survive_load(self, tmp_path: Path) -> None:
        """Lithuanian special characters survive the load path."""
        trickster = tmp_path / "trickster"
        # Build Lithuanian text with special chars via Unicode escapes
        # \u2013 = en-dash, \u201e = opening „, \u201c = closing "
        lithuanian_text = (
            "Tu esi Triksteris \u2013 gudr\u016bs ir \u0161elmi\u0161kas persona\u017eas.\n"
            "\u201eSveiki, mokiniai!\u201c \u2013 taip prasideda kiekviena pamoka.\n"
            "Naudok \u0105\u010d\u0119\u0117\u012f\u0161\u0173\u016b\u017e simbolius laisvai."
        )
        write_prompt_file(trickster / "persona_base.md", lithuanian_text)
        write_prompt_file(trickster / "behaviour_base.md", "Elgesys")
        write_prompt_file(trickster / "safety_base.md", "Saugumas")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert "\u0105\u010d\u0119\u0117\u012f\u0161\u0173\u016b\u017e" in result.persona
        assert "\u201eSveiki, mokiniai!\u201c" in result.persona
        assert "gudr\u016bs ir \u0161elmi\u0161kas" in result.persona

    def test_lithuanian_quotes_preserved(self, tmp_path: Path) -> None:
        """Lithuanian opening and closing quotes are preserved."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_base.md", "\u201eAr tikrai?\u201c \u2013 paklaus\u0117 Triksteris.")
        write_prompt_file(trickster / "behaviour_base.md", "b")
        write_prompt_file(trickster / "safety_base.md", "s")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.persona == "\u201eAr tikrai?\u201c \u2013 paklaus\u0117 Triksteris."


# ---------------------------------------------------------------------------
# Empty file detection
# ---------------------------------------------------------------------------


class TestEmptyFileDetection:
    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        """A completely empty file is treated as absent."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_base.md", "")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.persona is None

    def test_whitespace_only_file_returns_none(self, tmp_path: Path) -> None:
        """A file with only whitespace is treated as absent."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_base.md", "   \n\t\n  ")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.persona is None

    def test_file_with_content_is_stripped(self, tmp_path: Path) -> None:
        """Content is stripped of leading/trailing whitespace."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_base.md", "\n  Persona content  \n\n")
        write_prompt_file(trickster / "behaviour_base.md", "b")
        write_prompt_file(trickster / "safety_base.md", "s")

        loader = PromptLoader(tmp_path)
        result = loader.load_trickster_prompts("gemini")

        assert result.persona == "Persona content"


# ---------------------------------------------------------------------------
# Validation: AI tasks
# ---------------------------------------------------------------------------


class TestValidationAiTasks:
    def test_all_base_prompts_present_returns_empty(
        self, tmp_path: Path, make_cartridge
    ) -> None:
        """No errors when all mandatory base prompts exist and are non-empty."""
        setup_base_prompts(tmp_path)
        cartridge = make_cartridge()

        loader = PromptLoader(tmp_path)
        errors = loader.validate_task_prompts(cartridge)

        assert errors == []

    def test_missing_persona_returns_error(
        self, tmp_path: Path, make_cartridge
    ) -> None:
        """Missing persona_base.md produces an error."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "behaviour_base.md", "behaviour")
        write_prompt_file(trickster / "safety_base.md", "safety")
        cartridge = make_cartridge()

        loader = PromptLoader(tmp_path)
        errors = loader.validate_task_prompts(cartridge)

        assert len(errors) == 1
        assert "persona_base.md" in errors[0]
        assert cartridge.task_id in errors[0]

    def test_missing_all_base_prompts_returns_three_errors(
        self, tmp_path: Path, make_cartridge
    ) -> None:
        """Missing all three base prompts returns three errors."""
        cartridge = make_cartridge()

        loader = PromptLoader(tmp_path)
        errors = loader.validate_task_prompts(cartridge)

        assert len(errors) == 3
        assert any("persona_base.md" in e for e in errors)
        assert any("behaviour_base.md" in e for e in errors)
        assert any("safety_base.md" in e for e in errors)

    def test_empty_base_prompt_returns_error(
        self, tmp_path: Path, make_cartridge
    ) -> None:
        """Base prompt file that exists but is empty produces an error."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_base.md", "")  # empty
        write_prompt_file(trickster / "behaviour_base.md", "behaviour")
        write_prompt_file(trickster / "safety_base.md", "safety")
        cartridge = make_cartridge()

        loader = PromptLoader(tmp_path)
        errors = loader.validate_task_prompts(cartridge)

        assert len(errors) == 1
        assert "persona_base.md" in errors[0]
        assert "empty" in errors[0]

    def test_whitespace_only_base_prompt_returns_error(
        self, tmp_path: Path, make_cartridge
    ) -> None:
        """Base prompt file with only whitespace produces an error."""
        trickster = tmp_path / "trickster"
        write_prompt_file(trickster / "persona_base.md", "   \n  ")
        write_prompt_file(trickster / "behaviour_base.md", "behaviour")
        write_prompt_file(trickster / "safety_base.md", "safety")
        cartridge = make_cartridge()

        loader = PromptLoader(tmp_path)
        errors = loader.validate_task_prompts(cartridge)

        assert len(errors) == 1
        assert "empty" in errors[0]


# ---------------------------------------------------------------------------
# Validation: static tasks
# ---------------------------------------------------------------------------


class TestValidationStaticTasks:
    def test_static_task_no_validation(self, tmp_path: Path, make_cartridge) -> None:
        """Static task (no ai_config) returns empty errors — no validation needed."""
        cartridge = make_cartridge(task_type="static", ai_config=None)

        loader = PromptLoader(tmp_path)
        errors = loader.validate_task_prompts(cartridge)

        assert errors == []

    def test_hybrid_with_no_ai_phases_no_validation(
        self, tmp_path: Path, make_cartridge
    ) -> None:
        """Hybrid task with all phases static returns no errors."""
        # Override phases to have no AI phases
        phases = [
            {
                "id": "phase_intro",
                "title": "Intro",
                "is_ai_phase": False,
                "interaction": {
                    "type": "button",
                    "choices": [
                        {
                            "label": "Tęsti",
                            "target_phase": "phase_end",
                        },
                    ],
                },
            },
            {
                "id": "phase_end",
                "title": "End",
                "is_terminal": True,
                "evaluation_outcome": "trickster_wins",
            },
        ]
        cartridge = make_cartridge(phases=phases, initial_phase="phase_intro")

        loader = PromptLoader(tmp_path)
        errors = loader.validate_task_prompts(cartridge)

        assert errors == []

    def test_ai_driven_without_ai_config_no_validation(
        self, tmp_path: Path, make_cartridge
    ) -> None:
        """ai_driven task with ai_config=None returns no errors (defensive)."""
        cartridge = make_cartridge(task_type="ai_driven", ai_config=None)

        loader = PromptLoader(tmp_path)
        errors = loader.validate_task_prompts(cartridge)

        assert errors == []


# ---------------------------------------------------------------------------
# Cache key correctness
# ---------------------------------------------------------------------------


class TestCacheKeyCorrectness:
    def test_same_provider_different_tasks_are_distinct(
        self, tmp_path: Path
    ) -> None:
        """(gemini, "task-001") and (gemini, "task-002") are distinct entries."""
        setup_base_prompts(tmp_path)
        write_prompt_file(
            tmp_path / "tasks" / "task-001" / "trickster_base.md",
            "override for task 001",
        )
        write_prompt_file(
            tmp_path / "tasks" / "task-002" / "trickster_base.md",
            "override for task 002",
        )

        loader = PromptLoader(tmp_path)
        r1 = loader.load_trickster_prompts("gemini", task_id="task-001")
        r2 = loader.load_trickster_prompts("gemini", task_id="task-002")

        assert r1.task_override == "override for task 001"
        assert r2.task_override == "override for task 002"
        assert r1 is not r2

    def test_different_providers_same_task_are_distinct(
        self, tmp_path: Path
    ) -> None:
        """(gemini, "task-001") and (anthropic, "task-001") are distinct entries."""
        setup_base_prompts(tmp_path)
        task_dir = tmp_path / "tasks" / "task-001"
        write_prompt_file(task_dir / "trickster_gemini.md", "gemini override")
        write_prompt_file(task_dir / "trickster_claude.md", "claude override")

        loader = PromptLoader(tmp_path)
        gemini = loader.load_trickster_prompts("gemini", task_id="task-001")
        anthropic = loader.load_trickster_prompts("anthropic", task_id="task-001")

        assert gemini.task_override == "gemini override"
        assert anthropic.task_override == "claude override"
        assert gemini is not anthropic
