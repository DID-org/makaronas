"""Integration tests for Phase 1c mode-specific persona prompt files.

Verifies that the four mode prompt files (presenting, chat_participant,
narrator, commenter) load correctly through PromptLoader, contain
distinct Lithuanian content, and assemble correctly in ContextManager.
"""

from pathlib import Path

import pytest

from backend.ai.context import ContextManager
from backend.ai.prompts import PromptLoader
from backend.config import PROJECT_ROOT
from backend.tests.conftest import setup_base_prompts, write_prompt_file


_MODES = ["presenting", "chat_participant", "narrator", "commenter"]
_MIN_LENGTH = 50


@pytest.fixture
def loader() -> PromptLoader:
    """Creates a PromptLoader pointed at the real prompts directory."""
    return PromptLoader(PROJECT_ROOT / "prompts")


# ---------------------------------------------------------------------------
# PromptLoader integration — each mode file loads successfully
# ---------------------------------------------------------------------------


class TestModeFilesLoad:
    """Each mode file loads as non-None, non-empty mode_behaviour."""

    @pytest.mark.parametrize("mode", _MODES)
    def test_mode_loads_non_none(self, loader: PromptLoader, mode: str) -> None:
        """Mode file produces a non-None mode_behaviour."""
        prompts = loader.load_trickster_prompts("gemini", persona_mode=mode)
        assert prompts.mode_behaviour is not None, (
            f"mode_behaviour is None for mode '{mode}'"
        )

    @pytest.mark.parametrize("mode", _MODES)
    def test_mode_content_meaningful(self, loader: PromptLoader, mode: str) -> None:
        """Mode file contains meaningful content (>{} chars)."""
        prompts = loader.load_trickster_prompts("gemini", persona_mode=mode)
        assert prompts.mode_behaviour is not None
        assert len(prompts.mode_behaviour) > _MIN_LENGTH, (
            f"mode_behaviour for '{mode}' too short: {len(prompts.mode_behaviour)}"
        )

    @pytest.mark.parametrize("mode", _MODES)
    def test_base_fields_unchanged(self, loader: PromptLoader, mode: str) -> None:
        """Base prompt fields still present when loading with mode."""
        prompts = loader.load_trickster_prompts("gemini", persona_mode=mode)
        assert prompts.persona is not None
        assert prompts.behaviour is not None
        assert prompts.safety is not None


# ---------------------------------------------------------------------------
# Distinctness — all four modes produce different content
# ---------------------------------------------------------------------------


class TestModeDistinctness:
    """All four modes are distinct from each other."""

    def test_all_modes_distinct(self, loader: PromptLoader) -> None:
        """Each mode_behaviour content is unique across all four modes."""
        contents = []
        for mode in _MODES:
            prompts = loader.load_trickster_prompts("gemini", persona_mode=mode)
            assert prompts.mode_behaviour is not None
            contents.append(prompts.mode_behaviour)
        assert len(set(contents)) == 4, "Not all mode contents are distinct"


# ---------------------------------------------------------------------------
# Lithuanian character integrity
# ---------------------------------------------------------------------------


class TestModeLithuanianChars:
    """Lithuanian characters survive the load cycle for mode files."""

    _LT_CHARS = [
        "\u0105",  # ą
        "\u0161",  # š
        "\u017e",  # ž
    ]

    @pytest.mark.parametrize("mode", _MODES)
    def test_contains_lt_chars(self, loader: PromptLoader, mode: str) -> None:
        """Mode file contains Lithuanian diacritical characters."""
        prompts = loader.load_trickster_prompts("gemini", persona_mode=mode)
        assert prompts.mode_behaviour is not None
        found = [c for c in self._LT_CHARS if c in prompts.mode_behaviour]
        assert len(found) >= 2, (
            f"Mode '{mode}' has too few Lithuanian chars: found {found}"
        )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Existing calls without persona_mode still work."""

    def test_no_mode_returns_none(self, loader: PromptLoader) -> None:
        """Loader without persona_mode returns mode_behaviour=None."""
        prompts = loader.load_trickster_prompts("gemini")
        assert prompts.mode_behaviour is None


# ---------------------------------------------------------------------------
# ContextManager assembly — mode content in system prompt
# ---------------------------------------------------------------------------


class TestModeContextAssembly:
    """Mode content appears correctly in assembled system prompts."""

    @pytest.mark.parametrize("mode", _MODES)
    def test_mode_content_in_system_prompt(
        self, tmp_path: Path, make_session, make_cartridge, mode: str,
    ) -> None:
        """Mode-specific content appears in the assembled system prompt."""
        setup_base_prompts(tmp_path)
        # Use real mode files by copying content from the real prompts dir
        real_loader = PromptLoader(PROJECT_ROOT / "prompts")
        real_prompts = real_loader.load_trickster_prompts("gemini", persona_mode=mode)
        assert real_prompts.mode_behaviour is not None

        # Write the real mode content into the tmp_path test tree
        write_prompt_file(
            tmp_path / "trickster" / f"persona_{mode}_base.md",
            real_prompts.mode_behaviour,
        )

        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

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

        # A distinctive phrase from the mode file should be in the system prompt
        assert real_prompts.mode_behaviour in result.system_prompt

    @pytest.mark.parametrize("mode", _MODES)
    def test_mode_layer_ordering(
        self, tmp_path: Path, make_session, make_cartridge, mode: str,
    ) -> None:
        """Mode content appears after persona and before behaviour."""
        setup_base_prompts(tmp_path)
        marker = f"MODE_MARKER_{mode.upper()}"
        write_prompt_file(
            tmp_path / "trickster" / f"persona_{mode}_base.md", marker,
        )

        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

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

        prompt = result.system_prompt
        persona_pos = prompt.index("Test persona content.")
        mode_pos = prompt.index(marker)
        behaviour_pos = prompt.index("Test behaviour content.")
        assert persona_pos < mode_pos < behaviour_pos


# ---------------------------------------------------------------------------
# Content checks — key elements present per mode
# ---------------------------------------------------------------------------


class TestModeContentElements:
    """Each mode file contains expected key elements for its mode."""

    def test_presenting_has_ownership(self, loader: PromptLoader) -> None:
        """Presenting mode references content ownership/authorship."""
        prompts = loader.load_trickster_prompts("gemini", persona_mode="presenting")
        assert prompts.mode_behaviour is not None
        content = prompts.mode_behaviour.lower()
        assert "author" in content or "created" in content

    def test_chat_participant_has_frame_maintenance(
        self, loader: PromptLoader,
    ) -> None:
        """Chat participant mode references staying in character."""
        prompts = loader.load_trickster_prompts(
            "gemini", persona_mode="chat_participant",
        )
        assert prompts.mode_behaviour is not None
        content = prompts.mode_behaviour.lower()
        assert "character" in content or "mask" in content

    def test_chat_participant_has_multi_character(
        self, loader: PromptLoader,
    ) -> None:
        """Chat participant mode supports multi-character scenarios."""
        prompts = loader.load_trickster_prompts(
            "gemini", persona_mode="chat_participant",
        )
        assert prompts.mode_behaviour is not None
        content = prompts.mode_behaviour.lower()
        assert "multi" in content or "multiple" in content  # multi-character

    def test_narrator_has_socratic(self, loader: PromptLoader) -> None:
        """Narrator mode references Socratic/guide approach."""
        prompts = loader.load_trickster_prompts("gemini", persona_mode="narrator")
        assert prompts.mode_behaviour is not None
        content = prompts.mode_behaviour.lower()
        assert "guide" in content or "socratic" in content or "narrator" in content

    def test_commenter_has_brevity(self, loader: PromptLoader) -> None:
        """Commenter mode references short/brief responses."""
        prompts = loader.load_trickster_prompts("gemini", persona_mode="commenter")
        assert prompts.mode_behaviour is not None
        content = prompts.mode_behaviour.lower()
        assert "brief" in content or "short" in content  # brevity
