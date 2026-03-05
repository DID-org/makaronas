"""Tests for clean task context assembly (Phase 3a)."""

import logging
from pathlib import Path

import pytest

from backend.ai.context import ContextManager
from backend.ai.prompts import PromptLoader
from backend.tests.conftest import setup_base_prompts, write_prompt_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLEAN_EVAL = {
    "patterns_embedded": [],
    "checklist": [],
    "pass_conditions": {
        "trickster_wins": "Mokinys neteisingai apkaltino turin\u012f manipuliacija",
        "partial": "Mokinys abejojo, bet link\u0119s laikyti turin\u012f legitiminiu",
        "trickster_loses": "Mokinys teisingai atpa\u017eino legitim\u0173 turin\u012f",
    },
}

_CLEAN_PROMPT_CONTENT = (
    "\u0160is turinys yra legitimus. "
    "Gink jo patikimum\u0105 ir kvestionuok mokinio kaltinimus."
)


def _setup_clean_prompt(prompts_dir: Path) -> None:
    """Creates the clean_task_base.md prompt file."""
    write_prompt_file(
        prompts_dir / "trickster" / "clean_task_base.md",
        _CLEAN_PROMPT_CONTENT,
    )


# ---------------------------------------------------------------------------
# Clean task context in system prompt
# ---------------------------------------------------------------------------


class TestCleanTaskContext:
    """Tests that clean task cartridges produce correct context assembly."""

    def test_clean_task_context_in_system_prompt(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Assembling with is_clean=True includes clean task prompt content."""
        setup_base_prompts(tmp_path)
        _setup_clean_prompt(tmp_path)

        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        session = make_session(current_task=cartridge.task_id, current_phase="phase_ai")

        ctx = cm.assemble_trickster_call(
            session, cartridge, "gemini", exchange_count=1, min_exchanges=2,
        )

        assert _CLEAN_PROMPT_CONTENT in ctx.system_prompt
        assert "Svaraus turinio kontekstas" in ctx.system_prompt

    def test_clean_task_no_adversarial_framing(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Clean task context does NOT contain adversarial evaluation headers."""
        setup_base_prompts(tmp_path)
        _setup_clean_prompt(tmp_path)

        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        session = make_session(current_task=cartridge.task_id, current_phase="phase_ai")

        ctx = cm.assemble_trickster_call(
            session, cartridge, "gemini", exchange_count=1, min_exchanges=2,
        )

        assert "Vertinimo kriterijai" not in ctx.system_prompt
        assert "Kontrolinis sarasas" not in ctx.system_prompt

    def test_adversarial_task_unchanged(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Standard adversarial cartridge produces unchanged framing."""
        setup_base_prompts(tmp_path)

        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        cartridge = make_cartridge()  # default is_clean=False
        session = make_session(current_task=cartridge.task_id, current_phase="phase_ai")

        ctx = cm.assemble_trickster_call(
            session, cartridge, "gemini", exchange_count=1, min_exchanges=2,
        )

        assert "Uzduoties kontekstas" in ctx.system_prompt
        assert "Vertinimo kriterijai" in ctx.system_prompt
        assert "Svaraus turinio kontekstas" not in ctx.system_prompt

    def test_adversarial_task_no_clean_prompt(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Adversarial tasks do not load clean task prompt content."""
        setup_base_prompts(tmp_path)
        _setup_clean_prompt(tmp_path)

        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        cartridge = make_cartridge()  # default is_clean=False
        session = make_session(current_task=cartridge.task_id, current_phase="phase_ai")

        ctx = cm.assemble_trickster_call(
            session, cartridge, "gemini", exchange_count=1, min_exchanges=2,
        )

        assert _CLEAN_PROMPT_CONTENT not in ctx.system_prompt


class TestCleanTaskFallback:
    """Tests graceful degradation when clean task prompt is missing."""

    def test_missing_prompt_no_crash(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Assembly succeeds even without clean_task_base.md on disk."""
        setup_base_prompts(tmp_path)
        # Deliberately do NOT create clean_task_base.md.

        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        session = make_session(current_task=cartridge.task_id, current_phase="phase_ai")

        ctx = cm.assemble_trickster_call(
            session, cartridge, "gemini", exchange_count=1, min_exchanges=2,
        )

        # Still produces valid context with pass conditions.
        assert "Svaraus turinio kontekstas" in ctx.system_prompt
        assert "Triksteris laimi" in ctx.system_prompt

    def test_missing_prompt_logs_warning(
        self, tmp_path: Path, make_session, make_cartridge, caplog,
    ) -> None:
        """Missing clean_task_base.md logs a warning."""
        setup_base_prompts(tmp_path)

        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        session = make_session(current_task=cartridge.task_id, current_phase="phase_ai")

        with caplog.at_level(logging.WARNING, logger="backend.ai.context"):
            cm.assemble_trickster_call(
                session, cartridge, "gemini", exchange_count=1, min_exchanges=2,
            )

        assert any("Clean task prompt file not found" in r.message for r in caplog.records)


class TestCleanTaskPassConditions:
    """Tests that pass conditions appear in clean task context."""

    def test_pass_conditions_present(
        self, tmp_path: Path, make_session, make_cartridge,
    ) -> None:
        """Clean task context includes pass_conditions (inverted semantics)."""
        setup_base_prompts(tmp_path)
        _setup_clean_prompt(tmp_path)

        loader = PromptLoader(tmp_path)
        cm = ContextManager(loader)

        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        session = make_session(current_task=cartridge.task_id, current_phase="phase_ai")

        ctx = cm.assemble_trickster_call(
            session, cartridge, "gemini", exchange_count=1, min_exchanges=2,
        )

        assert "Triksteris laimi" in ctx.system_prompt
        assert "Triksteris pralaimi" in ctx.system_prompt
        assert "apkaltino" in ctx.system_prompt  # trickster_wins text


class TestParadigmNeutralSignals:
    """Tests that behaviour_base.md signal descriptions are paradigm-neutral."""

    def test_no_manipulation_reference_in_signals(self) -> None:
        """Signal descriptions do not contain 'manipuliacija' or '\u0161ablon'."""
        behaviour_path = (
            Path(__file__).parent.parent.parent
            / "prompts" / "trickster" / "behaviour_base.md"
        )
        content = behaviour_path.read_text(encoding="utf-8")

        # Extract only the signal description section (lines around transition_phase).
        signal_section = ""
        in_signal_section = False
        for line in content.splitlines():
            if '"understood"' in line or '"partial"' in line or '"max_reached"' in line:
                in_signal_section = True
            if in_signal_section:
                signal_section += line + "\n"
                # Stop after max_reached block ends.
                if '"max_reached"' in line:
                    # Include the next line(s) that are continuations.
                    continue
            if in_signal_section and line.strip() == "":
                break

        assert "manipuliacija" not in signal_section
        assert "\u0161ablon" not in signal_section  # "šablon" (patterns)


class TestCleanCartridgeFixture:
    """Tests that the clean cartridge override produces a valid cartridge."""

    def test_clean_cartridge_valid(self, make_cartridge) -> None:
        """make_cartridge(is_clean=True, evaluation=...) produces valid cartridge."""
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)

        assert cartridge.is_clean is True
        assert len(cartridge.evaluation.patterns_embedded) == 0
        assert len(cartridge.evaluation.checklist) == 0
        assert cartridge.evaluation.pass_conditions.trickster_wins != ""


class TestCleanTaskPromptEncoding:
    """Tests that the clean task prompt file contains valid Lithuanian chars."""

    def test_lithuanian_chars_in_clean_prompt(self) -> None:
        """clean_task_base.md contains Lithuanian diacritical characters."""
        prompt_path = (
            Path(__file__).parent.parent.parent
            / "prompts" / "trickster" / "clean_task_base.md"
        )
        content = prompt_path.read_text(encoding="utf-8")

        # Lithuanian-specific characters that should be present.
        lt_chars = set("\u0105\u010d\u0119\u0117\u012f\u0161\u0173\u016b\u017e")
        found = {c for c in content if c in lt_chars}

        # At least 5 different Lithuanian chars should appear.
        assert len(found) >= 5, f"Only found {found} Lithuanian chars"
