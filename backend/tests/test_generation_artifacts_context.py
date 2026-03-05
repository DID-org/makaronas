"""Tests for generated artifacts injection into Trickster context (Phase 7c).

Unit tests for _build_generation_artifacts_context() and integration tests
verifying artifacts appear correctly in assembled system prompts.

Test categories:
T1: Empty artifacts -> None return
T2: Single artifact produces block
T3: Multiple artifacts show evolution
T4: Safety-redacted artifact marked
T5: Context fencing instruction present
T6: Lithuanian content
T7: Adversarial task with artifacts (integration)
T8: Clean task with artifacts (integration)
T9: No artifacts, no injection (integration)
T10: Artifacts in full system prompt (assemble_trickster_call)
T11: Artifacts absent from debrief (negative test)
"""

from pathlib import Path

import pytest

from backend.ai.context import ContextManager
from backend.ai.prompts import PromptLoader
from backend.schemas import GameSession
from backend.tests.conftest import setup_base_prompts


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _artifact(
    prompt: str = "Para\u0161yk \u012fra\u0161\u0105",
    text: str = "Sugeneruotas tekstas",
    timestamp: str = "2026-03-05T10:00:00",
    redacted: bool = False,
) -> dict:
    """Creates a generated_artifacts entry dict."""
    return {
        "student_prompt": prompt,
        "generated_text": text,
        "timestamp": timestamp,
        "safety_redacted": redacted,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
# T1: Empty artifacts -> None return
# ---------------------------------------------------------------------------


class TestEmptyArtifacts:
    """Empty generated_artifacts produces None."""

    def test_empty_list_returns_none(self, make_session):
        session = make_session(generated_artifacts=[])
        result = ContextManager._build_generation_artifacts_context(session)
        assert result is None

    def test_default_session_returns_none(self, make_session):
        session = make_session()
        result = ContextManager._build_generation_artifacts_context(session)
        assert result is None


# ---------------------------------------------------------------------------
# T2: Single artifact produces block
# ---------------------------------------------------------------------------


class TestSingleArtifact:
    """One artifact renders correctly."""

    def test_returns_string(self, make_session):
        session = make_session(generated_artifacts=[_artifact()])
        result = ContextManager._build_generation_artifacts_context(session)
        assert result is not None
        assert isinstance(result, str)

    def test_header_present(self, make_session):
        session = make_session(generated_artifacts=[_artifact()])
        result = ContextManager._build_generation_artifacts_context(session)
        assert "Mokinio sukurtas turinys" in result

    def test_attempt_numbered(self, make_session):
        session = make_session(generated_artifacts=[_artifact()])
        result = ContextManager._build_generation_artifacts_context(session)
        assert "Bandymas 1" in result

    def test_prompt_present(self, make_session):
        session = make_session(
            generated_artifacts=[_artifact(prompt="Mano nurodymas")]
        )
        result = ContextManager._build_generation_artifacts_context(session)
        assert "Mano nurodymas" in result
        assert "Mokinio nurodymas" in result

    def test_generated_text_present(self, make_session):
        session = make_session(
            generated_artifacts=[_artifact(text="Rezultatas")]
        )
        result = ContextManager._build_generation_artifacts_context(session)
        assert "Rezultatas" in result
        assert "Sugeneruotas turinys" in result


# ---------------------------------------------------------------------------
# T3: Multiple artifacts show evolution
# ---------------------------------------------------------------------------


class TestMultipleArtifacts:
    """Multiple artifacts render chronologically with numbered attempts."""

    def test_three_artifacts_numbered(self, make_session):
        artifacts = [
            _artifact(prompt=f"Bandymas nr {i}", text=f"Rezultatas {i}")
            for i in range(1, 4)
        ]
        session = make_session(generated_artifacts=artifacts)
        result = ContextManager._build_generation_artifacts_context(session)
        assert "Bandymas 1" in result
        assert "Bandymas 2" in result
        assert "Bandymas 3" in result

    def test_chronological_order(self, make_session):
        artifacts = [
            _artifact(prompt="Pirmas"),
            _artifact(prompt="Antras"),
            _artifact(prompt="Tre\u010dias"),
        ]
        session = make_session(generated_artifacts=artifacts)
        result = ContextManager._build_generation_artifacts_context(session)
        pos_first = result.index("Pirmas")
        pos_second = result.index("Antras")
        pos_third = result.index("Tre\u010dias")
        assert pos_first < pos_second < pos_third

    def test_all_prompts_and_texts_present(self, make_session):
        artifacts = [
            _artifact(prompt=f"P{i}", text=f"T{i}")
            for i in range(1, 4)
        ]
        session = make_session(generated_artifacts=artifacts)
        result = ContextManager._build_generation_artifacts_context(session)
        for i in range(1, 4):
            assert f"P{i}" in result
            assert f"T{i}" in result


# ---------------------------------------------------------------------------
# T4: Safety-redacted artifact marked
# ---------------------------------------------------------------------------


class TestSafetyRedacted:
    """Safety-redacted artifacts show marker instead of generated text."""

    def test_redacted_shows_marker(self, make_session):
        session = make_session(
            generated_artifacts=[_artifact(text="bad content", redacted=True)]
        )
        result = ContextManager._build_generation_artifacts_context(session)
        assert "saugumo sistema" in result
        assert "bad content" not in result

    def test_non_redacted_shows_text(self, make_session):
        session = make_session(
            generated_artifacts=[_artifact(text="good content", redacted=False)]
        )
        result = ContextManager._build_generation_artifacts_context(session)
        assert "good content" in result
        assert "saugumo sistema" not in result

    def test_mixed_artifacts(self, make_session):
        """One redacted, one not."""
        artifacts = [
            _artifact(text="good text", redacted=False),
            _artifact(text="bad text", redacted=True),
        ]
        session = make_session(generated_artifacts=artifacts)
        result = ContextManager._build_generation_artifacts_context(session)
        assert "good text" in result
        assert "bad text" not in result
        assert "saugumo sistema" in result


# ---------------------------------------------------------------------------
# T5: Context fencing instruction present
# ---------------------------------------------------------------------------


class TestContextFencing:
    """Fencing instruction prevents verbatim echo."""

    def test_niekada_prohibition(self, make_session):
        session = make_session(generated_artifacts=[_artifact()])
        result = ContextManager._build_generation_artifacts_context(session)
        assert "NIEKADA" in result

    def test_necituok_instruction(self, make_session):
        session = make_session(generated_artifacts=[_artifact()])
        result = ContextManager._build_generation_artifacts_context(session)
        assert "necituok" in result

    def test_instrukcija_label(self, make_session):
        session = make_session(generated_artifacts=[_artifact()])
        result = ContextManager._build_generation_artifacts_context(session)
        assert "INSTRUKCIJA" in result


# ---------------------------------------------------------------------------
# T6: Lithuanian content
# ---------------------------------------------------------------------------


class TestLithuanianContent:
    """Output uses Lithuanian headers and labels."""

    def test_header_is_lithuanian(self, make_session):
        session = make_session(generated_artifacts=[_artifact()])
        result = ContextManager._build_generation_artifacts_context(session)
        assert "Mokinio sukurtas turinys" in result

    def test_labels_are_lithuanian(self, make_session):
        session = make_session(generated_artifacts=[_artifact()])
        result = ContextManager._build_generation_artifacts_context(session)
        assert "Mokinio nurodymas" in result
        assert "Sugeneruotas turinys" in result


# ---------------------------------------------------------------------------
# T7: Adversarial task with artifacts (integration)
# ---------------------------------------------------------------------------


class TestAdversarialTaskWithArtifacts:
    """_build_task_context() includes artifacts for adversarial cartridge."""

    def test_artifacts_in_adversarial_context(
        self, make_session, make_cartridge, context_manager
    ):
        session = make_session(
            generated_artifacts=[_artifact(prompt="Test prompt")],
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )
        cartridge = make_cartridge(is_clean=False)
        result = context_manager._build_task_context(
            session, cartridge, "gemini"
        )
        assert "Mokinio sukurtas turinys" in result
        assert "Test prompt" in result
        # Also contains adversarial context
        assert "Uzduoties kontekstas" in result


# ---------------------------------------------------------------------------
# T8: Clean task with artifacts (integration)
# ---------------------------------------------------------------------------


class TestCleanTaskWithArtifacts:
    """_build_task_context() includes artifacts for clean cartridge."""

    def test_artifacts_in_clean_context(
        self, make_session, make_cartridge, context_manager
    ):
        session = make_session(
            generated_artifacts=[_artifact(prompt="Clean test")],
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )
        cartridge = make_cartridge(
            is_clean=True,
            evaluation={
                "patterns_embedded": [],
                "checklist": [],
                "pass_conditions": {
                    "trickster_wins": "Nepavyko",
                    "partial": "I\u0161 dalies",
                    "trickster_loses": "Pavyko",
                },
            },
        )
        result = context_manager._build_task_context(
            session, cartridge, "gemini"
        )
        assert "Mokinio sukurtas turinys" in result
        assert "Clean test" in result
        # Also contains clean context
        assert "Svaraus turinio kontekstas" in result


# ---------------------------------------------------------------------------
# T9: No artifacts, no injection (integration)
# ---------------------------------------------------------------------------


class TestNoArtifactsNoInjection:
    """_build_task_context() unchanged when no artifacts."""

    def test_no_artifacts_section(
        self, make_session, make_cartridge, context_manager
    ):
        session = make_session(
            generated_artifacts=[],
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )
        cartridge = make_cartridge()
        result = context_manager._build_task_context(
            session, cartridge, "gemini"
        )
        assert "Mokinio sukurtas turinys" not in result


# ---------------------------------------------------------------------------
# T10: Artifacts in full system prompt (assemble_trickster_call)
# ---------------------------------------------------------------------------


class TestFullSystemPromptIntegration:
    """assemble_trickster_call includes artifacts in system prompt."""

    def test_artifacts_in_trickster_system_prompt(
        self, make_session, make_cartridge, context_manager
    ):
        session = make_session(
            generated_artifacts=[
                _artifact(prompt="Pirmas bandymas", text="Rezultatas 1"),
                _artifact(prompt="Antras bandymas", text="Rezultatas 2"),
            ],
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )
        cartridge = make_cartridge()
        ctx = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=0, min_exchanges=2,
        )
        assert "Mokinio sukurtas turinys" in ctx.system_prompt
        assert "Pirmas bandymas" in ctx.system_prompt
        assert "Antras bandymas" in ctx.system_prompt
        assert "NIEKADA" in ctx.system_prompt


# ---------------------------------------------------------------------------
# T11: Artifacts absent from debrief (negative test)
# ---------------------------------------------------------------------------


class TestDebriefExclusion:
    """assemble_debrief_call does NOT include generation artifacts."""

    def test_no_artifacts_in_debrief(
        self, make_session, make_cartridge, context_manager
    ):
        session = make_session(
            generated_artifacts=[
                _artifact(prompt="Should not appear"),
            ],
            current_task="test-ai-task-001",
            current_phase="phase_ai",
        )
        cartridge = make_cartridge()
        ctx = context_manager.assemble_debrief_call(
            session, cartridge, "gemini"
        )
        assert "Mokinio sukurtas turinys" not in ctx.system_prompt
        assert "Should not appear" not in ctx.system_prompt
