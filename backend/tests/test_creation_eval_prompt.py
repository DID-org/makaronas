"""Tests for creation evaluation prompt architecture (Phase 9a).

Tests cover prompt loading, Lithuanian encoding, context assembly
(injection when artifacts exist, absence when they don't), signal
semantics, and graceful fallback.
"""

import logging
from pathlib import Path

import pytest

from backend.ai.context import ContextManager
from backend.ai.prompts import PromptLoader
from backend.tests.conftest import setup_base_prompts


# ---------------------------------------------------------------------------
# Helpers
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
def prompts_dir_with_creation_eval(prompts_dir) -> Path:
    """Base prompts + creation eval prompt file."""
    src = Path("/home/vinga/projects/makaronas/prompts/trickster/creation_eval_base.md")
    dest = prompts_dir / "trickster" / "creation_eval_base.md"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return prompts_dir


@pytest.fixture
def context_manager(prompts_dir_with_creation_eval) -> ContextManager:
    """ContextManager backed by temp prompts including creation eval."""
    loader = PromptLoader(prompts_dir_with_creation_eval)
    return ContextManager(loader)


@pytest.fixture
def context_manager_no_creation_eval(prompts_dir) -> ContextManager:
    """ContextManager backed by temp prompts WITHOUT creation eval."""
    loader = PromptLoader(prompts_dir)
    return ContextManager(loader)


# ---------------------------------------------------------------------------
# T1: Prompt loading
# ---------------------------------------------------------------------------


class TestPromptLoading:
    """Creation eval prompt loads correctly through PromptLoader."""

    def test_loads_when_file_exists(self, prompts_dir_with_creation_eval):
        loader = PromptLoader(prompts_dir_with_creation_eval)
        result = loader.load_creation_eval_prompt("gemini")
        assert result is not None
        assert len(result) > 0

    def test_returns_none_when_file_missing(self, prompts_dir):
        loader = PromptLoader(prompts_dir)
        result = loader.load_creation_eval_prompt("gemini")
        assert result is None

    def test_content_is_lithuanian(self, prompts_dir_with_creation_eval):
        loader = PromptLoader(prompts_dir_with_creation_eval)
        result = loader.load_creation_eval_prompt("gemini")
        assert result is not None
        # Verify Lithuanian keywords from the coaching framework
        assert "vertink" in result.lower() or "vertinimo" in result.lower()
        assert "metodologij" in result.lower() or "metodolog" in result.lower()


# ---------------------------------------------------------------------------
# T2: Encoding verification
# ---------------------------------------------------------------------------


class TestEncoding:
    """Creation eval prompt preserves Lithuanian diacritics."""

    def test_contains_lithuanian_diacritics(self, prompts_dir_with_creation_eval):
        loader = PromptLoader(prompts_dir_with_creation_eval)
        result = loader.load_creation_eval_prompt("gemini")
        assert result is not None
        lt_chars = set("\u0105\u010d\u0119\u0117\u012f\u0161\u0173\u016b\u017e")
        found = set(result) & lt_chars
        assert len(found) >= 3, f"Expected >= 3 Lithuanian diacritics, found: {found}"


# ---------------------------------------------------------------------------
# T3: Context assembly — artifacts trigger injection
# ---------------------------------------------------------------------------


class TestArtifactsTriggerInjection:
    """Creation eval prompt appears only when artifacts exist."""

    def test_injected_when_artifacts_present(
        self, context_manager, make_session, make_cartridge,
    ):
        session = make_session(generated_artifacts=[_artifact()])
        cartridge = make_cartridge()
        result = context_manager._build_task_context(session, cartridge, "gemini")
        assert "K\u016brimo vertinimas" in result or "meistras" in result

    def test_absent_when_artifacts_empty(
        self, context_manager, make_session, make_cartridge,
    ):
        session = make_session(generated_artifacts=[])
        cartridge = make_cartridge()
        result = context_manager._build_task_context(session, cartridge, "gemini")
        assert "K\u016brimo vertinimas" not in result

    def test_absent_when_artifacts_default(
        self, context_manager, make_session, make_cartridge,
    ):
        session = make_session()
        cartridge = make_cartridge()
        result = context_manager._build_task_context(session, cartridge, "gemini")
        assert "K\u016brimo vertinimas" not in result


# ---------------------------------------------------------------------------
# T4: Context assembly — creation eval after artifacts
# ---------------------------------------------------------------------------


class TestOrderingAfterArtifacts:
    """Creation eval content appears after artifacts data."""

    def test_creation_eval_after_artifacts(
        self, context_manager, make_session, make_cartridge,
    ):
        session = make_session(generated_artifacts=[_artifact()])
        cartridge = make_cartridge()
        result = context_manager._build_task_context(session, cartridge, "gemini")
        # Artifacts section header from Phase 7c
        artifacts_marker = "Mokinio sukurtas turinys"
        # Creation eval header
        creation_eval_marker = "K\u016brimo vertinimas"
        assert artifacts_marker in result
        assert creation_eval_marker in result
        assert result.index(artifacts_marker) < result.index(creation_eval_marker)


# ---------------------------------------------------------------------------
# T5: Context assembly — with adversarial cartridge
# ---------------------------------------------------------------------------


class TestAdversarialCartridgeWithArtifacts:
    """Adversarial cartridge + artifacts includes both contexts."""

    def test_adversarial_plus_creation_eval(
        self, context_manager, make_session, make_cartridge,
    ):
        session = make_session(generated_artifacts=[_artifact()])
        cartridge = make_cartridge(is_clean=False)
        result = context_manager._build_task_context(session, cartridge, "gemini")
        # Adversarial task context marker (ASCII ž in static method)
        assert "Uzduoties kontekstas" in result
        # Creation eval present
        assert "meistras" in result or "K\u016brimo vertinimas" in result


# ---------------------------------------------------------------------------
# T6: Context assembly — with clean cartridge
# ---------------------------------------------------------------------------


class TestCleanCartridgeWithArtifacts:
    """Clean cartridge + artifacts includes both contexts."""

    def test_clean_plus_creation_eval(
        self, context_manager, make_session, make_cartridge,
    ):
        session = make_session(generated_artifacts=[_artifact()])
        cartridge = make_cartridge(
            is_clean=True,
            evaluation={
                "patterns_embedded": [],
                "checklist": [],
                "pass_conditions": {
                    "trickster_wins": "Mokinys neteisingai apkaltino \u0161var\u0173 turin\u012f",
                    "partial": "Mokinys nebuvo tikras",
                    "trickster_loses": "Mokinys teisingai atpa\u017eino legitimum\u0105",
                },
            },
        )
        result = context_manager._build_task_context(session, cartridge, "gemini")
        # Clean task context marker
        assert "Svaraus turinio kontekstas" in result
        # Creation eval present
        assert "meistras" in result or "K\u016brimo vertinimas" in result


# ---------------------------------------------------------------------------
# T7: Full system prompt integration
# ---------------------------------------------------------------------------


class TestFullSystemPromptIntegration:
    """assemble_trickster_call includes creation eval; debrief does not."""

    def test_trickster_call_includes_creation_eval(
        self, context_manager, make_session, make_cartridge,
    ):
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
            generated_artifacts=[_artifact()],
        )
        cartridge = make_cartridge()
        assembled = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=0, min_exchanges=2,
        )
        assert "K\u016brimo vertinimas" in assembled.system_prompt or \
               "meistras" in assembled.system_prompt

    def test_debrief_excludes_creation_eval(
        self, context_manager, make_session, make_cartridge,
    ):
        session = make_session(
            current_task="test-ai-task-001",
            current_phase="phase_ai",
            generated_artifacts=[_artifact()],
        )
        cartridge = make_cartridge()
        assembled = context_manager.assemble_debrief_call(
            session, cartridge, "gemini",
        )
        # Debrief uses its own path, not _build_task_context
        assert "K\u016brimo vertinimas" not in assembled.system_prompt


# ---------------------------------------------------------------------------
# T8: Signal semantics verification
# ---------------------------------------------------------------------------


class TestSignalSemantics:
    """The creation eval prompt contains all three signal concepts."""

    def test_signal_concepts_present(self, prompts_dir_with_creation_eval):
        loader = PromptLoader(prompts_dir_with_creation_eval)
        result = loader.load_creation_eval_prompt("gemini")
        assert result is not None
        text = result.lower()
        # Understood / comprehension
        assert "supratim" in text, "Missing 'understood' signal concept"
        # Partial
        assert "dalini" in text, "Missing 'partial' signal concept"
        # Max reached / inability
        assert "nepavyko" in text or "negal\u0117jo" in text, \
            "Missing 'max_reached' signal concept"


# ---------------------------------------------------------------------------
# T9: Graceful fallback
# ---------------------------------------------------------------------------


class TestGracefulFallback:
    """Missing creation eval prompt degrades gracefully."""

    def test_no_crash_when_prompt_missing(
        self, context_manager_no_creation_eval, make_session, make_cartridge,
    ):
        session = make_session(generated_artifacts=[_artifact()])
        cartridge = make_cartridge()
        # Should not raise
        result = context_manager_no_creation_eval._build_task_context(
            session, cartridge, "gemini",
        )
        # Artifacts data still present
        assert "Mokinio sukurtas turinys" in result
        # Creation eval absent
        assert "K\u016brimo vertinimas" not in result

    def test_warning_logged_when_prompt_missing(
        self, context_manager_no_creation_eval, make_session, make_cartridge,
        caplog,
    ):
        session = make_session(generated_artifacts=[_artifact()])
        cartridge = make_cartridge()
        with caplog.at_level(logging.WARNING, logger="backend.ai.context"):
            context_manager_no_creation_eval._build_task_context(
                session, cartridge, "gemini",
            )
        assert any(
            "Creation eval prompt not found" in r.message for r in caplog.records
        )
