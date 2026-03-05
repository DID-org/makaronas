"""Tests for context-isolated generation assembly and session artifacts."""

from typing import Any

import pytest

from backend.ai.context import (
    AssembledContext,
    ContextManager,
    _GENERATION_SYSTEM_PROMPT,
)
from backend.ai.prompts import PromptLoader
from backend.schemas import GameSession
from backend.tests.conftest import setup_base_prompts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context_manager(tmp_path) -> ContextManager:
    """Creates a ContextManager with base prompts (required by constructor)."""
    setup_base_prompts(tmp_path)
    loader = PromptLoader(tmp_path)
    return ContextManager(loader)


# ---------------------------------------------------------------------------
# Generation context assembly
# ---------------------------------------------------------------------------


class TestAssembleGenerationCall:
    """Tests for ContextManager.assemble_generation_call()."""

    def test_returns_assembled_context(self, tmp_path) -> None:
        """Returns a valid AssembledContext."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source text", "student instruction")
        assert isinstance(result, AssembledContext)

    def test_system_prompt_is_neutral(self, tmp_path) -> None:
        """System prompt is the neutral generation prompt."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source", "prompt")
        assert result.system_prompt == _GENERATION_SYSTEM_PROMPT

    def test_system_prompt_is_lithuanian(self, tmp_path) -> None:
        """System prompt contains Lithuanian text."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source", "prompt")
        # Lithuanian diacritics present
        assert "lietuvi\u0173" in result.system_prompt

    def test_system_prompt_is_short(self, tmp_path) -> None:
        """System prompt is concise (under 300 chars)."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source", "prompt")
        assert len(result.system_prompt) < 300

    def test_tools_is_none(self, tmp_path) -> None:
        """No tools in generation context (no transition tool)."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source", "prompt")
        assert result.tools is None

    def test_messages_has_two_user_entries(self, tmp_path) -> None:
        """Messages contain exactly two user messages."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source text", "student prompt")
        assert len(result.messages) == 2
        assert all(m["role"] == "user" for m in result.messages)

    def test_source_content_is_first_message(self, tmp_path) -> None:
        """Source content appears as the first user message."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("My source article", "Do something")
        assert result.messages[0]["content"] == "My source article"

    def test_student_prompt_is_second_message(self, tmp_path) -> None:
        """Student prompt appears as the second user message."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("Source", "Generate a post about X")
        assert result.messages[1]["content"] == "Generate a post about X"

    def test_preserves_content_exactly(self, tmp_path) -> None:
        """Content strings are passed through without modification."""
        cm = _make_context_manager(tmp_path)
        source = "Straipsnis apie klimato kait\u0105 ir jos poveik\u012f \u017eem\u0117s \u016bkiui."
        prompt = "Para\u0161yk \u012fra\u0161\u0105 socialiniame tinkle."
        result = cm.assemble_generation_call(source, prompt)
        assert result.messages[0]["content"] == source
        assert result.messages[1]["content"] == prompt


# ---------------------------------------------------------------------------
# Context isolation (the critical invariant)
# ---------------------------------------------------------------------------


class TestContextIsolation:
    """Verifies the generation context contains ZERO teaching data."""

    def test_no_trickster_persona(self, tmp_path) -> None:
        """System prompt does not contain Trickster persona markers."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source", "prompt")
        prompt_lower = result.system_prompt.lower()
        for forbidden in ["triksteris", "trickster", "persona", "makaronas"]:
            assert forbidden not in prompt_lower, (
                f"Found forbidden term '{forbidden}' in generation system prompt"
            )

    def test_no_evaluation_terms(self, tmp_path) -> None:
        """System prompt does not contain evaluation-related terms."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source", "prompt")
        prompt_lower = result.system_prompt.lower()
        for forbidden in ["vertinimas", "rubric", "evaluation", "patterns_embedded",
                          "checklist"]:
            assert forbidden not in prompt_lower, (
                f"Found forbidden term '{forbidden}' in generation system prompt"
            )

    def test_no_teaching_context(self, tmp_path) -> None:
        """System prompt does not contain teaching/manipulation terms."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source", "prompt")
        prompt_lower = result.system_prompt.lower()
        for forbidden in ["manipuliacija", "mokymas", "pedagoginis"]:
            assert forbidden not in prompt_lower, (
                f"Found forbidden term '{forbidden}' in generation system prompt"
            )

    def test_no_safety_config(self, tmp_path) -> None:
        """System prompt does not contain safety configuration terms."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source", "prompt")
        prompt_lower = result.system_prompt.lower()
        for forbidden in ["content_boundaries", "intensity_ceiling", "safety_config"]:
            assert forbidden not in prompt_lower

    def test_no_task_history(self, tmp_path) -> None:
        """System prompt does not contain task history terms."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source", "prompt")
        prompt_lower = result.system_prompt.lower()
        for forbidden in ["task_history", "ankstesn", "de-escalation"]:
            assert forbidden not in prompt_lower

    def test_messages_contain_only_provided_content(self, tmp_path) -> None:
        """Messages contain only the source content and student prompt."""
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source only", "prompt only")
        all_content = " ".join(m["content"] for m in result.messages)
        assert "source only" in all_content
        assert "prompt only" in all_content
        # No other substantial content injected
        assert len(result.messages) == 2

    def test_does_not_use_prompt_loader(self, tmp_path) -> None:
        """Generation call works even with a loader that has no task prompts."""
        # The loader has only base prompts — no task-specific files.
        # If assemble_generation_call() tried to load prompts, it would
        # get None values. The test passes if no loading occurs.
        cm = _make_context_manager(tmp_path)
        result = cm.assemble_generation_call("source", "prompt")
        assert result.system_prompt == _GENERATION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# GameSession.generated_artifacts field
# ---------------------------------------------------------------------------


class TestGeneratedArtifactsField:
    """Tests for the generated_artifacts field on GameSession."""

    def test_defaults_to_empty_list(self, make_session) -> None:
        """New sessions have an empty generated_artifacts list."""
        session = make_session()
        assert session.generated_artifacts == []
        assert isinstance(session.generated_artifacts, list)

    def test_appendable(self, make_session) -> None:
        """Artifacts can be appended to the list."""
        session = make_session()
        artifact = {
            "student_prompt": "Generate a post",
            "generated_text": "Breaking news...",
            "timestamp": "2026-03-05T12:34:56Z",
        }
        session.generated_artifacts.append(artifact)
        assert len(session.generated_artifacts) == 1
        assert session.generated_artifacts[0]["student_prompt"] == "Generate a post"

    def test_multiple_artifacts(self, make_session) -> None:
        """Multiple artifacts accumulate."""
        session = make_session()
        for i in range(3):
            session.generated_artifacts.append({
                "student_prompt": f"Prompt {i}",
                "generated_text": f"Output {i}",
                "timestamp": f"2026-03-05T12:0{i}:00Z",
            })
        assert len(session.generated_artifacts) == 3

    def test_serialization_roundtrip(self, make_session) -> None:
        """Session with artifacts survives JSON serialization round-trip."""
        session = make_session()
        session.generated_artifacts.append({
            "student_prompt": "Para\u0161yk \u012fra\u0161\u0105",
            "generated_text": "Naujausi tyrimai rodo...",
            "timestamp": "2026-03-05T12:34:56Z",
        })
        data = session.model_dump(mode="json")
        restored = GameSession.model_validate(data)
        assert restored.generated_artifacts == session.generated_artifacts

    def test_backward_compatibility(self) -> None:
        """Sessions created without the field deserialize correctly."""
        # Simulate old session data without generated_artifacts
        old_data: dict[str, Any] = {
            "session_id": "old-session",
            "student_id": "student-1",
            "school_id": "school-1",
        }
        session = GameSession.model_validate(old_data)
        assert session.generated_artifacts == []
