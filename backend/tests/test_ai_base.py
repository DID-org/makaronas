"""Tests for backend.ai.providers.base — stream event types and AIProvider ABC."""

import dataclasses
from collections.abc import AsyncIterator

from backend.ai.providers.base import (
    AIProvider,
    ModelConfig,
    StreamEvent,
    TextChunk,
    ToolCallEvent,
    UsageInfo,
)


# ---------------------------------------------------------------------------
# TextChunk
# ---------------------------------------------------------------------------


class TestTextChunk:
    """Streamed text fragment — frozen dataclass."""

    def test_construction(self) -> None:
        tc = TextChunk(text="Hello")
        assert tc.text == "Hello"

    def test_frozen(self) -> None:
        tc = TextChunk(text="Hello")
        try:
            tc.text = "Changed"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

    def test_empty_text(self) -> None:
        tc = TextChunk(text="")
        assert tc.text == ""

    def test_equality(self) -> None:
        assert TextChunk(text="a") == TextChunk(text="a")
        assert TextChunk(text="a") != TextChunk(text="b")


# ---------------------------------------------------------------------------
# ToolCallEvent
# ---------------------------------------------------------------------------


class TestToolCallEvent:
    """Tool/function call event — frozen dataclass with dict arguments."""

    def test_construction(self) -> None:
        tce = ToolCallEvent(function_name="transition_phase", arguments={"signal": "understood"})
        assert tce.function_name == "transition_phase"
        assert tce.arguments == {"signal": "understood"}

    def test_frozen(self) -> None:
        tce = ToolCallEvent(function_name="test", arguments={})
        try:
            tce.function_name = "changed"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

    def test_empty_arguments(self) -> None:
        tce = ToolCallEvent(function_name="test", arguments={})
        assert tce.arguments == {}

    def test_nested_arguments(self) -> None:
        args = {"signal": "partial", "metadata": {"reason": "incomplete"}}
        tce = ToolCallEvent(function_name="test", arguments=args)
        assert tce.arguments["metadata"]["reason"] == "incomplete"


# ---------------------------------------------------------------------------
# UsageInfo
# ---------------------------------------------------------------------------


class TestUsageInfo:
    """Token usage from a completed AI call — frozen dataclass."""

    def test_construction(self) -> None:
        ui = UsageInfo(prompt_tokens=100, completion_tokens=50)
        assert ui.prompt_tokens == 100
        assert ui.completion_tokens == 50

    def test_frozen(self) -> None:
        ui = UsageInfo(prompt_tokens=100, completion_tokens=50)
        try:
            ui.prompt_tokens = 200  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

    def test_zero_tokens(self) -> None:
        ui = UsageInfo(prompt_tokens=0, completion_tokens=0)
        assert ui.prompt_tokens == 0
        assert ui.completion_tokens == 0


# ---------------------------------------------------------------------------
# StreamEvent union type
# ---------------------------------------------------------------------------


class TestStreamEvent:
    """StreamEvent = TextChunk | ToolCallEvent — isinstance dispatch."""

    def test_text_chunk_is_stream_event(self) -> None:
        tc = TextChunk(text="hello")
        assert isinstance(tc, TextChunk)

    def test_tool_call_is_stream_event(self) -> None:
        tce = ToolCallEvent(function_name="test", arguments={})
        assert isinstance(tce, ToolCallEvent)

    def test_isinstance_dispatch(self) -> None:
        """Consumers use isinstance() to discriminate stream events."""
        events: list[StreamEvent] = [
            TextChunk(text="Hello"),
            ToolCallEvent(function_name="transition_phase", arguments={"signal": "understood"}),
            TextChunk(text=" world"),
        ]
        text_parts = []
        tool_calls = []
        for event in events:
            if isinstance(event, TextChunk):
                text_parts.append(event.text)
            elif isinstance(event, ToolCallEvent):
                tool_calls.append(event.function_name)

        assert text_parts == ["Hello", " world"]
        assert tool_calls == ["transition_phase"]

    def test_usage_info_not_stream_event(self) -> None:
        """UsageInfo is separate — not part of the StreamEvent union."""
        ui = UsageInfo(prompt_tokens=10, completion_tokens=5)
        assert not isinstance(ui, TextChunk)
        assert not isinstance(ui, ToolCallEvent)


# ---------------------------------------------------------------------------
# AIProvider ABC — verify subclassing contract
# ---------------------------------------------------------------------------


class TestAIProviderABC:
    """AIProvider can be subclassed with correct method signatures."""

    def test_cannot_instantiate_abstract(self) -> None:
        """ABC cannot be instantiated directly."""
        try:
            AIProvider()  # type: ignore[abstract]
            assert False, "Expected TypeError"
        except TypeError:
            pass

    def test_subclass_with_methods(self) -> None:
        """A concrete subclass with both methods can be instantiated."""

        class DummyProvider(AIProvider):
            async def stream(
                self,
                *,
                system_prompt: str,
                messages: list[dict[str, str]],
                model_config: ModelConfig,
                tools: list[dict] | None = None,
            ) -> AsyncIterator[StreamEvent]:
                yield TextChunk(text="test")

            async def complete(
                self,
                *,
                system_prompt: str,
                messages: list[dict[str, str]],
                model_config: ModelConfig,
                tools: list[dict] | None = None,
            ) -> tuple[str, UsageInfo]:
                return "test", UsageInfo(prompt_tokens=0, completion_tokens=0)

        provider = DummyProvider()
        assert isinstance(provider, AIProvider)
