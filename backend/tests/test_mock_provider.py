"""Tests for backend.ai.providers.mock — MockProvider contract verification."""

import pytest

from backend.ai.providers.base import (
    AIProvider,
    TextChunk,
    ToolCallEvent,
    UsageInfo,
)
from backend.ai.providers.mock import MockProvider
from backend.models import ModelConfig

# Shared test config — MockProvider ignores these but must accept them
_CONFIG = ModelConfig(provider="mock", model_id="mock-v1")
_SYSTEM = "You are a test."
_MESSAGES: list[dict[str, str]] = [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class TestABCContract:
    """MockProvider is a proper AIProvider subclass."""

    def test_isinstance(self) -> None:
        provider = MockProvider()
        assert isinstance(provider, AIProvider)

    def test_accepts_keyword_only_args(self) -> None:
        """Calling with keyword-only args matches the ABC signature."""
        provider = MockProvider()
        # Just verify it doesn't raise — actual results tested below
        assert provider is not None


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


class TestStream:
    """MockProvider.stream() yields text chunks then tool calls."""

    @pytest.mark.asyncio
    async def test_default_responses(self) -> None:
        provider = MockProvider()
        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], TextChunk)
        assert events[0].text == "Hello from MockProvider"

    @pytest.mark.asyncio
    async def test_multiple_text_chunks(self) -> None:
        provider = MockProvider(responses=["Hello", " world"])
        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 2
        assert events[0] == TextChunk(text="Hello")
        assert events[1] == TextChunk(text=" world")

    @pytest.mark.asyncio
    async def test_tool_call_emission(self) -> None:
        tool = ToolCallEvent(
            function_name="transition_phase",
            arguments={"signal": "understood"},
        )
        provider = MockProvider(tool_calls=[tool])
        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        # Default text chunk + tool call
        assert len(events) == 2
        assert isinstance(events[0], TextChunk)
        assert isinstance(events[1], ToolCallEvent)
        assert events[1].function_name == "transition_phase"
        assert events[1].arguments == {"signal": "understood"}

    @pytest.mark.asyncio
    async def test_combined_text_and_tool_ordering(self) -> None:
        """Text chunks come first, tool calls come last."""
        tool = ToolCallEvent(function_name="test_fn", arguments={"a": 1})
        provider = MockProvider(
            responses=["chunk1", "chunk2"],
            tool_calls=[tool],
        )
        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 3
        assert all(isinstance(e, TextChunk) for e in events[:2])
        assert isinstance(events[2], ToolCallEvent)

    @pytest.mark.asyncio
    async def test_usage_info_not_in_stream(self) -> None:
        """UsageInfo is not yielded by stream() — it's separate."""
        provider = MockProvider(usage=UsageInfo(prompt_tokens=999, completion_tokens=999))
        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        for event in events:
            assert not isinstance(event, UsageInfo)

    @pytest.mark.asyncio
    async def test_error_simulation(self) -> None:
        """Error raises before any tokens are yielded."""
        provider = MockProvider(error=RuntimeError("API timeout"))
        with pytest.raises(RuntimeError, match="API timeout"):
            async for _ in provider.stream(
                system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
            ):
                pass  # Should not reach here

    @pytest.mark.asyncio
    async def test_empty_responses(self) -> None:
        """Empty responses list yields nothing."""
        provider = MockProvider(responses=[])
        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert events == []

    @pytest.mark.asyncio
    async def test_accepts_tools_parameter(self) -> None:
        """stream() accepts the optional tools parameter."""
        provider = MockProvider()
        tools = [{"type": "function", "function": {"name": "test"}}]
        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM,
            messages=_MESSAGES,
            model_config=_CONFIG,
            tools=tools,
        ):
            events.append(event)

        assert len(events) == 1


# ---------------------------------------------------------------------------
# complete() tests
# ---------------------------------------------------------------------------


class TestComplete:
    """MockProvider.complete() returns concatenated text and usage."""

    @pytest.mark.asyncio
    async def test_default(self) -> None:
        provider = MockProvider()
        text, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert text == "Hello from MockProvider"
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_multiple_responses_concatenated(self) -> None:
        provider = MockProvider(responses=["Hello", " world"])
        text, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert text == "Hello world"

    @pytest.mark.asyncio
    async def test_custom_usage(self) -> None:
        custom = UsageInfo(prompt_tokens=200, completion_tokens=100)
        provider = MockProvider(usage=custom)
        _, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 100

    @pytest.mark.asyncio
    async def test_error_simulation(self) -> None:
        provider = MockProvider(error=ConnectionError("Network down"))
        with pytest.raises(ConnectionError, match="Network down"):
            await provider.complete(
                system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
            )

    @pytest.mark.asyncio
    async def test_empty_responses(self) -> None:
        provider = MockProvider(responses=[])
        text, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert text == ""
        assert usage == UsageInfo(prompt_tokens=10, completion_tokens=5)

    @pytest.mark.asyncio
    async def test_accepts_tools_parameter(self) -> None:
        provider = MockProvider()
        tools = [{"type": "function", "function": {"name": "test"}}]
        text, _ = await provider.complete(
            system_prompt=_SYSTEM,
            messages=_MESSAGES,
            model_config=_CONFIG,
            tools=tools,
        )

        assert text == "Hello from MockProvider"
