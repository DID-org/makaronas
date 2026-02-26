"""Tests for backend.ai.providers.anthropic — AnthropicProvider contract verification.

All tests mock the anthropic SDK client. No real API calls.
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import anthropic

from backend.ai.providers.base import (
    AIProvider,
    TextChunk,
    ToolCallEvent,
    UsageInfo,
)
from backend.ai.providers.anthropic import (
    AnthropicProvider,
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_TEMPERATURE,
    _build_tools,
    _is_retryable,
)
from backend.models import ModelConfig

# Shared test config
_CONFIG = ModelConfig(provider="anthropic", model_id="claude-test-model")
_CONFIG_WITH_THINKING = ModelConfig(
    provider="anthropic", model_id="claude-test-model", thinking_budget=1024
)
_SYSTEM = "You are a test assistant."
_MESSAGES: list[dict[str, str]] = [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_text_delta_event(text: str) -> MagicMock:
    """Creates a mock streaming event for a text delta."""
    event = MagicMock()
    event.type = "content_block_delta"
    delta = MagicMock()
    delta.type = "text_delta"
    delta.text = text
    event.delta = delta
    return event


def _make_other_event(event_type: str = "message_start") -> MagicMock:
    """Creates a mock streaming event that is not a text delta."""
    event = MagicMock()
    event.type = event_type
    return event


def _make_text_block(text: str) -> MagicMock:
    """Creates a mock content block with type="text"."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(name: str, input_dict: dict) -> MagicMock:
    """Creates a mock content block with type="tool_use"."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_dict
    return block


def _make_usage(input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    """Creates a mock Usage object."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


def _make_message(
    content_blocks: list | None = None,
    usage: MagicMock | None = None,
) -> MagicMock:
    """Creates a mock Message with content blocks and usage."""
    if content_blocks is None:
        content_blocks = []
    if usage is None:
        usage = _make_usage()
    message = MagicMock()
    message.content = content_blocks
    message.usage = usage
    return message


class _MockStream:
    """Mock for the stream object inside the async context manager.

    Supports ``async for event in stream`` iteration and
    ``await stream.get_final_message()``.
    """

    def __init__(self, events: list, final_message: MagicMock) -> None:
        self._events = events
        self._final_message = final_message

    def __aiter__(self):
        return self._iter_events()

    async def _iter_events(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final_message


class _MockStreamContext:
    """Mock async context manager for ``client.messages.stream()``."""

    def __init__(self, stream: _MockStream) -> None:
        self._stream = stream

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *args):
        pass


def _make_stream_context(
    events: list,
    message: MagicMock | None = None,
) -> _MockStreamContext:
    """Creates a mock stream context manager.

    Args:
        events: List of mock streaming events to yield.
        message: Mock final message. Defaults to empty message with usage.
    """
    if message is None:
        message = _make_message()
    stream = _MockStream(events, message)
    return _MockStreamContext(stream)


def _make_anthropic_error(
    error_cls: type,
    status_code: int,
    message: str = "error",
) -> Exception:
    """Creates a mock Anthropic API error.

    Anthropic errors require an httpx.Response object in their constructor.
    """
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.headers = {}
    mock_response.request = MagicMock(spec=httpx.Request)
    return error_cls(
        message,
        response=mock_response,
        body={"error": {"message": message}},
    )


def _make_provider() -> AnthropicProvider:
    """Creates an AnthropicProvider with a mocked client."""
    with patch("backend.ai.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")
    return provider


def _setup_stream(
    provider: AnthropicProvider,
    events: list,
    message: MagicMock | None = None,
) -> MagicMock:
    """Configures the provider's mocked client to return streaming events."""
    ctx = _make_stream_context(events, message)
    provider._client.messages.stream = MagicMock(return_value=ctx)
    return provider._client.messages.stream


def _setup_complete(
    provider: AnthropicProvider,
    response: MagicMock,
) -> AsyncMock:
    """Configures the provider's mocked client to return a complete response."""
    provider._client.messages.create = AsyncMock(return_value=response)
    return provider._client.messages.create


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class TestABCContract:
    """AnthropicProvider is a proper AIProvider subclass."""

    def test_isinstance(self) -> None:
        provider = _make_provider()
        assert isinstance(provider, AIProvider)

    def test_has_stream_method(self) -> None:
        provider = _make_provider()
        assert hasattr(provider, "stream")
        assert inspect.isasyncgenfunction(provider.stream)

    def test_has_complete_method(self) -> None:
        provider = _make_provider()
        assert hasattr(provider, "complete")
        assert asyncio.iscoroutinefunction(provider.complete)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestBuildTools:
    """Tests for tool definition conversion."""

    def test_none_returns_none(self) -> None:
        assert _build_tools(None) is None

    def test_empty_list_returns_none(self) -> None:
        assert _build_tools([]) is None

    def test_single_tool_conversion(self) -> None:
        tools = [
            {
                "name": "transition_phase",
                "description": "Signal phase transition",
                "parameters": {
                    "type": "object",
                    "properties": {"signal": {"type": "string"}},
                },
            }
        ]
        result = _build_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "transition_phase"
        assert result[0]["description"] == "Signal phase transition"
        assert "input_schema" in result[0]
        assert result[0]["input_schema"]["type"] == "object"
        assert "parameters" not in result[0]

    def test_multiple_tools(self) -> None:
        tools = [
            {"name": "tool_a", "description": "A"},
            {"name": "tool_b", "description": "B", "parameters": {"type": "object"}},
        ]
        result = _build_tools(tools)
        assert len(result) == 2
        assert result[0]["name"] == "tool_a"
        assert result[1]["name"] == "tool_b"

    def test_missing_description_defaults_empty(self) -> None:
        tools = [{"name": "test_fn"}]
        result = _build_tools(tools)
        assert result[0]["description"] == ""

    def test_missing_parameters_defaults_empty_dict(self) -> None:
        tools = [{"name": "test_fn", "description": "test"}]
        result = _build_tools(tools)
        assert result[0]["input_schema"] == {}


class TestIsRetryable:
    """Tests for the _is_retryable helper."""

    def test_rate_limit_error_is_retryable(self) -> None:
        exc = _make_anthropic_error(anthropic.RateLimitError, 429, "rate limit")
        assert _is_retryable(exc) is True

    def test_internal_server_error_is_retryable(self) -> None:
        exc = _make_anthropic_error(anthropic.InternalServerError, 500, "server error")
        assert _is_retryable(exc) is True

    def test_bad_request_is_not_retryable(self) -> None:
        exc = _make_anthropic_error(anthropic.BadRequestError, 400, "bad request")
        assert _is_retryable(exc) is False

    def test_auth_error_is_not_retryable(self) -> None:
        exc = _make_anthropic_error(
            anthropic.AuthenticationError, 401, "unauthorized"
        )
        assert _is_retryable(exc) is False

    def test_permission_denied_is_not_retryable(self) -> None:
        exc = _make_anthropic_error(
            anthropic.PermissionDeniedError, 403, "forbidden"
        )
        assert _is_retryable(exc) is False

    def test_not_found_is_not_retryable(self) -> None:
        exc = _make_anthropic_error(anthropic.NotFoundError, 404, "not found")
        assert _is_retryable(exc) is False

    def test_generic_exception_is_not_retryable(self) -> None:
        assert _is_retryable(RuntimeError("oops")) is False


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


class TestStream:
    """AnthropicProvider.stream() yields text chunks and tool call events."""

    @pytest.mark.asyncio
    async def test_text_streaming(self) -> None:
        """Multiple text deltas yield TextChunk events in order."""
        provider = _make_provider()
        events = [
            _make_text_delta_event("Hello "),
            _make_text_delta_event("world"),
        ]
        message = _make_message(usage=_make_usage())
        _setup_stream(provider, events, message)

        result = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            result.append(event)

        assert len(result) == 2
        assert result[0] == TextChunk(text="Hello ")
        assert result[1] == TextChunk(text="world")

    @pytest.mark.asyncio
    async def test_tool_call_interception(self) -> None:
        """Tool use blocks in final message yield ToolCallEvent."""
        provider = _make_provider()
        tool_block = _make_tool_use_block(
            "transition_phase", {"signal": "understood"}
        )
        message = _make_message([tool_block], _make_usage())
        _setup_stream(provider, [], message)

        result = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            result.append(event)

        assert len(result) == 1
        assert isinstance(result[0], ToolCallEvent)
        assert result[0].function_name == "transition_phase"
        assert result[0].arguments == {"signal": "understood"}

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_calls(self) -> None:
        """Text streaming followed by tool call extraction."""
        provider = _make_provider()
        events = [_make_text_delta_event("Some text")]
        tool_block = _make_tool_use_block(
            "transition_phase", {"signal": "partial"}
        )
        text_block = _make_text_block("Some text")
        message = _make_message([text_block, tool_block], _make_usage())
        _setup_stream(provider, events, message)

        result = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            result.append(event)

        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert isinstance(result[1], ToolCallEvent)

    @pytest.mark.asyncio
    async def test_non_text_events_skipped(self) -> None:
        """Non-text-delta events are silently skipped."""
        provider = _make_provider()
        events = [
            _make_other_event("message_start"),
            _make_text_delta_event("visible"),
            _make_other_event("content_block_stop"),
        ]
        message = _make_message(usage=_make_usage())
        _setup_stream(provider, events, message)

        result = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            result.append(event)

        assert len(result) == 1
        assert result[0] == TextChunk(text="visible")

    @pytest.mark.asyncio
    async def test_empty_response(self) -> None:
        """No events and no content blocks yields nothing."""
        provider = _make_provider()
        message = _make_message([], _make_usage())
        _setup_stream(provider, [], message)

        result = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            result.append(event)

        assert result == []

    @pytest.mark.asyncio
    async def test_usage_extracted_from_final_message(self) -> None:
        """_last_usage is populated from the final message's usage."""
        provider = _make_provider()
        events = [_make_text_delta_event("text")]
        message = _make_message(usage=_make_usage(200, 100))
        _setup_stream(provider, events, message)

        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass

        assert provider._last_usage is not None
        assert provider._last_usage.prompt_tokens == 200
        assert provider._last_usage.completion_tokens == 100

    @pytest.mark.asyncio
    async def test_retry_on_429(self) -> None:
        """Retries after 429 rate limit, succeeds on second attempt."""
        provider = _make_provider()
        events = [_make_text_delta_event("recovered")]
        message = _make_message(usage=_make_usage())
        ctx = _make_stream_context(events, message)

        call_count = 0
        rate_limit_exc = _make_anthropic_error(
            anthropic.RateLimitError, 429, "rate limit"
        )

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise rate_limit_exc
            return ctx

        provider._client.messages.stream = MagicMock(side_effect=side_effect)

        result = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            result.append(event)

        assert call_count == 2
        assert len(result) == 1
        assert result[0] == TextChunk(text="recovered")

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self) -> None:
        """Retries after InternalServerError, succeeds on second attempt."""
        provider = _make_provider()
        events = [_make_text_delta_event("recovered")]
        message = _make_message(usage=_make_usage())
        ctx = _make_stream_context(events, message)

        call_count = 0
        server_exc = _make_anthropic_error(
            anthropic.InternalServerError, 500, "internal"
        )

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise server_exc
            return ctx

        provider._client.messages.stream = MagicMock(side_effect=side_effect)

        result = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            result.append(event)

        assert call_count == 2
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(self) -> None:
        """BadRequestError propagates immediately without retry."""
        provider = _make_provider()
        exc = _make_anthropic_error(
            anthropic.BadRequestError, 400, "bad request"
        )
        provider._client.messages.stream = MagicMock(side_effect=exc)

        with pytest.raises(anthropic.BadRequestError):
            async for _ in provider.stream(
                system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
            ):
                pass

    @pytest.mark.asyncio
    async def test_auth_error_propagates(self) -> None:
        """AuthenticationError propagates immediately."""
        provider = _make_provider()
        exc = _make_anthropic_error(
            anthropic.AuthenticationError, 401, "unauthorized"
        )
        provider._client.messages.stream = MagicMock(side_effect=exc)

        with pytest.raises(anthropic.AuthenticationError):
            async for _ in provider.stream(
                system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
            ):
                pass

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        """After max retries, the error propagates."""
        provider = _make_provider()
        exc = _make_anthropic_error(
            anthropic.RateLimitError, 429, "rate limit"
        )
        provider._client.messages.stream = MagicMock(side_effect=exc)

        with pytest.raises(anthropic.RateLimitError):
            async for _ in provider.stream(
                system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
            ):
                pass

    @pytest.mark.asyncio
    async def test_system_prompt_passed(self) -> None:
        """System prompt is passed as the 'system' parameter."""
        provider = _make_provider()
        message = _make_message(usage=_make_usage())
        mock_fn = _setup_stream(provider, [], message)

        async for _ in provider.stream(
            system_prompt="Custom system prompt",
            messages=_MESSAGES,
            model_config=_CONFIG,
        ):
            pass

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["system"] == "Custom system prompt"

    @pytest.mark.asyncio
    async def test_messages_passed_without_role_conversion(self) -> None:
        """Messages pass through as-is — no role mapping needed."""
        provider = _make_provider()
        message = _make_message(usage=_make_usage())
        mock_fn = _setup_stream(provider, [], message)

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=messages, model_config=_CONFIG
        ):
            pass

        mock_fn.assert_called_once()
        passed_messages = mock_fn.call_args.kwargs["messages"]
        assert passed_messages[0]["role"] == "user"
        assert passed_messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_tool_definitions_converted(self) -> None:
        """Tool dicts have 'parameters' converted to 'input_schema'."""
        provider = _make_provider()
        message = _make_message(usage=_make_usage())
        mock_fn = _setup_stream(provider, [], message)

        tools = [
            {
                "name": "transition_phase",
                "description": "Signal phase transition",
                "parameters": {
                    "type": "object",
                    "properties": {"signal": {"type": "string"}},
                },
            }
        ]
        async for _ in provider.stream(
            system_prompt=_SYSTEM,
            messages=_MESSAGES,
            model_config=_CONFIG,
            tools=tools,
        ):
            pass

        call_kwargs = mock_fn.call_args.kwargs
        passed_tools = call_kwargs["tools"]
        assert len(passed_tools) == 1
        assert passed_tools[0]["name"] == "transition_phase"
        assert "input_schema" in passed_tools[0]
        assert "parameters" not in passed_tools[0]

    @pytest.mark.asyncio
    async def test_no_tools_omitted_from_kwargs(self) -> None:
        """When tools=None, 'tools' key is not in the SDK call kwargs."""
        provider = _make_provider()
        message = _make_message(usage=_make_usage())
        mock_fn = _setup_stream(provider, [], message)

        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass

        call_kwargs = mock_fn.call_args.kwargs
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_temperature_default(self) -> None:
        """Temperature defaults to 0.85."""
        provider = _make_provider()
        message = _make_message(usage=_make_usage())
        mock_fn = _setup_stream(provider, [], message)

        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass

        assert mock_fn.call_args.kwargs["temperature"] == 0.85

    @pytest.mark.asyncio
    async def test_max_tokens_default(self) -> None:
        """max_tokens defaults to _DEFAULT_MAX_TOKENS."""
        provider = _make_provider()
        message = _make_message(usage=_make_usage())
        mock_fn = _setup_stream(provider, [], message)

        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass

        assert mock_fn.call_args.kwargs["max_tokens"] == _DEFAULT_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_model_id_passed(self) -> None:
        """model_config.model_id is passed as the 'model' parameter."""
        provider = _make_provider()
        message = _make_message(usage=_make_usage())
        mock_fn = _setup_stream(provider, [], message)

        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass

        assert mock_fn.call_args.kwargs["model"] == "claude-test-model"

    @pytest.mark.asyncio
    async def test_thinking_budget_ignored(self) -> None:
        """thinking_budget > 0 doesn't cause errors or modify SDK call."""
        provider = _make_provider()
        message = _make_message(usage=_make_usage())
        mock_fn = _setup_stream(provider, [], message)

        async for _ in provider.stream(
            system_prompt=_SYSTEM,
            messages=_MESSAGES,
            model_config=_CONFIG_WITH_THINKING,
        ):
            pass

        # Verify no thinking-related params in the SDK call
        call_kwargs = mock_fn.call_args.kwargs
        assert "thinking" not in call_kwargs
        assert "thinking_budget" not in call_kwargs


# ---------------------------------------------------------------------------
# complete() tests
# ---------------------------------------------------------------------------


class TestComplete:
    """AnthropicProvider.complete() returns full text and usage info."""

    @pytest.mark.asyncio
    async def test_full_text_returned(self) -> None:
        """Concatenates text from all text content blocks."""
        provider = _make_provider()
        response = _make_message(
            [_make_text_block("Hello "), _make_text_block("world")],
            _make_usage(150, 75),
        )
        _setup_complete(provider, response)

        text, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert text == "Hello world"
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 75

    @pytest.mark.asyncio
    async def test_usage_extraction(self) -> None:
        """UsageInfo is extracted from response.usage."""
        provider = _make_provider()
        response = _make_message(
            [_make_text_block("text")],
            _make_usage(300, 200),
        )
        _setup_complete(provider, response)

        _, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert usage == UsageInfo(prompt_tokens=300, completion_tokens=200)

    @pytest.mark.asyncio
    async def test_thinking_budget_ignored(self) -> None:
        """thinking_budget > 0 doesn't cause errors."""
        provider = _make_provider()
        response = _make_message(
            [_make_text_block("text")],
            _make_usage(),
        )
        _setup_complete(provider, response)

        text, _ = await provider.complete(
            system_prompt=_SYSTEM,
            messages=_MESSAGES,
            model_config=_CONFIG_WITH_THINKING,
        )

        assert text == "text"

    @pytest.mark.asyncio
    async def test_retry_on_429(self) -> None:
        """Retries after 429, succeeds on second attempt."""
        provider = _make_provider()
        response = _make_message(
            [_make_text_block("ok")], _make_usage()
        )
        rate_limit_exc = _make_anthropic_error(
            anthropic.RateLimitError, 429, "rate limit"
        )

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise rate_limit_exc
            return response

        provider._client.messages.create = AsyncMock(side_effect=side_effect)

        text, _ = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert text == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(self) -> None:
        """BadRequestError propagates immediately."""
        provider = _make_provider()
        exc = _make_anthropic_error(
            anthropic.BadRequestError, 400, "bad request"
        )
        provider._client.messages.create = AsyncMock(side_effect=exc)

        with pytest.raises(anthropic.BadRequestError):
            await provider.complete(
                system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
            )

    @pytest.mark.asyncio
    async def test_system_prompt_and_messages(self) -> None:
        """Verifies system prompt and messages are passed to SDK."""
        provider = _make_provider()
        response = _make_message(
            [_make_text_block("text")], _make_usage()
        )
        mock_fn = _setup_complete(provider, response)

        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        await provider.complete(
            system_prompt="Be helpful", messages=messages, model_config=_CONFIG
        )

        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["system"] == "Be helpful"
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_tool_use_blocks_in_response(self) -> None:
        """Tool use blocks are ignored — only text blocks contribute to text."""
        provider = _make_provider()
        response = _make_message(
            [
                _make_text_block("Some text"),
                _make_tool_use_block("transition_phase", {"signal": "understood"}),
            ],
            _make_usage(),
        )
        _setup_complete(provider, response)

        text, _ = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert text == "Some text"

    @pytest.mark.asyncio
    async def test_empty_content(self) -> None:
        """Empty content blocks produce empty text."""
        provider = _make_provider()
        response = _make_message([], _make_usage(10, 5))
        _setup_complete(provider, response)

        text, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert text == ""
        assert usage == UsageInfo(prompt_tokens=10, completion_tokens=5)


# ---------------------------------------------------------------------------
# Streaming usage tracking
# ---------------------------------------------------------------------------


class TestStreamingUsage:
    """Tests for _last_usage attribute populated after streaming."""

    @pytest.mark.asyncio
    async def test_usage_populated_after_stream(self) -> None:
        """_last_usage holds correct UsageInfo after stream consumption."""
        provider = _make_provider()
        events = [_make_text_delta_event("token")]
        message = _make_message(usage=_make_usage(500, 250))
        _setup_stream(provider, events, message)

        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass

        assert provider._last_usage == UsageInfo(
            prompt_tokens=500, completion_tokens=250
        )

    @pytest.mark.asyncio
    async def test_usage_reset_between_calls(self) -> None:
        """_last_usage is reset to None at the start of each stream call."""
        provider = _make_provider()

        # First call — sets usage
        events1 = [_make_text_delta_event("first")]
        message1 = _make_message(usage=_make_usage(100, 50))
        _setup_stream(provider, events1, message1)
        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass
        assert provider._last_usage is not None

        # Second call — different usage
        events2 = [_make_text_delta_event("second")]
        message2 = _make_message(usage=_make_usage(200, 100))
        _setup_stream(provider, events2, message2)
        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass
        assert provider._last_usage == UsageInfo(
            prompt_tokens=200, completion_tokens=100
        )

    @pytest.mark.asyncio
    async def test_usage_none_initially(self) -> None:
        """_last_usage starts as None."""
        provider = _make_provider()
        assert provider._last_usage is None


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------


class TestClientInit:
    """Tests for client creation and configuration."""

    def test_sdk_retry_disabled(self) -> None:
        """SDK built-in retry is disabled via max_retries=0."""
        with patch(
            "backend.ai.providers.anthropic.anthropic.AsyncAnthropic"
        ) as mock_client_cls:
            AnthropicProvider(api_key="test-key")

        mock_client_cls.assert_called_once()
        call_kwargs = mock_client_cls.call_args.kwargs
        assert call_kwargs["max_retries"] == 0

    def test_api_key_passed(self) -> None:
        """API key is passed to the client."""
        with patch(
            "backend.ai.providers.anthropic.anthropic.AsyncAnthropic"
        ) as mock_client_cls:
            AnthropicProvider(api_key="my-secret-key")

        assert mock_client_cls.call_args.kwargs["api_key"] == "my-secret-key"
