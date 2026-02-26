"""Tests for backend.ai.providers.gemini — GeminiProvider contract verification.

All tests mock the google-genai SDK client. No real API calls.
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from backend.ai.providers.base import (
    AIProvider,
    TextChunk,
    ToolCallEvent,
    UsageInfo,
)
from backend.ai.providers.gemini import (
    GeminiProvider,
    _DEFAULT_TEMPERATURE,
    _build_config,
    _build_contents,
    _build_tools,
)
from backend.models import ModelConfig

# Shared test config
_CONFIG = ModelConfig(provider="gemini", model_id="gemini-test-model")
_CONFIG_WITH_THINKING = ModelConfig(
    provider="gemini", model_id="gemini-test-model", thinking_budget=1024
)
_SYSTEM = "You are a test assistant."
_MESSAGES: list[dict[str, str]] = [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_text_part(text: str, thought: bool = False) -> MagicMock:
    """Creates a mock Part with text content."""
    part = MagicMock()
    part.text = text
    part.thought = thought
    part.function_call = None
    return part


def _make_function_call_part(name: str, args: dict) -> MagicMock:
    """Creates a mock Part with a function call."""
    fc = MagicMock()
    fc.name = name
    fc.args = args

    part = MagicMock()
    part.text = None
    part.thought = False
    part.function_call = fc
    return part


def _make_chunk(
    parts: list | None = None,
    usage_metadata: MagicMock | None = None,
    empty_candidates: bool = False,
) -> MagicMock:
    """Creates a mock GenerateContentResponse chunk."""
    chunk = MagicMock()

    if empty_candidates:
        chunk.candidates = []
        chunk.usage_metadata = usage_metadata
        return chunk

    if parts is None:
        parts = []

    content = MagicMock()
    content.parts = parts

    candidate = MagicMock()
    candidate.content = content

    chunk.candidates = [candidate]
    chunk.usage_metadata = usage_metadata
    return chunk


def _make_usage(prompt: int = 100, completion: int = 50) -> MagicMock:
    """Creates a mock UsageMetadata."""
    usage = MagicMock()
    usage.prompt_token_count = prompt
    usage.candidates_token_count = completion
    return usage


async def _async_iter(items: list):
    """Creates an async iterator from a list of items."""
    for item in items:
        yield item


def _make_provider() -> GeminiProvider:
    """Creates a GeminiProvider with a mocked client."""
    with patch("backend.ai.providers.gemini.genai.Client"):
        provider = GeminiProvider(api_key="test-key")
    return provider


def _setup_stream(provider: GeminiProvider, chunks: list) -> AsyncMock:
    """Configures the provider's mocked client to return streaming chunks."""
    mock_stream = _async_iter(chunks)
    provider._client.aio.models.generate_content_stream = AsyncMock(
        return_value=mock_stream
    )
    return provider._client.aio.models.generate_content_stream


def _setup_complete(provider: GeminiProvider, response: MagicMock) -> AsyncMock:
    """Configures the provider's mocked client to return a complete response."""
    provider._client.aio.models.generate_content = AsyncMock(return_value=response)
    return provider._client.aio.models.generate_content


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class TestABCContract:
    """GeminiProvider is a proper AIProvider subclass."""

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


class TestBuildContents:
    """Tests for message conversion to Gemini Content objects."""

    def test_user_role_preserved(self) -> None:
        contents = _build_contents([{"role": "user", "content": "Hello"}])
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello"

    def test_assistant_mapped_to_model(self) -> None:
        contents = _build_contents([{"role": "assistant", "content": "Hi"}])
        assert len(contents) == 1
        assert contents[0].role == "model"

    def test_multi_turn(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        contents = _build_contents(messages)
        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"

    def test_empty_messages(self) -> None:
        contents = _build_contents([])
        assert contents == []


class TestBuildTools:
    """Tests for tool definition conversion."""

    def test_none_returns_none(self) -> None:
        assert _build_tools(None) is None

    def test_empty_list_returns_none(self) -> None:
        assert _build_tools([]) is None

    def test_single_tool(self) -> None:
        tools = [
            {
                "name": "transition_phase",
                "description": "Signals a phase transition",
                "parameters": {"type": "object", "properties": {"signal": {"type": "string"}}},
            }
        ]
        result = _build_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], types.Tool)
        assert len(result[0].function_declarations) == 1
        assert result[0].function_declarations[0].name == "transition_phase"

    def test_multiple_tools(self) -> None:
        tools = [
            {"name": "tool_a", "description": "A"},
            {"name": "tool_b", "description": "B", "parameters": {"type": "object"}},
        ]
        result = _build_tools(tools)
        assert len(result[0].function_declarations) == 2


class TestBuildConfig:
    """Tests for GenerateContentConfig assembly."""

    def test_system_prompt_set(self) -> None:
        config = _build_config("Test system", _CONFIG, None)
        assert config.system_instruction == "Test system"

    def test_temperature_default(self) -> None:
        config = _build_config("Test", _CONFIG, None)
        assert config.temperature == _DEFAULT_TEMPERATURE

    def test_thinking_budget_passed(self) -> None:
        config = _build_config("Test", _CONFIG_WITH_THINKING, None)
        assert config.thinking_config.thinking_budget == 1024

    def test_thinking_budget_zero(self) -> None:
        config = _build_config("Test", _CONFIG, None)
        assert config.thinking_config.thinking_budget == 0

    def test_no_tools_means_no_tool_config(self) -> None:
        config = _build_config("Test", _CONFIG, None)
        assert config.tools is None

    def test_tools_set_with_auto_calling_disabled(self) -> None:
        tools = [{"name": "test_fn", "description": "test"}]
        config = _build_config("Test", _CONFIG, tools)
        assert config.tools is not None
        assert config.automatic_function_calling is not None
        assert config.automatic_function_calling.disable is True


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


class TestStream:
    """GeminiProvider.stream() yields text chunks and tool call events."""

    @pytest.mark.asyncio
    async def test_text_streaming(self) -> None:
        """Multiple text chunks yield TextChunk events in order."""
        provider = _make_provider()
        chunks = [
            _make_chunk([_make_text_part("Hello ")]),
            _make_chunk([_make_text_part("world")]),
        ]
        _setup_stream(provider, chunks)

        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 2
        assert events[0] == TextChunk(text="Hello ")
        assert events[1] == TextChunk(text="world")

    @pytest.mark.asyncio
    async def test_tool_call_interception(self) -> None:
        """Function call parts yield ToolCallEvent with correct fields."""
        provider = _make_provider()
        chunks = [
            _make_chunk([
                _make_function_call_part("transition_phase", {"signal": "understood"})
            ]),
        ]
        _setup_stream(provider, chunks)

        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].function_name == "transition_phase"
        assert events[0].arguments == {"signal": "understood"}

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_calls(self) -> None:
        """Text followed by a tool call yields both event types."""
        provider = _make_provider()
        chunks = [
            _make_chunk([_make_text_part("Some text")]),
            _make_chunk([
                _make_function_call_part("transition_phase", {"signal": "partial"})
            ]),
        ]
        _setup_stream(provider, chunks)

        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], TextChunk)
        assert isinstance(events[1], ToolCallEvent)

    @pytest.mark.asyncio
    async def test_thinking_parts_skipped(self) -> None:
        """Parts with thought=True are not yielded."""
        provider = _make_provider()
        chunks = [
            _make_chunk([
                _make_text_part("thinking...", thought=True),
                _make_text_part("visible text"),
            ]),
        ]
        _setup_stream(provider, chunks)

        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0] == TextChunk(text="visible text")

    @pytest.mark.asyncio
    async def test_empty_candidates_handled(self) -> None:
        """Chunks with empty candidates are skipped gracefully."""
        provider = _make_provider()
        chunks = [
            _make_chunk(empty_candidates=True),
            _make_chunk([_make_text_part("after safety block")]),
        ]
        _setup_stream(provider, chunks)

        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0] == TextChunk(text="after safety block")

    @pytest.mark.asyncio
    async def test_usage_extracted_from_final_chunk(self) -> None:
        """_last_usage is populated from the final chunk's usage_metadata."""
        provider = _make_provider()
        usage = _make_usage(prompt=200, completion=100)
        chunks = [
            _make_chunk([_make_text_part("text")]),
            _make_chunk([_make_text_part("more")], usage_metadata=usage),
        ]
        _setup_stream(provider, chunks)

        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass

        assert provider._last_usage is not None
        assert provider._last_usage.prompt_tokens == 200
        assert provider._last_usage.completion_tokens == 100

    @pytest.mark.asyncio
    async def test_usage_none_before_streaming(self) -> None:
        """_last_usage is None before stream is consumed."""
        provider = _make_provider()
        assert provider._last_usage is None

    @pytest.mark.asyncio
    async def test_retry_on_429(self) -> None:
        """Retries after 429 rate limit, succeeds on second attempt."""
        provider = _make_provider()
        chunks = [_make_chunk([_make_text_part("recovered")])]

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise genai_errors.ClientError(429, {"error": {"message": "rate limit"}})
            return _async_iter(chunks)

        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=side_effect
        )

        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert call_count == 2
        assert len(events) == 1
        assert events[0] == TextChunk(text="recovered")

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self) -> None:
        """Retries after ServerError, succeeds on second attempt."""
        provider = _make_provider()
        chunks = [_make_chunk([_make_text_part("recovered")])]

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise genai_errors.ServerError(500, {"error": {"message": "internal"}})
            return _async_iter(chunks)

        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=side_effect
        )

        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert call_count == 2
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(self) -> None:
        """400/403 errors propagate immediately without retry."""
        provider = _make_provider()

        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=genai_errors.ClientError(
                403, {"error": {"message": "forbidden"}}
            )
        )

        with pytest.raises(genai_errors.ClientError) as exc_info:
            async for _ in provider.stream(
                system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
            ):
                pass

        assert exc_info.value.code == 403

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        """After max retries, the error propagates."""
        provider = _make_provider()

        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=genai_errors.ClientError(
                429, {"error": {"message": "rate limit"}}
            )
        )

        with pytest.raises(genai_errors.ClientError) as exc_info:
            async for _ in provider.stream(
                system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
            ):
                pass

        assert exc_info.value.code == 429

    @pytest.mark.asyncio
    async def test_system_prompt_passed(self) -> None:
        """System prompt is included in GenerateContentConfig."""
        provider = _make_provider()
        _setup_stream(provider, [])
        mock_fn = provider._client.aio.models.generate_content_stream

        async for _ in provider.stream(
            system_prompt="Custom system prompt",
            messages=_MESSAGES,
            model_config=_CONFIG,
        ):
            pass

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args
        config = call_kwargs.kwargs["config"]
        assert config.system_instruction == "Custom system prompt"

    @pytest.mark.asyncio
    async def test_message_role_mapping(self) -> None:
        """'assistant' role maps to 'model' in SDK call."""
        provider = _make_provider()
        _setup_stream(provider, [])
        mock_fn = provider._client.aio.models.generate_content_stream

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=messages, model_config=_CONFIG
        ):
            pass

        mock_fn.assert_called_once()
        contents = mock_fn.call_args.kwargs["contents"]
        assert contents[0].role == "user"
        assert contents[1].role == "model"

    @pytest.mark.asyncio
    async def test_tool_definitions_converted(self) -> None:
        """Tool dicts are converted to FunctionDeclaration in config."""
        provider = _make_provider()
        _setup_stream(provider, [])
        mock_fn = provider._client.aio.models.generate_content_stream

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

        config = mock_fn.call_args.kwargs["config"]
        assert config.tools is not None
        assert len(config.tools) == 1
        decls = config.tools[0].function_declarations
        assert decls[0].name == "transition_phase"

    @pytest.mark.asyncio
    async def test_thinking_budget_in_config(self) -> None:
        """ThinkingConfig is set from ModelConfig.thinking_budget."""
        provider = _make_provider()
        _setup_stream(provider, [])
        mock_fn = provider._client.aio.models.generate_content_stream

        async for _ in provider.stream(
            system_prompt=_SYSTEM,
            messages=_MESSAGES,
            model_config=_CONFIG_WITH_THINKING,
        ):
            pass

        config = mock_fn.call_args.kwargs["config"]
        assert config.thinking_config.thinking_budget == 1024

    @pytest.mark.asyncio
    async def test_temperature_default(self) -> None:
        """Temperature defaults to 0.85."""
        provider = _make_provider()
        _setup_stream(provider, [])
        mock_fn = provider._client.aio.models.generate_content_stream

        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass

        config = mock_fn.call_args.kwargs["config"]
        assert config.temperature == 0.85

    @pytest.mark.asyncio
    async def test_model_id_passed(self) -> None:
        """model_config.model_id is passed as the model parameter."""
        provider = _make_provider()
        _setup_stream(provider, [])
        mock_fn = provider._client.aio.models.generate_content_stream

        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass

        assert mock_fn.call_args.kwargs["model"] == "gemini-test-model"

    @pytest.mark.asyncio
    async def test_null_content_candidate_handled(self) -> None:
        """Candidate with None content is skipped."""
        provider = _make_provider()
        candidate = MagicMock()
        candidate.content = None

        chunk = MagicMock()
        chunk.candidates = [candidate]
        chunk.usage_metadata = None

        _setup_stream(provider, [chunk])

        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert events == []

    @pytest.mark.asyncio
    async def test_function_call_with_none_args(self) -> None:
        """Function call with None args yields empty dict."""
        provider = _make_provider()
        fc = MagicMock()
        fc.name = "test_fn"
        fc.args = None

        part = MagicMock()
        part.text = None
        part.thought = False
        part.function_call = fc

        chunks = [_make_chunk([part])]
        _setup_stream(provider, chunks)

        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0].arguments == {}


# ---------------------------------------------------------------------------
# complete() tests
# ---------------------------------------------------------------------------


class TestComplete:
    """GeminiProvider.complete() returns full text and usage info."""

    @pytest.mark.asyncio
    async def test_full_text_returned(self) -> None:
        """Concatenates text from all parts."""
        provider = _make_provider()
        response = _make_chunk(
            [_make_text_part("Hello "), _make_text_part("world")],
            usage_metadata=_make_usage(150, 75),
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
        """UsageInfo is extracted from response.usage_metadata."""
        provider = _make_provider()
        response = _make_chunk(
            [_make_text_part("text")],
            usage_metadata=_make_usage(300, 200),
        )
        _setup_complete(provider, response)

        _, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert usage == UsageInfo(prompt_tokens=300, completion_tokens=200)

    @pytest.mark.asyncio
    async def test_thinking_parts_excluded_from_text(self) -> None:
        """Thinking parts are not included in the response text."""
        provider = _make_provider()
        response = _make_chunk(
            [
                _make_text_part("thinking...", thought=True),
                _make_text_part("visible"),
            ],
            usage_metadata=_make_usage(),
        )
        _setup_complete(provider, response)

        text, _ = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert text == "visible"

    @pytest.mark.asyncio
    async def test_retry_on_429(self) -> None:
        """Retries after 429, succeeds on second attempt."""
        provider = _make_provider()
        response = _make_chunk(
            [_make_text_part("ok")], usage_metadata=_make_usage()
        )

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise genai_errors.ClientError(429, {"error": {"message": "rate limit"}})
            return response

        provider._client.aio.models.generate_content = AsyncMock(
            side_effect=side_effect
        )

        text, _ = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert text == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(self) -> None:
        """400 error propagates immediately."""
        provider = _make_provider()
        provider._client.aio.models.generate_content = AsyncMock(
            side_effect=genai_errors.ClientError(
                400, {"error": {"message": "bad request"}}
            )
        )

        with pytest.raises(genai_errors.ClientError) as exc_info:
            await provider.complete(
                system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
            )

        assert exc_info.value.code == 400

    @pytest.mark.asyncio
    async def test_system_prompt_and_messages(self) -> None:
        """Verifies system prompt and messages are passed to SDK."""
        provider = _make_provider()
        response = _make_chunk(
            [_make_text_part("text")], usage_metadata=_make_usage()
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
        assert call_kwargs["config"].system_instruction == "Be helpful"
        contents = call_kwargs["contents"]
        assert contents[0].role == "user"
        assert contents[1].role == "model"

    @pytest.mark.asyncio
    async def test_empty_candidates(self) -> None:
        """Empty candidates produces empty text."""
        provider = _make_provider()
        response = _make_chunk(empty_candidates=True, usage_metadata=_make_usage())
        _setup_complete(provider, response)

        text, _ = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert text == ""

    @pytest.mark.asyncio
    async def test_no_usage_metadata(self) -> None:
        """Missing usage_metadata returns zero tokens."""
        provider = _make_provider()
        response = _make_chunk([_make_text_part("text")])
        response.usage_metadata = None
        _setup_complete(provider, response)

        _, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        )

        assert usage == UsageInfo(prompt_tokens=0, completion_tokens=0)


# ---------------------------------------------------------------------------
# Streaming usage tracking
# ---------------------------------------------------------------------------


class TestStreamingUsage:
    """Tests for _last_usage attribute populated after streaming."""

    @pytest.mark.asyncio
    async def test_usage_populated_after_stream(self) -> None:
        """_last_usage holds correct UsageInfo after stream consumption."""
        provider = _make_provider()
        chunks = [
            _make_chunk([_make_text_part("token")]),
            _make_chunk(
                [_make_text_part("done")],
                usage_metadata=_make_usage(500, 250),
            ),
        ]
        _setup_stream(provider, chunks)

        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass

        assert provider._last_usage == UsageInfo(prompt_tokens=500, completion_tokens=250)

    @pytest.mark.asyncio
    async def test_usage_reset_between_calls(self) -> None:
        """_last_usage is reset to None at the start of each stream call."""
        provider = _make_provider()

        # First call — sets usage
        chunks1 = [
            _make_chunk(
                [_make_text_part("first")],
                usage_metadata=_make_usage(100, 50),
            ),
        ]
        _setup_stream(provider, chunks1)
        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass
        assert provider._last_usage is not None

        # Second call — no usage metadata in chunks
        chunks2 = [_make_chunk([_make_text_part("second")])]
        _setup_stream(provider, chunks2)
        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=_MESSAGES, model_config=_CONFIG
        ):
            pass
        # _last_usage should be None since no usage metadata in this stream
        assert provider._last_usage is None

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
        """SDK built-in retry is disabled via HttpRetryOptions."""
        with patch("backend.ai.providers.gemini.genai.Client") as mock_client_cls:
            GeminiProvider(api_key="test-key")

        mock_client_cls.assert_called_once()
        call_kwargs = mock_client_cls.call_args.kwargs
        http_options = call_kwargs["http_options"]
        assert http_options.retry_options.attempts == 1

    def test_api_key_passed(self) -> None:
        """API key is passed to the client."""
        with patch("backend.ai.providers.gemini.genai.Client") as mock_client_cls:
            GeminiProvider(api_key="my-secret-key")

        assert mock_client_cls.call_args.kwargs["api_key"] == "my-secret-key"
