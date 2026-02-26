"""Anthropic Claude AI provider using the anthropic SDK.

Implements the AIProvider contract for Anthropic's Claude model family.
Handles streaming and non-streaming calls, tool call interception,
and exponential backoff retry for transient errors.

Streaming uses raw event iteration (``async for event in stream``)
rather than ``text_stream`` which does not exist in anthropic SDK v0.84.
Tool calls are extracted from ``get_final_message()`` to avoid manual
JSON delta accumulation. Mid-stream retries may duplicate tokens —
acceptable for MVP (see plan §6.6).

Tier 2 service — imports from base.py (Tier 1) + anthropic SDK.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

import anthropic

from backend.ai.providers.base import (
    AIProvider,
    ModelConfig,
    StreamEvent,
    TextChunk,
    ToolCallEvent,
    UsageInfo,
)

logger = logging.getLogger(__name__)

_DEFAULT_TEMPERATURE = 0.85
_DEFAULT_MAX_TOKENS = 4096  # Sufficient for trickster dialogue (200-500 tokens typical)
_MAX_RETRIES = 2  # 3 total attempts
_BACKOFF_BASE = 1.0  # seconds — doubles each retry


def _is_retryable(exc: Exception) -> bool:
    """Checks whether an SDK error is transient and worth retrying.

    Retries on:
    - RateLimitError (429)
    - InternalServerError (500+)

    All other API errors (400, 401, 403, 404) propagate immediately.
    """
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.InternalServerError):
        return True
    return False


def _build_tools(tools: list[dict] | None) -> list[dict] | None:
    """Converts generic tool dicts to Anthropic tool format.

    Input format:  [{"name": "...", "description": "...", "parameters": {...}}]
    Output format: [{"name": "...", "description": "...", "input_schema": {...}}]
    """
    if not tools:
        return None
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {}),
        }
        for t in tools
    ]


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider using the anthropic SDK.

    Implements streaming and non-streaming AI calls through Anthropic's
    Messages API. Handles retry with exponential backoff for transient
    errors (429, 5xx) and surfaces tool calls as ToolCallEvent.

    Args:
        api_key: Anthropic API key for Claude access.
    """

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            max_retries=0,
        )
        self._last_usage: UsageInfo | None = None

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streams a response as text chunks and tool call events.

        Retries on transient errors (429, 5xx) with exponential backoff.
        After full consumption, ``_last_usage`` holds the UsageInfo from
        the final message.

        Uses raw event iteration (not ``text_stream`` which does not exist
        in SDK v0.84). Tool calls are extracted from ``get_final_message()``
        after all events are consumed.

        Args:
            system_prompt: The assembled system instruction.
            messages: Conversation history as {"role": ..., "content": ...} dicts.
            model_config: Provider-specific configuration (model ID, thinking budget).
            tools: Optional tool definitions for function calling.

        Yields:
            TextChunk for text tokens, ToolCallEvent for tool invocations.
        """
        self._last_usage = None
        anthropic_tools = _build_tools(tools)

        if model_config.thinking_budget > 0:
            logger.debug(
                "thinking_budget=%d ignored for Anthropic provider",
                model_config.thinking_budget,
            )

        kwargs: dict = {
            "model": model_config.model_id,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": _DEFAULT_MAX_TOKENS,
            "temperature": _DEFAULT_TEMPERATURE,
        }
        if anthropic_tools is not None:
            kwargs["tools"] = anthropic_tools

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                backoff = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Anthropic stream retry %d/%d after %.1fs backoff",
                    attempt,
                    _MAX_RETRIES,
                    backoff,
                )
                await asyncio.sleep(backoff)
            try:
                async with self._client.messages.stream(**kwargs) as stream:
                    # Yield text deltas during streaming
                    async for event in stream:
                        if (
                            event.type == "content_block_delta"
                            and event.delta.type == "text_delta"
                        ):
                            yield TextChunk(text=event.delta.text)

                    # Get final message for tool calls + usage
                    message = await stream.get_final_message()

                    # Extract tool calls from content blocks
                    for block in message.content:
                        if block.type == "tool_use":
                            yield ToolCallEvent(
                                function_name=block.name,
                                arguments=block.input,
                            )

                    # Extract usage
                    self._last_usage = UsageInfo(
                        prompt_tokens=message.usage.input_tokens,
                        completion_tokens=message.usage.output_tokens,
                    )

                # Stream completed successfully — no retry needed
                return

            except anthropic.APIStatusError as exc:
                if not _is_retryable(exc) or attempt == _MAX_RETRIES:
                    raise
                last_exc = exc

        # Should not reach here, but just in case
        if last_exc is not None:  # pragma: no cover
            raise last_exc

    async def complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        tools: list[dict] | None = None,
    ) -> tuple[str, UsageInfo]:
        """Returns the full response text and usage info (non-streaming).

        Retries on transient errors (429, 5xx) with exponential backoff.

        Args:
            system_prompt: The assembled system instruction.
            messages: Conversation history as {"role": ..., "content": ...} dicts.
            model_config: Provider-specific configuration (model ID, thinking budget).
            tools: Optional tool definitions for function calling.

        Returns:
            Tuple of (full response text, token usage information).
        """
        anthropic_tools = _build_tools(tools)

        if model_config.thinking_budget > 0:
            logger.debug(
                "thinking_budget=%d ignored for Anthropic provider",
                model_config.thinking_budget,
            )

        kwargs: dict = {
            "model": model_config.model_id,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": _DEFAULT_MAX_TOKENS,
            "temperature": _DEFAULT_TEMPERATURE,
        }
        if anthropic_tools is not None:
            kwargs["tools"] = anthropic_tools

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                backoff = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Anthropic complete retry %d/%d after %.1fs backoff",
                    attempt,
                    _MAX_RETRIES,
                    backoff,
                )
                await asyncio.sleep(backoff)
            try:
                response = await self._client.messages.create(**kwargs)

                # Concatenate text from all text content blocks
                parts_text = []
                for block in response.content:
                    if block.type == "text":
                        parts_text.append(block.text)

                full_text = "".join(parts_text)

                # Extract usage
                usage = UsageInfo(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                )

                return full_text, usage

            except anthropic.APIStatusError as exc:
                if not _is_retryable(exc) or attempt == _MAX_RETRIES:
                    raise
                last_exc = exc

        # Should not reach here, but just in case
        if last_exc is not None:  # pragma: no cover
            raise last_exc
        raise RuntimeError("Unreachable")  # pragma: no cover
