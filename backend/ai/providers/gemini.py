"""Google Gemini AI provider using the google-genai SDK.

Implements the AIProvider contract for Google's Gemini model family.
Handles streaming and non-streaming calls, tool call interception,
thinking part filtering, and exponential backoff retry for transient errors.

Tier 2 service — imports from base.py (Tier 1) + google-genai SDK.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

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
_MAX_RETRIES = 2  # 3 total attempts
_BACKOFF_BASE = 1.0  # seconds — doubles each retry


def _is_retryable(exc: Exception) -> bool:
    """Checks whether an SDK error is transient and worth retrying.

    Retries on:
    - ClientError with code 429 (rate limit)
    - Any ServerError (500, 502, 503, etc.)

    All other errors (400 bad request, 403 auth, etc.) propagate immediately.
    """
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError) and exc.code == 429:
        return True
    return False


def _build_contents(messages: list[dict[str, str]]) -> list[types.Content]:
    """Converts provider-neutral message dicts to Gemini Content objects.

    Role mapping: "user" → "user", "assistant" → "model".
    """
    role_map = {"user": "user", "assistant": "model"}
    contents = []
    for msg in messages:
        role = role_map.get(msg["role"], msg["role"])
        contents.append(
            types.Content(
                parts=[types.Part(text=msg["content"])],
                role=role,
            )
        )
    return contents


def _build_tools(tools: list[dict] | None) -> list[types.Tool] | None:
    """Converts generic tool dicts to Gemini Tool objects.

    Input format: [{"name": "...", "description": "...", "parameters": {...}}]
    """
    if not tools:
        return None
    declarations = []
    for tool_def in tools:
        declarations.append(
            types.FunctionDeclaration(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                parameters=tool_def.get("parameters"),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _build_config(
    system_prompt: str,
    model_config: ModelConfig,
    tools: list[dict] | None,
) -> types.GenerateContentConfig:
    """Builds the GenerateContentConfig for a Gemini API call."""
    gemini_tools = _build_tools(tools)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=_DEFAULT_TEMPERATURE,
        thinking_config=types.ThinkingConfig(
            thinking_budget=model_config.thinking_budget,
        ),
    )

    if gemini_tools is not None:
        config.tools = gemini_tools
        config.automatic_function_calling = (
            types.AutomaticFunctionCallingConfig(disable=True)
        )

    return config


class GeminiProvider(AIProvider):
    """Gemini AI provider using the google-genai SDK.

    Implements streaming and non-streaming AI calls through Google's
    Gemini API. Handles retry with exponential backoff for transient
    errors (429, 5xx) and surfaces tool calls as ToolCallEvent.

    Args:
        api_key: Google API key for Gemini access.
    """

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                api_version="v1alpha",
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
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
        Thinking parts are silently filtered. Empty candidates are skipped.
        After full consumption, ``_last_usage`` holds the UsageInfo from the
        final streaming chunk.

        Args:
            system_prompt: The assembled system instruction.
            messages: Conversation history as {"role": ..., "content": ...} dicts.
            model_config: Provider-specific configuration (model ID, thinking budget).
            tools: Optional tool definitions for function calling.

        Yields:
            TextChunk for text tokens, ToolCallEvent for tool invocations.
        """
        self._last_usage = None
        contents = _build_contents(messages)
        config = _build_config(system_prompt, model_config, tools)

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                backoff = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini stream retry %d/%d after %.1fs backoff",
                    attempt,
                    _MAX_RETRIES,
                    backoff,
                )
                await asyncio.sleep(backoff)
            try:
                response_stream = await self._client.aio.models.generate_content_stream(
                    model=model_config.model_id,
                    contents=contents,
                    config=config,
                )
                async for chunk in response_stream:
                    # Extract usage from the final chunk
                    if chunk.usage_metadata is not None:
                        prompt = chunk.usage_metadata.prompt_token_count or 0
                        completion = chunk.usage_metadata.candidates_token_count or 0
                        self._last_usage = UsageInfo(
                            prompt_tokens=prompt,
                            completion_tokens=completion,
                        )

                    # Handle empty candidates (safety blocks)
                    if not chunk.candidates:
                        continue

                    for candidate in chunk.candidates:
                        if candidate.content is None or candidate.content.parts is None:
                            continue
                        for part in candidate.content.parts:
                            # Skip thinking parts
                            if getattr(part, "thought", False):
                                continue

                            # Tool call
                            if part.function_call is not None:
                                yield ToolCallEvent(
                                    function_name=part.function_call.name or "",
                                    arguments=dict(part.function_call.args or {}),
                                )
                            # Text content
                            elif part.text is not None:
                                yield TextChunk(text=part.text)

                # Stream completed successfully — no retry needed
                return

            except (genai_errors.ClientError, genai_errors.ServerError) as exc:
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
        contents = _build_contents(messages)
        config = _build_config(system_prompt, model_config, tools)

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                backoff = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini complete retry %d/%d after %.1fs backoff",
                    attempt,
                    _MAX_RETRIES,
                    backoff,
                )
                await asyncio.sleep(backoff)
            try:
                response = await self._client.aio.models.generate_content(
                    model=model_config.model_id,
                    contents=contents,
                    config=config,
                )

                # Extract text from all non-thinking parts
                parts_text = []
                if response.candidates:
                    for candidate in response.candidates:
                        if candidate.content is None or candidate.content.parts is None:
                            continue
                        for part in candidate.content.parts:
                            if getattr(part, "thought", False):
                                continue
                            if part.text is not None:
                                parts_text.append(part.text)

                full_text = "".join(parts_text)

                # Extract usage
                prompt_tokens = 0
                completion_tokens = 0
                if response.usage_metadata is not None:
                    prompt_tokens = response.usage_metadata.prompt_token_count or 0
                    completion_tokens = (
                        response.usage_metadata.candidates_token_count or 0
                    )

                usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                return full_text, usage

            except (genai_errors.ClientError, genai_errors.ServerError) as exc:
                if not _is_retryable(exc) or attempt == _MAX_RETRIES:
                    raise
                last_exc = exc

        # Should not reach here, but just in case
        if last_exc is not None:  # pragma: no cover
            raise last_exc
        raise RuntimeError("Unreachable")  # pragma: no cover
