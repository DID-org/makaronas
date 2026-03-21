"""Google Gemini AI provider using the google-genai SDK.

Implements the AIProvider contract for Google's Gemini model family.
Handles streaming and non-streaming calls, tool call interception,
thinking part filtering, and exponential backoff retry for transient errors.

Tier 2 service — imports from base.py (Tier 1) + google-genai SDK.
"""

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from backend.ai.providers.base import (
    AIProvider,
    Message,
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


def _build_contents(messages: list[Message]) -> list[types.Content]:
    """Converts provider-neutral message dicts to Gemini Content objects.

    Handles both text-only content (str) and multimodal content (list of
    type-discriminated parts). Images are decoded from base64 to raw bytes
    for the Gemini SDK's inline_data format.

    Role mapping: "user" -> "user", "assistant" -> "model".
    Unknown content part types are skipped with a warning log.
    """
    role_map = {"user": "user", "assistant": "model"}
    contents = []
    for msg in messages:
        role = role_map.get(msg["role"], msg["role"])
        content = msg["content"]

        if isinstance(content, str):
            parts = [types.Part(text=content)]
        else:
            parts = []
            for part_data in content:
                part_type = part_data.get("type")
                if part_type == "text":
                    parts.append(types.Part(text=part_data["text"]))
                elif part_type == "image":
                    media_type = part_data.get("media_type")
                    data = part_data.get("data")
                    if media_type and data:
                        parts.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=media_type,
                                    data=base64.b64decode(data),
                                )
                            )
                        )
                    else:
                        logger.warning(
                            "Skipping image part with missing media_type or data"
                        )
                else:
                    logger.warning(
                        "Skipping unknown content part type: %s", part_type
                    )

        contents.append(types.Content(parts=parts, role=role))
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
    force_tool: bool = False,
) -> types.GenerateContentConfig:
    """Builds the GenerateContentConfig for a Gemini API call.

    Args:
        system_prompt: System instruction text.
        model_config: Provider-specific configuration.
        tools: Optional tool definitions for function calling.
        force_tool: If True, sets function calling mode to ANY,
            forcing the model to produce a tool call. Use when the
            model should signal a transition rather than continue
            talking.
    """
    gemini_tools = _build_tools(tools)

    # Build thinking config — prefer thinking_level (string) over thinking_budget (int)
    thinking_config = None
    if model_config.thinking_level:
        level_map = {
            "low": types.ThinkingLevel.LOW,
            "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH,
            "minimal": types.ThinkingLevel.MINIMAL,
        }
        level = level_map.get(model_config.thinking_level)
        if level:
            thinking_config = types.ThinkingConfig(thinkingLevel=level)
    if thinking_config is None and model_config.thinking_budget:
        thinking_config = types.ThinkingConfig(
            thinking_budget=model_config.thinking_budget,
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=_DEFAULT_TEMPERATURE,
        thinking_config=thinking_config,
    )

    if gemini_tools is not None:
        config.tools = gemini_tools
        config.automatic_function_calling = (
            types.AutomaticFunctionCallingConfig(disable=True)
        )
        if force_tool:
            config.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                )
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
        if not api_key:
            raise ValueError("API key must not be empty for Gemini provider")
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
        messages: list[Message],
        model_config: ModelConfig,
        tools: list[dict] | None = None,
        force_tool: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Streams a response as text chunks and tool call events.

        Retries on transient errors (429, 5xx) with exponential backoff.
        Thinking parts are silently filtered. Empty candidates are skipped.
        After full consumption, ``_last_usage`` holds the UsageInfo from the
        final streaming chunk.

        Args:
            system_prompt: The assembled system instruction.
            messages: Conversation history as Message dicts (text-only or multimodal).
            model_config: Provider-specific configuration (model ID, thinking budget).
            tools: Optional tool definitions for function calling.

        Yields:
            TextChunk for text tokens, ToolCallEvent for tool invocations.
        """
        self._last_usage = None
        contents = _build_contents(messages)
        config = _build_config(system_prompt, model_config, tools, force_tool=force_tool)

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
        messages: list[Message],
        model_config: ModelConfig,
        tools: list[dict] | None = None,
        force_tool: bool = False,
    ) -> tuple[str, UsageInfo]:
        """Returns the full response text and usage info (non-streaming).

        Retries on transient errors (429, 5xx) with exponential backoff.

        Args:
            system_prompt: The assembled system instruction.
            messages: Conversation history as Message dicts (text-only or multimodal).
            model_config: Provider-specific configuration (model ID, thinking budget).
            tools: Optional tool definitions for function calling.
            force_tool: If True, forces the model to produce a tool call.

        Returns:
            Tuple of (full response text, token usage information).
        """
        contents = _build_contents(messages)
        config = _build_config(system_prompt, model_config, tools, force_tool=force_tool)

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
