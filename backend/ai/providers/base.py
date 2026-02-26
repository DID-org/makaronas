"""Base AI provider interface and stream event types.

Defines the contract that every AI provider implementation (Gemini, Anthropic,
Mock) must satisfy. Stream event types provide type-safe discrimination between
text chunks and tool call events during streaming.

Tier 1 leaf — imports only stdlib and backend.models (also Tier 1).
No schemas, no config, no framework imports.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from backend.models import ModelConfig


# ---------------------------------------------------------------------------
# Stream event types — the contract between providers and consumers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextChunk:
    """A piece of streamed text from the AI provider."""

    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    """A tool/function call emitted by the model during streaming.

    Normalized from both Gemini and Anthropic SDK formats into a
    provider-neutral shape. The trickster engine intercepts these
    to handle phase transitions.
    """

    function_name: str
    arguments: dict


@dataclass(frozen=True)
class UsageInfo:
    """Token usage from a completed AI call.

    Used for cost logging (Framework Principle 9) and budget tracking.
    """

    prompt_tokens: int
    completion_tokens: int


# Union type for stream events — consumers use isinstance() to dispatch
StreamEvent = TextChunk | ToolCallEvent


# ---------------------------------------------------------------------------
# AIProvider ABC — the interface every provider implements
# ---------------------------------------------------------------------------


class AIProvider(ABC):
    """Abstract base for AI model providers.

    Concrete implementations (GeminiProvider, AnthropicProvider, MockProvider)
    implement stream() and complete() to talk to their respective APIs.

    Tier 1 leaf — imports only stdlib and this module's own types.
    No project imports beyond backend.models.ModelConfig.
    """

    @abstractmethod
    async def stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streams a response as text chunks and tool call events.

        Args:
            system_prompt: The assembled system instruction.
            messages: Conversation history as {"role": ..., "content": ...} dicts.
            model_config: Provider-specific configuration (model ID, thinking budget, etc.).
            tools: Optional tool definitions for function calling.

        Yields:
            TextChunk for text tokens, ToolCallEvent for tool invocations.
        """

    @abstractmethod
    async def complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        tools: list[dict] | None = None,
    ) -> tuple[str, UsageInfo]:
        """Returns the full response text and usage info (non-streaming).

        Used by the evaluation engine (V6) for rubric analysis where
        streaming is unnecessary.

        Args:
            system_prompt: The assembled system instruction.
            messages: Conversation history as {"role": ..., "content": ...} dicts.
            model_config: Provider-specific configuration (model ID, thinking budget, etc.).
            tools: Optional tool definitions for function calling.

        Returns:
            Tuple of (full response text, token usage information).
        """
