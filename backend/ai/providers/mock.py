"""Mock AI provider for testing and development.

Deterministic, zero-cost AIProvider implementation that yields configurable
canned responses. Used by:
- Every V3 test phase (via conftest.mock_provider fixture)
- Development mode for team members without API keys
- Reference implementation of the AIProvider contract

Tier 2 service — imports only from base.py (Tier 1).
"""

from collections.abc import AsyncIterator

from backend.ai.providers.base import (
    AIProvider,
    ModelConfig,
    StreamEvent,
    TextChunk,
    ToolCallEvent,
    UsageInfo,
)

_DEFAULT_RESPONSES = ["Hello from MockProvider"]
_DEFAULT_USAGE = UsageInfo(prompt_tokens=10, completion_tokens=5)


class MockProvider(AIProvider):
    """Deterministic AI provider for testing.

    Yields configurable canned text chunks, optional tool call events,
    and configurable usage info. Can simulate errors.

    Args:
        responses: Text strings to yield as TextChunk events. Defaults to
            a single "Hello from MockProvider" chunk.
        tool_calls: ToolCallEvents to yield after all text chunks.
        usage: Token usage returned by complete(). Defaults to 10/5.
        error: If set, both stream() and complete() raise this immediately.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        tool_calls: list[ToolCallEvent] | None = None,
        usage: UsageInfo | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = responses if responses is not None else list(_DEFAULT_RESPONSES)
        self.tool_calls = tool_calls or []
        self.usage = usage or _DEFAULT_USAGE
        self.error = error

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yields canned text chunks followed by tool call events.

        Raises configured error before yielding anything if error is set.
        """
        if self.error is not None:
            raise self.error

        for text in self.responses:
            yield TextChunk(text=text)

        for tool_call in self.tool_calls:
            yield tool_call

    async def complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        tools: list[dict] | None = None,
    ) -> tuple[str, UsageInfo]:
        """Returns concatenated responses and configured usage info.

        Raises configured error immediately if error is set.
        """
        if self.error is not None:
            raise self.error

        full_text = "".join(self.responses)
        return full_text, self.usage
