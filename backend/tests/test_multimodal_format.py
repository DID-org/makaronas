"""Tests for Phase 2a: Provider-neutral multimodal message format.

Verifies:
- Message type alias is importable from base.py
- Text-only messages (backward compat) still work with all providers
- Multimodal messages (text + image content parts) are accepted
- AssembledContext accepts multimodal messages
"""

import pytest

from backend.ai.context import AssembledContext
from backend.ai.providers.base import Message, TextChunk
from backend.ai.providers.mock import MockProvider
from backend.models import ModelConfig

_CONFIG = ModelConfig(provider="mock", model_id="mock-v1")
_SYSTEM = "You are a test."


# ---------------------------------------------------------------------------
# Message type alias
# ---------------------------------------------------------------------------


class TestMessageAlias:
    """Message type alias is available and communicates intent."""

    def test_importable(self) -> None:
        assert Message is not None

    def test_text_only_message_is_valid(self) -> None:
        """A text-only message matches the Message type."""
        msg: Message = {"role": "user", "content": "hello"}
        assert msg["role"] == "user"
        assert msg["content"] == "hello"

    def test_multimodal_message_is_valid(self) -> None:
        """A multimodal message with content parts matches the Message type."""
        msg: Message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image"},
                {
                    "type": "image",
                    "media_type": "image/jpeg",
                    "data": "aGVsbG8=",  # base64 "hello"
                },
            ],
        }
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "image"


# ---------------------------------------------------------------------------
# MockProvider backward compatibility (text-only)
# ---------------------------------------------------------------------------


class TestTextOnlyBackwardCompat:
    """Existing text-only messages continue to work unchanged."""

    @pytest.mark.asyncio
    async def test_stream_text_only(self) -> None:
        provider = MockProvider()
        messages: list[Message] = [{"role": "user", "content": "Hello"}]
        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=messages, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], TextChunk)
        assert events[0].text == "Hello from MockProvider"

    @pytest.mark.asyncio
    async def test_complete_text_only(self) -> None:
        provider = MockProvider()
        messages: list[Message] = [{"role": "user", "content": "Hello"}]
        text, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=messages, model_config=_CONFIG
        )

        assert text == "Hello from MockProvider"
        assert usage.prompt_tokens == 10


# ---------------------------------------------------------------------------
# MockProvider multimodal acceptance
# ---------------------------------------------------------------------------


class TestMultimodalAcceptance:
    """MockProvider accepts multimodal messages without error."""

    @pytest.mark.asyncio
    async def test_stream_multimodal(self) -> None:
        """Multimodal messages don't crash stream()."""
        provider = MockProvider()
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {
                        "type": "image",
                        "media_type": "image/jpeg",
                        "data": "aGVsbG8=",
                    },
                ],
            }
        ]
        events = []
        async for event in provider.stream(
            system_prompt=_SYSTEM, messages=messages, model_config=_CONFIG
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], TextChunk)

    @pytest.mark.asyncio
    async def test_complete_multimodal(self) -> None:
        """Multimodal messages don't crash complete()."""
        provider = MockProvider()
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo=",
                    },
                ],
            }
        ]
        text, usage = await provider.complete(
            system_prompt=_SYSTEM, messages=messages, model_config=_CONFIG
        )

        assert text == "Hello from MockProvider"
        assert usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_mixed_text_and_multimodal_messages(self) -> None:
        """A conversation with both text-only and multimodal messages works."""
        provider = MockProvider()
        messages: list[Message] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Now look at this"},
                    {
                        "type": "image",
                        "media_type": "image/webp",
                        "data": "d2VicA==",
                    },
                ],
            },
        ]
        text, _ = await provider.complete(
            system_prompt=_SYSTEM, messages=messages, model_config=_CONFIG
        )
        assert text == "Hello from MockProvider"


# ---------------------------------------------------------------------------
# AssembledContext multimodal
# ---------------------------------------------------------------------------


class TestAssembledContextMultimodal:
    """AssembledContext accepts multimodal messages."""

    def test_text_only_still_works(self) -> None:
        ctx = AssembledContext(
            system_prompt="test",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
        )
        assert len(ctx.messages) == 1
        assert ctx.messages[0]["content"] == "hello"

    def test_multimodal_messages(self) -> None:
        ctx = AssembledContext(
            system_prompt="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image",
                            "media_type": "image/jpeg",
                            "data": "abc=",
                        },
                    ],
                }
            ],
            tools=None,
        )
        assert len(ctx.messages) == 1
        assert isinstance(ctx.messages[0]["content"], list)

    def test_frozen(self) -> None:
        """AssembledContext remains frozen with multimodal messages."""
        ctx = AssembledContext(
            system_prompt="test",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                }
            ],
            tools=None,
        )
        with pytest.raises(AttributeError):
            ctx.system_prompt = "changed"  # type: ignore[misc]
