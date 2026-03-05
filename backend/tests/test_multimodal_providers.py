"""Tests for Phase 2b: Provider-specific multimodal content mapping.

Tests GeminiProvider._build_contents(), AnthropicProvider._prepare_messages(),
and MockProvider.last_messages capture. All tests are unit-level — no SDK
clients or API calls needed.

Gemini and Anthropic tests are skipped if the respective SDK is not installed.
MockProvider tests always run (no SDK dependency).
"""

import base64
import logging

import pytest

from backend.ai.providers.mock import MockProvider
from backend.models import ModelConfig

# Shared constants
_CONFIG = ModelConfig(provider="mock", model_id="mock-v1")
_SYSTEM = "You are a test."
_SAMPLE_IMAGE_B64 = base64.b64encode(b"fake-image-bytes").decode()

# Conditional SDK availability flags
try:
    import google.genai  # noqa: F401
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

try:
    import anthropic  # noqa: F401
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


# ---------------------------------------------------------------------------
# GeminiProvider: _build_contents()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_GENAI, reason="google-genai SDK not installed")
class TestGeminiBuildContents:
    """Unit tests for Gemini multimodal content mapping."""

    @pytest.fixture(autouse=True)
    def _import_gemini(self):
        from backend.ai.providers.gemini import _build_contents
        self._build_contents = _build_contents

    def test_text_only_unchanged(self) -> None:
        """Text-only messages produce a single text Part — unchanged behavior."""
        messages = [{"role": "user", "content": "hello"}]
        result = self._build_contents(messages)

        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].parts) == 1
        assert result[0].parts[0].text == "hello"

    def test_multimodal_text_and_image(self) -> None:
        """Multimodal message with text + image produces two Parts."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {
                        "type": "image",
                        "media_type": "image/jpeg",
                        "data": _SAMPLE_IMAGE_B64,
                    },
                ],
            }
        ]
        result = self._build_contents(messages)

        assert len(result) == 1
        parts = result[0].parts
        assert len(parts) == 2
        assert parts[0].text == "look at this"
        assert parts[1].inline_data is not None
        assert parts[1].inline_data.mime_type == "image/jpeg"

    def test_image_data_is_decoded_bytes(self) -> None:
        """Image data in the Blob is raw bytes, not a base64 string."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "media_type": "image/png",
                        "data": _SAMPLE_IMAGE_B64,
                    },
                ],
            }
        ]
        result = self._build_contents(messages)
        blob_data = result[0].parts[0].inline_data.data
        assert isinstance(blob_data, bytes)
        assert blob_data == b"fake-image-bytes"

    def test_image_only_message(self) -> None:
        """Content list with only an image part produces single image Part."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "media_type": "image/webp",
                        "data": _SAMPLE_IMAGE_B64,
                    },
                ],
            }
        ]
        result = self._build_contents(messages)
        assert len(result[0].parts) == 1
        assert result[0].parts[0].inline_data is not None

    def test_mixed_conversation(self) -> None:
        """Text-only msg + multimodal msg in same conversation."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I see the image"},
                    {
                        "type": "image",
                        "media_type": "image/jpeg",
                        "data": _SAMPLE_IMAGE_B64,
                    },
                ],
            },
        ]
        result = self._build_contents(messages)

        assert len(result) == 2
        # First: text-only
        assert len(result[0].parts) == 1
        assert result[0].parts[0].text == "hi"
        assert result[0].role == "user"
        # Second: multimodal
        assert len(result[1].parts) == 2
        assert result[1].role == "model"

    def test_unknown_type_skipped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown content part types are skipped with a warning."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "audio", "data": "something"},
                ],
            }
        ]
        with caplog.at_level(logging.WARNING):
            result = self._build_contents(messages)

        assert len(result[0].parts) == 1
        assert result[0].parts[0].text == "hello"
        assert "Skipping unknown content part type: audio" in caplog.text

    def test_image_missing_data_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Image part with missing data is skipped with a warning."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "media_type": "image/jpeg"},
                ],
            }
        ]
        with caplog.at_level(logging.WARNING):
            result = self._build_contents(messages)

        assert len(result[0].parts) == 0
        assert "missing media_type or data" in caplog.text

    def test_image_missing_media_type_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Image part with missing media_type is skipped with a warning."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "data": _SAMPLE_IMAGE_B64},
                ],
            }
        ]
        with caplog.at_level(logging.WARNING):
            result = self._build_contents(messages)

        assert len(result[0].parts) == 0
        assert "missing media_type or data" in caplog.text


# ---------------------------------------------------------------------------
# AnthropicProvider: _prepare_messages()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="anthropic SDK not installed")
class TestAnthropicPrepareMessages:
    """Unit tests for Anthropic multimodal message preparation."""

    @pytest.fixture(autouse=True)
    def _import_anthropic(self):
        from backend.ai.providers.anthropic import _prepare_messages
        self._prepare_messages = _prepare_messages

    def test_text_only_passthrough(self) -> None:
        """Text-only messages pass through unchanged."""
        messages = [{"role": "user", "content": "hello"}]
        result = self._prepare_messages(messages)

        assert len(result) == 1
        assert result[0] is messages[0]
        assert result[0]["content"] == "hello"

    def test_multimodal_image_gains_source_wrapper(self) -> None:
        """Image parts get the Anthropic source envelope."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {
                        "type": "image",
                        "media_type": "image/jpeg",
                        "data": _SAMPLE_IMAGE_B64,
                    },
                ],
            }
        ]
        result = self._prepare_messages(messages)

        assert len(result) == 1
        blocks = result[0]["content"]
        assert len(blocks) == 2

        assert blocks[0] == {"type": "text", "text": "look at this"}

        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"]["type"] == "base64"
        assert blocks[1]["source"]["media_type"] == "image/jpeg"
        assert blocks[1]["source"]["data"] == _SAMPLE_IMAGE_B64

    def test_text_parts_within_multimodal_unchanged(self) -> None:
        """Text parts inside a multimodal message pass through as-is."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
        result = self._prepare_messages(messages)
        blocks = result[0]["content"]
        assert blocks[0] == {"type": "text", "text": "first"}
        assert blocks[1] == {"type": "text", "text": "second"}

    def test_mixed_conversation(self) -> None:
        """Per-message dispatch: text-only + multimodal in one conversation."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "here's the image"},
                    {
                        "type": "image",
                        "media_type": "image/png",
                        "data": _SAMPLE_IMAGE_B64,
                    },
                ],
            },
        ]
        result = self._prepare_messages(messages)

        assert len(result) == 2
        assert result[0]["content"] == "hi"
        assert len(result[1]["content"]) == 2
        assert result[1]["content"][1]["source"]["type"] == "base64"

    def test_unknown_type_skipped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown content part types are skipped with a warning."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "video", "url": "something"},
                ],
            }
        ]
        with caplog.at_level(logging.WARNING):
            result = self._prepare_messages(messages)

        blocks = result[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Skipping unknown content part type: video" in caplog.text

    def test_no_base64_decode(self) -> None:
        """Anthropic accepts base64 strings directly — no decoding."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "media_type": "image/jpeg",
                        "data": _SAMPLE_IMAGE_B64,
                    },
                ],
            }
        ]
        result = self._prepare_messages(messages)
        source_data = result[0]["content"][0]["source"]["data"]
        assert isinstance(source_data, str)
        assert source_data == _SAMPLE_IMAGE_B64

    def test_image_missing_data_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Image part with missing data is skipped."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "media_type": "image/jpeg"},
                ],
            }
        ]
        with caplog.at_level(logging.WARNING):
            result = self._prepare_messages(messages)

        assert len(result[0]["content"]) == 0
        assert "missing media_type or data" in caplog.text


# ---------------------------------------------------------------------------
# MockProvider: last_messages capture
# ---------------------------------------------------------------------------


class TestMockProviderCapture:
    """MockProvider captures messages for test assertions."""

    def test_last_messages_starts_none(self) -> None:
        provider = MockProvider()
        assert provider.last_messages is None

    @pytest.mark.asyncio
    async def test_stream_captures_messages(self) -> None:
        provider = MockProvider()
        messages = [{"role": "user", "content": "hello"}]
        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=messages, model_config=_CONFIG
        ):
            pass
        assert provider.last_messages is messages

    @pytest.mark.asyncio
    async def test_complete_captures_messages(self) -> None:
        provider = MockProvider()
        messages = [{"role": "user", "content": "hello"}]
        await provider.complete(
            system_prompt=_SYSTEM, messages=messages, model_config=_CONFIG
        )
        assert provider.last_messages is messages

    @pytest.mark.asyncio
    async def test_multimodal_messages_captured(self) -> None:
        """Multimodal content list structure is preserved in capture."""
        provider = MockProvider()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image",
                        "media_type": "image/jpeg",
                        "data": _SAMPLE_IMAGE_B64,
                    },
                ],
            }
        ]
        async for _ in provider.stream(
            system_prompt=_SYSTEM, messages=messages, model_config=_CONFIG
        ):
            pass

        assert provider.last_messages is messages
        assert isinstance(provider.last_messages[0]["content"], list)
        assert provider.last_messages[0]["content"][1]["type"] == "image"
