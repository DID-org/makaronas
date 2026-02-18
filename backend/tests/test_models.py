"""Tests for backend.models — Model ID registry."""

from backend import models
from backend.models import (
    CLAUDE_HAIKU,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
    GEMINI_FLASH,
    GEMINI_FLASH_LITE,
    GEMINI_PRO,
    MODEL_MAP,
)


# All public constants and their expected values
EXPECTED_CONSTANTS = {
    "CLAUDE_HAIKU": "claude-haiku-4-5-20251001",
    "CLAUDE_SONNET": "claude-sonnet-4-6",
    "CLAUDE_OPUS": "claude-opus-4-6",
    "GEMINI_FLASH_LITE": "gemini-flash-lite-latest",
    "GEMINI_FLASH": "gemini-3-flash-preview",
    "GEMINI_PRO": "gemini-3-pro-preview",
}


class TestModelConstants:
    """Tests for module-level model ID constants."""

    def test_claude_haiku_value(self) -> None:
        assert CLAUDE_HAIKU == "claude-haiku-4-5-20251001"

    def test_claude_sonnet_value(self) -> None:
        assert CLAUDE_SONNET == "claude-sonnet-4-6"

    def test_claude_opus_value(self) -> None:
        assert CLAUDE_OPUS == "claude-opus-4-6"

    def test_gemini_flash_lite_value(self) -> None:
        assert GEMINI_FLASH_LITE == "gemini-flash-lite-latest"

    def test_gemini_flash_value(self) -> None:
        assert GEMINI_FLASH == "gemini-3-flash-preview"

    def test_gemini_pro_value(self) -> None:
        assert GEMINI_PRO == "gemini-3-pro-preview"


class TestModelMap:
    """Tests for MODEL_MAP lookup dictionary."""

    def test_contains_all_constants(self) -> None:
        for name in EXPECTED_CONSTANTS:
            assert name in MODEL_MAP, f"MODEL_MAP missing key: {name}"

    def test_values_match_constants(self) -> None:
        for name, expected_value in EXPECTED_CONSTANTS.items():
            assert MODEL_MAP[name] == expected_value

    def test_map_values_match_module_constants(self) -> None:
        """MODEL_MAP values reference the actual constant variables, not duplicated strings."""
        for name in EXPECTED_CONSTANTS:
            assert MODEL_MAP[name] == getattr(models, name)

    def test_no_extra_keys(self) -> None:
        assert set(MODEL_MAP.keys()) == set(EXPECTED_CONSTANTS.keys())

    def test_no_duplicate_values(self) -> None:
        values = list(MODEL_MAP.values())
        assert len(values) == len(set(values)), "MODEL_MAP has duplicate model IDs"
