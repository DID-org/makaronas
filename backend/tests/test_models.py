"""Tests for backend.models — Model ID registry."""

import dataclasses

import pytest

from backend import models
from backend.models import (
    CLAUDE_HAIKU,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
    GEMINI_FLASH,
    GEMINI_FLASH_LITE,
    GEMINI_PRO,
    MODEL_MAP,
    TIER_MAP,
    ModelConfig,
    resolve_tier,
)


# All public constants and their expected values
EXPECTED_CONSTANTS = {
    "CLAUDE_HAIKU": "claude-haiku-4-5-20251001",
    "CLAUDE_SONNET": "claude-sonnet-4-6",
    "CLAUDE_OPUS": "claude-opus-4-6",
    "GEMINI_FLASH_LITE": "gemini-flash-lite-latest",
    "GEMINI_FLASH": "gemini-3-flash-preview",
    "GEMINI_PRO": "gemini-3.1-pro-preview",
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
        assert GEMINI_PRO == "gemini-3.1-pro-preview"


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


class TestModelConfig:
    """ModelConfig frozen dataclass — provider + model_id + thinking_budget."""

    def test_construction(self) -> None:
        mc = ModelConfig(provider="gemini", model_id="gemini-3-flash-preview")
        assert mc.provider == "gemini"
        assert mc.model_id == "gemini-3-flash-preview"

    def test_default_thinking_budget(self) -> None:
        mc = ModelConfig(provider="gemini", model_id="test-model")
        assert mc.thinking_budget == 0

    def test_explicit_thinking_budget(self) -> None:
        mc = ModelConfig(provider="gemini", model_id="test-model", thinking_budget=8192)
        assert mc.thinking_budget == 8192

    def test_frozen(self) -> None:
        mc = ModelConfig(provider="gemini", model_id="test-model")
        with pytest.raises(dataclasses.FrozenInstanceError):
            mc.provider = "anthropic"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ModelConfig(provider="gemini", model_id="test", thinking_budget=0)
        b = ModelConfig(provider="gemini", model_id="test", thinking_budget=0)
        assert a == b

    def test_inequality(self) -> None:
        a = ModelConfig(provider="gemini", model_id="test")
        b = ModelConfig(provider="anthropic", model_id="test")
        assert a != b


class TestTierMap:
    """TIER_MAP — maps capability tiers to ModelConfig instances."""

    def test_contains_all_tiers(self) -> None:
        expected_tiers = {"fast", "standard", "complex"}
        assert set(TIER_MAP.keys()) == expected_tiers

    def test_values_are_model_configs(self) -> None:
        for tier, config in TIER_MAP.items():
            assert isinstance(config, ModelConfig), f"TIER_MAP['{tier}'] is not ModelConfig"

    def test_fast_tier(self) -> None:
        mc = TIER_MAP["fast"]
        assert mc.provider == "gemini"
        assert mc.model_id == GEMINI_FLASH_LITE

    def test_standard_tier(self) -> None:
        mc = TIER_MAP["standard"]
        assert mc.provider == "gemini"
        assert mc.model_id == GEMINI_FLASH

    def test_complex_tier(self) -> None:
        mc = TIER_MAP["complex"]
        assert mc.provider == "gemini"
        assert mc.model_id == GEMINI_PRO

    def test_all_tiers_gemini_provider(self) -> None:
        """MVP strategy: all tiers use Gemini."""
        for tier, config in TIER_MAP.items():
            assert config.provider == "gemini", f"TIER_MAP['{tier}'] not gemini"


class TestResolveTier:
    """resolve_tier() — tier name to ModelConfig lookup."""

    def test_known_tier(self) -> None:
        mc = resolve_tier("standard")
        assert mc == TIER_MAP["standard"]

    def test_all_tiers_resolve(self) -> None:
        for tier in ("fast", "standard", "complex"):
            mc = resolve_tier(tier)
            assert isinstance(mc, ModelConfig)

    def test_unknown_tier_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            resolve_tier("unknown")

    def test_empty_tier_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            resolve_tier("")


class TestGeminiThinkingBudgetRemoved:
    """GEMINI_THINKING_BUDGET is absorbed into ModelConfig.thinking_budget."""

    def test_no_gemini_thinking_budget_attribute(self) -> None:
        assert not hasattr(models, "GEMINI_THINKING_BUDGET")
