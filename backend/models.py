"""Model ID registry — single source of truth for AI model identifiers.

Every AI call in the platform resolves its model ID through this module.
The rest of the codebase imports family-name constants from here — no raw
model ID strings anywhere else.

Three-layer abstraction:
  Layer 1: Cartridge declares a capability tier ("fast", "standard", "complex")
  Layer 2: TIER_MAP resolves tier → ModelConfig (team experiments here)
  Layer 3: Model ID constants (updated when providers release new versions)

To swap a model: change a TIER_MAP value below. One line changed,
eight hundred schools speak to a new mind.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Layer 3: Model IDs (update when providers release new versions)
# ---------------------------------------------------------------------------

# --- Claude models ---
CLAUDE_HAIKU: str = "claude-haiku-4-5-20251001"
CLAUDE_SONNET: str = "claude-sonnet-4-6"
CLAUDE_OPUS: str = "claude-opus-4-6"

# --- Gemini models ---
GEMINI_FLASH_LITE: str = "gemini-flash-lite-latest"
GEMINI_FLASH: str = "gemini-3-flash-preview"
GEMINI_PRO: str = "gemini-3.1-pro-preview"


# ---------------------------------------------------------------------------
# ModelConfig — bundles all provider-specific configuration for a tier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Bundles all provider-specific configuration for a model tier.

    Tier 1 leaf — no project imports. Constructed in TIER_MAP below,
    consumed by provider implementations via resolve_tier().
    """

    provider: str          # "gemini" or "anthropic"
    model_id: str          # e.g. "gemini-3-flash-preview"
    thinking_budget: int = 0  # Gemini thinking tokens (0 = off)


# ---------------------------------------------------------------------------
# Layer 2: Capability tier → ModelConfig (team experiments here)
# ---------------------------------------------------------------------------
# MVP strategy: tune everything to Gemini for speed and cost.
# - Flash (zero thinking) for most tasks — better instruction following than Lite
# - Flash Lite for simple/fast interactions — cheapest, good character work
# - Haiku available for experimentation but not primary target
#
# thinking_budget=0 on Flash: disables internal reasoning, reduces latency
# and cost while keeping Flash's improved instruction-following over Lite.

TIER_MAP: dict[str, ModelConfig] = {
    "fast": ModelConfig(provider="gemini", model_id=GEMINI_FLASH_LITE, thinking_budget=0),
    "standard": ModelConfig(provider="gemini", model_id=GEMINI_FLASH, thinking_budget=0),
    "complex": ModelConfig(provider="gemini", model_id=GEMINI_PRO, thinking_budget=0),
}


def resolve_tier(tier: str) -> ModelConfig:
    """Resolves a capability tier name to its ModelConfig.

    Args:
        tier: Capability tier name ("fast", "standard", "complex").

    Returns:
        The ModelConfig for the given tier.

    Raises:
        KeyError: If the tier name is not found in TIER_MAP.
    """
    return TIER_MAP[tier]


# ---------------------------------------------------------------------------
# Lookup map — env var value → actual model ID
# ---------------------------------------------------------------------------
# Keys match the constant names exactly (case-sensitive).
# Available for env-var-based model overrides if needed in the future.
MODEL_MAP: dict[str, str] = {
    "CLAUDE_HAIKU": CLAUDE_HAIKU,
    "CLAUDE_SONNET": CLAUDE_SONNET,
    "CLAUDE_OPUS": CLAUDE_OPUS,
    "GEMINI_FLASH_LITE": GEMINI_FLASH_LITE,
    "GEMINI_FLASH": GEMINI_FLASH,
    "GEMINI_PRO": GEMINI_PRO,
}
