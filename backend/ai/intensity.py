"""Heuristic intensity scoring for Trickster AI responses.

Scores adversarial pressure in Lithuanian text using weighted keyword
matching across severity categories. The scoring function is pure and
stateless — indicators are passed as a parameter, loaded once at startup.

Framework Principle 12: "The system tracks conversational intensity and
intervenes if the adversarial pressure crosses a threshold, regardless
of what the prompt says."
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger("makaronas.ai.intensity")


def load_intensity_indicators(indicators_path: Path) -> dict:
    """Loads intensity indicator data from JSON file.

    Args:
        indicators_path: Path to the intensity indicators JSON file.

    Returns:
        Parsed indicator dict with category keys.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is invalid JSON.
    """
    with open(indicators_path, encoding="utf-8") as f:
        return json.load(f)


def score_intensity(
    text: str,
    exchange_position: int,
    max_exchanges: int,
    indicators: dict,
) -> float:
    """Scores adversarial intensity of Trickster response text.

    Combines weighted keyword matching across severity categories with
    a position-based escalation bonus. Uses casefold for Lithuanian
    case-insensitive comparison (same pattern as safety.py).

    Args:
        text: The Trickster's response text (Lithuanian).
        exchange_position: Current exchange number (1-based).
        max_exchanges: Maximum exchanges for this task phase.
        indicators: Loaded intensity indicator data (from JSON).

    Returns:
        Float in range 1.0-5.0. Higher = more intense.
    """
    if not text:
        return 1.0

    base_score = _weighted_match_score(text, indicators)
    pos_bonus = _position_factor(
        exchange_position,
        max_exchanges,
        indicators.get("position_weight", 0.5),
        indicators.get("max_position_bonus", 1.0),
    )

    raw = 1.0 + base_score + pos_bonus
    return _clamp(raw, 1.0, 5.0)


def _weighted_match_score(text: str, indicators: dict) -> float:
    """Computes weighted match score across all indicator categories.

    Uses logarithmic diminishing returns within each category to prevent
    score explosion from multiple matches of the same severity.

    Args:
        text: Response text to score.
        indicators: Indicator data with categories.

    Returns:
        Score in range 0.0-4.0.
    """
    text_lower = text.casefold()
    categories = indicators.get("categories", {})
    total = 0.0

    for category in categories.values():
        weight = category.get("weight", 1.0)
        keywords = category.get("keywords", [])

        match_count = 0
        for keyword in keywords:
            if keyword.casefold() in text_lower:
                match_count += 1

        if match_count > 0:
            # Diminishing returns: log2(count + 1) so 1 match = 1.0,
            # 2 matches = 1.58, 4 matches = 2.32, etc.
            diminished = math.log2(match_count + 1)
            total += weight * diminished

    # Normalize to 0.0-4.0 range. The max theoretical score depends on
    # category count and weights, so we use a scaling factor that maps
    # typical high-intensity responses to ~4.0.
    max_weight = max(
        (c.get("weight", 1.0) for c in categories.values()),
        default=1.0,
    )
    category_count = len(categories)
    # Scale factor: if every category matched once at max weight
    normalizer = max_weight * category_count if category_count > 0 else 1.0
    normalized = (total / normalizer) * 4.0

    return min(normalized, 4.0)


def _position_factor(
    exchange_position: int,
    max_exchanges: int,
    position_weight: float,
    max_bonus: float,
) -> float:
    """Computes position-based escalation bonus.

    Later exchanges naturally carry more pressure — the conversation has
    been building. This adds a small additive bonus capped so position
    alone cannot push the score above ~2.5 (1.0 base + max_bonus).

    Args:
        exchange_position: Current exchange number (1-based).
        max_exchanges: Maximum exchanges for this phase.
        position_weight: How much position affects the score.
        max_bonus: Maximum position bonus.

    Returns:
        Bonus in range 0.0-max_bonus.
    """
    if max_exchanges <= 0:
        return 0.0

    # Fraction of conversation completed (0.0 to 1.0)
    progress = min(exchange_position / max_exchanges, 1.0)
    return progress * position_weight * max_bonus


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamps a value to the given range."""
    return max(minimum, min(value, maximum))
