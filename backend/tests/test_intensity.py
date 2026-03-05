"""Tests for heuristic intensity scoring (Phase 4a).

Pure function testing — no async, no mocking of external services.
"""

import json
import math
from pathlib import Path

import pytest

from backend.ai.intensity import load_intensity_indicators, score_intensity

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def real_indicators() -> dict:
    """Loads the real intensity_indicators.json for integration tests."""
    path = PROJECT_ROOT / "content" / "intensity_indicators.json"
    return load_intensity_indicators(path)


def _make_indicators(
    categories: dict | None = None,
    position_weight: float = 0.5,
    max_position_bonus: float = 1.0,
) -> dict:
    """Builds a minimal indicators dict for unit tests."""
    if categories is None:
        categories = {
            "challenge": {
                "weight": 1.0,
                "keywords": ["ar tikrai", "pagalvok"],
            },
            "dismissal": {
                "weight": 2.0,
                "keywords": ["nes\u0105mon\u0117", "naivumas"],
            },
            "confrontation": {
                "weight": 3.0,
                "keywords": ["tu klysti", "absurdas"],
            },
            "aggression": {
                "weight": 4.0,
                "keywords": ["kvailyst\u0117", "beviltis"],
            },
        }
    return {
        "categories": categories,
        "position_weight": position_weight,
        "max_position_bonus": max_position_bonus,
    }


# ---------------------------------------------------------------------------
# Scoring Correctness
# ---------------------------------------------------------------------------


class TestScoringCorrectness:
    """Verifies score_intensity returns correct values for various inputs."""

    def test_no_matches_returns_near_minimum(self):
        indicators = _make_indicators()
        # No keyword matches, only small position bonus at exchange 1/6
        score = score_intensity("Labas rytas, kaip sekasi?", 1, 6, indicators)
        assert score < 1.2

    def test_empty_text_returns_minimum(self):
        indicators = _make_indicators()
        assert score_intensity("", 1, 6, indicators) == 1.0

    def test_english_text_returns_near_minimum(self):
        indicators = _make_indicators()
        # No keyword matches, only small position bonus
        score = score_intensity("This is English text only.", 1, 6, indicators)
        assert score < 1.2

    def test_challenge_keywords_score_low(self):
        indicators = _make_indicators()
        text = "Ar tikrai taip manai? Pagalvok."
        score = score_intensity(text, 1, 6, indicators)
        assert 1.0 < score < 3.0

    def test_aggression_keywords_score_high(self):
        indicators = _make_indicators()
        text = "Tai kvailyst\u0117! Beviltis atvejis."
        score = score_intensity(text, 1, 6, indicators)
        # Aggression weight=4.0, 2 matches in a 4-category normalizer
        assert score >= 2.5

    def test_mixed_categories_higher_than_single(self):
        indicators = _make_indicators()
        single = score_intensity("Ar tikrai?", 1, 6, indicators)
        mixed = score_intensity(
            "Ar tikrai? Tai nes\u0105mon\u0117! Tu klysti!", 1, 6, indicators
        )
        assert mixed > single

    def test_multiple_matches_diminishing_returns(self):
        """Multiple matches in the same category don't scale linearly."""
        # Use low weight + multiple categories so scores don't saturate
        indicators = _make_indicators(
            categories={
                "target": {
                    "weight": 0.1,
                    "keywords": ["aaa", "bbb", "ccc", "ddd"],
                },
                "filler1": {"weight": 0.1, "keywords": ["zzz"]},
                "filler2": {"weight": 0.1, "keywords": ["yyy"]},
                "filler3": {"weight": 0.1, "keywords": ["xxx"]},
            }
        )
        one_match = score_intensity("aaa", 1, 6, indicators)
        four_matches = score_intensity("aaa bbb ccc ddd", 1, 6, indicators)
        # Both must be above base so we can compare meaningfully
        assert one_match > 1.0
        assert four_matches > one_match
        # Four matches should be less than 4x one match (diminishing returns)
        # log2(4+1) / log2(1+1) = 2.32 / 1.0 = 2.32x, not 4x
        ratio = (four_matches - 1.0) / (one_match - 1.0)
        assert ratio < 4.0

    def test_score_clamped_at_maximum(self):
        """Even extreme input stays within [1.0, 5.0]."""
        indicators = _make_indicators()
        # Text with every keyword from every category
        all_keywords = []
        for cat in indicators["categories"].values():
            all_keywords.extend(cat["keywords"])
        text = " ".join(all_keywords)
        score = score_intensity(text, 6, 6, indicators)
        assert 1.0 <= score <= 5.0

    def test_score_ordering_by_severity(self):
        """Higher severity categories produce higher scores."""
        indicators = _make_indicators()
        challenge_score = score_intensity("Ar tikrai?", 1, 6, indicators)
        aggression_score = score_intensity("Tai kvailyst\u0117!", 1, 6, indicators)
        assert aggression_score > challenge_score


# ---------------------------------------------------------------------------
# Case Folding (Lithuanian)
# ---------------------------------------------------------------------------


class TestCaseFolding:
    """Verifies Lithuanian case-insensitive matching via casefold."""

    def test_uppercase_matches(self):
        indicators = _make_indicators()
        lower = score_intensity("ar tikrai", 1, 6, indicators)
        upper = score_intensity("AR TIKRAI", 1, 6, indicators)
        assert lower == pytest.approx(upper, abs=0.01)

    def test_mixed_case_matches(self):
        indicators = _make_indicators()
        score = score_intensity("Ar Tikrai", 1, 6, indicators)
        assert score > 1.0

    def test_lithuanian_diacritics_case_insensitive(self):
        """Lithuanian characters (\u0105, \u010d, \u0119, \u0117, \u012f, \u0161, \u0173, \u016b, \u017e) casefold correctly."""
        indicators = _make_indicators(
            categories={
                "test": {
                    "weight": 2.0,
                    "keywords": ["nes\u0105mon\u0117"],
                }
            }
        )
        score = score_intensity("NES\u0104MON\u0116", 1, 6, indicators)
        assert score > 1.0


# ---------------------------------------------------------------------------
# Position Factor
# ---------------------------------------------------------------------------


class TestPositionFactor:
    """Verifies exchange position influences the score."""

    def test_later_position_scores_higher(self):
        indicators = _make_indicators()
        text = "Ar tikrai taip manai?"
        early = score_intensity(text, 1, 6, indicators)
        late = score_intensity(text, 6, 6, indicators)
        assert late > early

    def test_position_1_minimal_bonus(self):
        indicators = _make_indicators()
        text = "Ar tikrai?"
        at_1 = score_intensity(text, 1, 6, indicators)
        at_0_pos = score_intensity(text, 0, 6, indicators)
        # Position 0 and 1 should be very close (small difference)
        assert abs(at_1 - at_0_pos) < 0.5

    def test_max_exchanges_zero_no_crash(self):
        indicators = _make_indicators()
        score = score_intensity("Ar tikrai?", 3, 0, indicators)
        assert 1.0 <= score <= 5.0

    def test_position_alone_cannot_exceed_threshold(self):
        """Position bonus alone shouldn't push neutral text above ~2.5."""
        indicators = _make_indicators()
        score = score_intensity("Labas rytas!", 6, 6, indicators)
        # No keyword matches, only position bonus: 1.0 + position
        assert score <= 2.5


# ---------------------------------------------------------------------------
# Indicator Loading
# ---------------------------------------------------------------------------


class TestIndicatorLoading:
    """Verifies load_intensity_indicators from disk."""

    def test_loads_real_file(self, real_indicators):
        assert "categories" in real_indicators
        assert "position_weight" in real_indicators
        assert "max_position_bonus" in real_indicators

    def test_real_file_has_all_categories(self, real_indicators):
        cats = real_indicators["categories"]
        assert "challenge" in cats
        assert "dismissal" in cats
        assert "confrontation" in cats
        assert "aggression" in cats

    def test_each_category_has_keywords(self, real_indicators):
        for name, cat in real_indicators["categories"].items():
            assert "weight" in cat, f"{name} missing weight"
            assert "keywords" in cat, f"{name} missing keywords"
            assert len(cat["keywords"]) >= 15, (
                f"{name} has only {len(cat['keywords'])} keywords, expected >= 15"
            )

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_intensity_indicators(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_decode_error(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_intensity_indicators(bad_file)

    def test_lithuanian_characters_preserved(self, real_indicators):
        """Verifies Lithuanian diacritics survived JSON loading."""
        all_keywords = []
        for cat in real_indicators["categories"].values():
            all_keywords.extend(cat["keywords"])
        text = " ".join(all_keywords)
        # Lithuanian diacritics should be present
        lithuanian_chars = set("\u0105\u010d\u0119\u0117\u012f\u0161\u0173\u016b\u017e")
        found = {c for c in text if c in lithuanian_chars}
        assert len(found) > 0, "No Lithuanian diacritics found in keywords"


# ---------------------------------------------------------------------------
# Non-Hardcoded Verification (Constraint #8)
# ---------------------------------------------------------------------------


class TestNotHardcoded:
    """Proves score_intensity uses the passed indicators, not internal lists."""

    def test_custom_indicators_change_score(self):
        """Different indicator dicts produce different scores for same text."""
        text = "specialus testas"

        default_indicators = _make_indicators()
        custom_indicators = _make_indicators(
            categories={
                "custom": {
                    "weight": 4.0,
                    "keywords": ["specialus testas"],
                }
            }
        )

        default_score = score_intensity(text, 1, 6, default_indicators)
        custom_score = score_intensity(text, 1, 6, custom_indicators)

        # Default indicators have no match for "specialus testas"
        assert default_score == pytest.approx(1.0, abs=0.1)
        # Custom indicators do match
        assert custom_score > 2.0

    def test_empty_categories_returns_base(self):
        """Indicators with no categories produce minimum + position only."""
        indicators = _make_indicators(categories={})
        score = score_intensity("Tu klysti!", 3, 6, indicators)
        # No categories to match => only position bonus
        assert score < 2.0

    def test_custom_position_weight(self):
        """Custom position_weight in indicators is respected."""
        text = "Labas"
        low_weight = _make_indicators(position_weight=0.0, max_position_bonus=1.0)
        high_weight = _make_indicators(position_weight=1.0, max_position_bonus=1.0)

        low_score = score_intensity(text, 6, 6, low_weight)
        high_score = score_intensity(text, 6, 6, high_weight)

        assert high_score > low_score


# ---------------------------------------------------------------------------
# Lithuanian Text Samples (Realistic Trickster Responses)
# ---------------------------------------------------------------------------


class TestLithuanianSamples:
    """Tests with realistic Trickster response snippets at different levels."""

    def test_neutral_informational(self, real_indicators):
        text = (
            "Pažiūrėkime į šį straipsnį atidžiau. "
            "Čia pateikiami keli teiginiai apie klimato kaitą."
        )
        score = score_intensity(text, 1, 6, real_indicators)
        assert score < 2.5

    def test_mild_pushback(self, real_indicators):
        text = (
            "Ar tikrai taip manai? Pagalvok dar kartą — "
            "ar esi tikras dėl šio teiginio?"
        )
        score = score_intensity(text, 2, 6, real_indicators)
        assert 1.5 < score < 3.5

    def test_moderate_challenge(self, real_indicators):
        text = (
            "Tu klysti. Tai nesąmonė — faktai rodo kitaip. "
            "Ar turi įrodymų savo pozicijai?"
        )
        score = score_intensity(text, 3, 6, real_indicators)
        assert score > 2.0

    def test_high_pressure(self, real_indicators):
        text = (
            "Tai visiškai kvaila! Neturi jokio supratimo. "
            "Tai apgailėtina — beviltiškas atvejis."
        )
        score = score_intensity(text, 5, 6, real_indicators)
        assert score > 3.0

    def test_ordering_low_to_high(self, real_indicators):
        """Score ordering matches intensity ordering."""
        neutral = "Pažiūrėkime į šį straipsnį."
        mild = "Ar tikrai taip manai? Pagalvok dar kartą."
        aggressive = "Tai kvailystė! Neturi supratimo. Beviltis."

        s_neutral = score_intensity(neutral, 2, 6, real_indicators)
        s_mild = score_intensity(mild, 2, 6, real_indicators)
        s_aggressive = score_intensity(aggressive, 2, 6, real_indicators)

        assert s_neutral < s_mild < s_aggressive
