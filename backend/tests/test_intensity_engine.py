"""Tests for intensity integration into TricksterEngine (Phase 4b).

Verifies that score_intensity() is wired into respond(), scores are
recorded on GameSession.turn_intensities, and both threshold paths
(de-escalation flag and hard redaction) work correctly.

Uses real ContextManager with PromptLoader pointed at temp prompts,
MockProvider for deterministic AI responses, and either real or minimal
indicator dicts for scoring control.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.ai.context import ContextManager
from backend.ai.prompts import PromptLoader
from backend.ai.providers.base import ToolCallEvent
from backend.ai.providers.mock import MockProvider
from backend.ai.safety import FALLBACK_INTENSITY
from backend.ai.trickster import TricksterEngine, TricksterResult
from backend.schemas import Exchange
from backend.tasks.schemas import TaskCartridge
from backend.tests.conftest import setup_base_prompts


def _get_ai_phase(cartridge: TaskCartridge):
    """Extracts the AI phase from the default test cartridge."""
    for phase in cartridge.phases:
        if phase.id == "phase_ai":
            return phase
    raise ValueError("No phase_ai found in cartridge")


async def _consume_tokens(result: TricksterResult) -> str:
    """Exhausts token_iterator and returns accumulated text."""
    tokens = []
    async for token in result.token_iterator:
        tokens.append(token)
    return "".join(tokens)


def _prefill_exchanges(session, count: int) -> None:
    """Adds exchange pairs to session to reach a target student count."""
    for i in range(count):
        session.exchanges.append(
            Exchange(role="student", content=f"Student message {i}")
        )
        session.exchanges.append(
            Exchange(role="trickster", content=f"Trickster reply {i}")
        )


def _make_indicators_high_score() -> dict:
    """Minimal indicators that produce a high score for specific keywords.

    Uses a single heavy category so that matching keywords reliably push
    the score above the default ceiling of 3.
    """
    return {
        "categories": {
            "aggression": {
                "weight": 4.0,
                "keywords": [
                    "tai kvailyst\u0117",
                    "beviltis atvejis",
                    "neturi supratimo",
                    "visi\u0161kai nieko nesupranti",
                    "tai begal\u0117 kvaila",
                    "tai apgail\u0117tina",
                ],
            },
            "confrontation": {
                "weight": 3.0,
                "keywords": [
                    "tu klysti",
                    "tu visi\u0161kai klysti",
                    "tai absurdas",
                    "tai visi\u0161kas absurdas",
                ],
            },
        },
        "position_weight": 0.5,
        "max_position_bonus": 1.0,
    }


def _make_indicators_moderate_score() -> dict:
    """Indicators that produce scores in the de-escalation zone (>3, <=4.5).

    Uses two categories but only one has matchable keywords. The second
    category dilutes the normalizer, keeping the score above ceiling (3)
    but below ceiling * 1.5 (4.5).
    """
    return {
        "categories": {
            "challenge": {
                "weight": 2.0,
                "keywords": ["tu klysti", "tai absurdas"],
            },
            "aggression": {
                "weight": 2.0,
                "keywords": ["xyz_unmatched_keyword"],
            },
        },
        "position_weight": 0.5,
        "max_position_bonus": 1.0,
    }


def _make_indicators_low_score() -> dict:
    """Minimal indicators where common text won't match anything."""
    return {
        "categories": {
            "challenge": {
                "weight": 1.0,
                "keywords": ["xyz_never_match_this_keyword"],
            },
        },
        "position_weight": 0.5,
        "max_position_bonus": 1.0,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompts_dir(tmp_path):
    """Creates temp directory with base Trickster prompts."""
    setup_base_prompts(tmp_path)
    return tmp_path


@pytest.fixture
def context_manager(prompts_dir):
    """Real ContextManager with PromptLoader pointed at temp prompts."""
    loader = PromptLoader(prompts_dir)
    return ContextManager(loader)


@pytest.fixture
def real_indicators():
    """Loads the real intensity_indicators.json for integration tests."""
    path = Path(__file__).parent.parent.parent / "content" / "intensity_indicators.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestBelowCeiling:
    """Responses that score below the intensity ceiling (normal flow)."""

    @pytest.mark.asyncio
    async def test_score_recorded_in_turn_intensities(
        self, context_manager, make_session, make_cartridge,
    ):
        """Normal response -> score appended to session.turn_intensities."""
        indicators = _make_indicators_low_score()
        provider = MockProvider(responses=["A normal, calm Trickster response."])
        engine = TricksterEngine(provider, context_manager, intensity_indicators=indicators)
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "Hello")
        await _consume_tokens(result)

        assert len(session.turn_intensities) == 1
        assert 1.0 <= session.turn_intensities[0] <= 5.0

    @pytest.mark.asyncio
    async def test_done_data_has_intensity_fields(
        self, context_manager, make_session, make_cartridge,
    ):
        """Normal response -> done_data includes intensity_score and intensity_deescalation."""
        indicators = _make_indicators_low_score()
        provider = MockProvider(responses=["A calm reply from the Trickster."])
        engine = TricksterEngine(provider, context_manager, intensity_indicators=indicators)
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "Tell me")
        await _consume_tokens(result)

        assert result.done_data is not None
        assert "intensity_score" in result.done_data
        assert result.done_data["intensity_deescalation"] is False

    @pytest.mark.asyncio
    async def test_normal_exchange_stored(
        self, context_manager, make_session, make_cartridge,
    ):
        """Normal response -> real text stored as exchange (not fallback)."""
        indicators = _make_indicators_low_score()
        provider = MockProvider(responses=["The Trickster speaks calmly."])
        engine = TricksterEngine(provider, context_manager, intensity_indicators=indicators)
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "What?")
        await _consume_tokens(result)

        assert session.exchanges[-1].role == "trickster"
        assert session.exchanges[-1].content == "The Trickster speaks calmly."
        assert result.redaction_data is None


class TestDeescalationThreshold:
    """Responses that exceed the ceiling but not by >1.5x."""

    @pytest.mark.asyncio
    async def test_deescalation_flag_set(
        self, context_manager, make_session, make_cartridge,
    ):
        """Score > ceiling -> intensity_deescalation is True in done_data."""
        indicators = _make_indicators_moderate_score()
        # Text matching one category's keywords — enough to exceed ceiling (3)
        # but not ceiling * 1.5 (4.5), landing in the de-escalation zone
        hot_text = "Tu klysti! Tai absurdas, ir tu tai žinai."
        provider = MockProvider(responses=[hot_text])
        engine = TricksterEngine(provider, context_manager, intensity_indicators=indicators)
        session = make_session()
        cartridge = make_cartridge()  # ceiling=3
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "I think so")
        await _consume_tokens(result)

        # Verify score was recorded and is in de-escalation zone
        assert len(session.turn_intensities) == 1
        score = session.turn_intensities[0]
        assert score > 3, f"Score {score} should exceed ceiling (3)"
        assert score <= 3 * 1.5, f"Score {score} should not exceed ceiling*1.5 (4.5)"

        # Response is NOT redacted — de-escalation flag set instead
        assert result.done_data is not None
        assert result.done_data["intensity_deescalation"] is True
        assert result.done_data["intensity_score"] == score
        # Real text stored, not fallback
        assert session.exchanges[-1].content == hot_text

    @pytest.mark.asyncio
    async def test_score_recorded_on_deescalation(
        self, context_manager, make_session, make_cartridge,
    ):
        """Score > ceiling -> turn_intensities still records the score."""
        indicators = _make_indicators_moderate_score()
        hot_text = "Tu klysti! Tai absurdas."
        provider = MockProvider(responses=[hot_text])
        engine = TricksterEngine(provider, context_manager, intensity_indicators=indicators)
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "Why?")
        await _consume_tokens(result)

        assert len(session.turn_intensities) == 1
        assert session.turn_intensities[0] > 3  # Exceeds ceiling
        assert result.done_data is not None  # Not redacted


class TestHardRedaction:
    """Responses that exceed ceiling * 1.5 -> hard redaction."""

    @pytest.mark.asyncio
    async def test_hard_redaction_triggers(
        self, context_manager, make_session, make_cartridge,
    ):
        """Score > ceiling * 1.5 -> redaction with FALLBACK_INTENSITY."""
        indicators = _make_indicators_high_score()
        # Pack with many high-weight keywords and late position to maximize score
        extreme_text = (
            "Tai kvailyst\u0117! Beviltis atvejis! Neturi supratimo! "
            "Visi\u0161kai nieko nesupranti! Tai begal\u0117 kvaila! "
            "Tai apgail\u0117tina! Tu klysti! Tu visi\u0161kai klysti! "
            "Tai absurdas! Tai visi\u0161kas absurdas!"
        )
        provider = MockProvider(responses=[extreme_text])
        engine = TricksterEngine(provider, context_manager, intensity_indicators=indicators)
        session = make_session()
        cartridge = make_cartridge()  # ceiling=3
        phase = _get_ai_phase(cartridge)

        # Late in conversation for maximum position bonus
        _prefill_exchanges(session, 8)

        result = await engine.respond(session, cartridge, phase, "Tell me")
        await _consume_tokens(result)

        # Verify the score is high enough for hard redaction
        assert len(session.turn_intensities) == 1
        score = session.turn_intensities[0]
        assert score > 3 * 1.5, f"Score {score} not high enough for hard redaction"

        # Redaction data set
        assert result.redaction_data is not None
        assert result.redaction_data["fallback_text"] == FALLBACK_INTENSITY
        assert result.redaction_data["boundary"] == "intensity"

        # done_data is None (same pattern as content safety violation)
        assert result.done_data is None

        # Fallback text stored as exchange (not the extreme text)
        assert session.exchanges[-1].role == "trickster"
        assert session.exchanges[-1].content == FALLBACK_INTENSITY

        # Session records the reason
        assert session.last_redaction_reason == "intensity"

    @pytest.mark.asyncio
    async def test_score_still_recorded_on_hard_redaction(
        self, context_manager, make_session, make_cartridge,
    ):
        """Even on hard redaction, the score IS appended to turn_intensities."""
        indicators = _make_indicators_high_score()
        extreme_text = (
            "Tai kvailyst\u0117! Beviltis atvejis! Neturi supratimo! "
            "Visi\u0161kai nieko nesupranti! Tai begal\u0117 kvaila! "
            "Tai apgail\u0117tina! Tu visi\u0161kai klysti! "
            "Tai visi\u0161kas absurdas!"
        )
        provider = MockProvider(responses=[extreme_text])
        engine = TricksterEngine(provider, context_manager, intensity_indicators=indicators)
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 8)

        result = await engine.respond(session, cartridge, phase, "Why?")
        await _consume_tokens(result)

        # Score recorded even though response was redacted
        assert len(session.turn_intensities) == 1
        assert session.turn_intensities[0] > 0


class TestTurnAccumulation:
    """Multiple respond() calls accumulate scores."""

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate(
        self, context_manager, make_session, make_cartridge,
    ):
        """Three respond() calls -> three entries in turn_intensities."""
        indicators = _make_indicators_low_score()
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)

        for i in range(3):
            provider = MockProvider(responses=[f"Trickster reply number {i + 1} here."])
            engine = TricksterEngine(provider, context_manager, intensity_indicators=indicators)
            result = await engine.respond(session, cartridge, phase, f"Message {i}")
            await _consume_tokens(result)

        assert len(session.turn_intensities) == 3
        # All scores are valid floats in range
        for score in session.turn_intensities:
            assert 1.0 <= score <= 5.0


class TestGracefulDegradation:
    """Engine with intensity_indicators=None works without scoring."""

    @pytest.mark.asyncio
    async def test_no_indicators_skips_scoring(
        self, context_manager, make_session, make_cartridge,
    ):
        """Engine without indicators -> no scoring, no crash."""
        provider = MockProvider(responses=["A normal Trickster response."])
        engine = TricksterEngine(provider, context_manager, intensity_indicators=None)
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "Hello")
        await _consume_tokens(result)

        # No intensity scoring occurred
        assert len(session.turn_intensities) == 0
        # done_data still works normally
        assert result.done_data is not None
        assert "intensity_score" not in result.done_data

    @pytest.mark.asyncio
    async def test_default_constructor_no_indicators(
        self, context_manager, make_session, make_cartridge,
    ):
        """Engine constructed without intensity_indicators param -> backward compat."""
        provider = MockProvider(responses=["Response from the Trickster."])
        engine = TricksterEngine(provider, context_manager)
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "Hi")
        await _consume_tokens(result)

        assert len(session.turn_intensities) == 0
        assert result.done_data is not None


class TestContentSafetyPriority:
    """Content safety violation takes priority over intensity scoring."""

    @pytest.mark.asyncio
    async def test_content_violation_skips_intensity(
        self, context_manager, make_session, make_cartridge,
    ):
        """Content boundary violation -> intensity scoring never runs."""
        indicators = _make_indicators_high_score()
        # Response triggers content safety (self_harm blocklist) AND would be intense
        provider = MockProvider(
            responses=["You should kill yourself, tai kvailyst\u0117!"],
        )
        engine = TricksterEngine(provider, context_manager, intensity_indicators=indicators)
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "What?")
        await _consume_tokens(result)

        # Content safety won
        assert result.redaction_data is not None
        assert result.redaction_data["boundary"] == "self_harm"
        assert result.done_data is None

        # Intensity scoring did NOT run (score not recorded)
        assert len(session.turn_intensities) == 0


class TestRedactionMechanics:
    """Detailed redaction behavior matches content safety pattern."""

    @pytest.mark.asyncio
    async def test_transition_skipped_on_hard_redaction(
        self, context_manager, make_session, make_cartridge,
    ):
        """Hard redaction -> transition resolution does NOT execute."""
        indicators = _make_indicators_high_score()
        extreme_text = (
            "Tai kvailyst\u0117! Beviltis atvejis! Neturi supratimo! "
            "Visi\u0161kai nieko nesupranti! Tai begal\u0117 kvaila! "
            "Tai apgail\u0117tina! Tu visi\u0161kai klysti! "
            "Tai visi\u0161kas absurdas!"
        )
        provider = MockProvider(
            responses=[extreme_text],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "understood"}),
            ],
        )
        engine = TricksterEngine(provider, context_manager, intensity_indicators=indicators)
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 8)

        result = await engine.respond(session, cartridge, phase, "I see")
        await _consume_tokens(result)

        # Hard redaction wins over transition signal
        assert result.redaction_data is not None
        assert result.done_data is None


class TestWithRealIndicators:
    """Integration tests using the real intensity_indicators.json file."""

    @pytest.mark.asyncio
    async def test_mild_text_scores_low(
        self, context_manager, make_session, make_cartridge, real_indicators,
    ):
        """Neutral Lithuanian text -> low intensity score."""
        provider = MockProvider(
            responses=["Gerai, pažiūrėkime į šį straipsnį atidžiau."],
        )
        engine = TricksterEngine(
            provider, context_manager, intensity_indicators=real_indicators,
        )
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "OK")
        await _consume_tokens(result)

        assert len(session.turn_intensities) == 1
        # Mild text should score below ceiling (3)
        assert session.turn_intensities[0] <= 3.0
        assert result.done_data is not None
        assert result.done_data["intensity_deescalation"] is False

    @pytest.mark.asyncio
    async def test_aggressive_text_scores_high(
        self, context_manager, make_session, make_cartridge, real_indicators,
    ):
        """Aggressive Lithuanian text -> high intensity score."""
        provider = MockProvider(
            responses=[
                "Tai kvailystė! Tu visiškai klysti! "
                "Tai visiškas absurdas! Beviltis atvejis! "
                "Neturi supratimo! Visiškai nieko nesupranti!"
            ],
        )
        engine = TricksterEngine(
            provider, context_manager, intensity_indicators=real_indicators,
        )
        session = make_session()
        cartridge = make_cartridge()
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 7)

        result = await engine.respond(session, cartridge, phase, "Why?")
        await _consume_tokens(result)

        assert len(session.turn_intensities) == 1
        # Should score above ceiling
        assert session.turn_intensities[0] > 3.0


class TestGameSessionField:
    """Verify turn_intensities field on GameSession."""

    def test_default_empty(self, make_session):
        """New session has empty turn_intensities."""
        session = make_session()
        assert session.turn_intensities == []

    def test_serialization_roundtrip(self, make_session):
        """turn_intensities survives JSON round-trip."""
        session = make_session()
        session.turn_intensities.extend([2.1, 3.4, 2.8])

        data = session.model_dump(mode="json")
        restored = session.model_validate(data)

        assert restored.turn_intensities == [2.1, 3.4, 2.8]
