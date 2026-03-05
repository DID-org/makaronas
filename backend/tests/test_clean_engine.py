"""Tests for TricksterEngine with clean (non-manipulative) task cartridges (Phase 3b).

Verifies that the engine handles clean tasks without any code-level awareness
of ``is_clean`` — the paradigm inversion lives entirely in context assembly
(Phase 3a) and cartridge-level ``ai_transitions``. The engine's signal routing
is paradigm-agnostic.

Uses real ContextManager with PromptLoader pointed at temp prompts,
and MockProvider for deterministic AI responses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.ai.context import ContextManager
from backend.ai.prompts import PromptLoader
from backend.ai.providers.base import ToolCallEvent
from backend.ai.providers.mock import MockProvider
from backend.ai.trickster import DebriefResult, TricksterEngine, TricksterResult
from backend.schemas import Exchange
from backend.tasks.schemas import TaskCartridge
from backend.tests.conftest import setup_base_prompts, write_prompt_file


# ---------------------------------------------------------------------------
# Helpers (local, same pattern as test_trickster.py)
# ---------------------------------------------------------------------------

_CLEAN_EVAL = {
    "patterns_embedded": [],
    "checklist": [],
    "pass_conditions": {
        "trickster_wins": "Mokinys neteisingai apkaltino turin\u012f manipuliacija",
        "partial": "Mokinys abejojo, bet link\u0119s laikyti turin\u012f legitiminiu",
        "trickster_loses": "Mokinys teisingai atpa\u017eino legitim\u0173 turin\u012f",
    },
}

_CLEAN_PROMPT_CONTENT = (
    "\u0160is turinys yra legitimus. "
    "Gink jo patikimum\u0105 ir kvestionuok mokinio kaltinimus."
)


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


async def _consume_debrief_tokens(result: DebriefResult) -> str:
    """Exhausts debrief token_iterator and returns accumulated text."""
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


def _setup_clean_prompt(prompts_dir: Path) -> None:
    """Creates the clean_task_base.md prompt file."""
    write_prompt_file(
        prompts_dir / "trickster" / "clean_task_base.md",
        _CLEAN_PROMPT_CONTENT,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompts_dir(tmp_path):
    """Creates temp directory with base Trickster prompts + clean task prompt."""
    setup_base_prompts(tmp_path)
    _setup_clean_prompt(tmp_path)
    return tmp_path


@pytest.fixture
def context_manager(prompts_dir):
    """Real ContextManager with PromptLoader pointed at temp prompts."""
    loader = PromptLoader(prompts_dir)
    return ContextManager(loader)


@pytest.fixture
def make_engine(context_manager):
    """Factory for TricksterEngine with configurable MockProvider."""

    def _make(**provider_kwargs) -> TricksterEngine:
        provider = MockProvider(**provider_kwargs)
        return TricksterEngine(provider, context_manager)

    return _make


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestCleanTaskRespondFlow:
    """Engine respond() with clean cartridge — tokens, exchanges, transitions."""

    @pytest.mark.asyncio
    async def test_clean_task_tokens_yielded(
        self, make_engine, make_session, make_cartridge,
    ):
        """respond() with clean cartridge yields token text correctly."""
        engine = make_engine(
            responses=["\u0160is straipsnis yra tikras ir patikimas."],
        )
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "Ar tai tiesa?")
        text = await _consume_tokens(result)

        assert text == "\u0160is straipsnis yra tikras ir patikimas."
        assert isinstance(result, TricksterResult)

    @pytest.mark.asyncio
    async def test_clean_task_exchanges_saved(
        self, make_engine, make_session, make_cartridge,
    ):
        """Student and trickster exchanges are saved after respond()."""
        engine = make_engine(
            responses=["Turinys yra patikimas, galite juo pasikliauti."],
        )
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(
            session, cartridge, phase, "Manau tai manipuliacija",
        )
        await _consume_tokens(result)

        assert len(session.exchanges) == 2
        assert session.exchanges[0].role == "student"
        assert session.exchanges[0].content == "Manau tai manipuliacija"
        assert session.exchanges[1].role == "trickster"

    @pytest.mark.asyncio
    async def test_clean_task_no_transition_early(
        self, make_engine, make_session, make_cartridge,
    ):
        """First exchange (below min_exchanges) produces no transition."""
        engine = make_engine(
            responses=["Atsakymas apie legitimaus turinio konteksta."],
        )
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        phase = _get_ai_phase(cartridge)

        result = await engine.respond(session, cartridge, phase, "Pirmas klausimas")
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] is None
        assert result.done_data["next_phase"] is None
        assert result.done_data["exchanges_count"] == 1

    @pytest.mark.asyncio
    async def test_clean_task_understood_signal(
        self, make_engine, make_session, make_cartridge,
    ):
        """'understood' signal maps to on_success -> phase_reveal_success.

        For clean tasks, this means the student correctly identified
        the content as legitimate.
        """
        engine = make_engine(
            responses=["Taip, teisingai supratote!"],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "understood"}),
            ],
        )
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)  # meet min_exchanges=2

        result = await engine.respond(session, cartridge, phase, "Turinys atrodo tikras")
        await _consume_tokens(result)

        assert result.done_data["phase_transition"] == "on_success"
        assert result.done_data["next_phase"] == "phase_reveal_success"

    @pytest.mark.asyncio
    async def test_clean_task_partial_signal(
        self, make_engine, make_session, make_cartridge,
    ):
        """'partial' signal maps to on_partial -> phase_reveal_partial."""
        engine = make_engine(
            responses=["Nesu tikras, ar supratote iki galo."],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "partial"}),
            ],
        )
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)

        result = await engine.respond(session, cartridge, phase, "Gal tai tiesa?")
        await _consume_tokens(result)

        assert result.done_data["phase_transition"] == "on_partial"
        assert result.done_data["next_phase"] == "phase_reveal_partial"

    @pytest.mark.asyncio
    async def test_clean_task_max_reached_signal(
        self, make_engine, make_session, make_cartridge,
    ):
        """'max_reached' signal maps to on_max_exchanges -> phase_reveal_timeout."""
        engine = make_engine(
            responses=["Laikas baigiasi, pabandykite dar karta."],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "max_reached"}),
            ],
        )
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase, "Tai tikrai manipuliacija!",
        )
        await _consume_tokens(result)

        assert result.done_data["phase_transition"] == "on_max_exchanges"
        assert result.done_data["next_phase"] == "phase_reveal_timeout"

    @pytest.mark.asyncio
    async def test_clean_task_max_exchanges_ceiling(
        self, context_manager, make_session, make_cartridge,
    ):
        """At max_exchanges with no tool call, on_max_exchanges fires automatically."""
        provider = MockProvider(
            responses=["Paskutinis atsakymas be pertraukos signalo."],
        )
        engine = TricksterEngine(provider, context_manager)
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        phase = _get_ai_phase(cartridge)

        # max_exchanges=10, pre-fill 9 -> this is message #10
        _prefill_exchanges(session, 9)

        result = await engine.respond(
            session, cartridge, phase, "Paskutine zinute",
        )
        await _consume_tokens(result)

        assert result.done_data["phase_transition"] == "on_max_exchanges"
        assert result.done_data["next_phase"] == "phase_reveal_timeout"
        assert result.done_data["exchanges_count"] == 10


class TestCleanTaskDebrief:
    """Debrief with clean cartridge — no crash, valid output."""

    @pytest.mark.asyncio
    async def test_clean_task_debrief_no_crash(
        self, make_engine, make_session, make_cartridge,
    ):
        """debrief() with clean cartridge returns DebriefResult with debrief_complete."""
        engine = make_engine(
            responses=["Aptarkime, ka pastebejote siame turinyje."],
        )
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        _prefill_exchanges(session, 3)

        result = await engine.debrief(session, cartridge)
        text = await _consume_debrief_tokens(result)

        assert isinstance(result, DebriefResult)
        assert text == "Aptarkime, ka pastebejote siame turinyje."
        assert result.done_data == {"debrief_complete": True}
        assert result.redaction_data is None

    @pytest.mark.asyncio
    async def test_clean_task_debrief_exchange_saved(
        self, make_engine, make_session, make_cartridge,
    ):
        """Debrief saves trickster exchange to session."""
        engine = make_engine(
            responses=["Debrief turinys apie svaru uzduoti."],
        )
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        _prefill_exchanges(session, 2)
        exchanges_before = len(session.exchanges)

        result = await engine.debrief(session, cartridge)
        await _consume_debrief_tokens(result)

        assert len(session.exchanges) == exchanges_before + 1
        assert session.exchanges[-1].role == "trickster"
        assert session.exchanges[-1].content == "Debrief turinys apie svaru uzduoti."

    @pytest.mark.asyncio
    async def test_clean_task_debrief_safety(
        self, make_engine, make_session, make_cartridge,
    ):
        """Debrief with clean cartridge passes safety check (no blocklist terms)."""
        engine = make_engine(
            responses=["Svaraus turinio vertinimas ir aptarimas."],
        )
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        _prefill_exchanges(session, 2)

        result = await engine.debrief(session, cartridge)
        await _consume_debrief_tokens(result)

        assert result.done_data == {"debrief_complete": True}
        assert result.redaction_data is None


class TestCleanTaskPromptSnapshot:
    """Prompt snapshotting works identically for clean tasks."""

    @pytest.mark.asyncio
    async def test_clean_task_snapshots_prompts(
        self, make_engine, make_session, make_cartridge,
    ):
        """First respond() with clean cartridge populates session.prompt_snapshots."""
        engine = make_engine(
            responses=["Atsakymas apie patikima turini."],
        )
        session = make_session()
        cartridge = make_cartridge(is_clean=True, evaluation=_CLEAN_EVAL)
        phase = _get_ai_phase(cartridge)

        assert session.prompt_snapshots is None

        result = await engine.respond(session, cartridge, phase, "Sveiki")
        await _consume_tokens(result)

        assert session.prompt_snapshots is not None
        assert "persona" in session.prompt_snapshots
        assert "behaviour" in session.prompt_snapshots
        assert "safety" in session.prompt_snapshots
