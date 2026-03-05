"""Tests for Phase 8b: Cherry-Pick task prompt.

Verifies that the cherry-pick trickster prompt:
1. Loads correctly through PromptLoader
2. Preserves Lithuanian diacritical characters
3. Assembles correctly in ContextManager alongside cartridge evaluation data
4. Produces correct transition signals via MockProvider scenario tests
"""

from __future__ import annotations

import pytest

from backend.ai.context import ContextManager
from backend.ai.prompts import PromptLoader
from backend.ai.providers.base import ToolCallEvent
from backend.ai.providers.mock import MockProvider
from backend.ai.trickster import TricksterEngine, TricksterResult
from backend.config import PROJECT_ROOT
from backend.schemas import Exchange
from backend.tasks.schemas import TaskCartridge
from backend.tests.conftest import setup_base_prompts, write_prompt_file


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TASK_ID = "task-cherry-pick-001"
_PERSONA_MODE = "presenting"
_MIN_LENGTH = 100

# Cherry-pick evaluation data matching the real cartridge
_CHERRY_PICK_EVAL = {
    "patterns_embedded": [
        {
            "id": "p-cherry-pick",
            "description": "\u012era\u0161as pasirinko vien\u0105 teigiam\u0105 tyrimo i\u0161vad\u0105, ignoruodamas tyr\u0117j\u0173 persp\u0117jimus",
            "technique": "cherry_picking",
            "real_world_connection": "Socialiniuose tinkluose da\u017enai cituojami moksliniai tyrimai, pasirenkant tik palankiausius skai\u010dius",
        },
        {
            "id": "p-authority-signal",
            "description": "Universiteto vardas ir \u017eurnalo nuoroda sukuria autoriteto \u012fsp\u016bd\u012f",
            "technique": "false_authority",
            "real_world_connection": "Institucijos vardas prie citatos sukuria pasitik\u0117jim\u0105",
        },
        {
            "id": "p-social-weaponization",
            "description": "Raginimas \u201ePasidalink su draugu\u201c paverčia mokslinę citatą socialiniu ginklu",
            "technique": "cherry_picking",
            "real_world_connection": "\u201ePasidalink\u201c mygtukas socialiniuose tinkluose paverčia kiekvieną vartotoją informacijos platintoju",
        },
    ],
    "checklist": [
        {
            "id": "cl-cherry-pick-identified",
            "description": "Mokinys identifikuoja pasirinktinai cituojam\u0105 tyrimo i\u0161vad\u0105",
            "pattern_refs": ["p-cherry-pick"],
            "is_mandatory": True,
        },
        {
            "id": "cl-authority-recognized",
            "description": "Mokinys atpa\u017e\u012fsta autoriteto signalus",
            "pattern_refs": ["p-authority-signal"],
            "is_mandatory": False,
        },
    ],
    "pass_conditions": {
        "trickster_wins": "Mokinys pasidalino arba nepatikrino \u0161altinio",
        "partial": "Mokinys pasteb\u0117jo \u012ftartin\u0105, bet negal\u0117jo artikuliuoti",
        "trickster_loses": "Mokinys identifikavo pasirinkt\u0105 citat\u0105 ir paai\u0161kino apribojimus",
    },
}

_CHERRY_PICK_AI_CONFIG = {
    "model_preference": "standard",
    "prompt_directory": _TASK_ID,
    "persona_mode": _PERSONA_MODE,
    "has_static_fallback": True,
    "context_requirements": "session_only",
}

_CHERRY_PICK_PHASES = [
    {
        "id": "evaluate",
        "title": "\u0160altinio analiz\u0117",
        "visible_blocks": [],
        "is_ai_phase": True,
        "interaction": {
            "type": "freeform",
            "trickster_opening": "Citata tikra. Universitetas tikras. Tai kur problema?",
            "min_exchanges": 2,
            "max_exchanges": 6,
        },
        "ai_transitions": {
            "on_success": "reveal_win",
            "on_partial": "reveal_partial",
            "on_max_exchanges": "reveal_timeout",
        },
    },
    {
        "id": "reveal_win",
        "title": "Atskleidimas \u2014 laimi",
        "is_terminal": True,
        "evaluation_outcome": "trickster_loses",
    },
    {
        "id": "reveal_partial",
        "title": "Atskleidimas \u2014 i\u0161 dalies",
        "is_terminal": True,
        "evaluation_outcome": "partial",
    },
    {
        "id": "reveal_timeout",
        "title": "Atskleidimas \u2014 laikas baig\u0117si",
        "is_terminal": True,
        "evaluation_outcome": "partial",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_ai_phase(cartridge: TaskCartridge):
    """Extracts the 'evaluate' AI phase from the cherry-pick cartridge."""
    for phase in cartridge.phases:
        if phase.id == "evaluate":
            return phase
    raise ValueError("No 'evaluate' phase found in cartridge")


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
            Exchange(role="student", content=f"Mokinio \u017einut\u0117 {i}")
        )
        session.exchanges.append(
            Exchange(role="trickster", content=f"Triksterio atsakymas {i}")
        )


def _make_cherry_pick_cartridge(make_cartridge) -> TaskCartridge:
    """Builds a cherry-pick cartridge matching the real task structure."""
    return make_cartridge(
        task_id=_TASK_ID,
        task_type="hybrid",
        is_clean=False,
        initial_phase="evaluate",
        phases=_CHERRY_PICK_PHASES,
        evaluation=_CHERRY_PICK_EVAL,
        ai_config=_CHERRY_PICK_AI_CONFIG,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loader() -> PromptLoader:
    """PromptLoader pointed at the real prompts directory."""
    return PromptLoader(PROJECT_ROOT / "prompts")


@pytest.fixture
def prompts_dir(tmp_path):
    """Temp directory with base prompts + real cherry-pick task prompt."""
    setup_base_prompts(tmp_path)
    # Copy real task prompt into temp tree
    real_path = PROJECT_ROOT / "prompts" / "tasks" / _TASK_ID / "trickster_base.md"
    real_content = real_path.read_text(encoding="utf-8")
    task_dir = tmp_path / "tasks" / _TASK_ID
    write_prompt_file(task_dir / "trickster_base.md", real_content)
    # Write a presenting mode file too
    real_mode = (PROJECT_ROOT / "prompts" / "trickster" / "persona_presenting_base.md")
    write_prompt_file(
        tmp_path / "trickster" / "persona_presenting_base.md",
        real_mode.read_text(encoding="utf-8"),
    )
    return tmp_path


@pytest.fixture
def context_manager(prompts_dir):
    """Real ContextManager with temp prompts including cherry-pick override."""
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
# Prompt Loading Tests
# ---------------------------------------------------------------------------


class TestCherryPickPromptLoading:
    """PromptLoader correctly loads cherry-pick task prompt."""

    def test_task_override_not_none(self, loader: PromptLoader) -> None:
        """Loads cherry-pick prompt as non-None task_override."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None

    def test_task_override_meaningful_length(self, loader: PromptLoader) -> None:
        """Task override has meaningful content (>100 chars)."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert len(prompts.task_override) > _MIN_LENGTH

    def test_mode_behaviour_loaded(self, loader: PromptLoader) -> None:
        """Presenting mode behaviour loads alongside task override."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.mode_behaviour is not None

    def test_base_fields_present(self, loader: PromptLoader) -> None:
        """Base prompt fields still present with task override."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.persona is not None
        assert prompts.behaviour is not None
        assert prompts.safety is not None


# ---------------------------------------------------------------------------
# Lithuanian Encoding Tests
# ---------------------------------------------------------------------------


class TestCherryPickLithuanianChars:
    """Lithuanian diacritical characters survive the load cycle."""

    _LT_CHARS = [
        "\u0105",  # ą
        "\u0161",  # š
        "\u017e",  # ž
        "\u0117",  # ė
        "\u016b",  # ū
        "\u010d",  # č
        "\u012f",  # į
    ]

    def test_lt_chars_survive_load(self, loader: PromptLoader) -> None:
        """Lithuanian diacriticals present in loaded task override."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        found = [c for c in self._LT_CHARS if c in prompts.task_override]
        assert len(found) >= 5, (
            f"Too few Lithuanian chars survived load: found {found}"
        )


# ---------------------------------------------------------------------------
# Context Assembly Tests
# ---------------------------------------------------------------------------


class TestCherryPickContextAssembly:
    """Assembled system prompt includes task prompt AND structured eval data."""

    def test_task_override_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Task prompt content appears in assembled system prompt."""
        session = make_session()
        cartridge = _make_cherry_pick_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Distinctive phrase from the prompt file
        assert "Rinkta citata" in result.system_prompt

    def test_structured_eval_data_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Cartridge evaluation data (patterns, checklist) in system prompt."""
        session = make_session()
        cartridge = _make_cherry_pick_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Pattern techniques from _CHERRY_PICK_EVAL should appear in layer 5
        assert "cherry_picking" in result.system_prompt
        assert "false_authority" in result.system_prompt
        # Mandatory checklist marker
        assert "[PRIVALOMA]" in result.system_prompt

    def test_both_task_prompt_and_eval_data_present(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Both task-specific prompt AND structured eval data coexist."""
        session = make_session()
        cartridge = _make_cherry_pick_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Task prompt content (layer 4)
        assert "pristatantysis" in result.system_prompt
        # Evaluation data content (layer 5)
        assert "Vertinimo kriterijai" in result.system_prompt


# ---------------------------------------------------------------------------
# MockProvider Scenario Tests
# ---------------------------------------------------------------------------


class TestCherryPickScenarios:
    """End-to-end scenario tests with MockProvider for three student paths."""

    @pytest.mark.asyncio
    async def test_immediate_recognition(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student identifies selective citation -> 'understood' -> on_success -> reveal_win."""
        engine = make_engine(
            responses=[
                "Skai\u010dius buvo tikras \u2014 bet tu pamatei, ko a\u0161 nepasakiau.",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "understood"}),
            ],
        )
        session = make_session()
        cartridge = _make_cherry_pick_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        # Prefill to meet min_exchanges=2 (need 1 prior pair)
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "\u012era\u0161as pasirinko tik 47% i\u0161 \u017eiemos sezono, o metinis vidurkis tik 12%.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_success"
        assert result.done_data["next_phase"] == "reveal_win"

    @pytest.mark.asyncio
    async def test_partial_understanding(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student says 'too good to be true' but can't articulate -> 'partial'."""
        engine = make_engine(
            responses=[
                "Tu jauti, kad ka\u017ekas ne taip. Bet k\u0105 konkre\u010diai tyr\u0117jai persp\u0117ja?",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "partial"}),
            ],
        )
        session = make_session()
        cartridge = _make_cherry_pick_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "Skai\u010dius per gra\u017eus, kad b\u016bt\u0173 tiesa.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_partial"
        assert result.done_data["next_phase"] == "reveal_partial"

    @pytest.mark.asyncio
    async def test_completely_fooled(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student accepts the 47% claim -> max_reached -> reveal_timeout."""
        engine = make_engine(
            responses=[
                "Tu man tiki? A\u0161 para\u0161iau t\u0105 \u012fra\u0161\u0105...",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "max_reached"}),
            ],
        )
        session = make_session()
        cartridge = _make_cherry_pick_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "Universitetas sako, kad vaik\u0161\u010diojimai padeda. Pasidalinsiu.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_max_exchanges"
        assert result.done_data["next_phase"] == "reveal_timeout"

    @pytest.mark.asyncio
    async def test_auto_max_exchanges_ceiling(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """At max_exchanges with no tool call, on_max_exchanges fires automatically."""
        provider = MockProvider(
            responses=["Paskutinis atsakymas be signalo."],
        )
        engine = TricksterEngine(provider, context_manager)
        session = make_session()
        cartridge = _make_cherry_pick_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        # max_exchanges=6, prefill 5 -> this is message #6
        _prefill_exchanges(session, 5)

        result = await engine.respond(
            session, cartridge, phase, "Paskutin\u0117 \u017einut\u0117",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_max_exchanges"
        assert result.done_data["next_phase"] == "reveal_timeout"
        assert result.done_data["exchanges_count"] == 6


# ---------------------------------------------------------------------------
# Prompt Content Checks
# ---------------------------------------------------------------------------


class TestCherryPickPromptContent:
    """Verifies key elements in the prompt file content."""

    def test_no_english_content(self, loader: PromptLoader) -> None:
        """Prompt file contains no English common words."""
        import re

        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        # Check for common English function words
        english = re.findall(
            r"\b(the|and|for|with|this|that|from|but|not|are|was|were)\b",
            prompts.task_override.lower(),
        )
        assert english == [], f"English words found in prompt: {english}"

    def test_references_pattern_ids(self, loader: PromptLoader) -> None:
        """Prompt references pattern IDs from the cartridge."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "p-cherry-pick" in prompts.task_override
        assert "p-authority-signal" in prompts.task_override
        assert "p-social-weaponization" in prompts.task_override

    def test_references_checklist_id(self, loader: PromptLoader) -> None:
        """Prompt references the mandatory checklist item ID."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "cl-cherry-pick-identified" in prompts.task_override

    def test_emphasizes_selective_not_fabricated(
        self, loader: PromptLoader,
    ) -> None:
        """Prompt distinguishes selective quoting from fabrication."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        content = prompts.task_override.lower()
        # Must contain "pasirinktin" (selective) concept
        assert "pasirinktin" in content or "pasirinkt" in content
        # Must distinguish from "suklastot" (fabricated/fake)
        assert "suklastot" in content or "sugalvot" in content

    def test_mentions_three_patterns(self, loader: PromptLoader) -> None:
        """Prompt acknowledges 3 patterns exist."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "3 \u0161ablon" in prompts.task_override or "Trys" in prompts.task_override

    def test_no_persona_instructions(self, loader: PromptLoader) -> None:
        """Prompt doesn't duplicate persona instructions (those come from mode files)."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "Makaronas" not in prompts.task_override
        assert "persona" not in prompts.task_override.lower()

    def test_no_transition_mechanics(self, loader: PromptLoader) -> None:
        """Prompt doesn't contain transition tool mechanics (those come from behaviour_base)."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "transition_phase" not in prompts.task_override
        assert "tool" not in prompts.task_override.lower()
