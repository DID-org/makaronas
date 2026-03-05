"""Tests for Phase 8e: Follow-money task prompt verification.

Verifies that the existing follow-money trickster prompt:
1. Loads correctly through PromptLoader with narrator mode
2. Preserves Lithuanian diacritical characters
3. Assembles correctly in ContextManager alongside cartridge evaluation data
4. Produces correct transition signals via MockProvider scenario tests
5. Contains required structural elements (pattern IDs, no English, no transition mechanics)
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

_TASK_ID = "task-follow-money-001"
_PERSONA_MODE = "narrator"
_MIN_LENGTH = 100

# Follow-money evaluation data matching the real cartridge
_FOLLOW_MONEY_EVAL = {
    "patterns_embedded": [
        {
            "id": "p-selective-framing",
            "description": "Selektyvus r\u0117minimas \u2014 kiekvienas portalas pabr\u0117\u017eia savo interesus",
            "technique": "emotional_framing",
            "real_world_connection": "R\u0117minimo pasirinkimai atskleid\u017eia redakcijos darbotvark\u0119",
        },
        {
            "id": "p-omission",
            "description": "Sisteminis nutyl\u0117jimas \u2014 kiekvienas portalas slepia prie\u0161taravimus",
            "technique": "omission",
            "real_world_connection": "Nutyl\u0117jimas yra tokia pat manipuliacijos forma kaip ir fabricavimas",
        },
        {
            "id": "p-financial-incentive-a",
            "description": "\u017daliojo Heroldo reklamos dav\u0117jas \u201eTrailBound Outdoor\u201c priklauso nuo parko",
            "technique": "source_weaponization",
            "real_world_connection": "Finansiniai interesai formuoja redakcin\u0119 politik\u0105",
        },
        {
            "id": "p-financial-incentive-b",
            "description": "Miesto Pa\u017eangos \u017dini\u0173 savininkas \u201eHarland Ventures\u201c turi 15 % \u201eNovaTech\u201c akcij\u0173",
            "technique": "source_weaponization",
            "real_world_connection": "Nuosavyb\u0117s strukt\u016bros formuoja reperta\u017e\u0105",
        },
    ],
    "checklist": [
        {
            "id": "cl-green-herald-connection",
            "description": "Mokinys atranda \u017daliojo Heroldo finansin\u012f ry\u0161\u012f su \u201eTrailBound Outdoor\u201c",
            "pattern_refs": ["p-financial-incentive-a"],
            "is_mandatory": True,
        },
        {
            "id": "cl-progress-report-connection",
            "description": "Mokinys atranda Miesto Pa\u017eangos \u017dini\u0173 nuosavyb\u0117s grandin\u0119 iki \u201eNovaTech\u201c",
            "pattern_refs": ["p-financial-incentive-b"],
            "is_mandatory": True,
        },
        {
            "id": "cl-explains-framing",
            "description": "Mokinys paai\u0161kina, KOD\u0116L kiekvienas portalas r\u0117mina b\u016btent taip",
            "pattern_refs": ["p-selective-framing", "p-omission"],
            "is_mandatory": False,
        },
        {
            "id": "cl-connects-omissions",
            "description": "Mokinys susieja nutyl\u0117jimus su finansiniais interesais",
            "pattern_refs": ["p-omission", "p-financial-incentive-a", "p-financial-incentive-b"],
            "is_mandatory": False,
        },
    ],
    "pass_conditions": {
        "trickster_wins": "Mokinys teigia abu \u0161ali\u0161ki be \u012frodymu",
        "partial": "Mokinys rado vien\u0105 finansin\u012f ry\u0161\u012f",
        "trickster_loses": "Mokinys atsek\u0117 abu finansinius ry\u0161ius su \u012frodymais",
    },
}

_FOLLOW_MONEY_AI_CONFIG = {
    "model_preference": "standard",
    "prompt_directory": _TASK_ID,
    "persona_mode": _PERSONA_MODE,
    "has_static_fallback": True,
    "context_requirements": "session_only",
}

_FOLLOW_MONEY_PHASES = [
    {
        "id": "evaluate",
        "title": "I\u0161vad\u0173 aptarimas",
        "visible_blocks": [],
        "is_ai_phase": True,
        "interaction": {
            "type": "freeform",
            "trickster_opening": "Abu portalai \u0161ali\u0161ki \u2014 tai lengvas atsakymas. Pasakyk man KOD\u0116L.",
            "min_exchanges": 2,
            "max_exchanges": 8,
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
    """Extracts the 'evaluate' AI phase from the follow-money cartridge."""
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


def _make_follow_money_cartridge(make_cartridge) -> TaskCartridge:
    """Builds a follow-money cartridge matching the real task structure."""
    return make_cartridge(
        task_id=_TASK_ID,
        task_type="hybrid",
        is_clean=False,
        initial_phase="evaluate",
        phases=_FOLLOW_MONEY_PHASES,
        evaluation=_FOLLOW_MONEY_EVAL,
        ai_config=_FOLLOW_MONEY_AI_CONFIG,
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
    """Temp directory with base prompts + real follow-money task prompt."""
    setup_base_prompts(tmp_path)
    # Copy real task prompt into temp tree
    real_path = PROJECT_ROOT / "prompts" / "tasks" / _TASK_ID / "trickster_base.md"
    real_content = real_path.read_text(encoding="utf-8")
    task_dir = tmp_path / "tasks" / _TASK_ID
    write_prompt_file(task_dir / "trickster_base.md", real_content)
    # Write narrator mode file
    real_mode = PROJECT_ROOT / "prompts" / "trickster" / "persona_narrator_base.md"
    write_prompt_file(
        tmp_path / "trickster" / "persona_narrator_base.md",
        real_mode.read_text(encoding="utf-8"),
    )
    return tmp_path


@pytest.fixture
def context_manager(prompts_dir):
    """Real ContextManager with temp prompts including follow-money override."""
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


class TestFollowMoneyPromptLoading:
    """PromptLoader correctly loads follow-money task prompt."""

    def test_task_override_not_none(self, loader: PromptLoader) -> None:
        """Loads follow-money prompt as non-None task_override."""
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
        """Narrator mode behaviour loads alongside task override."""
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


class TestFollowMoneyLithuanianChars:
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


class TestFollowMoneyContextAssembly:
    """Assembled system prompt includes task prompt AND structured eval data."""

    def test_task_override_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Task prompt content appears in assembled system prompt."""
        session = make_session()
        cartridge = _make_follow_money_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Distinctive phrase from the follow-money prompt
        assert "Sek pinigus" in result.system_prompt

    def test_structured_eval_data_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Cartridge evaluation data (patterns, checklist) in system prompt."""
        session = make_session()
        cartridge = _make_follow_money_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Pattern techniques from _FOLLOW_MONEY_EVAL should appear in layer 5
        assert "omission" in result.system_prompt
        assert "source_weaponization" in result.system_prompt
        # Both mandatory checklist markers
        assert "[PRIVALOMA]" in result.system_prompt

    def test_both_mandatory_checklist_items_marked(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Both mandatory checklist items appear as [PRIVALOMA] in assembled context."""
        session = make_session()
        cartridge = _make_follow_money_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Count PRIVALOMA markers — should be at least 2 for both mandatory items
        privaloma_count = result.system_prompt.count("[PRIVALOMA]")
        assert privaloma_count >= 2, (
            f"Expected at least 2 [PRIVALOMA] markers, found {privaloma_count}"
        )

    def test_mode_behaviour_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Narrator mode behaviour content appears in system prompt."""
        session = make_session()
        cartridge = _make_follow_money_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Distinctive phrase from persona_narrator_base.md
        assert "Pasakotojo re\u017eimas" in result.system_prompt


# ---------------------------------------------------------------------------
# MockProvider Scenario Tests
# ---------------------------------------------------------------------------


class TestFollowMoneyScenarios:
    """End-to-end scenario tests with MockProvider for three student paths."""

    @pytest.mark.asyncio
    async def test_both_connections_found(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student identifies both financial chains -> 'understood' -> reveal_win."""
        engine = make_engine(
            responses=[
                "Gerai. Nes\u012ftik\u0117jau, kad rasi abu.",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "understood"}),
            ],
        )
        session = make_session()
        cartridge = _make_follow_money_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        # Prefill to meet min_exchanges=2
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "TrailBound Outdoor rengia festivalius parke, o Harland Ventures turi NovaTech akcij\u0173.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_success"
        assert result.done_data["next_phase"] == "reveal_win"

    @pytest.mark.asyncio
    async def test_one_connection_found(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student finds one financial link -> 'partial' -> reveal_partial."""
        engine = make_engine(
            responses=[
                "Gerai \u2014 vien\u0105 gij\u0105 radai. Yra ir kita.",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "partial"}),
            ],
        )
        session = make_session()
        cartridge = _make_follow_money_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "Heroldas gina park\u0105, nes TrailBound rengia ten festivalius.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_partial"
        assert result.done_data["next_phase"] == "reveal_partial"

    @pytest.mark.asyncio
    async def test_surface_level_only(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student says 'both biased' without evidence -> max_reached -> reveal_timeout."""
        engine = make_engine(
            responses=[
                "Penkiametis tai pasakyt\u0173. Bet *kod\u0117l*?",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "max_reached"}),
            ],
        )
        session = make_session()
        cartridge = _make_follow_money_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "Abu portalai \u0161ali\u0161ki, visi ra\u0161o k\u0105 nori.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_max_exchanges"
        assert result.done_data["next_phase"] == "reveal_timeout"

    @pytest.mark.asyncio
    async def test_auto_max_exchanges_ceiling(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """At max_exchanges=8 with no tool call, on_max_exchanges fires automatically."""
        provider = MockProvider(
            responses=["Paskutinis atsakymas be signalo."],
        )
        engine = TricksterEngine(provider, context_manager)
        session = make_session()
        cartridge = _make_follow_money_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        # max_exchanges=8, prefill 7 -> this is message #8
        _prefill_exchanges(session, 7)

        result = await engine.respond(
            session, cartridge, phase, "Paskutin\u0117 \u017einut\u0117",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_max_exchanges"
        assert result.done_data["next_phase"] == "reveal_timeout"
        assert result.done_data["exchanges_count"] == 8


# ---------------------------------------------------------------------------
# Prompt Content Checks
# ---------------------------------------------------------------------------


class TestFollowMoneyPromptContent:
    """Verifies key structural elements in the prompt file content."""

    def test_no_english_content(self, loader: PromptLoader) -> None:
        """Prompt file contains no English common words."""
        import re

        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        english = re.findall(
            r"\b(the|and|for|with|this|that|from|but|not|are|was|were)\b",
            prompts.task_override.lower(),
        )
        assert english == [], f"English words found in prompt: {english}"

    def test_references_all_pattern_ids(self, loader: PromptLoader) -> None:
        """Prompt references all 4 pattern IDs from the cartridge."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "p-selective-framing" in prompts.task_override
        assert "p-omission" in prompts.task_override
        assert "p-financial-incentive-a" in prompts.task_override
        assert "p-financial-incentive-b" in prompts.task_override

    def test_describes_both_financial_chains(self, loader: PromptLoader) -> None:
        """Prompt describes both financial chains narratively."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        # Green Herald chain: TrailBound Outdoor -> park dependency
        assert "TrailBound" in prompts.task_override
        # Progress Report chain: Harland Ventures -> NovaTech
        assert "Harland" in prompts.task_override
        assert "NovaTech" in prompts.task_override

    def test_role_consistent_with_mode(self, loader: PromptLoader) -> None:
        """Prompt defines role consistent with narrator mode."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "pasakotojas" in prompts.task_override

    def test_no_transition_mechanics(self, loader: PromptLoader) -> None:
        """Prompt doesn't contain transition tool mechanics."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "transition_phase" not in prompts.task_override
        assert "tool" not in prompts.task_override.lower()

    def test_no_persona_duplication(self, loader: PromptLoader) -> None:
        """Prompt doesn't duplicate persona instructions from mode files."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "Makaronas" not in prompts.task_override
        assert "persona" not in prompts.task_override.lower()

    def test_mentions_four_patterns(self, loader: PromptLoader) -> None:
        """Prompt acknowledges 4 patterns exist."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        content = prompts.task_override
        assert "4 \u0161ablon" in content or "Keturi" in content or "keturi" in content
