"""Tests for Phase 8e: Clickbait-trap task prompt verification.

Verifies that the existing clickbait-trap trickster prompt:
1. Loads correctly through PromptLoader with presenting mode
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

_TASK_ID = "task-clickbait-trap-001"
_PERSONA_MODE = "presenting"
_MIN_LENGTH = 100

# Clickbait-trap evaluation data matching the real cartridge
_CLICKBAIT_EVAL = {
    "patterns_embedded": [
        {
            "id": "p-emotional-lang",
            "description": "Emocin\u0117 kalba antra\u0161t\u0117je \u2014 \u201eprover\u017eis\u201c, \u201erevoliucinis\u201c, \u201enulin\u0117s emisijos\u201c",
            "technique": "emotional_framing",
            "real_world_connection": "Naujienu antra\u0161t\u0117s naudoja emocinius \u017eod\u017eius",
        },
        {
            "id": "p-urgency",
            "description": "Dirbtinis skubumas \u2014 \u201eprover\u017eis\u201c ir \u201enulin\u0117s emisijos\u201c sukuria \u012fsp\u016bd\u012f",
            "technique": "manufactured_deadline",
            "real_world_connection": "\u017dodis \u201eprover\u017eis\u201c ver\u010dia galvoti, kad ka\u017ekas jau \u012fvyko",
        },
        {
            "id": "p-headline-contradiction",
            "description": "Antra\u0161t\u0117s-straipsnio prie\u0161taravimas",
            "technique": "headline_manipulation",
            "real_world_connection": "Atotr\u016bkis tarp antra\u0161t\u0117s pa\u017eado ir straipsnio tikrov\u0117s",
        },
        {
            "id": "p-snippet-framing",
            "description": "Fragmento r\u0117minimas \u2014 trumpas apra\u0161ymas pasirenka klaid\u017eiausi\u0105 kamp\u0105",
            "technique": "cherry_picking",
            "real_world_connection": "Anonsai ir fragmentai atrenka informacij\u0105",
        },
    ],
    "checklist": [
        {
            "id": "cl-headline-contradiction",
            "description": "Mokinys identifikuoja antra\u0161t\u0117s ir straipsnio prie\u0161taravim\u0105",
            "pattern_refs": ["p-headline-contradiction"],
            "is_mandatory": True,
        },
        {
            "id": "cl-emotional-language",
            "description": "Mokinys atpa\u017e\u012fsta emocin\u0117s kalbos \u0161ablonus",
            "pattern_refs": ["p-emotional-lang", "p-urgency"],
            "is_mandatory": False,
        },
        {
            "id": "cl-snippet-framing",
            "description": "Mokinys paai\u0161kina fragmento r\u0117minim\u0105",
            "pattern_refs": ["p-snippet-framing"],
            "is_mandatory": False,
        },
    ],
    "pass_conditions": {
        "trickster_wins": "Mokinys neper\u017ei\u016br\u0117jo viso straipsnio",
        "partial": "Mokinys perskait\u0117 straipsn\u012f ir aptiko dal\u012f manipuliacijos",
        "trickster_loses": "Mokinys identifikavo antra\u0161t\u0117s-straipsnio prie\u0161taravim\u0105",
    },
}

_CLICKBAIT_AI_CONFIG = {
    "model_preference": "standard",
    "prompt_directory": _TASK_ID,
    "persona_mode": _PERSONA_MODE,
    "has_static_fallback": True,
    "context_requirements": "session_only",
}

_CLICKBAIT_PHASES = [
    {
        "id": "evaluate",
        "title": "Straipsnio analiz\u0117",
        "visible_blocks": [],
        "is_ai_phase": True,
        "interaction": {
            "type": "freeform",
            "trickster_opening": "Na \u2014 perskaitei vis\u0105 straipsn\u012f. Tai pasakyk.",
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
    """Extracts the 'evaluate' AI phase from the clickbait-trap cartridge."""
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


def _make_clickbait_cartridge(make_cartridge) -> TaskCartridge:
    """Builds a clickbait-trap cartridge matching the real task structure."""
    return make_cartridge(
        task_id=_TASK_ID,
        task_type="hybrid",
        is_clean=False,
        initial_phase="evaluate",
        phases=_CLICKBAIT_PHASES,
        evaluation=_CLICKBAIT_EVAL,
        ai_config=_CLICKBAIT_AI_CONFIG,
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
    """Temp directory with base prompts + real clickbait-trap task prompt."""
    setup_base_prompts(tmp_path)
    # Copy real task prompt into temp tree
    real_path = PROJECT_ROOT / "prompts" / "tasks" / _TASK_ID / "trickster_base.md"
    real_content = real_path.read_text(encoding="utf-8")
    task_dir = tmp_path / "tasks" / _TASK_ID
    write_prompt_file(task_dir / "trickster_base.md", real_content)
    # Write presenting mode file
    real_mode = PROJECT_ROOT / "prompts" / "trickster" / "persona_presenting_base.md"
    write_prompt_file(
        tmp_path / "trickster" / "persona_presenting_base.md",
        real_mode.read_text(encoding="utf-8"),
    )
    return tmp_path


@pytest.fixture
def context_manager(prompts_dir):
    """Real ContextManager with temp prompts including clickbait-trap override."""
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


class TestClickbaitTrapPromptLoading:
    """PromptLoader correctly loads clickbait-trap task prompt."""

    def test_task_override_not_none(self, loader: PromptLoader) -> None:
        """Loads clickbait-trap prompt as non-None task_override."""
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


class TestClickbaitTrapLithuanianChars:
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


class TestClickbaitTrapContextAssembly:
    """Assembled system prompt includes task prompt AND structured eval data."""

    def test_task_override_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Task prompt content appears in assembled system prompt."""
        session = make_session()
        cartridge = _make_clickbait_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Distinctive phrase from the clickbait-trap prompt
        assert "Pasp\u0119sti sp\u0105stai" in result.system_prompt

    def test_structured_eval_data_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Cartridge evaluation data (patterns, checklist) in system prompt."""
        session = make_session()
        cartridge = _make_clickbait_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Pattern techniques from _CLICKBAIT_EVAL should appear in layer 5
        assert "emotional_framing" in result.system_prompt
        assert "headline_manipulation" in result.system_prompt
        # Mandatory checklist marker
        assert "[PRIVALOMA]" in result.system_prompt

    def test_mode_behaviour_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Presenting mode behaviour content appears in system prompt."""
        session = make_session()
        cartridge = _make_clickbait_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Distinctive phrase from persona_presenting_base.md
        assert "Pristatančiojo režimas" in result.system_prompt


# ---------------------------------------------------------------------------
# MockProvider Scenario Tests
# ---------------------------------------------------------------------------


class TestClickbaitTrapScenarios:
    """End-to-end scenario tests with MockProvider for three student paths."""

    @pytest.mark.asyncio
    async def test_immediate_recognition(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student identifies headline contradiction -> 'understood' -> reveal_win."""
        engine = make_engine(
            responses=[
                "Tu teisingai pasteb\u0117jai \u2014 antra\u0161t\u0117 \u017ead\u0117jo vien\u0105, straipsnis sako kit\u0105.",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "understood"}),
            ],
        )
        session = make_session()
        cartridge = _make_clickbait_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        # Prefill to meet min_exchanges=2
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "Antra\u0161t\u0117 sako nulin\u0117s emisijos, bet straipsnis kalba apie angl\u012f.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_success"
        assert result.done_data["next_phase"] == "reveal_win"

    @pytest.mark.asyncio
    async def test_partial_understanding(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student sees 'something off' but can't articulate -> 'partial' -> reveal_partial."""
        engine = make_engine(
            responses=[
                "Tu matai, kad ka\u017ekas negerai, bet negali tiksliai pasakyti.",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "partial"}),
            ],
        )
        session = make_session()
        cartridge = _make_clickbait_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "Ka\u017ekas \u010dia neatitinka, bet ne\u017einau k\u0105 tiksliai pasakyti.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_partial"
        assert result.done_data["next_phase"] == "reveal_partial"

    @pytest.mark.asyncio
    async def test_completely_fooled(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student accepts headline claims at face value -> max_reached -> reveal_timeout."""
        engine = make_engine(
            responses=[
                "Tu vis dar tiki antra\u0161te. Gerai, lai taip b\u016bna.",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "max_reached"}),
            ],
        )
        session = make_session()
        cartridge = _make_clickbait_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "Tai puiku, nulin\u0117s emisijos baterija, ateitis jau \u010dia.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_max_exchanges"
        assert result.done_data["next_phase"] == "reveal_timeout"

    @pytest.mark.asyncio
    async def test_auto_max_exchanges_ceiling(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """At max_exchanges=6 with no tool call, on_max_exchanges fires automatically."""
        provider = MockProvider(
            responses=["Paskutinis atsakymas be signalo."],
        )
        engine = TricksterEngine(provider, context_manager)
        session = make_session()
        cartridge = _make_clickbait_cartridge(make_cartridge)
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


class TestClickbaitTrapPromptContent:
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
        assert "p-emotional-lang" in prompts.task_override
        assert "p-urgency" in prompts.task_override
        assert "p-headline-contradiction" in prompts.task_override
        assert "p-snippet-framing" in prompts.task_override

    def test_references_mandatory_checklist_id(self, loader: PromptLoader) -> None:
        """Prompt references the mandatory checklist item ID."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "cl-headline-contradiction" in prompts.task_override

    def test_role_consistent_with_mode(self, loader: PromptLoader) -> None:
        """Prompt defines role consistent with presenting mode."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "pristatantysis" in prompts.task_override

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
