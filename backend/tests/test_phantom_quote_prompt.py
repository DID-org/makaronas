"""Tests for Phase 8a: Phantom-Quote task prompt.

Verifies that the phantom-quote trickster prompt:
1. Loads correctly through PromptLoader
2. Preserves Lithuanian diacritical characters
3. Assembles correctly in ContextManager alongside cartridge evaluation data
4. Produces correct transition signals via MockProvider scenario tests
"""

from __future__ import annotations

from pathlib import Path

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

_TASK_ID = "task-phantom-quote-001"
_PERSONA_MODE = "presenting"
_MIN_LENGTH = 100

# Phantom-quote evaluation data matching the real cartridge
_PHANTOM_EVAL = {
    "patterns_embedded": [
        {
            "id": "p-fabricated-quote",
            "description": "Tiesiogin\u0117 citata, priskirta tyr\u0117jai, niekada nebuvo pasakyta",
            "technique": "phantom_quote",
            "real_world_connection": "Kabučių buvimas negarantuoja autiškumo",
        },
        {
            "id": "p-quotation-authority",
            "description": "Kabu\u010di\u0173 ir institucijos vardo derinys sukuria klaidinga \u012frodymo \u012fsp\u016bd\u012f",
            "technique": "false_authority",
            "real_world_connection": "Kabut\u0117s r\u0117i\u0161kia tiksliai tai buvo pasakyta",
        },
        {
            "id": "p-topic-proximity",
            "description": "\u0160altinio straipsnis kalba apie pana\u0161ia tema",
            "technique": "phantom_quote",
            "real_world_connection": "Fabrikuotos citatos remiasi tikrais tyrimais",
        },
    ],
    "checklist": [
        {
            "id": "cl-fabricated-quote-found",
            "description": "Mokinys identifikuoja, kad citata neegzistuoja \u0161altinyje",
            "pattern_refs": ["p-fabricated-quote"],
            "is_mandatory": True,
        },
        {
            "id": "cl-quotation-authority-explained",
            "description": "Mokinys paai\u0161kina kabu\u010di\u0173 autorite\u0105",
            "pattern_refs": ["p-quotation-authority"],
            "is_mandatory": False,
        },
    ],
    "pass_conditions": {
        "trickster_wins": "Mokinys kartoja tinklaraš\u010dio teigin\u012f",
        "partial": "Mokinys pasteb\u0117jo neatitikimų, bet negali tiksliai \u012fvardyti",
        "trickster_loses": "Mokinys identifikavo citat\u0105 kaip vi\u0161i\u0161kai i\u0161galvot\u0105",
    },
}

_PHANTOM_AI_CONFIG = {
    "model_preference": "standard",
    "prompt_directory": _TASK_ID,
    "persona_mode": _PERSONA_MODE,
    "has_static_fallback": True,
    "context_requirements": "session_only",
}

_PHANTOM_PHASES = [
    {
        "id": "evaluate",
        "title": "Tekst\u0173 analiz\u0117",
        "visible_blocks": [],
        "is_ai_phase": True,
        "interaction": {
            "type": "freeform",
            "trickster_opening": "Abu tekstai prie\u0161 tave. Kur \u010dia manipuliacija?",
            "min_exchanges": 3,
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
    """Extracts the 'evaluate' AI phase from the phantom-quote cartridge."""
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
            Exchange(role="student", content=f"Mokinio žinut\u0117 {i}")
        )
        session.exchanges.append(
            Exchange(role="trickster", content=f"Triksterio atsakymas {i}")
        )


def _make_phantom_cartridge(make_cartridge) -> TaskCartridge:
    """Builds a phantom-quote cartridge matching the real task structure."""
    return make_cartridge(
        task_id=_TASK_ID,
        task_type="ai_driven",
        is_clean=False,
        initial_phase="evaluate",
        phases=_PHANTOM_PHASES,
        evaluation=_PHANTOM_EVAL,
        ai_config=_PHANTOM_AI_CONFIG,
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
    """Temp directory with base prompts + real phantom-quote task prompt."""
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
    """Real ContextManager with temp prompts including phantom-quote override."""
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


class TestPhantomQuotePromptLoading:
    """PromptLoader correctly loads phantom-quote task prompt."""

    def test_task_override_not_none(self, loader: PromptLoader) -> None:
        """Loads phantom-quote prompt as non-None task_override."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None

    def test_task_override_meaningful_length(self, loader: PromptLoader) -> None:
        """Task override has meaningful content (>{} chars)."""
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


class TestPhantomQuoteLithuanianChars:
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


class TestPhantomQuoteContextAssembly:
    """Assembled system prompt includes task prompt AND structured eval data."""

    def test_task_override_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Task prompt content appears in assembled system prompt."""
        session = make_session()
        cartridge = _make_phantom_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=3,
        )

        # Distinctive phrase from the prompt file
        assert "Vaiduokli\u0161ka citata" in result.system_prompt

    def test_structured_eval_data_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Cartridge evaluation data (patterns, checklist) in system prompt."""
        session = make_session()
        cartridge = _make_phantom_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=3,
        )

        # Pattern descriptions from _PHANTOM_EVAL should appear in layer 5
        assert "phantom_quote" in result.system_prompt
        assert "false_authority" in result.system_prompt
        # Mandatory checklist marker
        assert "[PRIVALOMA]" in result.system_prompt

    def test_both_task_prompt_and_eval_data_present(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Both task-specific prompt AND structured eval data coexist."""
        session = make_session()
        cartridge = _make_phantom_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=3,
        )

        # Task prompt content (layer 4)
        assert "pristatantysis" in result.system_prompt
        # Evaluation data content (layer 5)
        assert "Vertinimo kriterijai" in result.system_prompt


# ---------------------------------------------------------------------------
# MockProvider Scenario Tests
# ---------------------------------------------------------------------------


class TestPhantomQuoteScenarios:
    """End-to-end scenario tests with MockProvider for three student paths."""

    @pytest.mark.asyncio
    async def test_immediate_recognition(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student identifies fabrication -> 'understood' -> on_success -> reveal_win."""
        engine = make_engine(
            responses=[
                "Kabu\u010d\u0117s buvo mano geriausias triukas \u2014 ir tu j\u012f permat\u0117i.",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "understood"}),
            ],
        )
        session = make_session()
        cartridge = _make_phantom_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        # Prefill to meet min_exchanges=3
        _prefill_exchanges(session, 2)

        result = await engine.respond(
            session, cartridge, phase,
            "Ta citata apie 4 valandas yra visi\u0161kai i\u0161galvota.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_success"
        assert result.done_data["next_phase"] == "reveal_win"

    @pytest.mark.asyncio
    async def test_partial_understanding(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student says 'out of context' not 'fabricated' -> 'partial' -> reveal_partial."""
        engine = make_engine(
            responses=[
                "Tu matai, kad ka\u017ekas negerai. Bet 'i\u0161 konteksto' ir "
                "'niekada nepasakyta' \u2014 tai ne tas pats.",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "partial"}),
            ],
        )
        session = make_session()
        cartridge = _make_phantom_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 2)

        result = await engine.respond(
            session, cartridge, phase,
            "Citata paimta i\u0161 konteksto, ji i\u0161kraipyta.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_partial"
        assert result.done_data["next_phase"] == "reveal_partial"

    @pytest.mark.asyncio
    async def test_completely_fooled(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student accepts fabricated claim -> max_reached -> reveal_timeout."""
        engine = make_engine(
            responses=[
                "Tikrai? Tu man tiki? A\u0161 ra\u0161iau t\u0105 straipsn\u012f...",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "max_reached"}),
            ],
        )
        session = make_session()
        cartridge = _make_phantom_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 2)

        result = await engine.respond(
            session, cartridge, phase,
            "Institutas sako, kad 4 valandos yra gerai.",
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
        cartridge = _make_phantom_cartridge(make_cartridge)
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


class TestPhantomQuotePromptContent:
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
        assert "p-fabricated-quote" in prompts.task_override
        assert "p-quotation-authority" in prompts.task_override
        assert "p-topic-proximity" in prompts.task_override

    def test_references_checklist_id(self, loader: PromptLoader) -> None:
        """Prompt references the mandatory checklist item ID."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "cl-fabricated-quote-found" in prompts.task_override

    def test_emphasizes_fabrication_not_distortion(
        self, loader: PromptLoader,
    ) -> None:
        """Prompt distinguishes fabrication from distortion/out-of-context."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        content = prompts.task_override.lower()
        # Must contain "i\u0161galvota" (fabricated) concept
        assert "i\u0161galvot" in content
        # Must distinguish from "i\u0161 konteksto" (out of context)
        assert "i\u0161 konteksto" in content or "i\u0161kraipyt" in content

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
        # Should not contain generic persona setup phrases
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
