"""Tests for Phase 8d: Misleading-Frame task prompt.

Verifies that the misleading-frame trickster prompt:
1. Loads correctly through PromptLoader
2. Preserves Lithuanian diacritical characters
3. Assembles correctly in ContextManager alongside cartridge evaluation data
4. Multimodal context assembly includes images from cartridge
5. Produces correct transition signals via MockProvider scenario tests
"""

from __future__ import annotations

import base64
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

_TASK_ID = "task-misleading-frame-001"
_PERSONA_MODE = "presenting"
_MIN_LENGTH = 100

# Misleading-frame evaluation data matching the real cartridge
_FRAME_EVAL = {
    "patterns_embedded": [
        {
            "id": "p-visual-crop",
            "description": "Nuotraukos apkarpymas pa\u0161alina kontekst\u0105",
            "technique": "emotional_framing",
            "real_world_connection": "Naujien\u0173 portalai naudoja apkarpytas nuotraukas",
        },
        {
            "id": "p-caption-bias",
            "description": "Antrašt\u0117s tekstas sustiprina vizualin\u012f klaidinimą",
            "technique": "emotional_framing",
            "real_world_connection": "Nuotraukos antrašt\u0117 formuoja suvokimą",
        },
        {
            "id": "p-fear-framing",
            "description": "\u017demas kampas ir tamsus apšvietimas aktyvuoja baim\u0119s suvokimą",
            "technique": "emotional_framing",
            "real_world_connection": "Vizualiniai sprendimai formuoja emocijas",
        },
    ],
    "checklist": [
        {
            "id": "cl-crop-changes-narrative",
            "description": "Mokinys identifikuoja, kaip apkarpymas keičia pasakojimą",
            "pattern_refs": ["p-visual-crop"],
            "is_mandatory": True,
        },
        {
            "id": "cl-caption-image-cooperation",
            "description": "Mokinys atpažįsta antraštės ir vaizdo bendradarbiavimą",
            "pattern_refs": ["p-caption-bias"],
            "is_mandatory": False,
        },
    ],
    "pass_conditions": {
        "trickster_wins": "Mokinys nepažino vizualinio rėminimo",
        "partial": "Mokinys pastebėjo skirtumą, bet neartikuliavo mechanizmo",
        "trickster_loses": "Mokinys identifikavo rėminimą kaip sąmoningą pasirinkimą",
    },
}

_FRAME_AI_CONFIG = {
    "model_preference": "standard",
    "prompt_directory": _TASK_ID,
    "persona_mode": _PERSONA_MODE,
    "has_static_fallback": True,
    "context_requirements": "session_only",
}

_FRAME_PRESENTATION_BLOCKS = [
    {
        "id": "image-misleading",
        "type": "image",
        "src": "misleading.png",
        "alt_text": "Tamsi, grūsta nuotrauka",
    },
    {
        "id": "image-context",
        "type": "image",
        "src": "context.png",
        "alt_text": "Plati, šviesi nuotrauka",
    },
    {
        "id": "scene-description",
        "type": "text",
        "text": "Dvi nuotraukos to paties įvykio.",
    },
]

_FRAME_PHASES = [
    {
        "id": "evaluate",
        "title": "Nuotraukų analizė",
        "visible_blocks": [
            "image-misleading",
            "image-context",
            "scene-description",
        ],
        "is_ai_phase": True,
        "interaction": {
            "type": "freeform",
            "trickster_opening": "Dvi nuotraukos to paties \u012fvykio. Koki\u0105 istorij\u0105 pasakoja kiekviena?",
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
        "title": "Atskleidimas \u2014 iš dalies",
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
    """Extracts the 'evaluate' AI phase from the misleading-frame cartridge."""
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


def _make_frame_cartridge(make_cartridge) -> TaskCartridge:
    """Builds a misleading-frame cartridge matching the real task structure."""
    return make_cartridge(
        task_id=_TASK_ID,
        task_type="ai_driven",
        is_clean=False,
        initial_phase="evaluate",
        phases=_FRAME_PHASES,
        evaluation=_FRAME_EVAL,
        ai_config=_FRAME_AI_CONFIG,
        presentation_blocks=_FRAME_PRESENTATION_BLOCKS,
    )


# Minimal valid 1x1 PNG (67 bytes)
_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"  # signature
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
    b"\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
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
    """Temp directory with base prompts + real misleading-frame task prompt."""
    setup_base_prompts(tmp_path)
    # Copy real task prompt into temp tree
    real_path = PROJECT_ROOT / "prompts" / "tasks" / _TASK_ID / "trickster_base.md"
    real_content = real_path.read_text(encoding="utf-8")
    task_dir = tmp_path / "tasks" / _TASK_ID
    write_prompt_file(task_dir / "trickster_base.md", real_content)
    # Write a presenting mode file too
    real_mode = PROJECT_ROOT / "prompts" / "trickster" / "persona_presenting_base.md"
    write_prompt_file(
        tmp_path / "trickster" / "persona_presenting_base.md",
        real_mode.read_text(encoding="utf-8"),
    )
    return tmp_path


@pytest.fixture
def context_manager(prompts_dir):
    """Real ContextManager with temp prompts including misleading-frame override."""
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


class TestMisleadingFramePromptLoading:
    """PromptLoader correctly loads misleading-frame task prompt."""

    def test_task_override_not_none(self, loader: PromptLoader) -> None:
        """Loads misleading-frame prompt as non-None task_override."""
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


class TestMisleadingFrameLithuanianChars:
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


class TestMisleadingFrameContextAssembly:
    """Assembled system prompt includes task prompt AND structured eval data."""

    def test_task_override_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Task prompt content appears in assembled system prompt."""
        session = make_session()
        cartridge = _make_frame_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Distinctive phrase from the prompt file
        assert "Klaidinantis kadras" in result.system_prompt

    def test_structured_eval_data_in_system_prompt(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Cartridge evaluation data (patterns, checklist) in system prompt."""
        session = make_session()
        cartridge = _make_frame_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Pattern IDs and technique from _FRAME_EVAL
        assert "emotional_framing" in result.system_prompt
        assert "p-visual-crop" in result.system_prompt
        # Mandatory checklist marker
        assert "[PRIVALOMA]" in result.system_prompt

    def test_both_task_prompt_and_eval_data_present(
        self, context_manager, make_session, make_cartridge,
    ) -> None:
        """Both task-specific prompt AND structured eval data coexist."""
        session = make_session()
        cartridge = _make_frame_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Task prompt content (layer 4)
        assert "pristatantysis" in result.system_prompt
        # Evaluation data content (layer 5)
        assert "Vertinimo kriterijai" in result.system_prompt


# ---------------------------------------------------------------------------
# Multimodal Context Assembly Tests
# ---------------------------------------------------------------------------


class TestMisleadingFrameMultimodal:
    """Multimodal context assembly includes images from cartridge."""

    def test_images_included_in_messages(
        self, tmp_path, make_session, make_cartridge,
    ) -> None:
        """Assemble includes multimodal image content when content_dir is set."""
        # Set up content directory with dummy image files
        content_dir = tmp_path / "content"
        assets_dir = content_dir / "tasks" / _TASK_ID / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "misleading.png").write_bytes(_MINIMAL_PNG)
        (assets_dir / "context.png").write_bytes(_MINIMAL_PNG)

        # Set up prompts directory
        prompts_dir = tmp_path / "prompts"
        setup_base_prompts(prompts_dir)
        real_path = PROJECT_ROOT / "prompts" / "tasks" / _TASK_ID / "trickster_base.md"
        task_prompt_dir = prompts_dir / "tasks" / _TASK_ID
        write_prompt_file(
            task_prompt_dir / "trickster_base.md",
            real_path.read_text(encoding="utf-8"),
        )
        real_mode = PROJECT_ROOT / "prompts" / "trickster" / "persona_presenting_base.md"
        write_prompt_file(
            prompts_dir / "trickster" / "persona_presenting_base.md",
            real_mode.read_text(encoding="utf-8"),
        )

        loader = PromptLoader(prompts_dir)
        cm = ContextManager(loader, content_dir=content_dir)

        session = make_session(current_phase="evaluate")
        cartridge = _make_frame_cartridge(make_cartridge)

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Find the multimodal message (has list content with image parts)
        image_messages = [
            m for m in result.messages
            if isinstance(m.get("content"), list)
        ]
        assert len(image_messages) >= 1, "No multimodal message found in messages"

        # Count image content parts
        image_parts = []
        for msg in image_messages:
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "image":
                    image_parts.append(part)

        assert len(image_parts) == 2, (
            f"Expected 2 image parts, got {len(image_parts)}"
        )

    def test_images_contain_base64_data(
        self, tmp_path, make_session, make_cartridge,
    ) -> None:
        """Image parts contain valid base64-encoded data."""
        content_dir = tmp_path / "content"
        assets_dir = content_dir / "tasks" / _TASK_ID / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "misleading.png").write_bytes(_MINIMAL_PNG)
        (assets_dir / "context.png").write_bytes(_MINIMAL_PNG)

        prompts_dir = tmp_path / "prompts"
        setup_base_prompts(prompts_dir)
        real_path = PROJECT_ROOT / "prompts" / "tasks" / _TASK_ID / "trickster_base.md"
        task_prompt_dir = prompts_dir / "tasks" / _TASK_ID
        write_prompt_file(
            task_prompt_dir / "trickster_base.md",
            real_path.read_text(encoding="utf-8"),
        )
        real_mode = PROJECT_ROOT / "prompts" / "trickster" / "persona_presenting_base.md"
        write_prompt_file(
            prompts_dir / "trickster" / "persona_presenting_base.md",
            real_mode.read_text(encoding="utf-8"),
        )

        loader = PromptLoader(prompts_dir)
        cm = ContextManager(loader, content_dir=content_dir)

        session = make_session(current_phase="evaluate")
        cartridge = _make_frame_cartridge(make_cartridge)

        result = cm.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # Extract image parts
        for msg in result.messages:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "image":
                        assert "data" in part
                        assert "media_type" in part
                        assert part["media_type"] == "image/png"
                        # Verify it's valid base64
                        decoded = base64.b64decode(part["data"])
                        assert len(decoded) > 0

    def test_no_images_without_content_dir(
        self, make_session, make_cartridge, context_manager,
    ) -> None:
        """Without content_dir, no multimodal messages are produced."""
        session = make_session(current_phase="evaluate")
        cartridge = _make_frame_cartridge(make_cartridge)

        result = context_manager.assemble_trickster_call(
            session, cartridge, "gemini",
            exchange_count=1, min_exchanges=2,
        )

        # No messages should have list content (multimodal)
        image_messages = [
            m for m in result.messages
            if isinstance(m.get("content"), list)
        ]
        assert len(image_messages) == 0


# ---------------------------------------------------------------------------
# MockProvider Scenario Tests
# ---------------------------------------------------------------------------


class TestMisleadingFrameScenarios:
    """End-to-end scenario tests with MockProvider for three student paths."""

    @pytest.mark.asyncio
    async def test_immediate_recognition(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student identifies intentional framing -> 'understood' -> on_success -> reveal_win."""
        engine = make_engine(
            responses=[
                "Tu pamatei per kadrą. Nuotrauka nemeluoja — bet apkarpymas pasirenka tiesą.",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "understood"}),
            ],
        )
        session = make_session()
        cartridge = _make_frame_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        # Prefill to meet min_exchanges=2
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "Pirmoji nuotrauka apkarpyta — kažkas pasirinko siaurą kadrą, kad sukurtų chaoso įspūdį.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_success"
        assert result.done_data["next_phase"] == "reveal_win"

    @pytest.mark.asyncio
    async def test_partial_understanding(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student says 'photos are different' without mechanism -> 'partial' -> reveal_partial."""
        engine = make_engine(
            responses=[
                "Tu matai skirtumą, bet neartikuliuoji mechanizmo.",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "partial"}),
            ],
        )
        session = make_session()
        cartridge = _make_frame_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "Nuotraukos skirtingos — viena baisesnė, kita ramesnė.",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_partial"
        assert result.done_data["next_phase"] == "reveal_partial"

    @pytest.mark.asyncio
    async def test_completely_fooled(
        self, make_engine, make_session, make_cartridge,
    ) -> None:
        """Student accepts misleading photo -> max_reached -> reveal_timeout."""
        engine = make_engine(
            responses=[
                "Tu tiki ta nuotrauka? Kas pasirinko tą kampą?",
            ],
            tool_calls=[
                ToolCallEvent("transition_phase", {"signal": "max_reached"}),
            ],
        )
        session = make_session()
        cartridge = _make_frame_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        _prefill_exchanges(session, 1)

        result = await engine.respond(
            session, cartridge, phase,
            "Matau chaosą mieste, daug žmonių gatvėse.",
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
        cartridge = _make_frame_cartridge(make_cartridge)
        phase = _get_ai_phase(cartridge)
        # max_exchanges=6, prefill 5 -> this is message #6
        _prefill_exchanges(session, 5)

        result = await engine.respond(
            session, cartridge, phase, "Paskutinė žinutė",
        )
        await _consume_tokens(result)

        assert result.done_data is not None
        assert result.done_data["phase_transition"] == "on_max_exchanges"
        assert result.done_data["next_phase"] == "reveal_timeout"
        assert result.done_data["exchanges_count"] == 6


# ---------------------------------------------------------------------------
# Prompt Content Checks
# ---------------------------------------------------------------------------


class TestMisleadingFramePromptContent:
    """Verifies key elements in the prompt file content."""

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

    def test_references_pattern_ids(self, loader: PromptLoader) -> None:
        """Prompt references all three pattern IDs from the cartridge."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "p-visual-crop" in prompts.task_override
        assert "p-caption-bias" in prompts.task_override
        assert "p-fear-framing" in prompts.task_override

    def test_references_checklist_id(self, loader: PromptLoader) -> None:
        """Prompt references the mandatory checklist item ID."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "cl-crop-changes-narrative" in prompts.task_override

    def test_distinguishes_observation_from_understanding(
        self, loader: PromptLoader,
    ) -> None:
        """Prompt distinguishes visual observation from understanding intentionality."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        content = prompts.task_override.lower()
        # Must contain intentionality concept
        assert "pasirinkimas" in content or "pasirinkimą" in content or "pasirinko" in content
        # Must contain observation vs understanding distinction
        assert "pastebėjimas" in content or "skiriasi" in content

    def test_mentions_three_patterns(self, loader: PromptLoader) -> None:
        """Prompt acknowledges 3 patterns exist."""
        prompts = loader.load_trickster_prompts(
            "gemini", task_id=_TASK_ID, persona_mode=_PERSONA_MODE,
        )
        assert prompts.task_override is not None
        assert "3 šablon" in prompts.task_override or "Trys" in prompts.task_override

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
