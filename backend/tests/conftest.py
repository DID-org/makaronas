"""Shared AI test fixtures for all V3 test phases.

Factory-pattern fixtures that return callables accepting **overrides.
Every V3 phase imports from here — no reinventing test scaffolding.

Fixtures:
    mock_provider: Factory for MockProvider instances
    make_session: Factory for valid GameSession instances
    make_cartridge: Factory for valid AI-capable TaskCartridge instances
    mock_registry: Pre-populated TaskRegistry with one default cartridge
"""

from pathlib import Path
from uuid import uuid4

import pytest

from backend.ai.providers.mock import MockProvider
from backend.schemas import GameSession
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import TaskCartridge


# ---------------------------------------------------------------------------
# MockProvider factory
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider():
    """Returns a factory function for creating MockProvider instances."""

    def _make(**kwargs) -> MockProvider:
        return MockProvider(**kwargs)

    return _make


# ---------------------------------------------------------------------------
# GameSession factory
# ---------------------------------------------------------------------------


@pytest.fixture
def make_session():
    """Returns a factory function for creating valid GameSession instances.

    Defaults produce a minimal valid session with unique IDs.
    Override any field via kwargs.
    """

    def _make(**overrides) -> GameSession:
        defaults = {
            "session_id": f"session-{uuid4().hex[:8]}",
            "student_id": f"student-{uuid4().hex[:8]}",
            "school_id": "school-test-001",
            "language": "lt",
        }
        defaults.update(overrides)
        return GameSession(**defaults)

    return _make


# ---------------------------------------------------------------------------
# TaskCartridge factory
# ---------------------------------------------------------------------------


def _build_cartridge_data(**overrides) -> dict:
    """Builds a minimal valid AI-capable cartridge dict.

    Includes:
    - One static intro phase with ButtonInteraction -> AI phase
    - One AI phase with FreeformInteraction + AiTransitions
    - One terminal reveal phase
    - AiConfig, SafetyConfig, EvaluationContract
    """
    task_id = overrides.pop("task_id", "test-ai-task-001")
    task_type = overrides.pop("task_type", "hybrid")
    ai_config = overrides.pop("ai_config", {
        "model_preference": "standard",
        "prompt_directory": task_id,
        "persona_mode": "chat_participant",
        "has_static_fallback": False,
        "context_requirements": "session_only",
    })

    data: dict = {
        "task_id": task_id,
        "task_type": task_type,
        "title": "AI testo užduotis",
        "description": "Minimali AI užduotis testavimui",
        "version": "1.0",
        "trigger": "urgency",
        "technique": "headline_manipulation",
        "medium": "article",
        "learning_objectives": ["Atpažinti manipuliaciją"],
        "difficulty": 3,
        "time_minutes": 15,
        "is_evergreen": True,
        "is_clean": False,
        "initial_phase": "phase_intro",
        "phases": [
            {
                "id": "phase_intro",
                "title": "Įvadas",
                "is_ai_phase": False,
                "interaction": {
                    "type": "button",
                    "choices": [
                        {
                            "label": "Pradėti pokalbį",
                            "target_phase": "phase_ai",
                            "context_label": "Mokinys pasirinko pradėti pokalbį",
                        },
                    ],
                },
            },
            {
                "id": "phase_ai",
                "title": "Pokalbis su Triksteriu",
                "is_ai_phase": True,
                "interaction": {
                    "type": "freeform",
                    "trickster_opening": "Sveiki! Pažiūrėkime į šį straipsnį...",
                    "min_exchanges": 2,
                    "max_exchanges": 10,
                },
                "ai_transitions": {
                    "on_success": "phase_reveal_success",
                    "on_max_exchanges": "phase_reveal_timeout",
                    "on_partial": "phase_reveal_partial",
                },
            },
            {
                "id": "phase_reveal_success",
                "title": "Atskleidimas – laimėjai",
                "is_terminal": True,
                "evaluation_outcome": "trickster_loses",
            },
            {
                "id": "phase_reveal_timeout",
                "title": "Atskleidimas – laikas baigėsi",
                "is_terminal": True,
                "evaluation_outcome": "trickster_wins",
            },
            {
                "id": "phase_reveal_partial",
                "title": "Atskleidimas – iš dalies",
                "is_terminal": True,
                "evaluation_outcome": "partial",
            },
        ],
        "evaluation": {
            "patterns_embedded": [
                {
                    "id": "p1",
                    "description": "Antraštė neatitinka turinio",
                    "technique": "headline_manipulation",
                    "real_world_connection": "Dažnai pastebima naujienų portaluose",
                },
            ],
            "checklist": [
                {
                    "id": "c1",
                    "description": "Atpažino antraštės ir turinio neatitikimą",
                    "pattern_refs": ["p1"],
                    "is_mandatory": True,
                },
            ],
            "pass_conditions": {
                "trickster_wins": "Mokinys nesugebėjo atpažinti manipuliacijos",
                "partial": "Mokinys pastebėjo kai kuriuos ženklus",
                "trickster_loses": "Mokinys aiškiai įvardijo manipuliacijos techniką",
            },
        },
        "reveal": {
            "key_lesson": "Antraštė buvo sukurta skubos jausmui sukelti",
        },
        "safety": {
            "content_boundaries": ["self_harm"],
            "intensity_ceiling": 3,
            "cold_start_safe": True,
        },
    }

    if ai_config is not None:
        data["ai_config"] = ai_config

    data.update(overrides)
    return data


@pytest.fixture
def make_cartridge():
    """Returns a factory function for creating valid AI-capable TaskCartridge instances.

    Defaults produce a hybrid cartridge with:
    - Static intro phase -> AI freeform phase -> three terminal reveal phases
    - AiConfig with model_preference="standard"
    - SafetyConfig with content_boundaries=["self_harm"], intensity_ceiling=3
    - EvaluationContract with one pattern, one checklist item, PassConditions
    Override any top-level field via kwargs.
    """

    def _make(**overrides) -> TaskCartridge:
        data = _build_cartridge_data(**overrides)
        return TaskCartridge.model_validate(data)

    return _make


# ---------------------------------------------------------------------------
# Mock TaskRegistry
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_registry(make_cartridge):
    """Returns a TaskRegistry pre-populated with one default cartridge.

    Uses the manual population pattern established in V2 API tests.
    """
    cartridge = make_cartridge()
    registry = TaskRegistry(Path("/tmp"), Path("/tmp"))

    registry._by_id[cartridge.task_id] = cartridge
    registry._by_status.setdefault(cartridge.status, set()).add(cartridge.task_id)
    registry._by_trigger[cartridge.trigger].add(cartridge.task_id)
    registry._by_technique[cartridge.technique].add(cartridge.task_id)
    registry._by_medium[cartridge.medium].add(cartridge.task_id)
    for tag in cartridge.tags:
        registry._by_tag[tag].add(cartridge.task_id)

    return registry
