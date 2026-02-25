"""Task cartridge data models — the shared vocabulary of the Makaronas platform.

Defines the type system for task content: base type aliases, typed presentation
blocks (the open-type pattern), interaction types (the phase-level open-type
pattern), phase state machine models, evaluation contract, AI configuration,
reveal structure, safety config, and the top-level TaskCartridge model.

Includes routing logic that lets unknown block and interaction types pass
through instead of failing validation, and taxonomy-aware validators that
warn on unknown values via Pydantic's validation context.

This is a Tier 1 leaf module: imports only pydantic and stdlib.
No disk I/O, no framework imports. Everything else imports from here.

Usage:
    from backend.tasks.schemas import (
        TextBlock, ImageBlock, PresentationBlock, GenericBlock,
        TaskType, TaskStatus, ModelPreference, PersonaMode,
        ButtonInteraction, FreeformInteraction, InvestigationInteraction,
        InteractionConfig, GenericInteraction, AiTransitions, Phase,
        EmbeddedPattern, ChecklistItem, PassConditions, EvaluationContract,
        AiConfig, RevealContent, SafetyConfig, TaskCartridge,
        TaxonomyWarning, ContextRequirements,
    )
"""

from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, BeforeValidator, model_validator


# ---------------------------------------------------------------------------
# Base type aliases
# ---------------------------------------------------------------------------

TaskType = Literal["ai_driven", "static", "hybrid"]
"""Whether the task uses AI dialogue, static branching, or a mix."""

TaskStatus = Literal["active", "deprecated", "draft"]
"""Lifecycle status — active tasks are visible to teachers and students."""

ModelPreference = Literal["fast", "standard", "complex"]
"""Capability tier for AI model selection (Layer 1 of the three-layer abstraction)."""

PersonaMode = Literal["presenting", "chat_participant", "narrator", "commenter"]
"""How the Trickster presents itself within a task phase."""

Difficulty = Annotated[int, Field(ge=1, le=5)]
"""Task difficulty level, 1 (easiest) to 5 (hardest)."""


# ---------------------------------------------------------------------------
# Known presentation block types
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    """Text content — headlines, articles, snippets, captions."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["text"] = "text"
    text: str
    style: str | None = None


class ImageBlock(BaseModel):
    """Image content — misleading graphs, decontextualised photos.

    Accessibility: alt_text is required (Framework Principle 14).
    audio_description preserves deceptive framing for blind students
    in manipulation tasks.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["image"] = "image"
    src: str
    alt_text: str
    audio_description: str | None = None
    caption: str | None = None


class AudioBlock(BaseModel):
    """Audio content — voice notes, tone-of-voice manipulation, podcast clips.

    Accessibility: transcript is required (Framework Principle 14).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["audio"] = "audio"
    src: str
    transcript: str
    duration_seconds: int | None = None


class VideoTranscriptBlock(BaseModel):
    """Video transcript — full text of a video with optional source context."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["video_transcript"] = "video_transcript"
    transcript: str
    source_description: str | None = None


class MemeBlock(BaseModel):
    """Meme content — image with overlay text.

    Accessibility: alt_text is required (Framework Principle 14).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["meme"] = "meme"
    image_src: str
    top_text: str | None = None
    bottom_text: str | None = None
    alt_text: str
    audio_description: str | None = None


class ChatMessageBlock(BaseModel):
    """Chat message — one message in a group chat or DM thread."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["chat_message"] = "chat_message"
    username: str
    text: str
    timestamp: str | None = None
    is_highlighted: bool = False


class SocialPostBlock(BaseModel):
    """Social media post — evocative, not branded (Framework Principle 5)."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["social_post"] = "social_post"
    author: str
    text: str
    engagement: dict[str, Any] | None = None
    cited_source: str | None = None
    platform_hint: str | None = None


class SearchResultBlock(BaseModel):
    """Search result node in an investigation tree.

    Models a single search result. The tree structure emerges from
    child_queries linking results to further queries. Navigation logic
    belongs to Phase 1b's InvestigationInteraction.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["search_result"] = "search_result"
    query: str
    title: str
    snippet: str
    url: str | None = None
    is_key_finding: bool = False
    is_dead_end: bool = False
    child_queries: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Generic fallback block
# ---------------------------------------------------------------------------


class GenericBlock(BaseModel):
    """Fallback for unknown block types — preserves type and data without validation.

    When a cartridge contains a block type not in the known set, the routing
    function collects all fields except id and type into the data dict. This
    keeps the authoring format flat while giving GenericBlock a consistent shape.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Block type registry and routing
# ---------------------------------------------------------------------------

KNOWN_BLOCK_TYPES: dict[str, type[BaseModel]] = {
    "text": TextBlock,
    "image": ImageBlock,
    "audio": AudioBlock,
    "video_transcript": VideoTranscriptBlock,
    "meme": MemeBlock,
    "chat_message": ChatMessageBlock,
    "social_post": SocialPostBlock,
    "search_result": SearchResultBlock,
}
"""Maps type strings to their Pydantic model classes. Extend this to add new known types."""


def _route_presentation_block(data: Any) -> Any:
    """Routes block data to the correct model based on the type string.

    Handles two input shapes:
    1. Raw dict from JSON (authoring format): {"id": "...", "type": "text", "text": "..."}
    2. Already-constructed model instance from Python code

    For unknown types, collects all fields except id and type into a data dict
    before constructing GenericBlock. If the dict already has a data field and
    no other extra keys, treats it as pre-serialized GenericBlock format.
    """
    if isinstance(data, BaseModel):
        return data

    if not isinstance(data, dict):
        raise ValueError("PresentationBlock must be a dict or model instance")

    block_type = data.get("type")
    if block_type is None:
        raise ValueError("PresentationBlock requires a 'type' field")

    if data.get("id") is None:
        raise ValueError("PresentationBlock requires an 'id' field")

    model_cls = KNOWN_BLOCK_TYPES.get(block_type)
    if model_cls is not None:
        return model_cls.model_validate(data)

    # Unknown type → GenericBlock
    # Detect whether this is already in GenericBlock serialized form
    # (has 'data' dict and no other extra keys) vs flat authoring form
    known_keys = {"id", "type", "data"}
    extra_keys = set(data.keys()) - known_keys
    if not extra_keys and isinstance(data.get("data"), dict):
        return GenericBlock.model_validate(data)

    # Flat authoring format — collect extra fields into data
    extra = {k: v for k, v in data.items() if k not in ("id", "type")}
    return GenericBlock.model_validate({
        "id": data["id"],
        "type": block_type,
        "data": extra,
    })


PresentationBlock = Annotated[
    Union[
        TextBlock,
        ImageBlock,
        AudioBlock,
        VideoTranscriptBlock,
        MemeBlock,
        ChatMessageBlock,
        SocialPostBlock,
        SearchResultBlock,
        GenericBlock,
    ],
    BeforeValidator(_route_presentation_block),
]
"""Open-type union for presentation blocks.

Known type strings route to their specific model with full validation.
Unknown type strings route to GenericBlock — no ValidationError.
Use this type annotation in any model that contains presentation blocks.
"""


# ---------------------------------------------------------------------------
# Known interaction types
# ---------------------------------------------------------------------------


class ButtonChoice(BaseModel):
    """A single button option within a ButtonInteraction.

    Carries the display label, the target phase to transition to, and an
    optional context_label recorded to session history for AI continuity
    in hybrid tasks.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    target_phase: str
    context_label: str | None = None


class ButtonInteraction(BaseModel):
    """Button interaction — the student picks from a set of choices.

    Each choice carries a label, a target phase, and an optional context
    label for hybrid tasks where AI phases need to know what was clicked.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["button"] = "button"
    choices: list[ButtonChoice] = Field(default_factory=list)


class FreeformInteraction(BaseModel):
    """Freeform AI dialogue — multi-turn conversation with the Trickster.

    Exchange bounds control the conversation length. An exchange is one
    student turn + one Trickster turn. min_exchanges prevents premature
    evaluation; max_exchanges triggers the on_max_exchanges transition.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["freeform"] = "freeform"
    trickster_opening: str
    min_exchanges: int = Field(ge=1)
    max_exchanges: int = Field(ge=1)

    @model_validator(mode="before")
    @classmethod
    def check_exchange_bounds(cls, values: Any) -> Any:
        """Ensures min_exchanges <= max_exchanges."""
        if isinstance(values, dict):
            min_ex = values.get("min_exchanges")
            max_ex = values.get("max_exchanges")
            if min_ex is not None and max_ex is not None and min_ex > max_ex:
                raise ValueError(
                    f"min_exchanges ({min_ex}) must not exceed "
                    f"max_exchanges ({max_ex})"
                )
        return values


class InvestigationInteraction(BaseModel):
    """Investigation interaction — search tree navigation.

    Governs navigation rules for the investigation tree. The tree data
    lives in SearchResultBlock instances; this config controls which
    queries start available, where to go when done, and the minimum
    key findings threshold.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["investigation"] = "investigation"
    starting_queries: list[str] = Field(min_length=1)
    submit_target: str
    min_key_findings: int = 0


# ---------------------------------------------------------------------------
# Generic fallback interaction
# ---------------------------------------------------------------------------


class GenericInteraction(BaseModel):
    """Fallback for unknown interaction types — preserves type and config.

    When a cartridge contains an interaction type not in the known set,
    the routing function collects all fields except type into the config
    dict. This keeps the authoring format flat while giving
    GenericInteraction a consistent shape.
    """

    model_config = ConfigDict(frozen=True)

    type: str
    config: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Interaction type registry and routing
# ---------------------------------------------------------------------------

KNOWN_INTERACTION_TYPES: dict[str, type[BaseModel]] = {
    "button": ButtonInteraction,
    "freeform": FreeformInteraction,
    "investigation": InvestigationInteraction,
}
"""Maps interaction type strings to their Pydantic model classes."""


def _route_interaction(data: Any) -> Any:
    """Routes interaction data to the correct model based on the type string.

    Handles two input shapes:
    1. Raw dict from JSON (authoring format): {"type": "button", "choices": [...]}
    2. Already-constructed model instance from Python code

    For unknown types, collects all fields except type into a config dict
    before constructing GenericInteraction. If the dict already has a config
    field and no other extra keys, treats it as pre-serialized format.
    """
    if isinstance(data, BaseModel):
        return data

    if not isinstance(data, dict):
        raise ValueError("InteractionConfig must be a dict or model instance")

    interaction_type = data.get("type")
    if interaction_type is None:
        raise ValueError("InteractionConfig requires a 'type' field")

    model_cls = KNOWN_INTERACTION_TYPES.get(interaction_type)
    if model_cls is not None:
        return model_cls.model_validate(data)

    # Unknown type → GenericInteraction
    # Detect whether this is already in GenericInteraction serialized form
    # (has 'config' dict and no other extra keys) vs flat authoring form
    known_keys = {"type", "config"}
    extra_keys = set(data.keys()) - known_keys
    if not extra_keys and isinstance(data.get("config"), dict):
        return GenericInteraction.model_validate(data)

    # Flat authoring format — collect extra fields into config
    extra = {k: v for k, v in data.items() if k != "type"}
    return GenericInteraction.model_validate({
        "type": interaction_type,
        "config": extra,
    })


InteractionConfig = Annotated[
    Union[
        ButtonInteraction,
        FreeformInteraction,
        InvestigationInteraction,
        GenericInteraction,
    ],
    BeforeValidator(_route_interaction),
]
"""Open-type union for interaction configs.

Known type strings route to their specific model with full validation.
Unknown type strings route to GenericInteraction — no ValidationError.
Use this type annotation in any model that contains an interaction config.
"""


# ---------------------------------------------------------------------------
# AI transitions and evaluation outcomes
# ---------------------------------------------------------------------------

EvaluationOutcome = Literal["trickster_wins", "partial", "trickster_loses"]
"""Named outcomes for terminal phases — what the evaluation concluded."""


class AiTransitions(BaseModel):
    """Maps Trickster engine signals to target phase IDs.

    All three fields are required. If a task author doesn't distinguish
    partial from max_exchanges, they map both to the same target phase.
    """

    model_config = ConfigDict(frozen=True)

    on_success: str
    on_max_exchanges: str
    on_partial: str


# ---------------------------------------------------------------------------
# Phase model — the state machine node
# ---------------------------------------------------------------------------


class Phase(BaseModel):
    """A single phase in the task state machine.

    Each phase defines what the student sees (visible_blocks), how they
    interact (interaction), where the flow goes (button targets or
    ai_transitions), and whether this is a terminal evaluation point.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: str
    title: str

    # Content
    visible_blocks: list[str] = Field(default_factory=list)
    trickster_content: str | None = None

    # Interaction
    is_ai_phase: bool = False
    interaction: InteractionConfig | None = None
    ai_transitions: AiTransitions | None = None

    # Terminal state
    is_terminal: bool = False
    evaluation_outcome: EvaluationOutcome | None = None


# ---------------------------------------------------------------------------
# Taxonomy warning
# ---------------------------------------------------------------------------


class TaxonomyWarning(UserWarning):
    """Emitted when a taxonomy value (trigger, technique, medium) is not in the known set.

    Catchable via ``warnings.catch_warnings(record=True)`` in tests and the loader.
    Not a validation error — unknown values are accepted, just flagged.
    """


# ---------------------------------------------------------------------------
# Taxonomy validation helper
# ---------------------------------------------------------------------------


def _warn_if_unknown_taxonomy(
    value: str, category: str, info: Any,
) -> None:
    """Emits a TaxonomyWarning if *value* is not in the taxonomy for *category*.

    Args:
        value: The string value to check (e.g. a trigger name).
        category: The taxonomy category key (e.g. "triggers", "techniques", "mediums").
        info: The Pydantic ``ValidationInfo`` object — carries context from ``model_validate()``.

    Skips silently when ``info.context`` is ``None`` or has no ``taxonomy`` key
    (unit tests constructing models directly).
    """
    ctx = getattr(info, "context", None)
    if ctx is None:
        return
    taxonomy = ctx.get("taxonomy")
    if taxonomy is None:
        return
    known = taxonomy.get(category)
    if known is None:
        return
    if value not in known:
        warnings.warn(
            f"Unknown {category} value '{value}' — not in taxonomy. "
            f"Known values: {sorted(known)}",
            TaxonomyWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Context requirements type alias
# ---------------------------------------------------------------------------

ContextRequirements = Literal["session_only", "learning_profile", "full_history"]
"""How much student context the AI needs for this task."""


# ---------------------------------------------------------------------------
# Evaluation contract models
# ---------------------------------------------------------------------------


class EmbeddedPattern(BaseModel):
    """A single manipulation pattern baked into the task content.

    Represents one technique the author embedded for the student to find.
    The ``technique`` field references the technique taxonomy — validated at
    the TaskCartridge level via context injection, not here.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    description: str
    technique: str
    real_world_connection: str


class ChecklistItem(BaseModel):
    """What the student should demonstrate to earn credit.

    Each item describes a learning outcome and links it to the patterns
    the student should have identified. Mandatory items must be met for
    a ``trickster_loses`` outcome.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    description: str
    pattern_refs: list[str] = Field(default_factory=list)
    is_mandatory: bool = False


class PassConditions(BaseModel):
    """Textual descriptions of each evaluation outcome for this task.

    Field names mirror the EvaluationOutcome literal values so V6's
    evaluation engine can look up descriptions by outcome name.
    """

    model_config = ConfigDict(frozen=True)

    trickster_wins: str
    partial: str
    trickster_loses: str


class EvaluationContract(BaseModel):
    """The rubric contract between the cartridge author and V6's scoring engine.

    Declares *what* to check — patterns embedded, student checklist, and
    outcome descriptions. V6 decides *how* to check it.
    """

    model_config = ConfigDict(frozen=True)

    patterns_embedded: list[EmbeddedPattern] = Field(default_factory=list)
    checklist: list[ChecklistItem] = Field(default_factory=list)
    pass_conditions: PassConditions


# ---------------------------------------------------------------------------
# AI configuration
# ---------------------------------------------------------------------------


class AiConfig(BaseModel):
    """How the task connects to the AI layer (V3/V5).

    Specifies the capability tier, prompt directory, persona mode, fallback
    availability, and how much student context the AI needs.
    """

    model_config = ConfigDict(frozen=True)

    model_preference: ModelPreference
    prompt_directory: str
    persona_mode: PersonaMode
    has_static_fallback: bool
    context_requirements: ContextRequirements


# ---------------------------------------------------------------------------
# Reveal content
# ---------------------------------------------------------------------------


class RevealContent(BaseModel):
    """Post-task reveal — the lesson behind the trick.

    Pre-authored by the content author. The AI personalizes *around*
    this content but the key lesson is always this exact text (Framework
    Principle 19: facts from data, personalization from AI).
    """

    model_config = ConfigDict(frozen=True)

    key_lesson: str
    additional_resources: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Safety configuration
# ---------------------------------------------------------------------------


class SafetyConfig(BaseModel):
    """Safety guardrails for this task (Framework Principle 12).

    Content boundaries define forbidden areas. Intensity ceiling caps
    adversarial pressure. cold_start_safe indicates whether this task
    is appropriate as a student's first encounter with the platform.
    """

    model_config = ConfigDict(frozen=True)

    content_boundaries: list[str] = Field(default_factory=list)
    intensity_ceiling: Annotated[int, Field(ge=1, le=5)]
    cold_start_safe: bool


# ---------------------------------------------------------------------------
# Top-level cartridge model
# ---------------------------------------------------------------------------


class TaskCartridge(BaseModel):
    """The root model for a task cartridge — the complete data contract.

    Every downstream consumer — loader, registry, teacher library, student
    endpoint, Trickster engine, evaluation engine, reveal, Composer — imports
    and consumes this model. Getting the field contracts right here prevents
    ripple effects across the entire platform.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    task_id: str
    task_type: TaskType
    title: str
    description: str
    version: str

    # Classification (taxonomy-aware — validated via context injection)
    trigger: str
    technique: str
    medium: str

    # Pedagogical metadata
    learning_objectives: list[str] = Field(min_length=1)
    difficulty: Difficulty
    time_minutes: int = Field(gt=0)
    is_evergreen: bool
    is_clean: bool

    # Tagging and filtering
    tags: list[str] = Field(default_factory=list)
    status: TaskStatus = "active"
    prerequisites: list[str] = Field(default_factory=list)

    # Localisation
    language: str = "lt"
    available_languages: list[str] = Field(default_factory=lambda: ["lt"])

    # Content
    presentation_blocks: list[PresentationBlock] = Field(default_factory=list)
    phases: list[Phase] = Field(default_factory=list)
    initial_phase: str

    # Evaluation
    evaluation: EvaluationContract

    # AI (optional — static tasks don't need it; required for ai_driven/hybrid is Phase 2b)
    ai_config: AiConfig | None = None

    # Reveal and safety
    reveal: RevealContent
    safety: SafetyConfig

    @model_validator(mode="wrap")
    @classmethod
    def _validate_cartridge(
        cls, values: Any, handler: Any, info: Any,
    ) -> TaskCartridge:
        """Cross-field and taxonomy validation for the top-level cartridge.

        Uses mode='wrap' so we get both the fully-constructed model (via handler)
        and the ValidationInfo (which carries taxonomy context from the loader).

        Checks:
        1. is_clean=True with non-empty patterns_embedded -> hard error
        2. is_clean=False with empty patterns_embedded -> warning
        3. Taxonomy validation for trigger, technique, medium
        """
        instance = handler(values)

        # is_clean cross-validation
        has_patterns = len(instance.evaluation.patterns_embedded) > 0
        if instance.is_clean and has_patterns:
            raise ValueError(
                "is_clean=True but evaluation.patterns_embedded is not empty — "
                "a clean task cannot have embedded manipulation patterns"
            )
        if not instance.is_clean and not has_patterns:
            warnings.warn(
                "is_clean=False but evaluation.patterns_embedded is empty — "
                "this may be a draft in progress",
                TaxonomyWarning,
                stacklevel=2,
            )

        # Taxonomy validation via context
        _warn_if_unknown_taxonomy(instance.trigger, "triggers", info)
        _warn_if_unknown_taxonomy(instance.technique, "techniques", info)
        _warn_if_unknown_taxonomy(instance.medium, "mediums", info)

        return instance
