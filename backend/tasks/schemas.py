"""Task cartridge data models — the shared vocabulary of the Makaronas platform.

Defines the type system for task content: base type aliases, typed presentation
blocks (the open-type pattern), and the routing logic that lets unknown block
types pass through instead of failing validation.

This is a Tier 1 leaf module: imports only pydantic and stdlib.
No disk I/O, no framework imports. Everything else imports from here.

Usage:
    from backend.tasks.schemas import (
        TextBlock, ImageBlock, PresentationBlock, GenericBlock,
        TaskType, TaskStatus, ModelPreference, PersonaMode,
    )
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, BeforeValidator


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
