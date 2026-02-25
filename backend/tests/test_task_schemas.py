"""Tests for backend.tasks.schemas — Task cartridge base types, presentation blocks,
interaction types, phase model, evaluation contract, AI config, and TaskCartridge."""

import warnings
from typing import Literal

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from backend.tasks.schemas import (
    AiConfig,
    AiTransitions,
    AudioBlock,
    ButtonChoice,
    ButtonInteraction,
    ChatMessageBlock,
    ChecklistItem,
    ContextRequirements,
    Difficulty,
    EmbeddedPattern,
    EvaluationContract,
    FreeformInteraction,
    GenericBlock,
    GenericInteraction,
    ImageBlock,
    InteractionConfig,
    InvestigationInteraction,
    KNOWN_BLOCK_TYPES,
    KNOWN_INTERACTION_TYPES,
    MemeBlock,
    ModelPreference,
    PassConditions,
    PersonaMode,
    Phase,
    PresentationBlock,
    RevealContent,
    SafetyConfig,
    SearchResultBlock,
    SocialPostBlock,
    TaskCartridge,
    TaskStatus,
    TaskType,
    TaxonomyWarning,
    TextBlock,
    VideoTranscriptBlock,
)

# TypeAdapter for validating PresentationBlock outside a model context
_block_adapter = TypeAdapter(PresentationBlock)


def _validate_block(data: dict) -> BaseModel:
    """Validates a raw dict as a PresentationBlock via TypeAdapter."""
    return _block_adapter.validate_python(data)


# ---------------------------------------------------------------------------
# Base type aliases
# ---------------------------------------------------------------------------


class TestTaskType:
    """TaskType Literal — ai_driven, static, hybrid."""

    _adapter = TypeAdapter(TaskType)

    def test_valid_values(self) -> None:
        for value in ("ai_driven", "static", "hybrid"):
            assert self._adapter.validate_python(value) == value

    def test_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python("interactive")


class TestTaskStatus:
    """TaskStatus Literal — active, deprecated, draft."""

    _adapter = TypeAdapter(TaskStatus)

    def test_valid_values(self) -> None:
        for value in ("active", "deprecated", "draft"):
            assert self._adapter.validate_python(value) == value

    def test_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python("archived")


class TestModelPreference:
    """ModelPreference Literal — fast, standard, complex."""

    _adapter = TypeAdapter(ModelPreference)

    def test_valid_values(self) -> None:
        for value in ("fast", "standard", "complex"):
            assert self._adapter.validate_python(value) == value

    def test_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python("turbo")


class TestPersonaMode:
    """PersonaMode Literal — presenting, chat_participant, narrator, commenter."""

    _adapter = TypeAdapter(PersonaMode)

    def test_valid_values(self) -> None:
        for value in ("presenting", "chat_participant", "narrator", "commenter"):
            assert self._adapter.validate_python(value) == value

    def test_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python("invisible")


class TestDifficulty:
    """Difficulty — constrained int 1-5."""

    _adapter = TypeAdapter(Difficulty)

    def test_valid_range(self) -> None:
        for value in (1, 2, 3, 4, 5):
            assert self._adapter.validate_python(value) == value

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python(0)

    def test_six_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python(6)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python(-1)


# ---------------------------------------------------------------------------
# TextBlock
# ---------------------------------------------------------------------------


class TestTextBlock:
    """TextBlock — text content with optional style hint."""

    def test_valid_construction(self) -> None:
        b = TextBlock(id="t1", text="Antraštė")
        assert b.id == "t1"
        assert b.type == "text"
        assert b.text == "Antraštė"
        assert b.style is None

    def test_with_style(self) -> None:
        b = TextBlock(id="t2", text="Turinys", style="headline")
        assert b.style == "headline"

    def test_missing_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TextBlock(id="t1")  # type: ignore[call-arg]

    def test_missing_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TextBlock(text="Hello")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        b = TextBlock(id="t1", text="Antraštė")
        with pytest.raises(ValidationError):
            b.text = "Changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        b = TextBlock(id="t1", text="Antraštė", style="headline")
        assert TextBlock.model_validate(b.model_dump()) == b

    def test_json_roundtrip(self) -> None:
        b = TextBlock(id="t1", text="Antraštė", style="headline")
        assert TextBlock.model_validate(b.model_dump(mode="json")) == b


# ---------------------------------------------------------------------------
# ImageBlock
# ---------------------------------------------------------------------------


class TestImageBlock:
    """ImageBlock — image with required alt_text (accessibility)."""

    def test_valid_construction(self) -> None:
        b = ImageBlock(id="i1", src="assets/graph.png", alt_text="Klaidingas grafikas")
        assert b.id == "i1"
        assert b.type == "image"
        assert b.src == "assets/graph.png"
        assert b.alt_text == "Klaidingas grafikas"
        assert b.audio_description is None
        assert b.caption is None

    def test_with_optional_fields(self) -> None:
        b = ImageBlock(
            id="i2",
            src="assets/photo.jpg",
            alt_text="Nuotrauka",
            audio_description="Grafikas rodo melagingą tendenciją",
            caption="Šaltinis: nežinomas",
        )
        assert b.audio_description == "Grafikas rodo melagingą tendenciją"
        assert b.caption == "Šaltinis: nežinomas"

    def test_missing_alt_text_rejected(self) -> None:
        with pytest.raises(ValidationError, match="alt_text"):
            ImageBlock(id="i1", src="assets/graph.png")  # type: ignore[call-arg]

    def test_missing_src_rejected(self) -> None:
        with pytest.raises(ValidationError, match="src"):
            ImageBlock(id="i1", alt_text="Alt")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        b = ImageBlock(id="i1", src="assets/graph.png", alt_text="Alt")
        with pytest.raises(ValidationError):
            b.src = "other.png"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        b = ImageBlock(
            id="i1", src="assets/graph.png", alt_text="Alt", caption="Cap"
        )
        assert ImageBlock.model_validate(b.model_dump()) == b

    def test_json_roundtrip(self) -> None:
        b = ImageBlock(id="i1", src="assets/graph.png", alt_text="Alt")
        assert ImageBlock.model_validate(b.model_dump(mode="json")) == b


# ---------------------------------------------------------------------------
# AudioBlock
# ---------------------------------------------------------------------------


class TestAudioBlock:
    """AudioBlock — audio with required transcript (accessibility)."""

    def test_valid_construction(self) -> None:
        b = AudioBlock(id="a1", src="assets/clip.mp3", transcript="Garso įrašo tekstas")
        assert b.id == "a1"
        assert b.type == "audio"
        assert b.src == "assets/clip.mp3"
        assert b.transcript == "Garso įrašo tekstas"
        assert b.duration_seconds is None

    def test_with_duration(self) -> None:
        b = AudioBlock(
            id="a2", src="assets/clip.mp3", transcript="Tekstas", duration_seconds=120
        )
        assert b.duration_seconds == 120

    def test_missing_transcript_rejected(self) -> None:
        with pytest.raises(ValidationError, match="transcript"):
            AudioBlock(id="a1", src="assets/clip.mp3")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        b = AudioBlock(id="a1", src="assets/clip.mp3", transcript="Tekstas")
        with pytest.raises(ValidationError):
            b.transcript = "Changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        b = AudioBlock(id="a1", src="assets/clip.mp3", transcript="Tekstas")
        assert AudioBlock.model_validate(b.model_dump()) == b

    def test_json_roundtrip(self) -> None:
        b = AudioBlock(
            id="a1", src="assets/clip.mp3", transcript="Tekstas", duration_seconds=60
        )
        assert AudioBlock.model_validate(b.model_dump(mode="json")) == b


# ---------------------------------------------------------------------------
# VideoTranscriptBlock
# ---------------------------------------------------------------------------


class TestVideoTranscriptBlock:
    """VideoTranscriptBlock — video transcript with optional source description."""

    def test_valid_construction(self) -> None:
        b = VideoTranscriptBlock(id="v1", transcript="Vaizdo įrašo tekstas")
        assert b.id == "v1"
        assert b.type == "video_transcript"
        assert b.transcript == "Vaizdo įrašo tekstas"
        assert b.source_description is None

    def test_with_source_description(self) -> None:
        b = VideoTranscriptBlock(
            id="v2", transcript="Tekstas", source_description="Naujienos reportažas"
        )
        assert b.source_description == "Naujienos reportažas"

    def test_missing_transcript_rejected(self) -> None:
        with pytest.raises(ValidationError, match="transcript"):
            VideoTranscriptBlock(id="v1")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        b = VideoTranscriptBlock(id="v1", transcript="Tekstas")
        with pytest.raises(ValidationError):
            b.transcript = "Changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        b = VideoTranscriptBlock(
            id="v1", transcript="Tekstas", source_description="Desc"
        )
        assert VideoTranscriptBlock.model_validate(b.model_dump()) == b

    def test_json_roundtrip(self) -> None:
        b = VideoTranscriptBlock(id="v1", transcript="Tekstas")
        assert VideoTranscriptBlock.model_validate(b.model_dump(mode="json")) == b


# ---------------------------------------------------------------------------
# MemeBlock
# ---------------------------------------------------------------------------


class TestMemeBlock:
    """MemeBlock — meme image with overlay text and required alt_text."""

    def test_valid_construction(self) -> None:
        b = MemeBlock(id="m1", image_src="assets/meme.jpg", alt_text="Memas")
        assert b.id == "m1"
        assert b.type == "meme"
        assert b.image_src == "assets/meme.jpg"
        assert b.alt_text == "Memas"
        assert b.top_text is None
        assert b.bottom_text is None

    def test_with_overlay_text(self) -> None:
        b = MemeBlock(
            id="m2",
            image_src="assets/meme.jpg",
            alt_text="Memas",
            top_text="Viršuje",
            bottom_text="Apačioje",
        )
        assert b.top_text == "Viršuje"
        assert b.bottom_text == "Apačioje"

    def test_missing_alt_text_rejected(self) -> None:
        with pytest.raises(ValidationError, match="alt_text"):
            MemeBlock(id="m1", image_src="assets/meme.jpg")  # type: ignore[call-arg]

    def test_missing_image_src_rejected(self) -> None:
        with pytest.raises(ValidationError, match="image_src"):
            MemeBlock(id="m1", alt_text="Memas")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        b = MemeBlock(id="m1", image_src="assets/meme.jpg", alt_text="Memas")
        with pytest.raises(ValidationError):
            b.alt_text = "Changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        b = MemeBlock(
            id="m1", image_src="assets/meme.jpg", alt_text="Memas", top_text="Top"
        )
        assert MemeBlock.model_validate(b.model_dump()) == b

    def test_json_roundtrip(self) -> None:
        b = MemeBlock(id="m1", image_src="assets/meme.jpg", alt_text="Memas")
        assert MemeBlock.model_validate(b.model_dump(mode="json")) == b


# ---------------------------------------------------------------------------
# ChatMessageBlock
# ---------------------------------------------------------------------------


class TestChatMessageBlock:
    """ChatMessageBlock — single message in a chat thread."""

    def test_valid_construction(self) -> None:
        b = ChatMessageBlock(id="c1", username="Jonas", text="Labas!")
        assert b.id == "c1"
        assert b.type == "chat_message"
        assert b.username == "Jonas"
        assert b.text == "Labas!"
        assert b.timestamp is None
        assert b.is_highlighted is False

    def test_with_optional_fields(self) -> None:
        b = ChatMessageBlock(
            id="c2",
            username="Ona",
            text="Žiūrėk!",
            timestamp="14:32",
            is_highlighted=True,
        )
        assert b.timestamp == "14:32"
        assert b.is_highlighted is True

    def test_missing_username_rejected(self) -> None:
        with pytest.raises(ValidationError, match="username"):
            ChatMessageBlock(id="c1", text="Hello")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        b = ChatMessageBlock(id="c1", username="Jonas", text="Labas!")
        with pytest.raises(ValidationError):
            b.text = "Changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        b = ChatMessageBlock(
            id="c1", username="Jonas", text="Labas!", is_highlighted=True
        )
        assert ChatMessageBlock.model_validate(b.model_dump()) == b

    def test_json_roundtrip(self) -> None:
        b = ChatMessageBlock(
            id="c1", username="Jonas", text="Labas!", timestamp="14:32"
        )
        assert ChatMessageBlock.model_validate(b.model_dump(mode="json")) == b


# ---------------------------------------------------------------------------
# SocialPostBlock
# ---------------------------------------------------------------------------


class TestSocialPostBlock:
    """SocialPostBlock — social media post, evocative not branded."""

    def test_valid_construction(self) -> None:
        b = SocialPostBlock(id="s1", author="Petras", text="Tai labai svarbu!")
        assert b.id == "s1"
        assert b.type == "social_post"
        assert b.author == "Petras"
        assert b.text == "Tai labai svarbu!"
        assert b.engagement is None
        assert b.cited_source is None
        assert b.platform_hint is None

    def test_with_optional_fields(self) -> None:
        b = SocialPostBlock(
            id="s2",
            author="Rūta",
            text="Dalinkitės!",
            engagement={"likes": 1523, "shares": 347},
            cited_source="https://example.com/article",
            platform_hint="social media",
        )
        assert b.engagement == {"likes": 1523, "shares": 347}
        assert b.cited_source == "https://example.com/article"
        assert b.platform_hint == "social media"

    def test_missing_author_rejected(self) -> None:
        with pytest.raises(ValidationError, match="author"):
            SocialPostBlock(id="s1", text="Hello")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        b = SocialPostBlock(id="s1", author="Petras", text="Turinys")
        with pytest.raises(ValidationError):
            b.author = "Changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        b = SocialPostBlock(
            id="s1",
            author="Petras",
            text="Turinys",
            engagement={"likes": 100},
        )
        assert SocialPostBlock.model_validate(b.model_dump()) == b

    def test_json_roundtrip(self) -> None:
        b = SocialPostBlock(
            id="s1",
            author="Petras",
            text="Turinys",
            engagement={"likes": 100, "comments": 5},
            platform_hint="forum",
        )
        assert SocialPostBlock.model_validate(b.model_dump(mode="json")) == b


# ---------------------------------------------------------------------------
# SearchResultBlock
# ---------------------------------------------------------------------------


class TestSearchResultBlock:
    """SearchResultBlock — investigation tree node."""

    def test_valid_construction(self) -> None:
        b = SearchResultBlock(
            id="sr1", query="duomenų centras", title="Rezultatas", snippet="Fragmentas"
        )
        assert b.id == "sr1"
        assert b.type == "search_result"
        assert b.query == "duomenų centras"
        assert b.title == "Rezultatas"
        assert b.snippet == "Fragmentas"
        assert b.url is None
        assert b.is_key_finding is False
        assert b.is_dead_end is False
        assert b.child_queries == []

    def test_with_full_fields(self) -> None:
        b = SearchResultBlock(
            id="sr2",
            query="energetika investicijos",
            title="Tyrimas",
            snippet="Tyrimo fragmentas",
            url="https://example.com/study",
            is_key_finding=True,
            is_dead_end=False,
            child_queries=["kas finansuoja", "lobistų veikla"],
        )
        assert b.is_key_finding is True
        assert b.child_queries == ["kas finansuoja", "lobistų veikla"]

    def test_dead_end_node(self) -> None:
        b = SearchResultBlock(
            id="sr3",
            query="sąmokslo teorija",
            title="Aklavietė",
            snippet="Nieko nerasta",
            is_dead_end=True,
        )
        assert b.is_dead_end is True
        assert b.child_queries == []

    def test_missing_query_rejected(self) -> None:
        with pytest.raises(ValidationError, match="query"):
            SearchResultBlock(
                id="sr1", title="T", snippet="S"
            )  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        b = SearchResultBlock(
            id="sr1", query="q", title="T", snippet="S"
        )
        with pytest.raises(ValidationError):
            b.title = "Changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        b = SearchResultBlock(
            id="sr1",
            query="q",
            title="T",
            snippet="S",
            child_queries=["a", "b"],
            is_key_finding=True,
        )
        assert SearchResultBlock.model_validate(b.model_dump()) == b

    def test_json_roundtrip(self) -> None:
        b = SearchResultBlock(
            id="sr1",
            query="q",
            title="T",
            snippet="S",
            url="https://example.com",
            child_queries=["a"],
        )
        assert SearchResultBlock.model_validate(b.model_dump(mode="json")) == b


# ---------------------------------------------------------------------------
# GenericBlock
# ---------------------------------------------------------------------------


class TestGenericBlock:
    """GenericBlock — fallback for unknown block types."""

    def test_valid_construction(self) -> None:
        b = GenericBlock(id="g1", type="hologram", data={"brightness": 0.8})
        assert b.id == "g1"
        assert b.type == "hologram"
        assert b.data == {"brightness": 0.8}

    def test_empty_data(self) -> None:
        b = GenericBlock(id="g2", type="unknown")
        assert b.data == {}

    def test_any_type_string(self) -> None:
        b = GenericBlock(id="g3", type="future_block_type")
        assert b.type == "future_block_type"

    def test_frozen(self) -> None:
        b = GenericBlock(id="g1", type="hologram", data={"x": 1})
        with pytest.raises(ValidationError):
            b.type = "other"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        b = GenericBlock(id="g1", type="hologram", data={"brightness": 0.8, "angle": 45})
        assert GenericBlock.model_validate(b.model_dump()) == b

    def test_json_roundtrip(self) -> None:
        b = GenericBlock(id="g1", type="hologram", data={"nested": {"a": 1}})
        assert GenericBlock.model_validate(b.model_dump(mode="json")) == b


# ---------------------------------------------------------------------------
# PresentationBlock routing
# ---------------------------------------------------------------------------


class TestPresentationBlockRouting:
    """Open-type routing — known types validate, unknown types pass through."""

    def test_known_type_routes_to_specific_model(self) -> None:
        result = _validate_block({"id": "t1", "type": "text", "text": "Hello"})
        assert isinstance(result, TextBlock)
        assert result.text == "Hello"

    def test_all_known_types_route_correctly(self) -> None:
        samples = {
            "text": {"id": "b1", "type": "text", "text": "Content"},
            "image": {"id": "b2", "type": "image", "src": "a.png", "alt_text": "Alt"},
            "audio": {"id": "b3", "type": "audio", "src": "a.mp3", "transcript": "T"},
            "video_transcript": {"id": "b4", "type": "video_transcript", "transcript": "T"},
            "meme": {"id": "b5", "type": "meme", "image_src": "m.jpg", "alt_text": "A"},
            "chat_message": {"id": "b6", "type": "chat_message", "username": "U", "text": "T"},
            "social_post": {"id": "b7", "type": "social_post", "author": "A", "text": "T"},
            "search_result": {"id": "b8", "type": "search_result", "query": "Q", "title": "T", "snippet": "S"},
        }
        for type_str, data in samples.items():
            result = _validate_block(data)
            expected_cls = KNOWN_BLOCK_TYPES[type_str]
            assert isinstance(result, expected_cls), f"Expected {expected_cls.__name__} for type '{type_str}'"

    def test_unknown_type_routes_to_generic_block(self) -> None:
        result = _validate_block({
            "id": "h1", "type": "hologram", "projection_angle": 45, "brightness": 0.8
        })
        assert isinstance(result, GenericBlock)
        assert result.type == "hologram"
        assert result.data == {"projection_angle": 45, "brightness": 0.8}

    def test_unknown_type_preserves_all_fields_in_data(self) -> None:
        result = _validate_block({
            "id": "x1", "type": "future_widget",
            "width": 100, "height": 200, "color": "red",
        })
        assert isinstance(result, GenericBlock)
        assert result.data == {"width": 100, "height": 200, "color": "red"}

    def test_missing_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            _validate_block({"id": "b1", "text": "No type"})

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _validate_block({"type": "text", "text": "No id"})

    def test_known_type_with_invalid_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            _validate_block({"id": "i1", "type": "image"})  # missing src, alt_text

    def test_model_instance_passes_through(self) -> None:
        original = TextBlock(id="t1", text="Already constructed")
        result = _validate_block(original)
        assert result is original

    def test_non_dict_non_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            _validate_block("not a dict")

    def test_multiple_unknown_types_coexist(self) -> None:
        r1 = _validate_block({"id": "a1", "type": "hologram", "angle": 90})
        r2 = _validate_block({"id": "a2", "type": "ar_overlay", "layer": "top"})
        assert isinstance(r1, GenericBlock)
        assert isinstance(r2, GenericBlock)
        assert r1.type == "hologram"
        assert r2.type == "ar_overlay"


# ---------------------------------------------------------------------------
# GenericBlock round-trip through routing
# ---------------------------------------------------------------------------


class TestGenericBlockRoundTrip:
    """GenericBlock must survive serialization → routing → equality."""

    def test_serialized_generic_block_round_trips(self) -> None:
        """A GenericBlock serialized with model_dump and re-routed stays identical."""
        original = _validate_block({
            "id": "h1", "type": "hologram", "projection_angle": 45, "brightness": 0.8
        })
        assert isinstance(original, GenericBlock)

        serialized = original.model_dump()
        restored = _validate_block(serialized)
        assert isinstance(restored, GenericBlock)
        assert restored == original

    def test_serialized_json_generic_block_round_trips(self) -> None:
        """JSON round-trip: model_dump(mode='json') → route → equality."""
        original = _validate_block({
            "id": "h1", "type": "hologram", "nested": {"a": [1, 2, 3]}
        })
        assert isinstance(original, GenericBlock)

        json_data = original.model_dump(mode="json")
        restored = _validate_block(json_data)
        assert isinstance(restored, GenericBlock)
        assert restored == original

    def test_generic_block_with_empty_data_round_trips(self) -> None:
        """Unknown type with no extra fields → GenericBlock(data={})."""
        original = _validate_block({"id": "g1", "type": "empty_block"})
        assert isinstance(original, GenericBlock)
        assert original.data == {}

        serialized = original.model_dump()
        restored = _validate_block(serialized)
        assert restored == original


# ---------------------------------------------------------------------------
# Accessibility validation
# ---------------------------------------------------------------------------


class TestAccessibilityValidation:
    """Framework Principle 14 — accessibility is schema-enforced."""

    def test_image_requires_alt_text(self) -> None:
        with pytest.raises(ValidationError, match="alt_text"):
            _validate_block({"id": "i1", "type": "image", "src": "a.png"})

    def test_meme_requires_alt_text(self) -> None:
        with pytest.raises(ValidationError, match="alt_text"):
            _validate_block({"id": "m1", "type": "meme", "image_src": "m.jpg"})

    def test_audio_requires_transcript(self) -> None:
        with pytest.raises(ValidationError, match="transcript"):
            _validate_block({"id": "a1", "type": "audio", "src": "a.mp3"})

    def test_image_with_audio_description_accepted(self) -> None:
        result = _validate_block({
            "id": "i1", "type": "image", "src": "a.png", "alt_text": "Alt",
            "audio_description": "Detailed description for screen readers",
        })
        assert isinstance(result, ImageBlock)
        assert result.audio_description == "Detailed description for screen readers"

    def test_text_does_not_require_alt_text(self) -> None:
        result = _validate_block({"id": "t1", "type": "text", "text": "Content"})
        assert isinstance(result, TextBlock)


# ---------------------------------------------------------------------------
# Mixed block list serialization
# ---------------------------------------------------------------------------


class TestMixedBlockListSerialization:
    """Lists of mixed block types must survive JSON round-trip."""

    def _make_mixed_list(self) -> list:
        """Creates a list of mixed known + unknown blocks via routing."""
        raw_blocks = [
            {"id": "t1", "type": "text", "text": "Antraštė", "style": "headline"},
            {"id": "i1", "type": "image", "src": "assets/graph.png", "alt_text": "Grafikas"},
            {"id": "c1", "type": "chat_message", "username": "Jonas", "text": "Labas!"},
            {"id": "h1", "type": "hologram", "angle": 90, "brightness": 0.5},
            {"id": "sr1", "type": "search_result", "query": "Q", "title": "T", "snippet": "S",
             "child_queries": ["a", "b"]},
        ]
        return [_validate_block(b) for b in raw_blocks]

    def test_mixed_list_types(self) -> None:
        blocks = self._make_mixed_list()
        assert isinstance(blocks[0], TextBlock)
        assert isinstance(blocks[1], ImageBlock)
        assert isinstance(blocks[2], ChatMessageBlock)
        assert isinstance(blocks[3], GenericBlock)
        assert isinstance(blocks[4], SearchResultBlock)

    def test_mixed_list_json_roundtrip(self) -> None:
        blocks = self._make_mixed_list()
        serialized = [b.model_dump(mode="json") for b in blocks]
        restored = [_validate_block(d) for d in serialized]

        assert len(restored) == len(blocks)
        for original, restored_block in zip(blocks, restored):
            assert type(original) is type(restored_block)
            assert original == restored_block

    def test_mixed_list_dict_roundtrip(self) -> None:
        blocks = self._make_mixed_list()
        serialized = [b.model_dump() for b in blocks]
        restored = [_validate_block(d) for d in serialized]

        for original, restored_block in zip(blocks, restored):
            assert original == restored_block


# ---------------------------------------------------------------------------
# KNOWN_BLOCK_TYPES registry
# ---------------------------------------------------------------------------


class TestKnownBlockTypes:
    """The type registry maps type strings to model classes."""

    def test_registry_has_eight_types(self) -> None:
        assert len(KNOWN_BLOCK_TYPES) == 8

    def test_registry_keys_match_type_fields(self) -> None:
        for type_str, model_cls in KNOWN_BLOCK_TYPES.items():
            # Each model's type field has a default matching its registry key
            field_info = model_cls.model_fields["type"]
            assert field_info.default == type_str, (
                f"{model_cls.__name__}.type default '{field_info.default}' "
                f"doesn't match registry key '{type_str}'"
            )

    def test_all_known_types_are_frozen(self) -> None:
        for type_str, model_cls in KNOWN_BLOCK_TYPES.items():
            config = model_cls.model_config
            assert config.get("frozen") is True, (
                f"{model_cls.__name__} is not frozen"
            )


# ===========================================================================
# Phase 1b — Interaction types, AiTransitions, Phase
# ===========================================================================

# TypeAdapter for validating InteractionConfig outside a model context
_interaction_adapter = TypeAdapter(InteractionConfig)


def _validate_interaction(data: dict) -> BaseModel:
    """Validates a raw dict as an InteractionConfig via TypeAdapter."""
    return _interaction_adapter.validate_python(data)


# ---------------------------------------------------------------------------
# ButtonChoice
# ---------------------------------------------------------------------------


class TestButtonChoice:
    """ButtonChoice — a single button option."""

    def test_valid_construction(self) -> None:
        c = ButtonChoice(label="Dalintis", target_phase="phase_bitten")
        assert c.label == "Dalintis"
        assert c.target_phase == "phase_bitten"
        assert c.context_label is None

    def test_with_context_label(self) -> None:
        c = ButtonChoice(
            label="Dalintis",
            target_phase="phase_bitten",
            context_label="Pasidalino neperskaitęs",
        )
        assert c.context_label == "Pasidalino neperskaitęs"

    def test_missing_label_rejected(self) -> None:
        with pytest.raises(ValidationError, match="label"):
            ButtonChoice(target_phase="p1")  # type: ignore[call-arg]

    def test_missing_target_phase_rejected(self) -> None:
        with pytest.raises(ValidationError, match="target_phase"):
            ButtonChoice(label="Click me")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        c = ButtonChoice(label="Dalintis", target_phase="phase_bitten")
        with pytest.raises(ValidationError):
            c.label = "Changed"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        c = ButtonChoice(
            label="Skaityti", target_phase="phase_read", context_label="Skaitė"
        )
        assert ButtonChoice.model_validate(c.model_dump(mode="json")) == c


# ---------------------------------------------------------------------------
# ButtonInteraction
# ---------------------------------------------------------------------------


class TestButtonInteraction:
    """ButtonInteraction — button choices with labels and target phases."""

    def test_valid_construction(self) -> None:
        b = ButtonInteraction(
            choices=[
                ButtonChoice(label="Dalintis", target_phase="phase_bitten"),
                ButtonChoice(label="Skaityti", target_phase="phase_read"),
            ]
        )
        assert b.type == "button"
        assert len(b.choices) == 2
        assert b.choices[0].label == "Dalintis"

    def test_empty_choices_accepted(self) -> None:
        b = ButtonInteraction()
        assert b.choices == []

    def test_frozen(self) -> None:
        b = ButtonInteraction(
            choices=[ButtonChoice(label="A", target_phase="p1")]
        )
        with pytest.raises(ValidationError):
            b.type = "other"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        b = ButtonInteraction(
            choices=[
                ButtonChoice(label="A", target_phase="p1", context_label="Chose A"),
                ButtonChoice(label="B", target_phase="p2"),
            ]
        )
        assert ButtonInteraction.model_validate(b.model_dump()) == b

    def test_json_roundtrip(self) -> None:
        b = ButtonInteraction(
            choices=[
                ButtonChoice(label="A", target_phase="p1"),
            ]
        )
        assert ButtonInteraction.model_validate(b.model_dump(mode="json")) == b


# ---------------------------------------------------------------------------
# FreeformInteraction
# ---------------------------------------------------------------------------


class TestFreeformInteraction:
    """FreeformInteraction — AI dialogue with exchange bounds."""

    def test_valid_construction(self) -> None:
        f = FreeformInteraction(
            trickster_opening="Ką pastebėjai?",
            min_exchanges=2,
            max_exchanges=6,
        )
        assert f.type == "freeform"
        assert f.trickster_opening == "Ką pastebėjai?"
        assert f.min_exchanges == 2
        assert f.max_exchanges == 6

    def test_missing_trickster_opening_rejected(self) -> None:
        with pytest.raises(ValidationError, match="trickster_opening"):
            FreeformInteraction(min_exchanges=1, max_exchanges=3)  # type: ignore[call-arg]

    def test_min_exchanges_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FreeformInteraction(
                trickster_opening="Hi", min_exchanges=0, max_exchanges=3
            )

    def test_max_exchanges_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FreeformInteraction(
                trickster_opening="Hi", min_exchanges=1, max_exchanges=0
            )

    def test_min_exceeds_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FreeformInteraction(
                trickster_opening="Hi", min_exchanges=5, max_exchanges=3
            )

    def test_min_equals_max_accepted(self) -> None:
        f = FreeformInteraction(
            trickster_opening="Hi", min_exchanges=3, max_exchanges=3
        )
        assert f.min_exchanges == 3
        assert f.max_exchanges == 3

    def test_frozen(self) -> None:
        f = FreeformInteraction(
            trickster_opening="Hi", min_exchanges=1, max_exchanges=3
        )
        with pytest.raises(ValidationError):
            f.trickster_opening = "Changed"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        f = FreeformInteraction(
            trickster_opening="Ką pastebėjai?",
            min_exchanges=2,
            max_exchanges=6,
        )
        assert FreeformInteraction.model_validate(f.model_dump(mode="json")) == f


# ---------------------------------------------------------------------------
# InvestigationInteraction
# ---------------------------------------------------------------------------


class TestInvestigationInteraction:
    """InvestigationInteraction — search tree navigation config."""

    def test_valid_construction(self) -> None:
        i = InvestigationInteraction(
            starting_queries=["duomenų centras", "finansavimas"],
            submit_target="phase_evaluation",
        )
        assert i.type == "investigation"
        assert i.starting_queries == ["duomenų centras", "finansavimas"]
        assert i.submit_target == "phase_evaluation"
        assert i.min_key_findings == 0

    def test_with_min_key_findings(self) -> None:
        i = InvestigationInteraction(
            starting_queries=["q1"],
            submit_target="phase_eval",
            min_key_findings=3,
        )
        assert i.min_key_findings == 3

    def test_empty_starting_queries_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InvestigationInteraction(
                starting_queries=[],
                submit_target="phase_eval",
            )

    def test_missing_submit_target_rejected(self) -> None:
        with pytest.raises(ValidationError, match="submit_target"):
            InvestigationInteraction(
                starting_queries=["q1"],
            )  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        i = InvestigationInteraction(
            starting_queries=["q1"], submit_target="phase_eval"
        )
        with pytest.raises(ValidationError):
            i.submit_target = "changed"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        i = InvestigationInteraction(
            starting_queries=["q1", "q2"],
            submit_target="phase_eval",
            min_key_findings=2,
        )
        assert InvestigationInteraction.model_validate(i.model_dump(mode="json")) == i


# ---------------------------------------------------------------------------
# GenericInteraction
# ---------------------------------------------------------------------------


class TestGenericInteraction:
    """GenericInteraction — fallback for unknown interaction types."""

    def test_valid_construction(self) -> None:
        g = GenericInteraction(
            type="timeline_scrub",
            config={"start_time": 0, "end_time": 120},
        )
        assert g.type == "timeline_scrub"
        assert g.config == {"start_time": 0, "end_time": 120}

    def test_empty_config(self) -> None:
        g = GenericInteraction(type="unknown_interaction")
        assert g.config == {}

    def test_any_type_string(self) -> None:
        g = GenericInteraction(type="future_interaction")
        assert g.type == "future_interaction"

    def test_frozen(self) -> None:
        g = GenericInteraction(type="timeline_scrub", config={"x": 1})
        with pytest.raises(ValidationError):
            g.type = "other"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        g = GenericInteraction(type="drag_sort", config={"items": [1, 2, 3]})
        assert GenericInteraction.model_validate(g.model_dump()) == g

    def test_json_roundtrip(self) -> None:
        g = GenericInteraction(
            type="highlight", config={"regions": [{"start": 0, "end": 10}]}
        )
        assert GenericInteraction.model_validate(g.model_dump(mode="json")) == g


# ---------------------------------------------------------------------------
# InteractionConfig routing
# ---------------------------------------------------------------------------


class TestInteractionConfigRouting:
    """Open-type routing — known interactions validate, unknown pass through."""

    def test_known_type_routes_to_button(self) -> None:
        result = _validate_interaction({
            "type": "button",
            "choices": [{"label": "A", "target_phase": "p1"}],
        })
        assert isinstance(result, ButtonInteraction)
        assert len(result.choices) == 1

    def test_known_type_routes_to_freeform(self) -> None:
        result = _validate_interaction({
            "type": "freeform",
            "trickster_opening": "Hi",
            "min_exchanges": 1,
            "max_exchanges": 3,
        })
        assert isinstance(result, FreeformInteraction)
        assert result.trickster_opening == "Hi"

    def test_known_type_routes_to_investigation(self) -> None:
        result = _validate_interaction({
            "type": "investigation",
            "starting_queries": ["q1"],
            "submit_target": "phase_eval",
        })
        assert isinstance(result, InvestigationInteraction)
        assert result.starting_queries == ["q1"]

    def test_all_known_types_route_correctly(self) -> None:
        samples = {
            "button": {
                "type": "button",
                "choices": [{"label": "Go", "target_phase": "p1"}],
            },
            "freeform": {
                "type": "freeform",
                "trickster_opening": "Hi",
                "min_exchanges": 1,
                "max_exchanges": 3,
            },
            "investigation": {
                "type": "investigation",
                "starting_queries": ["q1"],
                "submit_target": "phase_eval",
            },
        }
        for type_str, data in samples.items():
            result = _validate_interaction(data)
            expected_cls = KNOWN_INTERACTION_TYPES[type_str]
            assert isinstance(result, expected_cls), (
                f"Expected {expected_cls.__name__} for type '{type_str}'"
            )

    def test_unknown_type_routes_to_generic(self) -> None:
        result = _validate_interaction({
            "type": "timeline_scrub",
            "start_time": 0,
            "end_time": 120,
        })
        assert isinstance(result, GenericInteraction)
        assert result.type == "timeline_scrub"
        assert result.config == {"start_time": 0, "end_time": 120}

    def test_unknown_type_preserves_all_fields_in_config(self) -> None:
        result = _validate_interaction({
            "type": "drag_sort",
            "items": ["a", "b", "c"],
            "allow_reorder": True,
        })
        assert isinstance(result, GenericInteraction)
        assert result.config == {"items": ["a", "b", "c"], "allow_reorder": True}

    def test_missing_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            _validate_interaction({"choices": [{"label": "A", "target_phase": "p1"}]})

    def test_known_type_with_invalid_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            _validate_interaction({"type": "freeform"})  # missing required fields

    def test_model_instance_passes_through(self) -> None:
        original = ButtonInteraction(
            choices=[ButtonChoice(label="A", target_phase="p1")]
        )
        result = _validate_interaction(original)
        assert result is original

    def test_non_dict_non_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            _validate_interaction("not a dict")

    def test_multiple_unknown_types_coexist(self) -> None:
        r1 = _validate_interaction({"type": "timeline_scrub", "duration": 60})
        r2 = _validate_interaction({"type": "highlight", "color": "yellow"})
        assert isinstance(r1, GenericInteraction)
        assert isinstance(r2, GenericInteraction)
        assert r1.type == "timeline_scrub"
        assert r2.type == "highlight"

    def test_no_id_field_required(self) -> None:
        """Interactions do NOT have an id field — unlike PresentationBlocks."""
        result = _validate_interaction({
            "type": "button",
            "choices": [{"label": "A", "target_phase": "p1"}],
        })
        assert isinstance(result, ButtonInteraction)
        assert not hasattr(result, "id") or "id" not in result.model_fields


# ---------------------------------------------------------------------------
# GenericInteraction round-trip through routing
# ---------------------------------------------------------------------------


class TestGenericInteractionRoundTrip:
    """GenericInteraction must survive serialization → routing → equality."""

    def test_serialized_generic_interaction_round_trips(self) -> None:
        original = _validate_interaction({
            "type": "timeline_scrub", "start_time": 0, "end_time": 120
        })
        assert isinstance(original, GenericInteraction)

        serialized = original.model_dump()
        restored = _validate_interaction(serialized)
        assert isinstance(restored, GenericInteraction)
        assert restored == original

    def test_serialized_json_generic_interaction_round_trips(self) -> None:
        original = _validate_interaction({
            "type": "drag_sort", "nested": {"a": [1, 2, 3]}
        })
        assert isinstance(original, GenericInteraction)

        json_data = original.model_dump(mode="json")
        restored = _validate_interaction(json_data)
        assert isinstance(restored, GenericInteraction)
        assert restored == original

    def test_generic_interaction_with_empty_config_round_trips(self) -> None:
        original = _validate_interaction({"type": "empty_interaction"})
        assert isinstance(original, GenericInteraction)
        assert original.config == {}

        serialized = original.model_dump()
        restored = _validate_interaction(serialized)
        assert restored == original


# ---------------------------------------------------------------------------
# AiTransitions
# ---------------------------------------------------------------------------


class TestAiTransitions:
    """AiTransitions — maps engine signals to target phase IDs."""

    def test_valid_construction(self) -> None:
        t = AiTransitions(
            on_success="phase_reveal_win",
            on_max_exchanges="phase_reveal_timeout",
            on_partial="phase_reveal_partial",
        )
        assert t.on_success == "phase_reveal_win"
        assert t.on_max_exchanges == "phase_reveal_timeout"
        assert t.on_partial == "phase_reveal_partial"

    def test_missing_on_success_rejected(self) -> None:
        with pytest.raises(ValidationError, match="on_success"):
            AiTransitions(
                on_max_exchanges="p2", on_partial="p3"
            )  # type: ignore[call-arg]

    def test_missing_on_max_exchanges_rejected(self) -> None:
        with pytest.raises(ValidationError, match="on_max_exchanges"):
            AiTransitions(
                on_success="p1", on_partial="p3"
            )  # type: ignore[call-arg]

    def test_missing_on_partial_rejected(self) -> None:
        with pytest.raises(ValidationError, match="on_partial"):
            AiTransitions(
                on_success="p1", on_max_exchanges="p2"
            )  # type: ignore[call-arg]

    def test_same_target_for_all_accepted(self) -> None:
        t = AiTransitions(
            on_success="phase_reveal",
            on_max_exchanges="phase_reveal",
            on_partial="phase_reveal",
        )
        assert t.on_success == t.on_max_exchanges == t.on_partial

    def test_frozen(self) -> None:
        t = AiTransitions(
            on_success="p1", on_max_exchanges="p2", on_partial="p3"
        )
        with pytest.raises(ValidationError):
            t.on_success = "changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        t = AiTransitions(
            on_success="p1", on_max_exchanges="p2", on_partial="p3"
        )
        assert AiTransitions.model_validate(t.model_dump()) == t

    def test_json_roundtrip(self) -> None:
        t = AiTransitions(
            on_success="p1", on_max_exchanges="p2", on_partial="p3"
        )
        assert AiTransitions.model_validate(t.model_dump(mode="json")) == t


# ---------------------------------------------------------------------------
# EvaluationOutcome
# ---------------------------------------------------------------------------


class TestEvaluationOutcome:
    """EvaluationOutcome — named terminal phase outcomes."""

    _adapter = TypeAdapter(Literal["trickster_wins", "partial", "trickster_loses"])

    def test_valid_values(self) -> None:
        for value in ("trickster_wins", "partial", "trickster_loses"):
            assert self._adapter.validate_python(value) == value

    def test_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python("draw")


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


class TestPhase:
    """Phase — the core state machine node."""

    def test_valid_static_phase(self) -> None:
        p = Phase(
            id="phase_intro",
            title="Įvadas",
            visible_blocks=["block-headline", "block-snippet"],
            trickster_content="Pažiūrėk į šią antraštę...",
            interaction=ButtonInteraction(
                choices=[
                    ButtonChoice(label="Dalintis", target_phase="phase_bitten"),
                    ButtonChoice(label="Skaityti", target_phase="phase_read"),
                ]
            ),
        )
        assert p.id == "phase_intro"
        assert p.title == "Įvadas"
        assert p.visible_blocks == ["block-headline", "block-snippet"]
        assert p.is_ai_phase is False
        assert p.trickster_content == "Pažiūrėk į šią antraštę..."
        assert isinstance(p.interaction, ButtonInteraction)
        assert p.ai_transitions is None
        assert p.is_terminal is False
        assert p.evaluation_outcome is None

    def test_valid_ai_phase(self) -> None:
        p = Phase(
            id="phase_dialogue",
            title="Diskusija",
            visible_blocks=["block-article"],
            is_ai_phase=True,
            interaction=FreeformInteraction(
                trickster_opening="Ką pastebėjai?",
                min_exchanges=2,
                max_exchanges=6,
            ),
            ai_transitions=AiTransitions(
                on_success="phase_reveal_win",
                on_max_exchanges="phase_reveal_timeout",
                on_partial="phase_reveal_partial",
            ),
        )
        assert p.is_ai_phase is True
        assert isinstance(p.interaction, FreeformInteraction)
        assert isinstance(p.ai_transitions, AiTransitions)
        assert p.ai_transitions.on_success == "phase_reveal_win"

    def test_valid_terminal_phase(self) -> None:
        p = Phase(
            id="phase_reveal_win",
            title="Atskleidimas",
            visible_blocks=["block-reveal"],
            is_terminal=True,
            evaluation_outcome="trickster_loses",
        )
        assert p.is_terminal is True
        assert p.evaluation_outcome == "trickster_loses"
        assert p.interaction is None

    def test_terminal_without_evaluation_outcome_accepted(self) -> None:
        """Schema level doesn't enforce that terminal phases have outcomes.
        That's Phase 2b's business rule."""
        p = Phase(
            id="phase_end",
            title="Pabaiga",
            is_terminal=True,
        )
        assert p.is_terminal is True
        assert p.evaluation_outcome is None

    def test_non_terminal_with_evaluation_outcome_accepted(self) -> None:
        """Schema doesn't cross-validate terminal + outcome. Phase 2b does."""
        p = Phase(
            id="phase_mid",
            title="Vidurys",
            evaluation_outcome="partial",
        )
        assert p.is_terminal is False
        assert p.evaluation_outcome == "partial"

    def test_empty_visible_blocks_accepted(self) -> None:
        p = Phase(id="phase_chat", title="Pokalbis")
        assert p.visible_blocks == []

    def test_interaction_defaults_to_none(self) -> None:
        p = Phase(id="phase_end", title="Pabaiga")
        assert p.interaction is None

    def test_ai_transitions_defaults_to_none(self) -> None:
        p = Phase(id="phase_end", title="Pabaiga")
        assert p.ai_transitions is None

    def test_evaluation_outcome_defaults_to_none(self) -> None:
        p = Phase(id="phase_mid", title="Vidurys")
        assert p.evaluation_outcome is None

    def test_invalid_evaluation_outcome_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Phase(
                id="phase_end",
                title="Pabaiga",
                evaluation_outcome="draw",
            )

    def test_frozen(self) -> None:
        p = Phase(id="phase_intro", title="Įvadas")
        with pytest.raises(ValidationError):
            p.title = "Changed"  # type: ignore[misc]

    def test_missing_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="id"):
            Phase(title="No ID")  # type: ignore[call-arg]

    def test_missing_title_rejected(self) -> None:
        with pytest.raises(ValidationError, match="title"):
            Phase(id="phase_1")  # type: ignore[call-arg]

    def test_serialization_roundtrip(self) -> None:
        p = Phase(
            id="phase_intro",
            title="Įvadas",
            visible_blocks=["b1", "b2"],
            trickster_content="Text",
            interaction=ButtonInteraction(
                choices=[ButtonChoice(label="A", target_phase="p1")]
            ),
        )
        assert Phase.model_validate(p.model_dump()) == p

    def test_json_roundtrip(self) -> None:
        p = Phase(
            id="phase_dialogue",
            title="Diskusija",
            visible_blocks=["b1"],
            is_ai_phase=True,
            interaction=FreeformInteraction(
                trickster_opening="Hi",
                min_exchanges=2,
                max_exchanges=6,
            ),
            ai_transitions=AiTransitions(
                on_success="p1", on_max_exchanges="p2", on_partial="p3"
            ),
        )
        assert Phase.model_validate(p.model_dump(mode="json")) == p

    def test_json_roundtrip_terminal(self) -> None:
        p = Phase(
            id="phase_end",
            title="Pabaiga",
            is_terminal=True,
            evaluation_outcome="trickster_wins",
        )
        assert Phase.model_validate(p.model_dump(mode="json")) == p

    def test_interaction_from_raw_dict(self) -> None:
        """Phase should accept interaction as a raw dict (JSON deserialization)."""
        p = Phase.model_validate({
            "id": "phase_intro",
            "title": "Intro",
            "interaction": {
                "type": "button",
                "choices": [{"label": "Go", "target_phase": "p1"}],
            },
        })
        assert isinstance(p.interaction, ButtonInteraction)

    def test_interaction_unknown_type_from_raw_dict(self) -> None:
        """Phase accepts unknown interaction types via GenericInteraction."""
        p = Phase.model_validate({
            "id": "phase_custom",
            "title": "Custom",
            "interaction": {
                "type": "timeline_scrub",
                "start_time": 0,
                "end_time": 120,
            },
        })
        assert isinstance(p.interaction, GenericInteraction)
        assert p.interaction.type == "timeline_scrub"


# ---------------------------------------------------------------------------
# Mixed Phase list serialization
# ---------------------------------------------------------------------------


class TestMixedPhaseListSerialization:
    """Lists of phases with different interaction types survive JSON round-trip."""

    def _make_mixed_phases(self) -> list[Phase]:
        return [
            Phase(
                id="phase_intro",
                title="Intro",
                visible_blocks=["b1"],
                trickster_content="Look at this...",
                interaction=ButtonInteraction(
                    choices=[
                        ButtonChoice(label="Share", target_phase="phase_bitten"),
                        ButtonChoice(label="Read", target_phase="phase_read"),
                    ]
                ),
            ),
            Phase(
                id="phase_dialogue",
                title="Dialogue",
                visible_blocks=["b1", "b2"],
                is_ai_phase=True,
                interaction=FreeformInteraction(
                    trickster_opening="What did you notice?",
                    min_exchanges=2,
                    max_exchanges=6,
                ),
                ai_transitions=AiTransitions(
                    on_success="phase_win",
                    on_max_exchanges="phase_timeout",
                    on_partial="phase_partial",
                ),
            ),
            Phase(
                id="phase_win",
                title="Victory",
                visible_blocks=["b3"],
                is_terminal=True,
                evaluation_outcome="trickster_loses",
            ),
        ]

    def test_mixed_phases_json_roundtrip(self) -> None:
        phases = self._make_mixed_phases()
        serialized = [p.model_dump(mode="json") for p in phases]
        restored = [Phase.model_validate(d) for d in serialized]

        assert len(restored) == len(phases)
        for original, restored_phase in zip(phases, restored):
            assert original == restored_phase

    def test_mixed_phases_dict_roundtrip(self) -> None:
        phases = self._make_mixed_phases()
        serialized = [p.model_dump() for p in phases]
        restored = [Phase.model_validate(d) for d in serialized]

        for original, restored_phase in zip(phases, restored):
            assert original == restored_phase


# ---------------------------------------------------------------------------
# KNOWN_INTERACTION_TYPES registry
# ---------------------------------------------------------------------------


class TestKnownInteractionTypes:
    """The interaction type registry maps type strings to model classes."""

    def test_registry_has_three_types(self) -> None:
        assert len(KNOWN_INTERACTION_TYPES) == 3

    def test_registry_keys_match_type_fields(self) -> None:
        for type_str, model_cls in KNOWN_INTERACTION_TYPES.items():
            field_info = model_cls.model_fields["type"]
            assert field_info.default == type_str, (
                f"{model_cls.__name__}.type default '{field_info.default}' "
                f"doesn't match registry key '{type_str}'"
            )

    def test_all_known_types_are_frozen(self) -> None:
        for type_str, model_cls in KNOWN_INTERACTION_TYPES.items():
            config = model_cls.model_config
            assert config.get("frozen") is True, (
                f"{model_cls.__name__} is not frozen"
            )


# ===========================================================================
# Phase 1c: Evaluation contract, AI config, reveal, safety, TaskCartridge
# ===========================================================================


# ---------------------------------------------------------------------------
# ContextRequirements type alias
# ---------------------------------------------------------------------------


class TestContextRequirements:
    """ContextRequirements Literal — session_only, learning_profile, full_history."""

    _adapter = TypeAdapter(ContextRequirements)

    def test_valid_values(self) -> None:
        for value in ("session_only", "learning_profile", "full_history"):
            assert self._adapter.validate_python(value) == value

    def test_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python("invalid_context")


# ---------------------------------------------------------------------------
# EmbeddedPattern
# ---------------------------------------------------------------------------


class TestEmbeddedPattern:
    """EmbeddedPattern — a single manipulation pattern in task content."""

    def test_valid_construction(self) -> None:
        p = EmbeddedPattern(
            id="pattern-urgency",
            description="Artificial deadline",
            technique="manufactured_deadline",
            real_world_connection="News outlets use countdown timers",
        )
        assert p.id == "pattern-urgency"
        assert p.technique == "manufactured_deadline"

    def test_missing_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="id"):
            EmbeddedPattern(  # type: ignore[call-arg]
                description="d",
                technique="t",
                real_world_connection="r",
            )

    def test_missing_description_rejected(self) -> None:
        with pytest.raises(ValidationError, match="description"):
            EmbeddedPattern(  # type: ignore[call-arg]
                id="p1",
                technique="t",
                real_world_connection="r",
            )

    def test_missing_technique_rejected(self) -> None:
        with pytest.raises(ValidationError, match="technique"):
            EmbeddedPattern(  # type: ignore[call-arg]
                id="p1",
                description="d",
                real_world_connection="r",
            )

    def test_missing_real_world_connection_rejected(self) -> None:
        with pytest.raises(ValidationError, match="real_world_connection"):
            EmbeddedPattern(  # type: ignore[call-arg]
                id="p1",
                description="d",
                technique="t",
            )

    def test_frozen(self) -> None:
        p = EmbeddedPattern(
            id="p1", description="d", technique="t", real_world_connection="r",
        )
        with pytest.raises(ValidationError):
            p.id = "p2"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        p = EmbeddedPattern(
            id="pattern-urgency",
            description="Artificial deadline",
            technique="manufactured_deadline",
            real_world_connection="News outlets use countdown timers",
        )
        assert EmbeddedPattern.model_validate(p.model_dump(mode="json")) == p

    def test_serialization_roundtrip(self) -> None:
        p = EmbeddedPattern(
            id="p1", description="d", technique="t", real_world_connection="r",
        )
        assert EmbeddedPattern.model_validate(p.model_dump()) == p


# ---------------------------------------------------------------------------
# ChecklistItem
# ---------------------------------------------------------------------------


class TestChecklistItem:
    """ChecklistItem — what the student should demonstrate."""

    def test_valid_construction(self) -> None:
        item = ChecklistItem(
            id="check-headline",
            description="Recognized headline mismatch",
            pattern_refs=["pattern-headline"],
            is_mandatory=True,
        )
        assert item.id == "check-headline"
        assert item.is_mandatory is True

    def test_defaults(self) -> None:
        item = ChecklistItem(id="c1", description="d")
        assert item.pattern_refs == []
        assert item.is_mandatory is False

    def test_missing_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="id"):
            ChecklistItem(description="d")  # type: ignore[call-arg]

    def test_missing_description_rejected(self) -> None:
        with pytest.raises(ValidationError, match="description"):
            ChecklistItem(id="c1")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        item = ChecklistItem(id="c1", description="d")
        with pytest.raises(ValidationError):
            item.id = "c2"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        item = ChecklistItem(
            id="check-1",
            description="d",
            pattern_refs=["p1", "p2"],
            is_mandatory=True,
        )
        assert ChecklistItem.model_validate(item.model_dump(mode="json")) == item

    def test_serialization_roundtrip(self) -> None:
        item = ChecklistItem(id="c1", description="d")
        assert ChecklistItem.model_validate(item.model_dump()) == item


# ---------------------------------------------------------------------------
# PassConditions
# ---------------------------------------------------------------------------


class TestPassConditions:
    """PassConditions — textual descriptions of each evaluation outcome."""

    def test_valid_construction(self) -> None:
        pc = PassConditions(
            trickster_wins="Student shared without reading",
            partial="Student read but missed patterns",
            trickster_loses="Student identified techniques",
        )
        assert pc.trickster_wins == "Student shared without reading"

    def test_missing_trickster_wins_rejected(self) -> None:
        with pytest.raises(ValidationError, match="trickster_wins"):
            PassConditions(  # type: ignore[call-arg]
                partial="p",
                trickster_loses="l",
            )

    def test_missing_partial_rejected(self) -> None:
        with pytest.raises(ValidationError, match="partial"):
            PassConditions(  # type: ignore[call-arg]
                trickster_wins="w",
                trickster_loses="l",
            )

    def test_missing_trickster_loses_rejected(self) -> None:
        with pytest.raises(ValidationError, match="trickster_loses"):
            PassConditions(  # type: ignore[call-arg]
                trickster_wins="w",
                partial="p",
            )

    def test_frozen(self) -> None:
        pc = PassConditions(trickster_wins="w", partial="p", trickster_loses="l")
        with pytest.raises(ValidationError):
            pc.trickster_wins = "new"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        pc = PassConditions(trickster_wins="w", partial="p", trickster_loses="l")
        assert PassConditions.model_validate(pc.model_dump(mode="json")) == pc

    def test_serialization_roundtrip(self) -> None:
        pc = PassConditions(trickster_wins="w", partial="p", trickster_loses="l")
        assert PassConditions.model_validate(pc.model_dump()) == pc


# ---------------------------------------------------------------------------
# EvaluationContract
# ---------------------------------------------------------------------------


class TestEvaluationContract:
    """EvaluationContract — the rubric consumed by V6's scoring engine."""

    def _make_contract(self) -> EvaluationContract:
        return EvaluationContract(
            patterns_embedded=[
                EmbeddedPattern(
                    id="p1", description="d", technique="t", real_world_connection="r",
                ),
            ],
            checklist=[
                ChecklistItem(id="c1", description="d", pattern_refs=["p1"]),
            ],
            pass_conditions=PassConditions(
                trickster_wins="w", partial="p", trickster_loses="l",
            ),
        )

    def test_valid_construction(self) -> None:
        ec = self._make_contract()
        assert len(ec.patterns_embedded) == 1
        assert len(ec.checklist) == 1
        assert ec.pass_conditions.trickster_wins == "w"

    def test_empty_patterns_accepted(self) -> None:
        ec = EvaluationContract(
            patterns_embedded=[],
            checklist=[],
            pass_conditions=PassConditions(
                trickster_wins="w", partial="p", trickster_loses="l",
            ),
        )
        assert ec.patterns_embedded == []

    def test_missing_pass_conditions_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pass_conditions"):
            EvaluationContract(  # type: ignore[call-arg]
                patterns_embedded=[],
                checklist=[],
            )

    def test_frozen(self) -> None:
        ec = self._make_contract()
        with pytest.raises(ValidationError):
            ec.checklist = []  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        ec = self._make_contract()
        assert EvaluationContract.model_validate(ec.model_dump(mode="json")) == ec

    def test_serialization_roundtrip(self) -> None:
        ec = self._make_contract()
        assert EvaluationContract.model_validate(ec.model_dump()) == ec


# ---------------------------------------------------------------------------
# AiConfig
# ---------------------------------------------------------------------------


class TestAiConfig:
    """AiConfig — AI layer configuration for a task."""

    def _make_config(self) -> AiConfig:
        return AiConfig(
            model_preference="standard",
            prompt_directory="prompts/tasks/task-001",
            persona_mode="presenting",
            has_static_fallback=True,
            context_requirements="session_only",
        )

    def test_valid_construction(self) -> None:
        ac = self._make_config()
        assert ac.model_preference == "standard"
        assert ac.persona_mode == "presenting"
        assert ac.context_requirements == "session_only"

    def test_missing_model_preference_rejected(self) -> None:
        with pytest.raises(ValidationError, match="model_preference"):
            AiConfig(  # type: ignore[call-arg]
                prompt_directory="p",
                persona_mode="presenting",
                has_static_fallback=True,
                context_requirements="session_only",
            )

    def test_invalid_model_preference_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AiConfig(
                model_preference="turbo",  # type: ignore[arg-type]
                prompt_directory="p",
                persona_mode="presenting",
                has_static_fallback=True,
                context_requirements="session_only",
            )

    def test_invalid_persona_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AiConfig(
                model_preference="fast",
                prompt_directory="p",
                persona_mode="invalid",  # type: ignore[arg-type]
                has_static_fallback=True,
                context_requirements="session_only",
            )

    def test_invalid_context_requirements_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AiConfig(
                model_preference="fast",
                prompt_directory="p",
                persona_mode="presenting",
                has_static_fallback=True,
                context_requirements="invalid",  # type: ignore[arg-type]
            )

    def test_frozen(self) -> None:
        ac = self._make_config()
        with pytest.raises(ValidationError):
            ac.model_preference = "fast"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        ac = self._make_config()
        assert AiConfig.model_validate(ac.model_dump(mode="json")) == ac

    def test_serialization_roundtrip(self) -> None:
        ac = self._make_config()
        assert AiConfig.model_validate(ac.model_dump()) == ac


# ---------------------------------------------------------------------------
# RevealContent
# ---------------------------------------------------------------------------


class TestRevealContent:
    """RevealContent — post-task reveal content."""

    def test_valid_construction(self) -> None:
        rc = RevealContent(
            key_lesson="The headline was designed to trigger urgency",
            additional_resources=["https://example.com/media-literacy"],
        )
        assert rc.key_lesson.startswith("The headline")
        assert len(rc.additional_resources) == 1

    def test_defaults(self) -> None:
        rc = RevealContent(key_lesson="lesson")
        assert rc.additional_resources == []

    def test_missing_key_lesson_rejected(self) -> None:
        with pytest.raises(ValidationError, match="key_lesson"):
            RevealContent()  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        rc = RevealContent(key_lesson="lesson")
        with pytest.raises(ValidationError):
            rc.key_lesson = "new"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        rc = RevealContent(key_lesson="lesson", additional_resources=["a", "b"])
        assert RevealContent.model_validate(rc.model_dump(mode="json")) == rc

    def test_serialization_roundtrip(self) -> None:
        rc = RevealContent(key_lesson="lesson")
        assert RevealContent.model_validate(rc.model_dump()) == rc


# ---------------------------------------------------------------------------
# SafetyConfig
# ---------------------------------------------------------------------------


class TestSafetyConfig:
    """SafetyConfig — safety guardrails for a task."""

    def test_valid_construction(self) -> None:
        sc = SafetyConfig(
            content_boundaries=["no_real_harm", "no_violence"],
            intensity_ceiling=3,
            cold_start_safe=True,
        )
        assert sc.intensity_ceiling == 3
        assert sc.cold_start_safe is True

    def test_defaults_content_boundaries(self) -> None:
        sc = SafetyConfig(intensity_ceiling=1, cold_start_safe=False)
        assert sc.content_boundaries == []

    def test_intensity_ceiling_min(self) -> None:
        sc = SafetyConfig(intensity_ceiling=1, cold_start_safe=True)
        assert sc.intensity_ceiling == 1

    def test_intensity_ceiling_max(self) -> None:
        sc = SafetyConfig(intensity_ceiling=5, cold_start_safe=True)
        assert sc.intensity_ceiling == 5

    def test_intensity_ceiling_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SafetyConfig(intensity_ceiling=0, cold_start_safe=True)

    def test_intensity_ceiling_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SafetyConfig(intensity_ceiling=6, cold_start_safe=True)

    def test_missing_intensity_ceiling_rejected(self) -> None:
        with pytest.raises(ValidationError, match="intensity_ceiling"):
            SafetyConfig(cold_start_safe=True)  # type: ignore[call-arg]

    def test_missing_cold_start_safe_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cold_start_safe"):
            SafetyConfig(intensity_ceiling=3)  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        sc = SafetyConfig(intensity_ceiling=3, cold_start_safe=True)
        with pytest.raises(ValidationError):
            sc.intensity_ceiling = 1  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        sc = SafetyConfig(
            content_boundaries=["a"], intensity_ceiling=3, cold_start_safe=True,
        )
        assert SafetyConfig.model_validate(sc.model_dump(mode="json")) == sc

    def test_serialization_roundtrip(self) -> None:
        sc = SafetyConfig(intensity_ceiling=2, cold_start_safe=False)
        assert SafetyConfig.model_validate(sc.model_dump()) == sc


# ---------------------------------------------------------------------------
# TaskCartridge — helpers
# ---------------------------------------------------------------------------


def _minimal_cartridge_data(**overrides: object) -> dict:
    """Returns a minimal valid TaskCartridge dict for testing.

    Override any field by passing keyword arguments.
    """
    data: dict = {
        "task_id": "task-test-001",
        "task_type": "hybrid",
        "title": "Testas",
        "description": "Testo aprašymas",
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
        "evaluation": {
            "patterns_embedded": [
                {
                    "id": "p1",
                    "description": "Urgency pattern",
                    "technique": "manufactured_deadline",
                    "real_world_connection": "Common in news",
                },
            ],
            "checklist": [
                {
                    "id": "c1",
                    "description": "Identified urgency",
                    "pattern_refs": ["p1"],
                    "is_mandatory": True,
                },
            ],
            "pass_conditions": {
                "trickster_wins": "Mokinys pasidalino",
                "partial": "Mokinys perskaitė, bet praleido",
                "trickster_loses": "Mokinys atpažino technikas",
            },
        },
        "reveal": {"key_lesson": "Antraštė buvo sukurta skubos jausmui sukelti"},
        "safety": {
            "content_boundaries": ["no_real_harm"],
            "intensity_ceiling": 3,
            "cold_start_safe": True,
        },
    }
    data.update(overrides)
    return data


def _make_cartridge(**overrides: object) -> TaskCartridge:
    """Constructs a minimal valid TaskCartridge for testing."""
    return TaskCartridge.model_validate(_minimal_cartridge_data(**overrides))


# ---------------------------------------------------------------------------
# TaskCartridge — core
# ---------------------------------------------------------------------------


class TestTaskCartridge:
    """TaskCartridge — the top-level cartridge model."""

    def test_valid_construction(self) -> None:
        tc = _make_cartridge()
        assert tc.task_id == "task-test-001"
        assert tc.task_type == "hybrid"
        assert tc.difficulty == 3
        assert tc.time_minutes == 15

    def test_defaults(self) -> None:
        tc = _make_cartridge()
        assert tc.tags == []
        assert tc.status == "active"
        assert tc.prerequisites == []
        assert tc.language == "lt"
        assert tc.available_languages == ["lt"]
        assert tc.presentation_blocks == []
        assert tc.phases == []
        assert tc.ai_config is None

    def test_ai_config_none_accepted(self) -> None:
        tc = _make_cartridge(ai_config=None)
        assert tc.ai_config is None

    def test_ai_config_present(self) -> None:
        tc = _make_cartridge(
            ai_config={
                "model_preference": "standard",
                "prompt_directory": "prompts/tasks/task-test-001",
                "persona_mode": "presenting",
                "has_static_fallback": True,
                "context_requirements": "session_only",
            },
        )
        assert tc.ai_config is not None
        assert tc.ai_config.model_preference == "standard"

    def test_missing_task_id_rejected(self) -> None:
        data = _minimal_cartridge_data()
        del data["task_id"]
        with pytest.raises(ValidationError, match="task_id"):
            TaskCartridge.model_validate(data)

    def test_missing_evaluation_rejected(self) -> None:
        data = _minimal_cartridge_data()
        del data["evaluation"]
        with pytest.raises(ValidationError, match="evaluation"):
            TaskCartridge.model_validate(data)

    def test_missing_reveal_rejected(self) -> None:
        data = _minimal_cartridge_data()
        del data["reveal"]
        with pytest.raises(ValidationError, match="reveal"):
            TaskCartridge.model_validate(data)

    def test_missing_safety_rejected(self) -> None:
        data = _minimal_cartridge_data()
        del data["safety"]
        with pytest.raises(ValidationError, match="safety"):
            TaskCartridge.model_validate(data)

    def test_empty_learning_objectives_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_cartridge(learning_objectives=[])

    def test_time_minutes_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_cartridge(time_minutes=0)

    def test_time_minutes_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_cartridge(time_minutes=-1)

    def test_difficulty_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_cartridge(difficulty=0)
        with pytest.raises(ValidationError):
            _make_cartridge(difficulty=6)

    def test_invalid_task_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_cartridge(task_type="interactive")

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_cartridge(status="deleted")

    def test_frozen(self) -> None:
        tc = _make_cartridge()
        with pytest.raises(ValidationError):
            tc.task_id = "new"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        tc = _make_cartridge()
        assert TaskCartridge.model_validate(tc.model_dump(mode="json")) == tc

    def test_serialization_roundtrip(self) -> None:
        tc = _make_cartridge()
        assert TaskCartridge.model_validate(tc.model_dump()) == tc


# ---------------------------------------------------------------------------
# TaskCartridge — is_clean cross-validation
# ---------------------------------------------------------------------------


class TestTaskCartridgeIsClean:
    """is_clean cross-validation with evaluation.patterns_embedded."""

    def test_clean_true_empty_patterns_ok(self) -> None:
        """Clean task with no patterns — valid, no warning."""
        data = _minimal_cartridge_data(is_clean=True)
        data["evaluation"]["patterns_embedded"] = []
        data["evaluation"]["checklist"] = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tc = TaskCartridge.model_validate(data)
            assert tc.is_clean is True
            taxonomy_warnings = [
                x for x in w if issubclass(x.category, TaxonomyWarning)
            ]
            assert len(taxonomy_warnings) == 0

    def test_clean_true_with_patterns_hard_error(self) -> None:
        """Clean task with embedded patterns — logical contradiction, hard error."""
        data = _minimal_cartridge_data(is_clean=True)
        # patterns_embedded is non-empty in the default data
        with pytest.raises(ValidationError, match="is_clean=True"):
            TaskCartridge.model_validate(data)

    def test_clean_false_empty_patterns_warning(self) -> None:
        """Non-clean task with no patterns — warning (may be draft in progress)."""
        data = _minimal_cartridge_data(is_clean=False)
        data["evaluation"]["patterns_embedded"] = []
        data["evaluation"]["checklist"] = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tc = TaskCartridge.model_validate(data)
            assert tc.is_clean is False
            taxonomy_warnings = [
                x for x in w if issubclass(x.category, TaxonomyWarning)
            ]
            assert len(taxonomy_warnings) == 1
            assert "draft in progress" in str(taxonomy_warnings[0].message)

    def test_clean_false_with_patterns_ok(self) -> None:
        """Non-clean task with patterns — normal case, no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tc = _make_cartridge(is_clean=False)
            assert tc.is_clean is False
            taxonomy_warnings = [
                x for x in w if issubclass(x.category, TaxonomyWarning)
            ]
            assert len(taxonomy_warnings) == 0


# ---------------------------------------------------------------------------
# TaskCartridge — taxonomy validation
# ---------------------------------------------------------------------------


_TEST_TAXONOMY = {
    "triggers": {"urgency", "belonging", "injustice"},
    "techniques": {"headline_manipulation", "cherry_picking", "fabrication"},
    "mediums": {"article", "social_post", "chat"},
}


class TestTaskCartridgeTaxonomy:
    """Taxonomy-aware validation via Pydantic context injection."""

    def test_known_values_no_warning(self) -> None:
        """All taxonomy values known — no warnings emitted."""
        data = _minimal_cartridge_data(
            trigger="urgency",
            technique="headline_manipulation",
            medium="article",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TaskCartridge.model_validate(data, context={"taxonomy": _TEST_TAXONOMY})
            taxonomy_warnings = [
                x for x in w if issubclass(x.category, TaxonomyWarning)
            ]
            assert len(taxonomy_warnings) == 0

    def test_unknown_trigger_warning(self) -> None:
        """Unknown trigger emits TaxonomyWarning."""
        data = _minimal_cartridge_data(trigger="greed")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tc = TaskCartridge.model_validate(
                data, context={"taxonomy": _TEST_TAXONOMY},
            )
            taxonomy_warnings = [
                x for x in w if issubclass(x.category, TaxonomyWarning)
            ]
            # Filter to just trigger warnings (not is_clean warnings)
            trigger_warnings = [
                x for x in taxonomy_warnings if "trigger" in str(x.message).lower()
            ]
            assert len(trigger_warnings) == 1
            assert "greed" in str(trigger_warnings[0].message)
            # Model still constructs successfully
            assert tc.trigger == "greed"

    def test_unknown_technique_warning(self) -> None:
        """Unknown technique emits TaxonomyWarning."""
        data = _minimal_cartridge_data(technique="novel_technique")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TaskCartridge.model_validate(
                data, context={"taxonomy": _TEST_TAXONOMY},
            )
            technique_warnings = [
                x for x in w
                if issubclass(x.category, TaxonomyWarning)
                and "technique" in str(x.message).lower()
            ]
            assert len(technique_warnings) == 1
            assert "novel_technique" in str(technique_warnings[0].message)

    def test_unknown_medium_warning(self) -> None:
        """Unknown medium emits TaxonomyWarning."""
        data = _minimal_cartridge_data(medium="podcast")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TaskCartridge.model_validate(
                data, context={"taxonomy": _TEST_TAXONOMY},
            )
            medium_warnings = [
                x for x in w
                if issubclass(x.category, TaxonomyWarning)
                and "medium" in str(x.message).lower()
            ]
            assert len(medium_warnings) == 1
            assert "podcast" in str(medium_warnings[0].message)

    def test_no_context_no_warnings(self) -> None:
        """Without taxonomy context, no warnings are emitted."""
        data = _minimal_cartridge_data(trigger="unknown_trigger")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tc = TaskCartridge.model_validate(data)
            taxonomy_warnings = [
                x for x in w if issubclass(x.category, TaxonomyWarning)
            ]
            assert len(taxonomy_warnings) == 0
            assert tc.trigger == "unknown_trigger"

    def test_empty_taxonomy_context_no_warnings(self) -> None:
        """With empty taxonomy dict, no warnings (no known values to check against)."""
        data = _minimal_cartridge_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TaskCartridge.model_validate(data, context={"taxonomy": {}})
            taxonomy_warnings = [
                x for x in w if issubclass(x.category, TaxonomyWarning)
            ]
            assert len(taxonomy_warnings) == 0


# ---------------------------------------------------------------------------
# TaskCartridge — with nested content (blocks and phases)
# ---------------------------------------------------------------------------


class TestTaskCartridgeWithContent:
    """TaskCartridge with real blocks and phases — integration-level round-trip."""

    def _make_full_cartridge_data(self) -> dict:
        """Returns a realistic cartridge dict with blocks, phases, and all nested models."""
        return {
            "task_id": "task-clickbait-001",
            "task_type": "hybrid",
            "title": "Paspęsti spąstai",
            "description": "Ar atpažinsite manipuliaciją antraštėje?",
            "version": "1.0",
            "trigger": "urgency",
            "technique": "headline_manipulation",
            "medium": "article",
            "learning_objectives": [
                "Atpažinti emocines manipuliacijas antraštėse",
                "Palyginti antraštę su straipsnio turiniu",
            ],
            "difficulty": 3,
            "time_minutes": 15,
            "is_evergreen": True,
            "is_clean": False,
            "tags": ["urgency", "news"],
            "status": "active",
            "prerequisites": [],
            "language": "lt",
            "available_languages": ["lt"],
            "presentation_blocks": [
                {"id": "b-headline", "type": "text", "text": "SKUBU: naujas atradimas!"},
                {"id": "b-snippet", "type": "text", "text": "Mokslininkai teigia..."},
                {"id": "b-full", "type": "text", "text": "Ilgas straipsnio tekstas..."},
                {
                    "id": "b-image",
                    "type": "image",
                    "src": "assets/graph.png",
                    "alt_text": "Klaidinanti diagrama",
                },
            ],
            "phases": [
                {
                    "id": "phase_intro",
                    "title": "Įvadas",
                    "visible_blocks": ["b-headline", "b-snippet"],
                    "trickster_content": "Pažiūrėk į šią naujieną...",
                    "interaction": {
                        "type": "button",
                        "choices": [
                            {"label": "Dalintis", "target_phase": "phase_bitten",
                             "context_label": "Mokinys pasidalino"},
                            {"label": "Skaityti", "target_phase": "phase_read"},
                        ],
                    },
                },
                {
                    "id": "phase_bitten",
                    "title": "Triksteris laimi",
                    "visible_blocks": ["b-headline"],
                    "trickster_content": "Pasidalinai nepatikrinęs!",
                    "is_terminal": True,
                    "evaluation_outcome": "trickster_wins",
                },
                {
                    "id": "phase_read",
                    "title": "Skaitymas",
                    "visible_blocks": ["b-headline", "b-snippet", "b-full"],
                    "is_ai_phase": True,
                    "interaction": {
                        "type": "freeform",
                        "trickster_opening": "Ką pastebėjai?",
                        "min_exchanges": 2,
                        "max_exchanges": 6,
                    },
                    "ai_transitions": {
                        "on_success": "phase_win",
                        "on_max_exchanges": "phase_timeout",
                        "on_partial": "phase_partial",
                    },
                },
                {
                    "id": "phase_win",
                    "title": "Pergalė",
                    "visible_blocks": ["b-full"],
                    "is_terminal": True,
                    "evaluation_outcome": "trickster_loses",
                },
                {
                    "id": "phase_timeout",
                    "title": "Laikas baigėsi",
                    "visible_blocks": ["b-full"],
                    "is_terminal": True,
                    "evaluation_outcome": "partial",
                },
                {
                    "id": "phase_partial",
                    "title": "Dalinai",
                    "visible_blocks": ["b-full"],
                    "is_terminal": True,
                    "evaluation_outcome": "partial",
                },
            ],
            "initial_phase": "phase_intro",
            "evaluation": {
                "patterns_embedded": [
                    {
                        "id": "p-urgency",
                        "description": "Dirbtinis skubumo jausmas",
                        "technique": "manufactured_deadline",
                        "real_world_connection": "Naujienų portalai naudoja",
                    },
                    {
                        "id": "p-headline",
                        "description": "Antraštė neatitinka turinio",
                        "technique": "headline_manipulation",
                        "real_world_connection": "Clickbait strategija",
                    },
                ],
                "checklist": [
                    {
                        "id": "c-urgency",
                        "description": "Atpažino skubumo signalą",
                        "pattern_refs": ["p-urgency"],
                        "is_mandatory": True,
                    },
                    {
                        "id": "c-headline",
                        "description": "Palygino antraštę su turiniu",
                        "pattern_refs": ["p-headline"],
                        "is_mandatory": False,
                    },
                ],
                "pass_conditions": {
                    "trickster_wins": "Mokinys pasidalino neperskaitęs",
                    "partial": "Mokinys perskaitė, bet praleido raktus",
                    "trickster_loses": "Mokinys atpažino manipuliacijos technikas",
                },
            },
            "ai_config": {
                "model_preference": "standard",
                "prompt_directory": "prompts/tasks/task-clickbait-001",
                "persona_mode": "presenting",
                "has_static_fallback": True,
                "context_requirements": "session_only",
            },
            "reveal": {
                "key_lesson": "Antraštė buvo sukurta tam, kad sukeltų skubumo jausmą.",
                "additional_resources": [],
            },
            "safety": {
                "content_boundaries": ["no_real_harm", "no_violence"],
                "intensity_ceiling": 3,
                "cold_start_safe": True,
            },
        }

    def test_full_cartridge_construction(self) -> None:
        """A realistic cartridge with all nested models constructs correctly."""
        data = self._make_full_cartridge_data()
        tc = TaskCartridge.model_validate(data)

        assert tc.task_id == "task-clickbait-001"
        assert len(tc.presentation_blocks) == 4
        assert len(tc.phases) == 6
        assert len(tc.evaluation.patterns_embedded) == 2
        assert len(tc.evaluation.checklist) == 2
        assert tc.ai_config is not None
        assert tc.ai_config.model_preference == "standard"

    def test_full_cartridge_json_roundtrip(self) -> None:
        """The acid test — full cartridge survives JSON serialization and back."""
        data = self._make_full_cartridge_data()
        tc = TaskCartridge.model_validate(data)
        restored = TaskCartridge.model_validate(tc.model_dump(mode="json"))
        assert restored == tc

    def test_full_cartridge_dict_roundtrip(self) -> None:
        """Full cartridge survives Python dict serialization and back."""
        data = self._make_full_cartridge_data()
        tc = TaskCartridge.model_validate(data)
        restored = TaskCartridge.model_validate(tc.model_dump())
        assert restored == tc

    def test_blocks_routed_correctly(self) -> None:
        """Presentation blocks are routed to their correct model types."""
        data = self._make_full_cartridge_data()
        tc = TaskCartridge.model_validate(data)

        from backend.tasks.schemas import TextBlock, ImageBlock
        text_blocks = [b for b in tc.presentation_blocks if isinstance(b, TextBlock)]
        image_blocks = [b for b in tc.presentation_blocks if isinstance(b, ImageBlock)]
        assert len(text_blocks) == 3
        assert len(image_blocks) == 1

    def test_phases_have_correct_interactions(self) -> None:
        """Phase interaction types are correctly routed."""
        data = self._make_full_cartridge_data()
        tc = TaskCartridge.model_validate(data)

        intro = next(p for p in tc.phases if p.id == "phase_intro")
        assert isinstance(intro.interaction, ButtonInteraction)

        read = next(p for p in tc.phases if p.id == "phase_read")
        assert isinstance(read.interaction, FreeformInteraction)
        assert read.ai_transitions is not None

    def test_cartridge_with_unknown_block_type(self) -> None:
        """Unknown block types route to GenericBlock within a full cartridge."""
        data = self._make_full_cartridge_data()
        data["presentation_blocks"].append({
            "id": "b-custom",
            "type": "timeline",
            "events": [{"date": "2024-01-01", "label": "Event"}],
        })
        tc = TaskCartridge.model_validate(data)
        assert len(tc.presentation_blocks) == 5
        last = tc.presentation_blocks[-1]
        assert isinstance(last, GenericBlock)
        assert last.type == "timeline"
