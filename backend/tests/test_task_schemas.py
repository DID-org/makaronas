"""Tests for backend.tasks.schemas — Task cartridge base types, presentation blocks,
interaction types, and phase model."""

from typing import Literal

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from backend.tasks.schemas import (
    AiTransitions,
    AudioBlock,
    ButtonChoice,
    ButtonInteraction,
    ChatMessageBlock,
    Difficulty,
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
    PersonaMode,
    Phase,
    PresentationBlock,
    SearchResultBlock,
    SocialPostBlock,
    TaskStatus,
    TaskType,
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
