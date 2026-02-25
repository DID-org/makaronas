"""Tests for backend.tasks.schemas — Task cartridge base types and presentation blocks."""

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from backend.tasks.schemas import (
    AudioBlock,
    ChatMessageBlock,
    Difficulty,
    GenericBlock,
    ImageBlock,
    KNOWN_BLOCK_TYPES,
    MemeBlock,
    ModelPreference,
    PersonaMode,
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
