"""Tests for the JSON Schema authoring contract.

Validates that:
1. The schema file itself is valid JSON Schema (Draft 2020-12)
2. All reference cartridges pass schema validation
3. The template passes schema validation
4. Known error cases are rejected
5. Unknown block/interaction types pass (open type system)
"""

from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError, validate

from backend.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCHEMA_PATH = PROJECT_ROOT / "content" / "tasks" / "task.schema.json"
TEMPLATE_PATH = PROJECT_ROOT / "content" / "tasks" / "TEMPLATE" / "task.json"
CARTRIDGES_DIR = PROJECT_ROOT / "content" / "tasks"


@pytest.fixture(scope="module")
def schema() -> dict:
    """Loads the JSON Schema once per module."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def template_data() -> dict:
    """Loads the template cartridge once per module."""
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


def _discover_cartridges() -> list[Path]:
    """Finds all reference cartridge task.json files dynamically."""
    return sorted(
        p
        for p in CARTRIDGES_DIR.glob("task-*/task.json")
        if p.parent.name != "TEMPLATE"
    )


# ---------------------------------------------------------------------------
# 1. Schema self-validation
# ---------------------------------------------------------------------------


class TestSchemaSelfValidation:
    """Verifies the schema file is valid JSON Schema."""

    def test_schema_is_valid_json(self) -> None:
        """Schema file parses as valid JSON."""
        data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_schema_is_valid_draft_2020_12(self, schema: dict) -> None:
        """Schema conforms to Draft 2020-12 meta-schema."""
        Draft202012Validator.check_schema(schema)

    def test_schema_has_defs(self, schema: dict) -> None:
        """Schema has $defs section with expected type definitions."""
        assert "$defs" in schema
        expected_defs = [
            "TextBlock",
            "ImageBlock",
            "AudioBlock",
            "VideoTranscriptBlock",
            "MemeBlock",
            "ChatMessageBlock",
            "SocialPostBlock",
            "SearchResultBlock",
            "GenericBlock",
            "PresentationBlock",
            "ButtonChoice",
            "ButtonInteraction",
            "FreeformInteraction",
            "InvestigationInteraction",
            "GenericInteraction",
            "InteractionConfig",
            "AiTransitions",
            "Phase",
            "EmbeddedPattern",
            "ChecklistItem",
            "PassConditions",
            "EvaluationContract",
            "AiConfig",
            "RevealContent",
            "SafetyConfig",
        ]
        for name in expected_defs:
            assert name in schema["$defs"], f"Missing $defs/{name}"


# ---------------------------------------------------------------------------
# 2. Reference cartridge validation
# ---------------------------------------------------------------------------


class TestReferenceCartridges:
    """All 6 reference cartridges must pass schema validation."""

    @pytest.mark.parametrize(
        "cartridge_path",
        _discover_cartridges(),
        ids=lambda p: p.parent.name,
    )
    def test_cartridge_validates(
        self, schema: dict, cartridge_path: Path,
    ) -> None:
        """Each reference cartridge passes schema validation."""
        data = json.loads(cartridge_path.read_text(encoding="utf-8"))
        validate(instance=data, schema=schema)

    def test_at_least_6_cartridges_discovered(self) -> None:
        """Sanity check: we should find at least 6 reference cartridges."""
        cartridges = _discover_cartridges()
        assert len(cartridges) >= 6, (
            f"Expected >= 6 cartridges, found {len(cartridges)}: "
            f"{[p.parent.name for p in cartridges]}"
        )


# ---------------------------------------------------------------------------
# 3. Template validation
# ---------------------------------------------------------------------------


class TestTemplateValidation:
    """The starter template must pass schema validation."""

    def test_template_validates(
        self, schema: dict, template_data: dict,
    ) -> None:
        """Template passes schema validation."""
        validate(instance=template_data, schema=schema)

    def test_template_is_draft_status(self, template_data: dict) -> None:
        """Template starts with draft status."""
        assert template_data["status"] == "draft"

    def test_template_is_hybrid(self, template_data: dict) -> None:
        """Template uses the hybrid task type."""
        assert template_data["task_type"] == "hybrid"

    def test_template_has_terminal_phases(self, template_data: dict) -> None:
        """Template has at least one terminal phase."""
        terminals = [p for p in template_data["phases"] if p.get("is_terminal")]
        assert len(terminals) >= 1


# ---------------------------------------------------------------------------
# 4. Error case validation
# ---------------------------------------------------------------------------


class TestErrorCases:
    """Known-invalid structures must be rejected by the schema."""

    @pytest.fixture
    def valid_cartridge(self) -> dict:
        """Returns a copy of the first reference cartridge as a base."""
        cartridges = _discover_cartridges()
        assert cartridges, "No reference cartridges found"
        return json.loads(cartridges[0].read_text(encoding="utf-8"))

    def test_missing_task_id(self, schema: dict, valid_cartridge: dict) -> None:
        """Missing task_id should fail validation."""
        bad = copy.deepcopy(valid_cartridge)
        del bad["task_id"]
        with pytest.raises(ValidationError):
            validate(instance=bad, schema=schema)

    def test_missing_evaluation(self, schema: dict, valid_cartridge: dict) -> None:
        """Missing evaluation should fail validation."""
        bad = copy.deepcopy(valid_cartridge)
        del bad["evaluation"]
        with pytest.raises(ValidationError):
            validate(instance=bad, schema=schema)

    def test_difficulty_below_minimum(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """difficulty=0 should fail (minimum is 1)."""
        bad = copy.deepcopy(valid_cartridge)
        bad["difficulty"] = 0
        with pytest.raises(ValidationError):
            validate(instance=bad, schema=schema)

    def test_difficulty_above_maximum(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """difficulty=6 should fail (maximum is 5)."""
        bad = copy.deepcopy(valid_cartridge)
        bad["difficulty"] = 6
        with pytest.raises(ValidationError):
            validate(instance=bad, schema=schema)

    def test_empty_learning_objectives(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """Empty learning_objectives should fail (minItems: 1)."""
        bad = copy.deepcopy(valid_cartridge)
        bad["learning_objectives"] = []
        with pytest.raises(ValidationError):
            validate(instance=bad, schema=schema)

    def test_time_minutes_zero(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """time_minutes=0 should fail (exclusiveMinimum: 0)."""
        bad = copy.deepcopy(valid_cartridge)
        bad["time_minutes"] = 0
        with pytest.raises(ValidationError):
            validate(instance=bad, schema=schema)

    def test_invalid_task_type(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """Invalid task_type should fail."""
        bad = copy.deepcopy(valid_cartridge)
        bad["task_type"] = "interactive"
        with pytest.raises(ValidationError):
            validate(instance=bad, schema=schema)

    def test_invalid_status(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """Invalid status should fail."""
        bad = copy.deepcopy(valid_cartridge)
        bad["status"] = "published"
        with pytest.raises(ValidationError):
            validate(instance=bad, schema=schema)

    def test_intensity_ceiling_out_of_range(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """intensity_ceiling=6 should fail (maximum is 5)."""
        bad = copy.deepcopy(valid_cartridge)
        bad["safety"]["intensity_ceiling"] = 6
        with pytest.raises(ValidationError):
            validate(instance=bad, schema=schema)


# ---------------------------------------------------------------------------
# 5. Open type acceptance
# ---------------------------------------------------------------------------


class TestOpenTypeAcceptance:
    """Unknown block and interaction types must NOT cause validation failure."""

    @pytest.fixture
    def valid_cartridge(self) -> dict:
        """Returns a copy of the first reference cartridge."""
        cartridges = _discover_cartridges()
        assert cartridges, "No reference cartridges found"
        return json.loads(cartridges[0].read_text(encoding="utf-8"))

    def test_unknown_block_type_accepted(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """A block with an unknown type should pass via GenericBlock fallback."""
        data = copy.deepcopy(valid_cartridge)
        data["presentation_blocks"].append({
            "id": "hologram-1",
            "type": "hologram",
            "intensity": 5,
            "color": "blue",
        })
        validate(instance=data, schema=schema)

    def test_unknown_interaction_type_accepted(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """An interaction with an unknown type should pass via GenericInteraction."""
        data = copy.deepcopy(valid_cartridge)
        # Find a non-terminal phase and replace its interaction
        for phase in data["phases"]:
            if not phase.get("is_terminal", False) and phase.get("interaction"):
                phase["interaction"] = {
                    "type": "timeline_scrub",
                    "start": 0,
                    "end": 100,
                }
                break
        validate(instance=data, schema=schema)

    def test_unknown_taxonomy_values_accepted(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """Unknown trigger/technique/medium should pass (open taxonomy)."""
        data = copy.deepcopy(valid_cartridge)
        data["trigger"] = "nostalgia"
        data["technique"] = "appeal_to_tradition"
        data["medium"] = "hologram_display"
        validate(instance=data, schema=schema)

    def test_known_block_with_typo_rejected(
        self, schema: dict, valid_cartridge: dict,
    ) -> None:
        """A known block type with a typo in a field should be rejected.

        This tests that additionalProperties: false on known types catches typos.
        """
        data = copy.deepcopy(valid_cartridge)
        data["presentation_blocks"].append({
            "id": "bad-text",
            "type": "text",
            "textt": "This field name is wrong",
        })
        with pytest.raises(ValidationError):
            validate(instance=data, schema=schema)
