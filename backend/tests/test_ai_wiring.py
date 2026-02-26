"""Tests for Phase 6a: Service Wiring & Startup.

Covers:
- Provider factory routing (create_provider)
- DI wiring (get_prompt_loader, get_trickster_engine, 503 before init)
- Startup checks (API key warnings, prompt enforcement errors)
- Usage logging (log_ai_call structured output)
- Composer/Evaluator stub importability and interface shape
- Graceful degradation (check_ai_readiness)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.ai.providers.base import AIProvider
from backend.models import ModelConfig


# ---------------------------------------------------------------------------
# Helper: write prompt file
# ---------------------------------------------------------------------------


def _write(path: Path, content: str = "Test prompt content.") -> None:
    """Creates a file at path with the given content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _setup_base_prompts(prompts_dir: Path) -> None:
    """Creates the three mandatory base prompt files."""
    trickster = prompts_dir / "trickster"
    _write(trickster / "persona_base.md")
    _write(trickster / "behaviour_base.md")
    _write(trickster / "safety_base.md")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeSettings:
    """Minimal Settings-like object for tests."""

    google_api_key: str = "test-google-key"
    anthropic_api_key: str = "test-anthropic-key"
    app_env: str = "test"
    app_port: int = 8000
    log_level: str = "info"
    cors_origins: list = None
    default_language: str = "lt"
    supported_languages: list = None

    def __post_init__(self) -> None:
        # Frozen dataclass can't set defaults for mutable fields in __init__,
        # but we only read these in tests so None is fine.
        pass


@pytest.fixture
def settings_with_keys():
    """Returns FakeSettings with both API keys set."""
    return FakeSettings()


@pytest.fixture
def settings_no_google():
    """Returns FakeSettings with empty google_api_key."""
    return FakeSettings(google_api_key="")


@pytest.fixture
def settings_no_keys():
    """Returns FakeSettings with no API keys."""
    return FakeSettings(google_api_key="", anthropic_api_key="")


# ---------------------------------------------------------------------------
# Provider Factory Tests
# ---------------------------------------------------------------------------


_has_genai = True
try:
    from google import genai  # noqa: F401
except ModuleNotFoundError:
    _has_genai = False

_has_anthropic = True
try:
    import anthropic  # noqa: F401
except ModuleNotFoundError:
    _has_anthropic = False


class TestCreateProvider:
    """Tests for deps.create_provider() factory routing."""

    @pytest.mark.skipif(not _has_genai, reason="google-genai SDK not installed")
    def test_gemini_returns_gemini_provider(self, settings_with_keys):
        from backend.ai.providers.gemini import GeminiProvider
        from backend.api.deps import create_provider

        config = ModelConfig(provider="gemini", model_id="test-model")
        provider = create_provider(config, settings_with_keys)
        assert isinstance(provider, GeminiProvider)
        assert isinstance(provider, AIProvider)

    @pytest.mark.skipif(not _has_anthropic, reason="anthropic SDK not installed")
    def test_anthropic_returns_anthropic_provider(self, settings_with_keys):
        from backend.ai.providers.anthropic import AnthropicProvider
        from backend.api.deps import create_provider

        config = ModelConfig(provider="anthropic", model_id="test-model")
        provider = create_provider(config, settings_with_keys)
        assert isinstance(provider, AnthropicProvider)
        assert isinstance(provider, AIProvider)

    def test_unknown_provider_raises_value_error(self, settings_with_keys):
        from backend.api.deps import create_provider

        config = ModelConfig(provider="openai", model_id="test-model")
        with pytest.raises(ValueError, match="Unknown provider.*openai"):
            create_provider(config, settings_with_keys)

    @pytest.mark.skipif(not _has_genai, reason="google-genai SDK not installed")
    def test_empty_api_key_still_constructs(self, settings_no_google):
        """Empty API key doesn't crash at construction (lazy client)."""
        from backend.ai.providers.gemini import GeminiProvider
        from backend.api.deps import create_provider

        config = ModelConfig(provider="gemini", model_id="test-model")
        provider = create_provider(config, settings_no_google)
        assert isinstance(provider, GeminiProvider)

    def test_gemini_routing_without_sdk(self, settings_with_keys):
        """create_provider routes 'gemini' correctly (may fail on import)."""
        from backend.api.deps import create_provider

        config = ModelConfig(provider="gemini", model_id="test-model")
        if _has_genai:
            provider = create_provider(config, settings_with_keys)
            assert isinstance(provider, AIProvider)
        else:
            # Without SDK, import will fail — that's expected
            with pytest.raises((ModuleNotFoundError, ImportError)):
                create_provider(config, settings_with_keys)


# ---------------------------------------------------------------------------
# DI Wiring Tests
# ---------------------------------------------------------------------------


class TestDIWiring:
    """Tests for AI dependency providers (503 pattern, singleton access)."""

    def test_get_prompt_loader_returns_503_before_init(self):
        from backend.api import deps

        original = deps._prompt_loader
        try:
            deps._prompt_loader = None
            with pytest.raises(HTTPException) as exc_info:
                deps.get_prompt_loader()
            assert exc_info.value.status_code == 503
            assert "Prompt loader" in str(exc_info.value.detail)
        finally:
            deps._prompt_loader = original

    def test_get_trickster_engine_returns_503_before_init(self):
        from backend.api import deps

        original = deps._trickster_engine
        try:
            deps._trickster_engine = None
            with pytest.raises(HTTPException) as exc_info:
                deps.get_trickster_engine()
            assert exc_info.value.status_code == 503
            assert "Trickster engine" in str(exc_info.value.detail)
        finally:
            deps._trickster_engine = original

    def test_get_prompt_loader_returns_instance_after_init(self, tmp_path):
        from backend.ai.prompts import PromptLoader
        from backend.api import deps

        original = deps._prompt_loader
        try:
            loader = PromptLoader(tmp_path)
            deps._prompt_loader = loader
            result = deps.get_prompt_loader()
            assert result is loader
            assert isinstance(result, PromptLoader)
        finally:
            deps._prompt_loader = original

    def test_get_trickster_engine_returns_instance_after_init(self, tmp_path, mock_provider):
        from backend.ai.context import ContextManager
        from backend.ai.prompts import PromptLoader
        from backend.ai.trickster import TricksterEngine
        from backend.api import deps

        original = deps._trickster_engine
        try:
            loader = PromptLoader(tmp_path)
            cm = ContextManager(loader)
            engine = TricksterEngine(mock_provider(), cm)
            deps._trickster_engine = engine
            result = deps.get_trickster_engine()
            assert result is engine
            assert isinstance(result, TricksterEngine)
        finally:
            deps._trickster_engine = original


# ---------------------------------------------------------------------------
# Startup Checks Tests
# ---------------------------------------------------------------------------


class TestStartupAPIKeyCheck:
    """Tests for _check_api_keys() warning on missing keys."""

    def test_missing_google_key_logs_warning(self, settings_no_google, caplog):
        from backend.main import _check_api_keys
        from backend.models import TIER_MAP

        with caplog.at_level(logging.WARNING, logger="makaronas"):
            _check_api_keys(settings_no_google, TIER_MAP)

        assert any("GOOGLE_API_KEY" in r.message for r in caplog.records)
        assert any("gemini" in r.message for r in caplog.records)

    def test_all_keys_present_no_warning(self, settings_with_keys, caplog):
        from backend.main import _check_api_keys
        from backend.models import TIER_MAP

        with caplog.at_level(logging.WARNING, logger="makaronas"):
            _check_api_keys(settings_with_keys, TIER_MAP)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) == 0

    def test_anthropic_key_warning_when_tier_map_uses_anthropic(self, settings_no_keys, caplog):
        """If TIER_MAP references anthropic, missing key logs warning."""
        from backend.main import _check_api_keys

        custom_tier_map = {
            "fast": ModelConfig(provider="anthropic", model_id="test"),
        }
        with caplog.at_level(logging.WARNING, logger="makaronas"):
            _check_api_keys(settings_no_keys, custom_tier_map)

        assert any("ANTHROPIC_API_KEY" in r.message for r in caplog.records)


class TestStartupPromptEnforcement:
    """Tests for _check_prompt_enforcement() error logging."""

    def test_missing_prompts_log_error(self, tmp_path, make_cartridge, caplog):
        """Active AI cartridge with no prompt files → ERROR logged."""
        from backend.ai.prompts import PromptLoader
        from backend.api import deps
        from backend.main import _check_prompt_enforcement
        from backend.tasks.registry import TaskRegistry

        # Set up a registry with one AI cartridge (no prompt files on disk)
        cartridge = make_cartridge()
        registry = TaskRegistry(Path("/tmp"), Path("/tmp"))
        registry._by_id[cartridge.task_id] = cartridge
        registry._by_status.setdefault(cartridge.status, set()).add(cartridge.task_id)

        original_registry = deps._task_registry
        original_loader = deps._prompt_loader
        try:
            deps._task_registry = registry
            deps._prompt_loader = PromptLoader(tmp_path)  # Empty prompts dir

            with caplog.at_level(logging.ERROR, logger="makaronas"):
                _check_prompt_enforcement(deps)

            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert len(error_records) > 0
            assert any("Prompt enforcement" in r.message for r in error_records)
        finally:
            deps._task_registry = original_registry
            deps._prompt_loader = original_loader

    def test_static_only_cartridge_no_errors(self, tmp_path, caplog):
        """Static-only cartridge (no AI phases) → no prompt errors."""
        from backend.ai.prompts import PromptLoader
        from backend.api import deps
        from backend.main import _check_prompt_enforcement
        from backend.tasks.registry import TaskRegistry
        from backend.tasks.schemas import TaskCartridge

        # Build a static-only cartridge (no ai_config)
        data = {
            "task_id": "test-static-001",
            "task_type": "static",
            "title": "Statinis testas",
            "description": "Testas be AI",
            "version": "1.0",
            "trigger": "urgency",
            "technique": "headline_manipulation",
            "medium": "article",
            "learning_objectives": ["Test"],
            "difficulty": 1,
            "time_minutes": 5,
            "is_evergreen": True,
            "is_clean": False,
            "initial_phase": "p1",
            "phases": [
                {"id": "p1", "title": "Phase 1", "is_terminal": True},
            ],
            "evaluation": {
                "patterns_embedded": [],
                "checklist": [],
                "pass_conditions": {
                    "trickster_wins": "N/A",
                    "partial": "N/A",
                    "trickster_loses": "N/A",
                },
            },
            "safety": {
                "content_boundaries": [],
                "intensity_ceiling": 1,
                "cold_start_safe": True,
            },
            "reveal": {
                "key_lesson": "Test lesson",
            },
        }
        cartridge = TaskCartridge.model_validate(data)
        registry = TaskRegistry(Path("/tmp"), Path("/tmp"))
        registry._by_id[cartridge.task_id] = cartridge
        registry._by_status.setdefault(cartridge.status, set()).add(cartridge.task_id)

        original_registry = deps._task_registry
        original_loader = deps._prompt_loader
        try:
            deps._task_registry = registry
            deps._prompt_loader = PromptLoader(tmp_path)

            with caplog.at_level(logging.ERROR, logger="makaronas"):
                _check_prompt_enforcement(deps)

            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert len(error_records) == 0
        finally:
            deps._task_registry = original_registry
            deps._prompt_loader = original_loader

    def test_no_crash_when_registry_none(self, caplog):
        """No crash when registry is None (startup incomplete)."""
        from backend.api import deps
        from backend.main import _check_prompt_enforcement

        original_registry = deps._task_registry
        original_loader = deps._prompt_loader
        try:
            deps._task_registry = None
            deps._prompt_loader = None

            with caplog.at_level(logging.ERROR, logger="makaronas"):
                _check_prompt_enforcement(deps)

            # Should silently skip — no errors, no crashes
            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert len(error_records) == 0
        finally:
            deps._task_registry = original_registry
            deps._prompt_loader = original_loader


# ---------------------------------------------------------------------------
# Init AI Services Integration Test
# ---------------------------------------------------------------------------


class TestInitAIServices:
    """Tests for _init_ai_services() full wiring."""

    def test_init_sets_prompt_loader(self, tmp_path, monkeypatch):
        """After _init_ai_services(), prompt_loader is always set."""
        from backend.api import deps
        from backend.main import _init_ai_services

        # Save originals
        orig_loader = deps._prompt_loader
        orig_engine = deps._trickster_engine
        orig_registry = deps._task_registry

        try:
            # Reset singletons
            deps._prompt_loader = None
            deps._trickster_engine = None

            # Provide a registry so prompt enforcement runs
            from backend.tasks.registry import TaskRegistry

            deps._task_registry = TaskRegistry(Path("/tmp"), Path("/tmp"))

            # Patch PROJECT_ROOT and settings
            import backend.config as config_module

            monkeypatch.setattr(config_module, "_settings", None)
            monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
            monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

            # Set up prompts at PROJECT_ROOT / "prompts"
            prompts_parent = tmp_path
            actual_prompts = prompts_parent / "prompts"
            actual_prompts.mkdir(exist_ok=True)
            _setup_base_prompts(actual_prompts)
            monkeypatch.setattr("backend.config.PROJECT_ROOT", prompts_parent)

            _init_ai_services()

            from backend.ai.prompts import PromptLoader

            assert isinstance(deps._prompt_loader, PromptLoader)
            # Engine may or may not be set depending on SDK availability
            # (google-genai may not be installed in test env)
        finally:
            deps._prompt_loader = orig_loader
            deps._trickster_engine = orig_engine
            deps._task_registry = orig_registry
            monkeypatch.setattr(config_module, "_settings", None)

    @pytest.mark.skipif(not _has_genai, reason="google-genai SDK not installed")
    def test_init_sets_engine_with_sdk(self, tmp_path, monkeypatch):
        """With SDK available, engine is set after _init_ai_services()."""
        from backend.api import deps
        from backend.main import _init_ai_services

        orig_loader = deps._prompt_loader
        orig_engine = deps._trickster_engine
        orig_registry = deps._task_registry

        try:
            deps._prompt_loader = None
            deps._trickster_engine = None

            from backend.tasks.registry import TaskRegistry

            deps._task_registry = TaskRegistry(Path("/tmp"), Path("/tmp"))

            import backend.config as config_module

            monkeypatch.setattr(config_module, "_settings", None)
            monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
            monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

            prompts_parent = tmp_path
            actual_prompts = prompts_parent / "prompts"
            actual_prompts.mkdir(exist_ok=True)
            _setup_base_prompts(actual_prompts)
            monkeypatch.setattr("backend.config.PROJECT_ROOT", prompts_parent)

            _init_ai_services()

            from backend.ai.trickster import TricksterEngine

            assert isinstance(deps._trickster_engine, TricksterEngine)
        finally:
            deps._prompt_loader = orig_loader
            deps._trickster_engine = orig_engine
            deps._task_registry = orig_registry
            monkeypatch.setattr(config_module, "_settings", None)

    def test_init_no_crash_without_sdk(self, tmp_path, monkeypatch):
        """_init_ai_services() logs warning but doesn't crash if SDK unavailable."""
        from backend.api import deps
        from backend.main import _init_ai_services

        orig_loader = deps._prompt_loader
        orig_engine = deps._trickster_engine
        orig_registry = deps._task_registry

        try:
            deps._prompt_loader = None
            deps._trickster_engine = None

            from backend.tasks.registry import TaskRegistry

            deps._task_registry = TaskRegistry(Path("/tmp"), Path("/tmp"))

            import backend.config as config_module

            monkeypatch.setattr(config_module, "_settings", None)
            monkeypatch.setenv("GOOGLE_API_KEY", "")
            monkeypatch.setenv("ANTHROPIC_API_KEY", "")

            prompts_parent = tmp_path
            actual_prompts = prompts_parent / "prompts"
            actual_prompts.mkdir(exist_ok=True)
            _setup_base_prompts(actual_prompts)
            monkeypatch.setattr("backend.config.PROJECT_ROOT", prompts_parent)

            # Should not crash regardless of SDK availability
            _init_ai_services()

            from backend.ai.prompts import PromptLoader

            # Prompt loader is always set
            assert isinstance(deps._prompt_loader, PromptLoader)
        finally:
            deps._prompt_loader = orig_loader
            deps._trickster_engine = orig_engine
            deps._task_registry = orig_registry
            monkeypatch.setattr(config_module, "_settings", None)


# ---------------------------------------------------------------------------
# Usage Logging Tests
# ---------------------------------------------------------------------------


class TestUsageLogging:
    """Tests for log_ai_call() structured output."""

    def test_log_ai_call_emits_info_with_all_fields(self, caplog):
        from backend.ai.usage import log_ai_call

        with caplog.at_level(logging.INFO, logger="makaronas.ai.usage"):
            log_ai_call(
                model_id="gemini-3-flash-preview",
                prompt_tokens=100,
                completion_tokens=50,
                latency_ms=234.5,
                task_id="task-clickbait-trap-001",
                session_id="session-abc123",
                call_type="trickster",
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.INFO
        assert record.name == "makaronas.ai.usage"

        # Check message contains key info
        assert "trickster" in record.message
        assert "gemini-3-flash-preview" in record.message
        assert "100" in record.message
        assert "50" in record.message

    def test_log_ai_call_extra_fields_machine_parseable(self, caplog):
        from backend.ai.usage import log_ai_call

        with caplog.at_level(logging.INFO, logger="makaronas.ai.usage"):
            log_ai_call(
                model_id="test-model",
                prompt_tokens=200,
                completion_tokens=75,
                latency_ms=500.0,
                task_id="task-001",
                session_id="session-001",
                call_type="debrief",
            )

        record = caplog.records[0]
        # extra fields are set on the LogRecord
        assert record.model_id == "test-model"
        assert record.prompt_tokens == 200
        assert record.completion_tokens == 75
        assert record.latency_ms == 500.0
        assert record.task_id == "task-001"
        assert record.session_id == "session-001"
        assert record.call_type == "debrief"

    def test_logger_name_is_correct(self):
        from backend.ai.usage import logger

        assert logger.name == "makaronas.ai.usage"


# ---------------------------------------------------------------------------
# Composer/Evaluator Stub Tests
# ---------------------------------------------------------------------------


class TestComposerStub:
    """Tests that Composer stub is importable and has expected interface."""

    def test_import_composer(self):
        from backend.ai.composer import Composer

        assert Composer is not None

    def test_constructor_accepts_provider(self, mock_provider):
        from backend.ai.composer import Composer

        composer = Composer(provider=mock_provider())
        assert composer._provider is not None

    @pytest.mark.asyncio
    async def test_suggest_raises_not_implemented(self, mock_provider):
        from backend.ai.composer import Composer

        composer = Composer(provider=mock_provider())
        with pytest.raises(NotImplementedError, match="V8"):
            async for _ in composer.suggest(query="test"):
                pass

    @pytest.mark.asyncio
    async def test_explain_raises_not_implemented(self, mock_provider):
        from backend.ai.composer import Composer

        composer = Composer(provider=mock_provider())
        with pytest.raises(NotImplementedError, match="V8"):
            async for _ in composer.explain(task_id="test"):
                pass


class TestEvaluatorStub:
    """Tests that Evaluator stub is importable and has expected interface."""

    def test_import_evaluator(self):
        from backend.ai.evaluator import Evaluator

        assert Evaluator is not None

    def test_import_evaluation_result(self):
        from backend.ai.evaluator import EvaluationResult

        result = EvaluationResult(
            outcome="partial",
            summary="Test summary",
            details={"patterns": []},
        )
        assert result.outcome == "partial"

    def test_constructor_accepts_provider(self, mock_provider):
        from backend.ai.evaluator import Evaluator

        evaluator = Evaluator(provider=mock_provider())
        assert evaluator._provider is not None

    @pytest.mark.asyncio
    async def test_evaluate_raises_not_implemented(self, mock_provider):
        from backend.ai.evaluator import Evaluator

        evaluator = Evaluator(provider=mock_provider())
        with pytest.raises(NotImplementedError, match="V6"):
            await evaluator.evaluate(
                exchanges=[],
                evaluation_contract={},
                task_id="test",
            )


# ---------------------------------------------------------------------------
# Graceful Degradation Tests
# ---------------------------------------------------------------------------


class TestCheckAIReadiness:
    """Tests for check_ai_readiness() degradation helper."""

    def test_static_only_cartridge_returns_empty(self, settings_with_keys):
        """Static-only cartridge (no ai_config) → no issues."""
        from backend.api.deps import check_ai_readiness
        from backend.tasks.schemas import TaskCartridge

        data = {
            "task_id": "test-static-001",
            "task_type": "static",
            "title": "Test",
            "description": "Test",
            "version": "1.0",
            "trigger": "urgency",
            "technique": "headline_manipulation",
            "medium": "article",
            "learning_objectives": ["Test"],
            "difficulty": 1,
            "time_minutes": 5,
            "is_evergreen": True,
            "is_clean": False,
            "initial_phase": "p1",
            "phases": [{"id": "p1", "title": "Phase 1", "is_terminal": True}],
            "evaluation": {
                "patterns_embedded": [],
                "checklist": [],
                "pass_conditions": {
                    "trickster_wins": "N/A",
                    "partial": "N/A",
                    "trickster_loses": "N/A",
                },
            },
            "safety": {
                "content_boundaries": [],
                "intensity_ceiling": 1,
                "cold_start_safe": True,
            },
            "reveal": {
                "key_lesson": "Test lesson",
            },
        }
        cartridge = TaskCartridge.model_validate(data)
        issues = check_ai_readiness(cartridge, settings_with_keys)
        assert issues == []

    def test_ai_cartridge_missing_key_reports_issue(
        self, make_cartridge, settings_no_google
    ):
        """AI cartridge with missing API key → reports issue."""
        from backend.api import deps
        from backend.api.deps import check_ai_readiness

        # Need prompt_loader set for full check
        original = deps._prompt_loader
        try:
            from backend.ai.prompts import PromptLoader

            deps._prompt_loader = PromptLoader(Path("/tmp/nonexistent"))

            cartridge = make_cartridge()
            issues = check_ai_readiness(cartridge, settings_no_google)
            assert any("Missing API key" in issue for issue in issues)
        finally:
            deps._prompt_loader = original

    def test_ai_cartridge_missing_prompts_reports_issue(
        self, tmp_path, make_cartridge, settings_with_keys
    ):
        """AI cartridge with missing prompt files → reports issue."""
        from backend.ai.prompts import PromptLoader
        from backend.api import deps
        from backend.api.deps import check_ai_readiness

        original = deps._prompt_loader
        try:
            deps._prompt_loader = PromptLoader(tmp_path)  # Empty dir

            cartridge = make_cartridge()
            issues = check_ai_readiness(cartridge, settings_with_keys)
            # Should have prompt-related issues
            assert len(issues) > 0
            assert any("prompt" in issue.lower() or "missing" in issue.lower() for issue in issues)
        finally:
            deps._prompt_loader = original

    def test_ai_cartridge_all_ready_returns_empty(
        self, tmp_path, make_cartridge, settings_with_keys
    ):
        """AI cartridge with all requirements met → empty list."""
        from backend.ai.prompts import PromptLoader
        from backend.api import deps
        from backend.api.deps import check_ai_readiness

        # Set up valid prompts
        _setup_base_prompts(tmp_path)

        original = deps._prompt_loader
        try:
            deps._prompt_loader = PromptLoader(tmp_path)

            cartridge = make_cartridge()
            issues = check_ai_readiness(cartridge, settings_with_keys)
            assert issues == []
        finally:
            deps._prompt_loader = original

    def test_prompt_loader_not_initialized_reports_issue(
        self, make_cartridge, settings_with_keys
    ):
        """When prompt loader is None → reports issue."""
        from backend.api import deps
        from backend.api.deps import check_ai_readiness

        original = deps._prompt_loader
        try:
            deps._prompt_loader = None
            cartridge = make_cartridge()
            issues = check_ai_readiness(cartridge, settings_with_keys)
            assert any("Prompt loader not initialized" in issue for issue in issues)
        finally:
            deps._prompt_loader = original


# ---------------------------------------------------------------------------
# GetApiKeyForProvider Tests
# ---------------------------------------------------------------------------


class TestGetApiKeyForProvider:
    """Tests for _get_api_key_for_provider helper."""

    def test_gemini_returns_google_key(self, settings_with_keys):
        from backend.api.deps import _get_api_key_for_provider

        assert _get_api_key_for_provider("gemini", settings_with_keys) == "test-google-key"

    def test_anthropic_returns_anthropic_key(self, settings_with_keys):
        from backend.api.deps import _get_api_key_for_provider

        assert _get_api_key_for_provider("anthropic", settings_with_keys) == "test-anthropic-key"

    def test_unknown_provider_returns_empty(self, settings_with_keys):
        from backend.api.deps import _get_api_key_for_provider

        assert _get_api_key_for_provider("openai", settings_with_keys) == ""
