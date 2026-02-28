"""Tests for backend.config — Typed configuration from environment."""

import backend.config as config_module
import pytest
from backend.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resets the cached singleton before each test."""
    monkeypatch.setattr(config_module, "_settings", None)


@pytest.fixture()
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removes all Makaronas-related env vars so defaults are tested cleanly."""
    env_vars = [
        "APP_ENV", "APP_PORT", "LOG_LEVEL", "CORS_ORIGINS",
        "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
        "DEFAULT_LANGUAGE", "SUPPORTED_LANGUAGES",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


class TestDefaults:
    """Settings defaults when no env vars are set."""

    @pytest.mark.usefixtures("_clean_env")
    def test_app_defaults(self) -> None:
        s = get_settings()
        assert s.app_env == "development"
        assert s.app_port == 8000
        assert s.log_level == "info"

    @pytest.mark.usefixtures("_clean_env")
    def test_cors_origins_default(self) -> None:
        s = get_settings()
        assert s.cors_origins == ["http://localhost:3000", "http://localhost:5173"]

    @pytest.mark.usefixtures("_clean_env")
    def test_api_keys_default_empty(self) -> None:
        s = get_settings()
        assert s.google_api_key == ""
        assert s.anthropic_api_key == ""

    @pytest.mark.usefixtures("_clean_env")
    def test_language_defaults(self) -> None:
        s = get_settings()
        assert s.default_language == "lt"
        assert s.supported_languages == ["lt"]


class TestCommaSeparatedParsing:
    """Comma-separated env vars parse into lists correctly."""

    def test_cors_origins_multiple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com ,http://c.com")
        s = get_settings()
        assert s.cors_origins == ["http://a.com", "http://b.com", "http://c.com"]

    def test_supported_languages_multiple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUPPORTED_LANGUAGES", "lt, en, lv")
        s = get_settings()
        assert s.supported_languages == ["lt", "en", "lv"]

    def test_empty_items_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", "http://a.com,,http://b.com,")
        s = get_settings()
        assert s.cors_origins == ["http://a.com", "http://b.com"]


class TestSingleton:
    """get_settings() returns the same cached instance."""

    def test_same_instance(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_creates_new_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s1 = get_settings()
        monkeypatch.setattr(config_module, "_settings", None)
        s2 = get_settings()
        assert s1 is not s2


class TestEnvOverrides:
    """Environment variables override defaults."""

    def test_app_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        assert get_settings().app_env == "production"

    def test_app_port_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_PORT", "9000")
        assert get_settings().app_port == 9000

    def test_log_level_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        assert get_settings().log_level == "debug"

    def test_api_key_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
        assert get_settings().google_api_key == "test-key-123"


class TestSettingsImmutability:
    """Settings dataclass is frozen — no accidental mutation."""

    def test_cannot_mutate_field(self) -> None:
        s = get_settings()
        with pytest.raises(AttributeError):
            s.app_env = "production"  # type: ignore[misc]
