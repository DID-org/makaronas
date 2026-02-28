"""App configuration — environment variable loading with typed defaults.

Loads settings from .env file (via python-dotenv) and os.environ.
Real environment variables take precedence over .env file values.

Model routing goes through models.py TIER_MAP, not through config.
Config holds API keys (infrastructure secrets) and app-level settings.

Usage:
    from backend.config import get_settings
    settings = get_settings()
    print(settings.google_api_key)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Canonical project root — single source of truth (Framework P17).
# Computed once here; every other module imports it instead of deriving its own.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class Settings:
    """Typed configuration for the Makaronas platform.

    All fields have sensible defaults for local development.
    Model routing is handled by models.py TIER_MAP — not here.
    """

    # App
    app_env: str
    app_port: int
    log_level: str
    cors_origins: list[str]

    # AI — API keys only (model routing is in models.py TIER_MAP)
    google_api_key: str
    anthropic_api_key: str

    # Language
    default_language: str
    supported_languages: list[str]


def _split_csv(value: str) -> list[str]:
    """Splits a comma-separated string into a list of stripped, non-empty values."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_settings() -> Settings:
    """Loads configuration from .env file and environment variables.

    Returns:
        A fully resolved Settings instance.
    """
    load_dotenv(_DOTENV_PATH)

    return Settings(
        # App
        app_env=os.environ.get("APP_ENV", "development"),
        app_port=int(os.environ.get("APP_PORT", "8000")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
        cors_origins=_split_csv(
            os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
        ),
        # AI
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        # Language
        default_language=os.environ.get("DEFAULT_LANGUAGE", "lt"),
        supported_languages=_split_csv(
            os.environ.get("SUPPORTED_LANGUAGES", "lt")
        ),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Returns the singleton Settings instance. Loads .env on first call."""
    global _settings
    if _settings is None:
        _settings = _load_settings()
    return _settings
