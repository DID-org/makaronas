"""App configuration — environment variable loading with typed defaults.

Loads settings from .env file (via python-dotenv) and os.environ.
Real environment variables take precedence over .env file values.

Model name env vars (e.g. TRICKSTER_MODEL=GEMINI_FLASH) are resolved to
actual API model IDs at load time via MODEL_MAP from backend.models.

Usage:
    from backend.config import get_settings
    settings = get_settings()
    print(settings.trickster_model)  # "gemini-3-flash-preview"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from backend.models import MODEL_MAP

# Only load .env from the project root — don't traverse parent directories.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class Settings:
    """Typed configuration for the Makaronas platform.

    All fields have sensible defaults for local development.
    Model fields store resolved API model IDs (not family names).
    """

    # App
    app_env: str
    app_port: int
    log_level: str
    cors_origins: list[str]

    # AI
    ai_backend: str
    trickster_model: str
    composer_model: str
    evaluator_model: str
    google_api_key: str
    anthropic_api_key: str

    # Language
    default_language: str
    supported_languages: list[str]


def _resolve_model(env_var: str, value: str) -> str:
    """Resolves a family-name string to an actual model ID via MODEL_MAP.

    Args:
        env_var: Name of the environment variable (for error messages).
        value: The family-name value from the environment (e.g. "GEMINI_FLASH").

    Returns:
        The resolved model ID string.

    Raises:
        ValueError: If the value doesn't match any key in MODEL_MAP.
    """
    if value in MODEL_MAP:
        return MODEL_MAP[value]
    valid = ", ".join(sorted(MODEL_MAP.keys()))
    raise ValueError(
        f"Invalid value for {env_var}: {value!r}. "
        f"Valid options: {valid}"
    )


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
        ai_backend=os.environ.get("AI_BACKEND", "gemini"),
        trickster_model=_resolve_model(
            "TRICKSTER_MODEL",
            os.environ.get("TRICKSTER_MODEL", "GEMINI_FLASH"),
        ),
        composer_model=_resolve_model(
            "COMPOSER_MODEL",
            os.environ.get("COMPOSER_MODEL", "CLAUDE_SONNET"),
        ),
        evaluator_model=_resolve_model(
            "EVALUATOR_MODEL",
            os.environ.get("EVALUATOR_MODEL", "GEMINI_FLASH"),
        ),
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
