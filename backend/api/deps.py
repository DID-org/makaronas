"""Shared FastAPI dependencies — auth, database, session, storage, AI injection.

Module-level singletons for each service stub. Route handlers access them
via FastAPI's Depends() system — never by importing stubs directly. When the
team swaps a stub for a real implementation, they change the class here and
every downstream handler picks it up automatically.

TEAM: To wire your real services, replace the stub class on the right side
of each singleton assignment below. The get_* functions and all route
handlers stay unchanged.

Tier 2 service module: imports from hooks/* (Tier 2) and hooks/interfaces
(Tier 1), schemas (Tier 1), ai/* (Tier 2), models (Tier 1).

Usage:
    from backend.api.deps import get_current_user, get_database

    @router.get("/something")
    async def do_thing(
        user: User = Depends(get_current_user),
        db: DatabaseAdapter = Depends(get_database),
    ): ...
"""

import logging

from fastapi import Depends, Header, HTTPException

from backend.ai.prompts import PromptLoader
from backend.ai.providers.base import AIProvider
from backend.ai.trickster import TricksterEngine
from backend.config import Settings
from backend.hooks.auth import FakeAuthService
from backend.hooks.database import InMemoryStore
from backend.hooks.interfaces import (
    AuthService,
    DatabaseAdapter,
    FileStorage,
    SessionStore,
)
from backend.hooks.sessions import InMemorySessionStore
from backend.hooks.storage import LocalFileStorage
from backend.models import ModelConfig, resolve_tier
from backend.schemas import ApiError, ApiResponse, User
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import TaskCartridge

logger = logging.getLogger("makaronas")

# ---------------------------------------------------------------------------
# Service singletons — the swap point
# ---------------------------------------------------------------------------

# TEAM: Replace with your real implementations here.
_auth_service: AuthService = FakeAuthService()
_database: DatabaseAdapter = InMemoryStore()
_session_store: SessionStore = InMemorySessionStore()
_file_storage: FileStorage = LocalFileStorage()
_task_registry: TaskRegistry | None = None

# AI singletons — set by _init_ai_services() in main.py at startup
_prompt_loader: PromptLoader | None = None
_trickster_engine: TricksterEngine | None = None
_reload_all = None  # Set to a callable by _init_ai_services()


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


def get_auth_service() -> AuthService:
    """Returns the auth service singleton."""
    return _auth_service


def get_database() -> DatabaseAdapter:
    """Returns the database adapter singleton."""
    return _database


def get_session_store() -> SessionStore:
    """Returns the session store singleton."""
    return _session_store


def get_file_storage() -> FileStorage:
    """Returns the file storage singleton."""
    return _file_storage


def get_task_registry() -> TaskRegistry:
    """Returns the task registry singleton.

    Raises HTTPException(503) if the registry hasn't been loaded yet
    (startup not complete). This prevents requests from hitting a
    None registry during slow startup.
    """
    if _task_registry is None:
        raise HTTPException(
            status_code=503,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="SERVICE_UNAVAILABLE",
                    message="Task registry is not yet available. Server is starting up.",
                ),
            ).model_dump(),
        )
    return _task_registry


def get_prompt_loader() -> PromptLoader:
    """Returns the prompt loader singleton.

    Raises HTTPException(503) if the loader hasn't been initialized yet
    (startup not complete).
    """
    if _prompt_loader is None:
        raise HTTPException(
            status_code=503,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="SERVICE_UNAVAILABLE",
                    message="Prompt loader is not yet available. Server is starting up.",
                ),
            ).model_dump(),
        )
    return _prompt_loader


def get_trickster_engine() -> TricksterEngine:
    """Returns the trickster engine singleton.

    Raises HTTPException(503) if the engine hasn't been initialized yet
    (startup not complete).
    """
    if _trickster_engine is None:
        raise HTTPException(
            status_code=503,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="SERVICE_UNAVAILABLE",
                    message="Trickster engine is not yet available. Server is starting up.",
                ),
            ).model_dump(),
        )
    return _trickster_engine


# ---------------------------------------------------------------------------
# AI provider factory
# ---------------------------------------------------------------------------


def create_provider(model_config: ModelConfig, settings: Settings) -> AIProvider:
    """Routes a ModelConfig to the correct concrete provider instance.

    Args:
        model_config: The resolved tier configuration.
        settings: Application settings with API keys.

    Returns:
        A concrete AIProvider instance (GeminiProvider or AnthropicProvider).

    Raises:
        ValueError: If the provider name is not recognized.
    """
    # Local imports to avoid pulling SDK dependencies at module load time.
    # These are only needed when actually constructing providers.
    if model_config.provider == "gemini":
        from backend.ai.providers.gemini import GeminiProvider

        return GeminiProvider(api_key=settings.google_api_key)

    if model_config.provider == "anthropic":
        from backend.ai.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=settings.anthropic_api_key)

    raise ValueError(
        f"Unknown provider: {model_config.provider!r}. "
        f"Expected 'gemini' or 'anthropic'."
    )


# ---------------------------------------------------------------------------
# Graceful degradation helper
# ---------------------------------------------------------------------------


def check_ai_readiness(cartridge: TaskCartridge, settings: Settings) -> list[str]:
    """Checks whether a cartridge's AI phases can be served.

    Returns a list of human-readable issue descriptions. An empty list
    means the cartridge is ready for AI interaction.

    Called by the student endpoint (Phase 6b) before entering an AI phase
    to decide whether to proceed or fall back.

    Args:
        cartridge: The task cartridge to check.
        settings: Application settings with API keys.

    Returns:
        List of issues preventing AI phases from being served.
    """
    issues: list[str] = []

    # Static-only cartridges don't need AI
    if cartridge.ai_config is None:
        return issues

    has_ai_phase = any(p.is_ai_phase for p in cartridge.phases)
    if not has_ai_phase:
        return issues

    # Check API key for the cartridge's resolved provider
    try:
        model_config = resolve_tier(cartridge.ai_config.model_preference)
    except KeyError:
        issues.append(
            f"Unknown model tier: {cartridge.ai_config.model_preference!r}"
        )
        return issues

    api_key = _get_api_key_for_provider(model_config.provider, settings)
    if not api_key:
        issues.append(
            f"Missing API key for provider {model_config.provider!r} "
            f"(tier: {cartridge.ai_config.model_preference!r})"
        )

    # Check prompt existence via the module-level singleton
    if _prompt_loader is not None:
        prompt_errors = _prompt_loader.validate_task_prompts(cartridge)
        issues.extend(prompt_errors)
    else:
        issues.append("Prompt loader not initialized")

    return issues


def _get_api_key_for_provider(provider: str, settings: Settings) -> str:
    """Returns the API key for a given provider name.

    Args:
        provider: Provider identifier ("gemini" or "anthropic").
        settings: Application settings.

    Returns:
        The API key string (empty string if not configured).
    """
    if provider == "gemini":
        return settings.google_api_key
    if provider == "anthropic":
        return settings.anthropic_api_key
    return ""


# ---------------------------------------------------------------------------
# Auth dependency — used by route handlers
# ---------------------------------------------------------------------------


async def get_current_user(
    authorization: str | None = Header(default=None),
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """Extracts and validates a Bearer token from the Authorization header.

    Returns the authenticated User on success. Raises HTTPException(401)
    on missing header, malformed header, or invalid token.

    Args:
        authorization: The raw Authorization header value.
        auth_service: Injected auth service.

    Returns:
        The authenticated User.

    Raises:
        HTTPException: 401 with ApiResponse envelope on auth failure.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail=ApiResponse(
                ok=False,
                error=ApiError(code="UNAUTHORIZED", message="Missing authorization header."),
            ).model_dump(),
        )

    parts = authorization.split(" ", maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=401,
            detail=ApiResponse(
                ok=False,
                error=ApiError(code="UNAUTHORIZED", message="Invalid authorization header format."),
            ).model_dump(),
        )

    token = parts[1].strip()
    user = await auth_service.validate_token(token)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail=ApiResponse(
                ok=False,
                error=ApiError(code="UNAUTHORIZED", message="Invalid or expired token."),
            ).model_dump(),
        )

    return user
