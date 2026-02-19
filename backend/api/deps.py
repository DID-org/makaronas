"""Shared FastAPI dependencies — auth, database, session, storage injection.

Module-level singletons for each service stub. Route handlers access them
via FastAPI's Depends() system — never by importing stubs directly. When the
team swaps a stub for a real implementation, they change the class here and
every downstream handler picks it up automatically.

TEAM: To wire your real services, replace the stub class on the right side
of each singleton assignment below. The get_* functions and all route
handlers stay unchanged.

Tier 2 service module: imports from hooks/* (Tier 2) and hooks/interfaces
(Tier 1), schemas (Tier 1).

Usage:
    from backend.api.deps import get_current_user, get_database

    @router.get("/something")
    async def do_thing(
        user: User = Depends(get_current_user),
        db: DatabaseAdapter = Depends(get_database),
    ): ...
"""

from fastapi import Depends, Header, HTTPException

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
from backend.schemas import ApiError, ApiResponse, User

# ---------------------------------------------------------------------------
# Service singletons — the swap point
# ---------------------------------------------------------------------------

# TEAM: Replace with your real implementations here.
_auth_service: AuthService = FakeAuthService()
_database: DatabaseAdapter = InMemoryStore()
_session_store: SessionStore = InMemorySessionStore()
_file_storage: FileStorage = LocalFileStorage()


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
