"""Fake auth service — development stub for AuthService.

Accepts any non-empty token and returns a configurable test user. Empty
tokens return None (simulates a missing/invalid Authorization header).

TEAM: Replace this with your real auth provider (OAuth, JWT, SAML, etc.).
Subclass AuthService from backend.hooks.interfaces and implement
validate_token and get_user. The platform never touches tokens directly —
it gets a User back from your implementation.

Tier 2 service module: imports from backend.hooks.interfaces (Tier 1)
and backend.schemas (Tier 1).

Usage:
    from backend.hooks.auth import FakeAuthService

    auth = FakeAuthService()                          # default: student
    auth = FakeAuthService(default_role="teacher")    # teacher user
"""

from backend.hooks.interfaces import AuthService
from backend.schemas import User

_ROLE_NAMES: dict[str, str] = {
    "student": "Test Student",
    "teacher": "Test Teacher",
    "admin": "Test Admin",
}

_DEFAULT_SCHOOL_ID = "school-test-1"


class FakeAuthService(AuthService):
    """STUB — returns a test user for any non-empty token.

    Does not perform real authentication. Any non-empty string is treated
    as a valid token. The returned user's role is configurable at
    construction time.

    TEAM: Replace with your auth provider. Satisfy the AuthService
    interface from backend.hooks.interfaces.
    """

    def __init__(self, default_role: str = "student") -> None:
        """Initialises the fake auth service.

        Args:
            default_role: The role assigned to all returned users.
                Must be "student", "teacher", or "admin".
        """
        self._default_role = default_role

    async def validate_token(self, token: str) -> User | None:
        """Returns a test user for any non-empty token.

        Args:
            token: Any string. Non-empty → valid user, empty → None.

        Returns:
            A User with the configured role, or None if token is empty.
        """
        if not token:
            return None
        return User(
            id="fake-user-1",
            role=self._default_role,  # type: ignore[arg-type]
            name=_ROLE_NAMES.get(self._default_role, "Test User"),
            school_id=_DEFAULT_SCHOOL_ID,
        )

    async def get_user(self, user_id: str) -> User | None:
        """Returns a test user with the given ID.

        Args:
            user_id: The user identifier. Empty string → None.

        Returns:
            A User with the given id and configured role, or None if
            user_id is empty.
        """
        if not user_id:
            return None
        return User(
            id=user_id,
            role=self._default_role,  # type: ignore[arg-type]
            name=_ROLE_NAMES.get(self._default_role, "Test User"),
            school_id=_DEFAULT_SCHOOL_ID,
        )
