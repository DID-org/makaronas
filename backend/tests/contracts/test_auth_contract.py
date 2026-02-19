"""Contract tests for AuthService — behavioral specification.

Verifies that any AuthService implementation satisfies the token validation
and user lookup contracts. The contract is minimal by design: non-empty
token/user_id → User, empty → None. Real implementations will validate
JWTs, session cookies, etc. — but this behavioral boundary must hold.

Run against registered implementations:
    python -m pytest backend/tests/contracts/test_auth_contract.py -v
"""

import pytest

from backend.schemas import User


class TestAuthContract:
    """Behavioral contract for AuthService implementations."""

    # -- Token validation --------------------------------------------------

    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self, auth_service) -> None:
        """Non-empty token must return a User instance."""
        user = await auth_service.validate_token("any-valid-token")
        assert user is not None
        assert isinstance(user, User)

    @pytest.mark.asyncio
    async def test_valid_token_user_has_all_fields(self, auth_service) -> None:
        """Returned User must have all required fields as non-empty strings."""
        user = await auth_service.validate_token("test-token")
        assert user is not None
        assert isinstance(user.id, str) and user.id
        assert user.role in ("student", "teacher", "admin")
        assert isinstance(user.name, str) and user.name
        assert isinstance(user.school_id, str) and user.school_id

    @pytest.mark.asyncio
    async def test_empty_token_returns_none(self, auth_service) -> None:
        """Empty token must return None (invalid/missing auth)."""
        user = await auth_service.validate_token("")
        assert user is None

    # -- User lookup -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_user_returns_user_with_matching_id(
        self, auth_service
    ) -> None:
        """Non-empty user_id must return a User with that exact id."""
        user = await auth_service.get_user("user-42")
        assert user is not None
        assert isinstance(user, User)
        assert user.id == "user-42"

    @pytest.mark.asyncio
    async def test_get_user_has_all_fields(self, auth_service) -> None:
        """Returned User must have all required fields populated."""
        user = await auth_service.get_user("lookup-test")
        assert user is not None
        assert user.role in ("student", "teacher", "admin")
        assert isinstance(user.name, str) and user.name
        assert isinstance(user.school_id, str) and user.school_id

    @pytest.mark.asyncio
    async def test_get_user_empty_id_returns_none(self, auth_service) -> None:
        """Empty user_id must return None."""
        user = await auth_service.get_user("")
        assert user is None
