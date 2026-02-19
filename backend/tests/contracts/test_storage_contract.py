"""Contract tests for FileStorage — behavioral specification.

Verifies that any FileStorage implementation satisfies:
- URL generation (non-empty, contains task_id and filename)
- Store and retrieve consistency
- URL consistency between store_asset and get_asset_url
- Overwrite behavior

Does NOT verify that URLs are actually fetchable (that requires an HTTP
server — tested in Phase 4c endpoint tests). Only verifies URL format
consistency and store_asset success.

Run against registered implementations:
    python -m pytest backend/tests/contracts/test_storage_contract.py -v
"""

import pytest


class TestStorageContract:
    """Behavioral contract for FileStorage implementations."""

    # -- URL format --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_asset_url_returns_nonempty_string(
        self, file_storage
    ) -> None:
        """get_asset_url must return a non-empty string."""
        url = await file_storage.get_asset_url("task1", "image.png")
        assert isinstance(url, str)
        assert len(url) > 0

    @pytest.mark.asyncio
    async def test_get_asset_url_contains_task_and_filename(
        self, file_storage
    ) -> None:
        """URL must contain the task_id and filename (format may vary)."""
        url = await file_storage.get_asset_url("task-abc", "photo.jpg")
        assert "task-abc" in url
        assert "photo.jpg" in url

    # -- Store and retrieve ------------------------------------------------

    @pytest.mark.asyncio
    async def test_store_asset_returns_nonempty_url(self, file_storage) -> None:
        """store_asset must return a non-empty URL string."""
        url = await file_storage.store_asset("task1", "data.bin", b"some-bytes")
        assert isinstance(url, str)
        assert len(url) > 0

    @pytest.mark.asyncio
    async def test_store_then_get_url_contains_identifiers(
        self, file_storage
    ) -> None:
        """After storing, get_asset_url must return a URL with task_id and filename."""
        await file_storage.store_asset("task-x", "report.pdf", b"pdf-bytes")
        url = await file_storage.get_asset_url("task-x", "report.pdf")
        assert "task-x" in url
        assert "report.pdf" in url

    # -- URL consistency ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_store_and_get_urls_match(self, file_storage) -> None:
        """URL from store_asset must equal URL from get_asset_url for same asset."""
        store_url = await file_storage.store_asset(
            "task-match", "img.png", b"png-bytes"
        )
        get_url = await file_storage.get_asset_url("task-match", "img.png")
        assert store_url == get_url

    # -- Overwrite ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_overwrite_succeeds(self, file_storage) -> None:
        """Storing with same task_id+filename twice must succeed."""
        url1 = await file_storage.store_asset("task-ow", "file.txt", b"first")
        url2 = await file_storage.store_asset("task-ow", "file.txt", b"second")
        assert isinstance(url2, str)
        assert len(url2) > 0
        # URL should remain consistent after overwrite
        get_url = await file_storage.get_asset_url("task-ow", "file.txt")
        assert get_url == url2
