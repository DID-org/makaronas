"""Local file storage — development stub for FileStorage.

Reads and writes task assets from the local filesystem. Returns URL
paths that Phase 4c's static file route will serve. No cloud storage,
no CDN — just the local disk.

TEAM: Replace this with your cloud storage (S3, GCS, etc.). Subclass
FileStorage from backend.hooks.interfaces and implement get_asset_url
and store_asset. Your get_asset_url might return signed CDN URLs instead
of local API paths.

Tier 2 service module: imports from backend.hooks.interfaces (Tier 1).
No schema imports needed.

Usage:
    from backend.hooks.storage import LocalFileStorage

    storage = LocalFileStorage()                        # default: content/tasks
    storage = LocalFileStorage(base_path="/tmp/assets")  # custom path
    url = await storage.store_asset("task-1", "image.png", image_bytes)
"""

from pathlib import Path

from backend.hooks.interfaces import FileStorage


class LocalFileStorage(FileStorage):
    """STUB — serves assets from the local filesystem.

    Assets are stored under {base_path}/{task_id}/{filename}. URLs
    returned are API paths (/api/v1/assets/...) that Phase 4c's
    static file route will serve.

    TEAM: Replace with your cloud storage adapter. Satisfy the
    FileStorage interface from backend.hooks.interfaces.
    """

    def __init__(self, base_path: str = "content/tasks") -> None:
        """Initialises with a base directory for asset storage.

        Args:
            base_path: Root directory for task assets. Resolved to a
                Path object internally. Defaults to "content/tasks"
                relative to the working directory.
        """
        self._base_path = Path(base_path)

    async def get_asset_url(self, task_id: str, filename: str) -> str:
        """Returns the API URL for a task asset.

        Args:
            task_id: The task that owns the asset.
            filename: The asset filename.

        Returns:
            A URL path like /api/v1/assets/{task_id}/{filename}.
        """
        return f"/api/v1/assets/{task_id}/{filename}"

    async def store_asset(
        self, task_id: str, filename: str, data: bytes
    ) -> str:
        """Writes an asset to the local filesystem and returns its URL.

        Creates intermediate directories if they don't exist.

        Args:
            task_id: The task that owns the asset.
            filename: The asset filename.
            data: Raw bytes to write.

        Returns:
            A URL path like /api/v1/assets/{task_id}/{filename}.
        """
        target_dir = self._base_path / task_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        target_file.write_bytes(data)
        return f"/api/v1/assets/{task_id}/{filename}"
