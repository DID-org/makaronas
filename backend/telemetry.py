"""Session telemetry — collects anonymous student interaction data.

Writes incremental JSON files per session. Each task completion appends
to the session file. Session end writes the final summary.

Data is anonymous — no student names or IDs beyond the session UUID.
Files live in data/sessions/ and should be deleted after analysis.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.schemas import GameSession

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/sessions")


def _ensure_dir() -> None:
    """Creates the data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    """Returns the path for a session's telemetry file."""
    return DATA_DIR / f"{session_id}.json"


def _load_existing(session_id: str) -> dict:
    """Loads existing session telemetry or returns empty structure."""
    path = _session_path(session_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "session_id": session_id,
        "created_at": None,
        "tasks": [],
        "session_report": None,
        "completed": False,
    }


def save_task_completion(
    session: GameSession,
    task_id: str,
    phase_exchanges: list[dict],
    task_duration_ms: float | None = None,
) -> None:
    """Saves telemetry when a task completes (terminal phase reached).

    Called incrementally — each task completion appends to the file.
    Even if the student doesn't finish all tasks, we have data for
    every task they did complete.
    """
    _ensure_dir()

    data = _load_existing(session.session_id)
    data["created_at"] = session.created_at.isoformat()

    # Build task record
    task_record = {
        "task_id": task_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": task_duration_ms,
        "exchange_count": len([e for e in phase_exchanges if e.get("role") == "student"]),
        "exchanges": phase_exchanges,
    }

    # Get the latest task_history entry for this task (has evaluation outcome)
    for entry in reversed(session.task_history):
        if entry.get("task_id") == task_id:
            task_record["evaluation_outcome"] = entry.get("evaluation_outcome")
            task_record["intensity_score"] = entry.get("intensity_score")
            break

    # Avoid duplicates
    existing_ids = {t["task_id"] for t in data["tasks"]}
    if task_id not in existing_ids:
        data["tasks"].append(task_record)

    path = _session_path(session.session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Telemetry: saved task %s for session %s", task_id, session.session_id)


def save_session_end(
    session: GameSession,
    report_text: str | None = None,
) -> None:
    """Saves final session telemetry when all tasks are done.

    Enriches the existing file with the session report and marks
    the session as completed.
    """
    _ensure_dir()

    data = _load_existing(session.session_id)
    data["created_at"] = session.created_at.isoformat()
    data["completed"] = True
    data["completed_at"] = datetime.now(timezone.utc).isoformat()
    data["session_report"] = report_text
    data["total_tasks_completed"] = len(data["tasks"])
    data["task_history"] = session.task_history

    path = _session_path(session.session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Telemetry: session %s completed (%d tasks)", session.session_id, len(data["tasks"]))


def save_active_session(session: GameSession) -> None:
    """Dumps an active (incomplete) session for students who didn't finish.

    Called by the admin export endpoint.
    """
    _ensure_dir()

    data = _load_existing(session.session_id)
    data["created_at"] = session.created_at.isoformat()
    data["completed"] = False
    data["current_task"] = session.current_task
    data["current_phase"] = session.current_phase
    data["task_history"] = session.task_history
    data["dumped_at"] = datetime.now(timezone.utc).isoformat()

    # Save current exchanges (whatever task they're in the middle of)
    if session.exchanges:
        data["active_exchanges"] = [
            {
                "role": e.role,
                "content": e.content,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in session.exchanges
        ]

    path = _session_path(session.session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Telemetry: dumped active session %s", session.session_id)
