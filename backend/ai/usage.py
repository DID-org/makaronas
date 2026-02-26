"""Structured usage logging for AI calls.

Emits one structured log line per AI call with all fields needed for
cost analysis. Machine-parseable via the ``extra`` dict — standard JSON
log formatters (e.g., python-json-logger) pick these up automatically.

Logger name: ``makaronas.ai.usage``

Phase 6a does not configure a JSON formatter — it structures the data.
The team configures their preferred log formatter in production.

Vision ref: every AI call must log model_id, prompt_tokens,
completion_tokens, latency_ms, task_id, session_id, call_type.

Tier 2 service: imports only stdlib.
"""

import logging

logger = logging.getLogger("makaronas.ai.usage")


def log_ai_call(
    *,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    task_id: str,
    session_id: str,
    call_type: str,
) -> None:
    """Emits a structured INFO log for a completed AI call.

    All fields are passed as ``extra`` for machine-parseable output.
    The log message itself is a human-readable summary.

    Args:
        model_id: The model identifier used for this call.
        prompt_tokens: Number of input tokens consumed.
        completion_tokens: Number of output tokens generated.
        latency_ms: Wall-clock duration of the AI call in milliseconds.
        task_id: The cartridge task_id this call serves.
        session_id: The student session identifier.
        call_type: The type of AI call ("trickster" or "debrief").
    """
    logger.info(
        "AI call: %s %s tokens_in=%d tokens_out=%d latency=%.0fms task=%s session=%s",
        call_type,
        model_id,
        prompt_tokens,
        completion_tokens,
        latency_ms,
        task_id,
        session_id,
        extra={
            "model_id": model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "task_id": task_id,
            "session_id": session_id,
            "call_type": call_type,
        },
    )
