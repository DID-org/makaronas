"""SSE streaming utilities for AI response delivery.

Three-layer separation:
- format_sse_event: wire formatting (event + JSON data)
- stream_ai_response: orchestrates token → done/error lifecycle
- create_sse_response: wraps any SSE generator in the right HTTP response

Created: Phase 3b
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel
from starlette.responses import StreamingResponse

from backend.schemas import DoneEvent, ErrorEvent, TokenEvent

logger = logging.getLogger("makaronas.streaming")


def format_sse_event(event_type: str, data: BaseModel) -> str:
    """Formats a single SSE event string.

    Args:
        event_type: One of "token", "done", "error".
        data: A Pydantic model (TokenEvent, DoneEvent, or ErrorEvent).

    Returns:
        SSE-formatted string: "event: {type}\\ndata: {json}\\n\\n"
    """
    return f"event: {event_type}\ndata: {data.model_dump_json()}\n\n"


def create_sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    """Wraps an async generator of SSE-formatted strings in a StreamingResponse.

    Args:
        generator: Yields pre-formatted SSE event strings (output of format_sse_event).

    Returns:
        StreamingResponse with text/event-stream content type and proxy-safe headers.
    """
    return StreamingResponse(
        content=generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def stream_ai_response(
    token_iterator: AsyncIterator[str],
    done_data: dict[str, Any] | None = None,
    timeout_seconds: float = 30.0,
) -> AsyncGenerator[str, None]:
    """Turns a token iterator into a full SSE event stream.

    Consumes tokens from token_iterator, yields formatted TokenEvent SSE strings,
    accumulates full_text. On completion, yields a DoneEvent with full_text + done_data.
    On timeout or exception, yields an ErrorEvent with partial_text recovery.

    The generator catches all errors internally and never re-raises — once SSE
    streaming begins, the HTTP status is already 200. Errors are communicated
    via SSE error events, not HTTP status codes.

    Args:
        token_iterator: Async iterator yielding individual text tokens.
        done_data: Optional structured data for the DoneEvent payload.
        timeout_seconds: Maximum wall-clock time for the entire stream.

    Yields:
        SSE-formatted strings (token events, then one done or error event).
    """
    accumulated = []

    try:
        async with asyncio.timeout(timeout_seconds):
            async for token in token_iterator:
                accumulated.append(token)
                yield format_sse_event("token", TokenEvent(text=token))

    except TimeoutError:
        partial = "".join(accumulated)
        logger.warning(
            "AI stream timed out after %.1fs, partial_text length=%d",
            timeout_seconds,
            len(partial),
        )
        yield format_sse_event(
            "error",
            ErrorEvent(
                code="AI_TIMEOUT",
                message="The AI is taking too long. Try again, or continue to the next task.",
                partial_text=partial,
            ),
        )
        return

    except Exception as exc:
        partial = "".join(accumulated)
        logger.warning(
            "AI stream error: %s, partial_text length=%d",
            exc,
            len(partial),
        )
        yield format_sse_event(
            "error",
            ErrorEvent(
                code="STREAM_ERROR",
                message="Something went wrong with the AI response. Try again.",
                partial_text=partial,
            ),
        )
        return

    # Stream completed successfully — outside timeout scope so the done
    # event yield cannot be interrupted by asyncio.timeout.
    full_text = "".join(accumulated)
    yield format_sse_event(
        "done",
        DoneEvent(full_text=full_text, data=done_data or {}),
    )
