"""Tests for Phase 3b — SSE streaming infrastructure.

Covers: format_sse_event (wire format), stream_ai_response (happy path,
done_data, timeout, exception, empty stream), create_sse_response (headers).

All tests use explicit @pytest.mark.asyncio per strict mode (Phase 1a note).
"""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from starlette.responses import StreamingResponse

from backend.schemas import DoneEvent, ErrorEvent, TokenEvent
from backend.streaming import create_sse_response, format_sse_event, stream_ai_response


# ---------------------------------------------------------------------------
# Helpers: async iterators for testing
# ---------------------------------------------------------------------------


async def _token_iter(tokens: list[str]) -> AsyncIterator[str]:
    """Yields tokens from a list."""
    for t in tokens:
        yield t


async def _slow_iter(tokens: list[str], delay: float) -> AsyncIterator[str]:
    """Yields first token, then sleeps longer than expected timeout."""
    for t in tokens:
        yield t
        await asyncio.sleep(delay)


async def _failing_iter(tokens: list[str], exc: Exception) -> AsyncIterator[str]:
    """Yields tokens then raises an exception."""
    for t in tokens:
        yield t
    raise exc


async def _collect(gen: AsyncIterator[str]) -> list[str]:
    """Collects all items from an async generator into a list."""
    return [item async for item in gen]


def _parse_events(raw_events: list[str]) -> list[dict]:
    """Parses SSE-formatted strings into (event_type, data_dict) tuples."""
    parsed = []
    for raw in raw_events:
        lines = raw.strip().split("\n")
        event_type = None
        data_json = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_json = line[6:]
        parsed.append({"type": event_type, "data": json.loads(data_json)})
    return parsed


# ---------------------------------------------------------------------------
# format_sse_event
# ---------------------------------------------------------------------------


class TestFormatSseEvent:
    """format_sse_event — SSE wire format from Pydantic models."""

    def test_token_event_format(self) -> None:
        result = format_sse_event("token", TokenEvent(text="hello"))
        assert result == 'event: token\ndata: {"text":"hello"}\n\n'

    def test_done_event_format(self) -> None:
        result = format_sse_event(
            "done",
            DoneEvent(full_text="hello world", data={"task_complete": True}),
        )
        assert result.startswith("event: done\ndata: ")
        assert result.endswith("\n\n")

        data = json.loads(result.split("data: ")[1].strip())
        assert data["full_text"] == "hello world"
        assert data["data"]["task_complete"] is True

    def test_error_event_format(self) -> None:
        result = format_sse_event(
            "error",
            ErrorEvent(code="AI_TIMEOUT", message="Too slow", partial_text="hel"),
        )
        assert result.startswith("event: error\ndata: ")
        assert result.endswith("\n\n")

        data = json.loads(result.split("data: ")[1].strip())
        assert data["code"] == "AI_TIMEOUT"
        assert data["message"] == "Too slow"
        assert data["partial_text"] == "hel"

    def test_done_event_default_data(self) -> None:
        result = format_sse_event("done", DoneEvent(full_text="text"))
        data = json.loads(result.split("data: ")[1].strip())
        assert data["data"] == {}

    def test_error_event_default_partial_text(self) -> None:
        result = format_sse_event(
            "error", ErrorEvent(code="X", message="msg")
        )
        data = json.loads(result.split("data: ")[1].strip())
        assert data["partial_text"] == ""


# ---------------------------------------------------------------------------
# stream_ai_response
# ---------------------------------------------------------------------------


class TestStreamAiResponseHappyPath:
    """stream_ai_response — normal token → done flow."""

    @pytest.mark.asyncio
    async def test_tokens_then_done(self) -> None:
        tokens = ["You ", "found ", "it."]
        events = await _collect(
            stream_ai_response(_token_iter(tokens))
        )

        assert len(events) == 4  # 3 tokens + 1 done
        parsed = _parse_events(events)

        # Token events
        for i, token in enumerate(tokens):
            assert parsed[i]["type"] == "token"
            assert parsed[i]["data"]["text"] == token

        # Done event
        assert parsed[3]["type"] == "done"
        assert parsed[3]["data"]["full_text"] == "You found it."
        assert parsed[3]["data"]["data"] == {}

    @pytest.mark.asyncio
    async def test_done_with_structured_data(self) -> None:
        tokens = ["Yes"]
        done_data = {"task_complete": True, "score": 42}
        events = await _collect(
            stream_ai_response(_token_iter(tokens), done_data=done_data)
        )

        parsed = _parse_events(events)
        done = parsed[-1]
        assert done["type"] == "done"
        assert done["data"]["full_text"] == "Yes"
        assert done["data"]["data"]["task_complete"] is True
        assert done["data"]["data"]["score"] == 42

    @pytest.mark.asyncio
    async def test_empty_stream(self) -> None:
        events = await _collect(
            stream_ai_response(_token_iter([]))
        )

        assert len(events) == 1  # just done
        parsed = _parse_events(events)
        assert parsed[0]["type"] == "done"
        assert parsed[0]["data"]["full_text"] == ""


class TestStreamAiResponseTimeout:
    """stream_ai_response — timeout yields error event with partial text."""

    @pytest.mark.asyncio
    async def test_timeout_yields_error_with_partial_text(self) -> None:
        events = await _collect(
            stream_ai_response(
                _slow_iter(["fast", "slow"], delay=5.0),
                timeout_seconds=0.1,
            )
        )

        parsed = _parse_events(events)
        # Should have at least one token event, then an error
        error = parsed[-1]
        assert error["type"] == "error"
        assert error["data"]["code"] == "AI_TIMEOUT"
        assert "fast" in error["data"]["partial_text"]

    @pytest.mark.asyncio
    async def test_timeout_message_is_user_friendly(self) -> None:
        events = await _collect(
            stream_ai_response(
                _slow_iter(["x"], delay=5.0),
                timeout_seconds=0.1,
            )
        )

        parsed = _parse_events(events)
        error = parsed[-1]
        assert "taking too long" in error["data"]["message"].lower()


class TestStreamAiResponseException:
    """stream_ai_response — iterator exception yields error event."""

    @pytest.mark.asyncio
    async def test_exception_yields_error_with_partial_text(self) -> None:
        events = await _collect(
            stream_ai_response(
                _failing_iter(["partial "], RuntimeError("provider down"))
            )
        )

        parsed = _parse_events(events)
        assert len(parsed) == 2  # 1 token + 1 error

        token = parsed[0]
        assert token["type"] == "token"
        assert token["data"]["text"] == "partial "

        error = parsed[1]
        assert error["type"] == "error"
        assert error["data"]["code"] == "STREAM_ERROR"
        assert error["data"]["partial_text"] == "partial "

    @pytest.mark.asyncio
    async def test_exception_with_no_tokens_yields_empty_partial(self) -> None:
        events = await _collect(
            stream_ai_response(
                _failing_iter([], ValueError("empty boom"))
            )
        )

        parsed = _parse_events(events)
        assert len(parsed) == 1
        error = parsed[0]
        assert error["type"] == "error"
        assert error["data"]["code"] == "STREAM_ERROR"
        assert error["data"]["partial_text"] == ""


# ---------------------------------------------------------------------------
# create_sse_response
# ---------------------------------------------------------------------------


class TestCreateSseResponse:
    """create_sse_response — wraps generator with correct HTTP response."""

    def test_returns_streaming_response(self) -> None:
        async def empty_gen():
            return
            yield  # make it a generator

        resp = create_sse_response(empty_gen())
        assert isinstance(resp, StreamingResponse)

    def test_content_type_is_event_stream(self) -> None:
        async def empty_gen():
            return
            yield

        resp = create_sse_response(empty_gen())
        assert resp.media_type == "text/event-stream"

    def test_cache_control_header(self) -> None:
        async def empty_gen():
            return
            yield

        resp = create_sse_response(empty_gen())
        # Headers are in the raw headers list
        header_dict = dict(resp.headers)
        assert header_dict.get("cache-control") == "no-cache"

    def test_nginx_buffering_header(self) -> None:
        async def empty_gen():
            return
            yield

        resp = create_sse_response(empty_gen())
        header_dict = dict(resp.headers)
        assert header_dict.get("x-accel-buffering") == "no"

    @pytest.mark.asyncio
    async def test_response_body_contains_events(self) -> None:
        tokens = ["a", "b"]

        resp = create_sse_response(stream_ai_response(_token_iter(tokens)))

        # Consume the response body
        body_parts = []
        async for chunk in resp.body_iterator:
            body_parts.append(chunk)

        body = "".join(body_parts)
        assert "event: token" in body
        assert "event: done" in body
        assert '"full_text":"ab"' in body
