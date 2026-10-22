# ABOUTME: Tests for get_response_with_retry, verifying retry limits and behavior.
# ABOUTME: Ensures retries are bounded and the last exception is raised on exhaustion.

"""Tests for the retry utility."""

from __future__ import annotations

import logging

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from utils.llm.utils import (
    TransientProviderError,
    get_response_with_retry,
    response_to_plain_text,
)

_REQUEST = httpx.Request("POST", "https://provider.test/v1")


def _openai_status_error(cls, status: int):
    return cls("provider error", response=httpx.Response(status, request=_REQUEST), body=None)


def _anthropic_status_error(cls, status: int):
    return cls("provider error", response=httpx.Response(status, request=_REQUEST), body=None)


def _anthropic_stream_error(error_type: str):
    """Build the exception the Anthropic SDK raises for an SSE `error` event on an open stream."""
    body = {"type": "error", "error": {"type": error_type, "message": "Overloaded"}}
    return anthropic.Anthropic(api_key="x")._make_status_error(
        "Overloaded", body=body, response=httpx.Response(200, request=_REQUEST)
    )


def _openai_stream_error(error_type: str):
    """Build the exception the OpenAI SDK raises for an error payload inside an open stream."""
    return openai.APIError(
        "server error", request=_REQUEST, body={"type": error_type, "message": "server error"}
    )


def _call_raising(exceptions: list[BaseException], result: str = "ok"):
    """Return a callable that raises the given exceptions in order, then returns result."""
    attempts = {"count": 0}

    def call():
        attempts["count"] += 1
        if attempts["count"] <= len(exceptions):
            raise exceptions[attempts["count"] - 1]
        return result

    return call, attempts


def test_raises_after_max_retries():
    """Retrying must stop after max_retries and raise the last exception."""
    call, attempts = _call_raising([openai.APIConnectionError(request=_REQUEST)] * 5)

    with pytest.raises(openai.APIConnectionError):
        get_response_with_retry(call, wait_time=0, error_msg="test", max_retries=3)

    assert attempts["count"] == 3


def test_returns_on_success():
    """A successful call should return immediately without retrying."""
    result = get_response_with_retry(lambda: "ok", wait_time=0, error_msg="test")
    assert result == "ok"


@pytest.mark.parametrize(
    "exception",
    [
        openai.APIConnectionError(request=_REQUEST),
        openai.APITimeoutError(request=_REQUEST),
        _openai_status_error(openai.RateLimitError, 429),
        _openai_status_error(openai.InternalServerError, 503),
        _openai_status_error(openai.ConflictError, 409),
        pytest.param(_openai_stream_error("server_error"), id="openai.APIError-server_error"),
        anthropic.APIConnectionError(request=_REQUEST),
        _anthropic_status_error(anthropic.RateLimitError, 429),
        _anthropic_status_error(anthropic.InternalServerError, 500),
        _anthropic_status_error(anthropic.OverloadedError, 529),
        _anthropic_status_error(anthropic.ConflictError, 409),
        pytest.param(
            _anthropic_stream_error("overloaded_error"),
            id="anthropic.APIStatusError-overloaded_error",
        ),
        pytest.param(_anthropic_stream_error("api_error"), id="anthropic.APIStatusError-api_error"),
        pytest.param(
            _anthropic_stream_error("rate_limit_error"),
            id="anthropic.APIStatusError-rate_limit_error",
        ),
        genai_errors.ServerError(503, {"error": {"message": "unavailable"}}),
        genai_errors.ClientError(429, {"error": {"message": "quota"}}),
        httpx.ReadTimeout("read timed out", request=_REQUEST),
        httpx.RemoteProtocolError("peer closed connection", request=_REQUEST),
        httpx.ReadError("connection reset", request=_REQUEST),
        TransientProviderError("provider reported status=failed, error=server_error"),
    ],
    ids=lambda exc: f"{type(exc).__module__.split('.')[0]}.{type(exc).__name__}",
)
def test_transient_provider_failures_are_retried(exception):
    """Dropped connections, timeouts, rate limits and provider-side errors get another attempt."""
    call, attempts = _call_raising([exception, exception], result="recovered")

    result = get_response_with_retry(call, wait_time=0, error_msg="test", max_retries=5)

    assert result == "recovered"
    assert attempts["count"] == 3


@pytest.mark.parametrize(
    "exception",
    [
        _openai_status_error(openai.BadRequestError, 400),
        _openai_status_error(openai.AuthenticationError, 401),
        _openai_status_error(openai.PermissionDeniedError, 403),
        _openai_status_error(openai.NotFoundError, 404),
        pytest.param(
            _openai_stream_error("invalid_request_error"),
            id="openai.APIError-invalid_request_error",
        ),
        _anthropic_status_error(anthropic.BadRequestError, 400),
        pytest.param(
            _anthropic_stream_error("invalid_request_error"),
            id="anthropic.APIStatusError-invalid_request_error",
        ),
        genai_errors.ClientError(400, {"error": {"message": "invalid argument"}}),
        httpx.UnsupportedProtocol("unsupported scheme", request=_REQUEST),
        RuntimeError("OpenAI response incomplete (status=incomplete)"),
        ValueError("no text"),
    ],
    ids=lambda exc: f"{type(exc).__module__.split('.')[0]}.{type(exc).__name__}",
)
def test_failures_a_retry_cannot_fix_raise_on_the_first_attempt(exception):
    """Bad requests, auth failures, refused requests and our own errors are not retried."""
    call, attempts = _call_raising([exception])

    with pytest.raises(type(exception)):
        get_response_with_retry(call, wait_time=0, error_msg="test", max_retries=5)

    assert attempts["count"] == 1


def test_retry_log_reports_transient_failure_with_exception_type(caplog):
    """Retry logging should name the exception type and not label every error a rate limit."""
    caplog.set_level(logging.INFO, logger="utils.llm.utils")
    call, _ = _call_raising([anthropic.APIConnectionError(request=_REQUEST)])

    with pytest.raises(anthropic.APIConnectionError):
        get_response_with_retry(
            call,
            wait_time=0,
            error_msg="Anthropic API request failed.",
            max_retries=1,
        )

    assert "Anthropic API request failed. (attempt 1/1): APIConnectionError:" in caplog.text
    assert "rate limit" not in caplog.text.lower()


def test_response_to_plain_text_uses_json_dump_when_available():
    """Response debug formatting should be provider-agnostic."""

    class Response:
        def model_dump_json(self, indent: int) -> str:
            return f'{{"indent":{indent}}}'

    assert response_to_plain_text(Response()) == '{"indent":2}'
