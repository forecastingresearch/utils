# ABOUTME: Tests for get_response_with_retry, verifying retry limits and behavior.
# ABOUTME: Ensures retries are bounded and the last exception is raised on exhaustion.

"""Tests for the retry utility."""

from __future__ import annotations

import logging

import pytest

from utils.llm.utils import get_response_with_retry, response_to_plain_text


def test_raises_after_max_retries():
    """Retrying must stop after max_retries and raise the last exception."""
    attempts = 0

    def failing_call():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        get_response_with_retry(failing_call, wait_time=0, error_msg="test", max_retries=3)

    assert attempts == 3


def test_returns_on_success():
    """A successful call should return immediately without retrying."""
    result = get_response_with_retry(lambda: "ok", wait_time=0, error_msg="test")
    assert result == "ok"


def test_succeeds_after_transient_failures():
    """Should return the result once the call succeeds within the retry limit."""
    attempts = 0

    def flaky_call():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("transient")
        return "recovered"

    result = get_response_with_retry(flaky_call, wait_time=0, error_msg="test", max_retries=5)
    assert result == "recovered"
    assert attempts == 3


def test_repetitive_patterns_raises_final_exception():
    """The 'repetitive patterns' error should be treated like any provider error."""
    with pytest.raises(RuntimeError, match="repetitive patterns in input"):
        get_response_with_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("repetitive patterns in input")),
            wait_time=0,
            error_msg="test",
            max_retries=1,
        )


def test_retry_log_reports_generic_provider_failure_with_exception_type(caplog):
    """Retry logging should not label arbitrary provider errors as rate limits."""
    caplog.set_level(logging.INFO, logger="utils.llm.utils")

    with pytest.raises(ValueError, match="no text"):
        get_response_with_retry(
            lambda: (_ for _ in ()).throw(ValueError("no text")),
            wait_time=0,
            error_msg="Anthropic API request failed.",
            max_retries=1,
        )

    assert "Anthropic API request failed. (attempt 1/1): ValueError: no text" in caplog.text
    assert "rate limit" not in caplog.text.lower()


def test_response_to_plain_text_uses_json_dump_when_available():
    """Response debug formatting should be provider-agnostic."""

    class Response:
        def model_dump_json(self, indent: int) -> str:
            return f'{{"indent":{indent}}}'

    assert response_to_plain_text(Response()) == '{"indent":2}'
