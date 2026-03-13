# ABOUTME: Tests for get_response_with_retry, verifying retry limits and behavior.
# ABOUTME: Ensures retries are bounded and the last exception is raised on exhaustion.

"""Tests for the retry utility."""

from __future__ import annotations

import pytest

from utils.llm.utils import get_response_with_retry


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


def test_repetitive_patterns_returns_sentinel():
    """The 'repetitive patterns' error should return the sentinel string, not retry."""
    result = get_response_with_retry(
        lambda: (_ for _ in ()).throw(RuntimeError("repetitive patterns in input")),
        wait_time=0,
        error_msg="test",
    )
    assert result == "need_a_new_reformat_prompt"
