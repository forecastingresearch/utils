"""Test helpers for live LLM integration checks."""

from __future__ import annotations

from collections.abc import Callable


def assert_capital_of_france(get_response: Callable[[str], str]) -> str:
    """Assert that the callable returns Paris when asked for France's capital.

    Args:
        get_response: Callable that takes a prompt string and returns the model's response.

    Returns:
        The raw response returned by the callable.
    """
    prompt = "What is the capital of France?"
    response = get_response(prompt)

    assert isinstance(response, str)
    normalized = response.strip().lower()
    assert "paris" in normalized

    return response
