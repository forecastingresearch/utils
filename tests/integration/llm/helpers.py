"""Test helpers for live LLM integration checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


def assert_capital_of_france(get_response: Callable[[str], str]) -> str:
    """Assert that the callable returns Paris when asked for France's capital.

    Args:
        get_response: Callable that takes a prompt string and returns the model's response.

    Returns:
        The raw response returned by the callable.
    """
    prompt = "What is the capital of France?"
    response = get_response(prompt)

    assert isinstance(response, str), f"Expected string response, got {type(response)}: {response}"
    normalized = response.strip().lower()
    assert "paris" in normalized, (
        f"Expected 'paris' in response, but got: {response!r} " f"(normalized: {normalized!r})"
    )

    return response


class Person(BaseModel):
    """Simple test schema for structured output."""

    name: str
    age: int


def assert_structured_person_extraction(
    get_structured_response: Callable[[str, type[BaseModel], Any], BaseModel],
    **options: Any,
) -> BaseModel:
    """Assert that the callable returns structured output matching the Person schema.

    Args:
        get_structured_response: Callable that takes (prompt, schema, **options) and
            returns a validated BaseModel instance.
        **options: Additional options to pass to get_structured_response
            (e.g., temperature, max_tokens, wait_time).

    Returns:
        The validated Person instance returned by the callable.
    """
    prompt = "Extract the person's name and age from this text: " "John Smith is 30 years old."

    result = get_structured_response(prompt, Person, **options)

    assert isinstance(result, Person)
    assert result.name == "John Smith"
    assert result.age == 30

    return result
