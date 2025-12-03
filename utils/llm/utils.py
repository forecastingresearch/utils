"""Shared helpers for working with LLM providers."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class NonRetryableError(Exception):
    """Exception that should not be retried by the retry logic."""

    pass


def get_response_with_retry(
    api_call: Callable[[], str | Any],
    wait_time: int,
    error_msg: str,
) -> str | Any:
    """Execute an API call, retrying with a delay when errors occur.

    Args:
        api_call: The API call function to execute.
        wait_time: Time to wait between retries in seconds.
        error_msg: Message to log when retrying.

    Returns:
        The result of the API call.

    Raises:
        NonRetryableError: If a non-retryable error occurs (e.g., empty response, validation error).
        Exception: Other exceptions are retried indefinitely.
    """
    while True:
        try:
            return api_call()
        except NonRetryableError:
            # Don't retry non-retryable errors (empty responses, validation errors, etc.)
            raise
        except Exception as exc:  # noqa: BLE001 - retries must catch broad exceptions
            if "repetitive patterns" in str(exc):
                logger.info(
                    "Repetitive patterns detected in the prompt. Modifying prompt and retrying..."
                )
                return "need_a_new_reformat_prompt"

            logger.info("%s: %s", error_msg, exc)
            logger.info("Waiting for %s seconds before retrying...", wait_time)

            time.sleep(wait_time)


def create_json_prompt(prompt: str, json_schema: dict[str, Any]) -> str:
    """Create an enhanced prompt that requests JSON output matching a schema.

    Args:
        prompt: The original prompt.
        json_schema: The JSON schema to match.

    Returns:
        An enhanced prompt that includes instructions to return JSON.
    """
    return (
        f"{prompt}\n\n"
        f"Please respond with a valid JSON object matching this schema: {json.dumps(json_schema, indent=2)}\n"
        f"Respond with only the JSON object, no additional text."
    )


def extract_json_from_text(text: str, provider_name: str = "") -> str:
    """Extract JSON from text that may contain markdown code blocks or reasoning tags.

    Args:
        text: The text to extract JSON from.
        provider_name: Name of the provider (for error messages).

    Returns:
        The extracted JSON text.

    Raises:
        ValueError: If JSON cannot be extracted.
    """
    json_text = text.strip()

    # Remove reasoning tags if present (some models include reasoning tags)
    reasoning_tags = [
        ("<think>", "</think>"),
    ]
    for open_tag, close_tag in reasoning_tags:
        if open_tag in json_text:
            reasoning_end = json_text.find(close_tag)
            if reasoning_end != -1:
                json_text = json_text[reasoning_end + len(close_tag) :].strip()
                break

    # If we still have reasoning-like content (starts with reasoning text, not JSON),
    # try to find the JSON object/array in the response
    # Look for the last occurrence of { or [ which is likely the actual JSON response
    # (reasoning usually comes first, JSON comes last)
    if not (json_text.strip().startswith("{") or json_text.strip().startswith("[")):
        # Find the last JSON-like structure
        last_brace = json_text.rfind("{")
        last_bracket = json_text.rfind("[")
        if last_brace != -1 or last_bracket != -1:
            # Use whichever comes last (or whichever exists)
            json_start = max(
                (pos for pos in [last_brace, last_bracket] if pos != -1),
                default=0,
            )
            json_text = json_text[json_start:].strip()

    # Try to extract JSON from markdown code blocks
    if "```json" in json_text:
        # Extract JSON from markdown code block
        start = json_text.find("```json") + 7
        end = json_text.find("```", start)
        if end != -1:
            json_text = json_text[start:end].strip()
    elif "```" in json_text:
        # Extract JSON from generic code block
        start = json_text.find("```") + 3
        end = json_text.find("```", start)
        if end != -1:
            json_text = json_text[start:end].strip()

    # Remove any trailing markdown formatting (backticks, whitespace, etc.)
    json_text = json_text.rstrip("`").strip()

    # If there are still backticks at the end (on a new line), remove them
    while json_text.endswith("```"):
        json_text = json_text[:-3].rstrip()

    # Try to find JSON boundaries more robustly
    # Find the first { or [ and extract until the matching closing brace/bracket
    json_start = -1
    for char in ["{", "["]:
        pos = json_text.find(char)
        if pos != -1 and (json_start == -1 or pos < json_start):
            json_start = pos

    if json_start != -1:
        # Extract from the opening brace/bracket
        json_text = json_text[json_start:]
        # Find the matching closing brace/bracket
        bracket_count = 0
        brace_count = 0
        json_end = -1
        for i, char in enumerate(json_text):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and bracket_count == 0:
                    json_end = i + 1
                    break
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if brace_count == 0 and bracket_count == 0:
                    json_end = i + 1
                    break

        if json_end != -1:
            json_text = json_text[:json_end].strip()

    return json_text


def parse_and_validate_json(
    json_text: str, response_schema: type[BaseModel], provider_name: str, response_text: str = ""
) -> BaseModel:
    """Parse JSON text and validate it against a Pydantic schema.

    Args:
        json_text: The JSON text to parse.
        response_schema: The Pydantic schema to validate against.
        provider_name: Name of the provider (for error messages).
        response_text: The original response text (for error messages).

    Returns:
        A validated instance of response_schema.

    Raises:
        NonRetryableError: If JSON cannot be parsed or validated (these shouldn't be retried).
    """
    # Parse JSON from response
    try:
        json_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        error_text = response_text[:200] if response_text else json_text[:200]
        raise NonRetryableError(
            f"Failed to parse JSON from {provider_name} response: {e}. "
            f"Response text: {error_text}"
        ) from e

    # Validate against Pydantic schema
    try:
        return response_schema.model_validate(json_data)
    except ValidationError as e:
        raise NonRetryableError(
            f"{provider_name} response did not match expected schema: {e}. "
            f"Response data: {json_data}"
        ) from e
