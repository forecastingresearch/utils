"""Shared helpers for working with LLM providers."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


_DEFAULT_MAX_RETRIES: int = 5


def response_to_plain_text(response: Any) -> str:
    """Return a readable plain-text representation of a provider response."""
    for method_name in ("model_dump_json", "json", "to_json"):
        method = getattr(response, method_name, None)
        if callable(method):
            try:
                return str(method(indent=2))
            except TypeError:
                try:
                    return str(method())
                except Exception:  # noqa: BLE001 - best-effort debug formatting
                    continue
            except Exception:  # noqa: BLE001 - best-effort debug formatting
                continue
    return str(response)


def get_response_with_retry(
    api_call: Callable[[], str | Any],
    wait_time: int,
    error_msg: str,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> str | Any:
    """Execute an API call, retrying with a delay when errors occur.

    Args:
        api_call (Callable): The API call to execute.
        wait_time (int): Seconds to wait between retries.
        error_msg (str): Message prefix for retry log entries.
        max_retries (int): Maximum number of attempts before raising.
    """
    for attempt in range(max_retries):
        try:
            return api_call()
        except Exception as exc:  # noqa: BLE001 - retries must catch broad exceptions
            logger.info(
                "%s (attempt %d/%d): %s: %s",
                error_msg,
                attempt + 1,
                max_retries,
                type(exc).__name__,
                exc,
            )

            if attempt + 1 >= max_retries:
                raise

            logger.info("Waiting for %s seconds before retrying...", wait_time)
            time.sleep(wait_time)
