"""Shared helpers for working with LLM providers."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


_DEFAULT_MAX_RETRIES: int = 5


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
            if "repetitive patterns" in str(exc):
                logger.info(
                    "Repetitive patterns detected in the prompt. Modifying prompt and retrying..."
                )
                return "need_a_new_reformat_prompt"

            logger.info("%s (attempt %d/%d): %s", error_msg, attempt + 1, max_retries, exc)

            if attempt + 1 >= max_retries:
                raise

            logger.info("Waiting for %s seconds before retrying...", wait_time)
            time.sleep(wait_time)
