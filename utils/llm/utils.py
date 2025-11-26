"""Shared helpers for working with LLM providers."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def get_response_with_retry(
    api_call: Callable[[], str | Any],
    wait_time: int,
    error_msg: str,
) -> str | Any:
    """Execute an API call, retrying with a delay when errors occur."""
    while True:
        try:
            return api_call()
        except Exception as exc:  # noqa: BLE001 - retries must catch broad exceptions
            if "repetitive patterns" in str(exc):
                logger.info(
                    "Repetitive patterns detected in the prompt. Modifying prompt and retrying..."
                )
                return "need_a_new_reformat_prompt"

            logger.info("%s: %s", error_msg, exc)
            logger.info("Waiting for %s seconds before retrying...", wait_time)

            time.sleep(wait_time)
