"""Helpers for invoking Anthropic models with retry support."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from ..utils import response_to_plain_text
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """LLM provider that communicates with the Anthropic Messages API."""

    retry_message = "Anthropic API request failed."

    def __init__(self, *, api_key: str | None = None, default_wait_time: int | None = None) -> None:
        """Instantiate the Anthropic client using the provided API key.

        Args:
            api_key: Anthropic API key (e.g., "sk-ant-..."). If None, an error will be raised.
            default_wait_time: Optional custom backoff interval.

        Raises:
            ValueError: If api_key is None.
        """
        super().__init__(default_wait_time=default_wait_time)
        if api_key is None:
            raise ValueError(
                "API key required for AnthropicProvider. "
                "Call configure_api_keys() or provide api_key parameter."
            )
        self._anthropic_console = anthropic.Anthropic(api_key=api_key)

    def _call_model(self, *, model_id: str, prompt: str, options: dict[str, Any]) -> str:
        call_args: dict[str, Any] = {
            **options,
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        with self._anthropic_console.messages.stream(**call_args) as stream:
            stream.until_done()

            try:
                return stream.get_final_text().strip()
            except RuntimeError:
                logger.error(
                    "LLM provider response did not include text content. response=%s",
                    response_to_plain_text(stream.get_final_message()),
                )
                raise
