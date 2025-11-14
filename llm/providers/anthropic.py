"""Helpers for invoking Anthropic models with retry support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import anthropic

from ...helpers.constants import ANTHROPIC_API_KEY_SECRET_NAME
from ...keys.secrets import get_secret
from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """LLM provider that communicates with the Anthropic Messages API."""

    rate_limit_message = "Anthropic API request exceeded rate limit."

    def __init__(self) -> None:
        """Instantiate the Anthropic client using the configured API secret."""
        super().__init__()
        api_key = get_secret(ANTHROPIC_API_KEY_SECRET_NAME)
        self._anthropic_console = anthropic.Anthropic(api_key=api_key)

    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        temperature = options.get("temperature")
        max_tokens = options.get("max_tokens")
        model_name = model.full_name

        call_args: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        if temperature is not None:
            call_args["temperature"] = temperature
        if max_tokens is not None:
            call_args["max_tokens"] = max_tokens

        with self._anthropic_console.messages.stream(**call_args) as stream:
            stream.until_done()

        return stream.get_final_message().content[0].text
