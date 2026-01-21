"""Helpers for invoking Anthropic models with retry support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import anthropic

from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """LLM provider that communicates with the Anthropic Messages API."""

    rate_limit_message = "Anthropic API request exceeded rate limit."

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

    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        temperature = options.get("temperature")
        max_tokens = options.get("max_tokens")
        assert max_tokens is not None, "max_tokens is required for Anthropic models."
        tools = options.get("tools")
        model_name = model.full_name

        call_args: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            call_args["temperature"] = temperature
        if tools is not None:
            call_args["tools"] = tools

        with self._anthropic_console.messages.stream(**call_args) as stream:
            stream.until_done()

        message = stream.get_final_message()
        # Extract text from the last text block (web search responses have multiple blocks)
        text_blocks = [block for block in message.content if block.type == "text"]
        if not text_blocks:
            return ""
        return "".join(block.text for block in text_blocks)
