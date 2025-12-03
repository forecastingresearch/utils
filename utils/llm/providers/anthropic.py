"""Helpers for invoking Anthropic models with retry support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import anthropic
from pydantic import BaseModel

from ..utils import create_json_prompt, extract_json_from_text, parse_and_validate_json
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

        with self._anthropic_console.messages.stream(**call_args) as stream:
            stream.until_done()

        return stream.get_final_message().content[0].text

    def _call_model_structured(
        self,
        model: "Model",
        prompt: str,
        response_schema: type[BaseModel],
        **options: Any,
    ) -> BaseModel:
        """Execute a structured output request using JSON parsing fallback.

        Note: Anthropic's structured outputs feature requires SDK 0.73.0+ and
        is not yet fully supported. This implementation uses prompt enhancement
        to request JSON output, which is then parsed and validated.
        """
        temperature = options.get("temperature")
        max_tokens = options.get("max_tokens")
        assert max_tokens is not None, "max_tokens is required for Anthropic models."
        model_name = model.full_name

        # Convert Pydantic schema to JSON schema and enhance prompt
        json_schema = response_schema.model_json_schema()
        enhanced_prompt = create_json_prompt(prompt, json_schema)

        call_args: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": enhanced_prompt,
                },
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            call_args["temperature"] = temperature

        with self._anthropic_console.messages.stream(**call_args) as stream:
            stream.until_done()

        final_message = stream.get_final_message()
        response_text = final_message.content[0].text

        # Extract and validate JSON
        json_text = extract_json_from_text(response_text, provider_name="Anthropic")
        return parse_and_validate_json(json_text, response_schema, "Anthropic", response_text)
