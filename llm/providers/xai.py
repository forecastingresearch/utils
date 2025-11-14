"""Helpers for invoking xAI models with retry support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import openai

from ...helpers.constants import XAI_API_KEY_SECRET_NAME
from ...keys.secrets import get_secret
from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model


class XAIProvider(BaseLLMProvider):
    """LLM provider that wraps calls to the xAI chat completion endpoint."""

    rate_limit_message = "xAI API request exceeded rate limit."

    def __init__(self) -> None:
        """Instantiate the xAI client using the configured API secret."""
        super().__init__()
        api_key = get_secret(XAI_API_KEY_SECRET_NAME)
        self._xai_client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        temperature = options.get("temperature")
        max_tokens = options.get("max_tokens")
        model_name = model.full_name

        request_payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": temperature,
        }
        if temperature is not None:
            request_payload["temperature"] = temperature
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens

        response = self._xai_client.chat.completions.create(**request_payload)

        return response.choices[0].message.content
