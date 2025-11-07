"""Helpers for invoking Mistral models with retry support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union

from mistralai import Mistral, UserMessage  # type: ignore[import]

from ...helpers.constants import MISTRAL_API_KEY_SECRET_NAME
from ...keys.secrets import get_secret
from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model


class MistralProvider(BaseLLMProvider):
    """LLM provider that calls the Mistral chat completion API."""

    rate_limit_message = "Mistral API request exceeded rate limit."

    def __init__(self) -> None:
        """Instantiate the Mistral client using the configured API secret."""
        super().__init__()
        api_key = get_secret(MISTRAL_API_KEY_SECRET_NAME)
        self._mistral_client = Mistral(api_key=api_key)

    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        max_tokens = options.get("max_tokens")
        temperature = options.get("temperature", 0.8)  # TODO put defaults in constants?
        model_name = model.full_name

        messages: List[Union[Dict[str, str], UserMessage]] = [
            {"role": "user", "content": prompt},
        ]

        request_payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        if temperature is not None:
            request_payload["temperature"] = temperature
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens

        chat_response = self._mistral_client.chat.complete(**request_payload)

        return chat_response.choices[0].message.content
