"""Helpers for invoking Mistral models with retry support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union

from mistralai import Mistral, UserMessage  # type: ignore[import]

from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model


class MistralProvider(BaseLLMProvider):
    """LLM provider that calls the Mistral chat completion API."""

    rate_limit_message = "Mistral API request exceeded rate limit."

    def __init__(self, *, api_key: str | None = None, default_wait_time: int | None = None) -> None:
        """Instantiate the Mistral client using the provided API key.

        Args:
            api_key: Mistral API key. If None, an error will be raised.
            default_wait_time: Optional custom backoff interval.

        Raises:
            ValueError: If api_key is None.
        """
        super().__init__(default_wait_time=default_wait_time)
        if api_key is None:
            raise ValueError(
                "API key required for MistralProvider. "
                "Call configure_api_keys() or provide api_key parameter."
            )
        self._mistral_client = Mistral(api_key=api_key)

    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        max_tokens = options.get("max_tokens")
        temperature = options.get("temperature")
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

        content = chat_response.choices[0].message.content

        # Handle different content types: None, str, or list
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Content can be a list of content blocks, extract text from each
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                else:
                    text_parts.append(str(item))
            return "".join(text_parts)

        return str(content)
