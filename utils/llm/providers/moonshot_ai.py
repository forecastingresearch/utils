"""Helpers for invoking Moonshot AI models with retry support."""

from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI  # type: ignore[import]

from .base import BaseLLMProvider

MOONSHOT_AI_BASE_URL = "https://api.moonshot.ai/v1"


class MoonshotAIProvider(BaseLLMProvider):
    """LLM provider that wraps Moonshot AI's OpenAI-compatible chat API."""

    retry_message = "Moonshot AI API request failed."

    def __init__(self, *, api_key: str | None = None, default_wait_time: int | None = None) -> None:
        """Instantiate the Moonshot AI client using the provided API key.

        Args:
            api_key: Moonshot AI API key. If None, an error will be raised.
            default_wait_time: Optional custom backoff interval.

        Raises:
            ValueError: If api_key is None.
        """
        super().__init__(default_wait_time=default_wait_time)
        if api_key is None:
            raise ValueError(
                "API key required for MoonshotAIProvider. "
                "Call configure_api_keys() or provide api_key parameter."
            )
        self._moonshot_ai_client = OpenAI(
            api_key=api_key,
            base_url=MOONSHOT_AI_BASE_URL,
        )

    def _call_model(self, *, model_id: str, prompt: str, options: dict[str, Any]) -> str:
        request_payload: Dict[str, Any] = {
            **options,
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        response = self._moonshot_ai_client.chat.completions.create(**request_payload)
        message_content = response.choices[0].message.content
        if message_content is None:
            raise RuntimeError("Moonshot AI response did not include text content.")
        if isinstance(message_content, str):
            return message_content.strip()
        return str(message_content).strip()
