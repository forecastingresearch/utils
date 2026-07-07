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
        # Moonshot requires streaming: non-streamed requests buffer the whole completion
        # server-side, so the idle connection is dropped by intermediaries before the
        # response arrives. Streaming keeps bytes flowing and returns the assembled text.
        request_payload: Dict[str, Any] = {
            **options,
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }

        stream = self._moonshot_ai_client.chat.completions.create(**request_payload)
        content_parts = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                content_parts.append(delta_content)

        content = "".join(content_parts).strip()
        if not content:
            raise RuntimeError("Moonshot AI response did not include text content.")
        return content
