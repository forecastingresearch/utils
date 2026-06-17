"""Helpers for invoking Google Gemini models with retry support."""

from __future__ import annotations

from typing import Any, Dict

from google import genai
from google.genai import types

from ..utils import response_to_plain_text
from .base import BaseLLMProvider

_ROUTING_ENVELOPE_KEYS = frozenset({"model", "contents"})


class GoogleProvider(BaseLLMProvider):
    """LLM provider that wraps Google Gemini API calls."""

    retry_message = "Google AI API request failed."

    def __init__(self, *, api_key: str | None = None, default_wait_time: int | None = None) -> None:
        """Instantiate the Google client using the provided API key.

        Args:
            api_key: Google Gemini API key. If None, an error will be raised.
            default_wait_time: Optional custom backoff interval.

        Raises:
            ValueError: If api_key is None.
        """
        super().__init__(default_wait_time=default_wait_time)
        if api_key is None:
            raise ValueError(
                "API key required for GoogleProvider. "
                "Call configure_api_keys() or provide api_key parameter."
            )
        self._google_ai_client = genai.Client(api_key=api_key)

    def _call_model(self, *, model_id: str, prompt: str, options: dict[str, Any]) -> str:
        request_payload: Dict[str, Any] = {
            "model": model_id,
            "contents": prompt,
        }
        if options:
            config_options = {
                key: value for key, value in options.items() if key not in _ROUTING_ENVELOPE_KEYS
            }
            if config_options:
                request_payload["config"] = types.GenerateContentConfig(**config_options)

        response = self._google_ai_client.models.generate_content(**request_payload)
        text = response.text
        if text is None:
            raise RuntimeError(
                "Google response did not include text. "
                f"response={response_to_plain_text(response)}"
            )
        return text.strip()
