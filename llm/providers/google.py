"""Helpers for invoking Google Gemini models with retry support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from google import genai
from google.genai import types

from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model


class GoogleProvider(BaseLLMProvider):
    """LLM provider that wraps Google Gemini API calls."""

    rate_limit_message = "Google AI API request exceeded rate limit."

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

    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        temperature = options.get("temperature")
        model_name = model.full_name

        config_kwargs: Dict[str, Any] = {
            "candidate_count": 1,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        }
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        response = self._google_ai_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text
