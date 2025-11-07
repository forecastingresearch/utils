"""Helpers for invoking Google Gemini models with retry support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types

from ...helpers.constants import GOOGLE_GEMINI_API_KEY_SECRET_NAME
from ...keys.secrets import get_secret
from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model


class GoogleProvider(BaseLLMProvider):
    """LLM provider that wraps Google Gemini API calls."""

    rate_limit_message = "Google AI API request exceeded rate limit."

    def __init__(self) -> None:
        """Instantiate the Google client using the configured API secret."""
        super().__init__()
        api_key = get_secret(GOOGLE_GEMINI_API_KEY_SECRET_NAME)
        self._google_ai_client = genai.Client(api_key=api_key)

    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        temperature = options.get("temperature", 0.8)  # TODO put defaults in constants?
        model_name = model.full_name

        response = self._google_ai_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                candidate_count=1,
                temperature=temperature,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )
        return response.text
