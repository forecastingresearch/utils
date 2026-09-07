"""Helpers for invoking Meta models with retry support."""

from __future__ import annotations

from openai import OpenAI  # type: ignore[import]

from .openai import OpenAIProvider

META_MODEL_API_BASE_URL = "https://api.meta.ai/v1"


class MetaProvider(OpenAIProvider):
    """LLM provider that wraps the Meta Model API through its OpenAI-compatible Responses route."""

    retry_message = "Meta Model API request failed."

    def _create_client(self, api_key: str) -> OpenAI:
        """Return an OpenAI SDK client pointed at the Meta Model API."""
        return OpenAI(api_key=api_key, base_url=META_MODEL_API_BASE_URL)
