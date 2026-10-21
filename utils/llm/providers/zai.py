"""Helpers for invoking Z.ai GLM models with retry support."""

from __future__ import annotations

from .openai_compatible_chat import OpenAICompatibleChatProvider

ZAI_API_BASE_URL = "https://api.z.ai/api/paas/v4"


class ZAIProvider(OpenAICompatibleChatProvider):
    """LLM provider that wraps Z.ai's OpenAI-compatible chat completions API."""

    provider_name = "Z.ai"
    api_base_url = ZAI_API_BASE_URL
    retry_message = "Z.ai API request failed."
