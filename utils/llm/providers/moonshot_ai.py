"""Helpers for invoking Moonshot AI models with retry support."""

from __future__ import annotations

from .openai_compatible_chat import OpenAICompatibleChatProvider

MOONSHOT_AI_BASE_URL = "https://api.moonshot.ai/v1"


class MoonshotAIProvider(OpenAICompatibleChatProvider):
    """LLM provider that wraps Moonshot AI's OpenAI-compatible chat API."""

    provider_name = "Moonshot AI"
    api_base_url = MOONSHOT_AI_BASE_URL
    retry_message = "Moonshot AI API request failed."
