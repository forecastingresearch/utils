"""Unit tests for LLM provider API key handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from utils.llm.providers.anthropic import AnthropicProvider  # type: ignore[import]
from utils.llm.providers.google import GoogleProvider  # type: ignore[import]
from utils.llm.providers.mistral import MistralProvider  # type: ignore[import]
from utils.llm.providers.openai import OpenAIProvider  # type: ignore[import]
from utils.llm.providers.together import TogetherProvider  # type: ignore[import]
from utils.llm.providers.xai import XAIProvider  # type: ignore[import]


@pytest.mark.parametrize(
    "provider_class",
    [
        OpenAIProvider,
        AnthropicProvider,
        GoogleProvider,
        XAIProvider,
        MistralProvider,
        TogetherProvider,
    ],
)
def test_provider_requires_api_key(provider_class):
    """Test that providers raise ValueError when api_key is None."""
    with pytest.raises(ValueError, match="API key required"):
        provider_class(api_key=None)


@pytest.mark.parametrize(
    "provider_class,api_key",
    [
        (OpenAIProvider, "sk-test-openai-key"),
        (AnthropicProvider, "sk-ant-test-anthropic-key"),
        (GoogleProvider, "test-google-key"),
        (XAIProvider, "test-xai-key"),
        (MistralProvider, "test-mistral-key"),
        (TogetherProvider, "test-together-key"),
    ],
)
def test_provider_accepts_api_key(provider_class, api_key):
    """Test that providers can be instantiated with explicit API keys."""
    # Mock the underlying client initialization
    if provider_class == OpenAIProvider:
        with patch("utils.llm.providers.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            provider = provider_class(api_key=api_key)
            mock_openai.assert_called_once_with(api_key=api_key)
            assert provider is not None
    elif provider_class == AnthropicProvider:
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            provider = provider_class(api_key=api_key)
            mock_anthropic.assert_called_once_with(api_key=api_key)
            assert provider is not None
    elif provider_class == GoogleProvider:
        with patch("utils.llm.providers.google.genai.Client") as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            provider = provider_class(api_key=api_key)
            mock_client.assert_called_once_with(api_key=api_key)
            assert provider is not None
    elif provider_class == XAIProvider:
        with patch("utils.llm.providers.xai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            provider = provider_class(api_key=api_key)
            mock_openai.assert_called_once_with(api_key=api_key, base_url="https://api.x.ai/v1")
            assert provider is not None
    elif provider_class == MistralProvider:
        with patch("utils.llm.providers.mistral.Mistral") as mock_mistral:
            mock_client = MagicMock()
            mock_mistral.return_value = mock_client
            provider = provider_class(api_key=api_key)
            mock_mistral.assert_called_once_with(api_key=api_key)
            assert provider is not None
    elif provider_class == TogetherProvider:
        with patch("utils.llm.providers.together.Together") as mock_together:
            mock_client = MagicMock()
            mock_together.return_value = mock_client
            provider = provider_class(api_key=api_key)
            mock_together.assert_called_once_with(api_key=api_key)
            assert provider is not None
