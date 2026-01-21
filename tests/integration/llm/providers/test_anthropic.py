"""Integration tests for Anthropic model helpers."""

from __future__ import annotations

import pytest

import utils.llm.providers.anthropic as anthropic_module  # type: ignore[import]
from tests.integration.helpers import (
    assert_capital_of_france,
)
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]

ANTHROPIC_MODEL: Model | None = next(
    (model for model in MODELS if model.id == "claude-3-7-sonnet-20250219"), None
)
assert ANTHROPIC_MODEL is not None


@pytest.mark.integration
def test_anthropic_provider_get_response_live_call():
    """It invokes the live Anthropic API and returns text."""
    from utils.llm.model_registry import (
        _get_api_key_for_provider,  # type: ignore[import]
    )
    from utils.llm.providers.anthropic import AnthropicProvider  # type: ignore[import]

    # API keys are already configured by the session-scoped fixture
    api_key = _get_api_key_for_provider(AnthropicProvider)
    assert api_key is not None, "API key should be configured by fixture"
    provider = anthropic_module.AnthropicProvider(api_key=api_key)
    assert_capital_of_france(
        lambda prompt: provider.get_response(
            ANTHROPIC_MODEL,
            prompt,
            temperature=0,
            max_tokens=16,
            wait_time=1,
        )
    )


@pytest.mark.integration
def test_anthropic_provider_web_search():
    """It invokes web search and returns current information."""
    from utils.llm.model_registry import (
        _get_api_key_for_provider,  # type: ignore[import]
    )
    from utils.llm.providers.anthropic import AnthropicProvider  # type: ignore[import]

    api_key = _get_api_key_for_provider(AnthropicProvider)
    assert api_key is not None, "API key should be configured by fixture"
    provider = anthropic_module.AnthropicProvider(api_key=api_key)

    # Ask a question with a stable, verifiable answer
    prompt = "What is the official website URL for Python? Just respond with the URL."
    response = provider.get_response(
        ANTHROPIC_MODEL,
        prompt,
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 1,
            }
        ],
        max_tokens=256,
        wait_time=1,
    )

    assert isinstance(response, str)
    assert "python.org" in response.lower()
