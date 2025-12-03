"""Integration tests for xAI model helpers."""

from __future__ import annotations

import pytest

import utils.llm.providers.xai as xai_module  # type: ignore[import]
from tests.integration.helpers import (  # type: ignore[import]
    assert_capital_of_france,
    assert_structured_person_extraction,
)
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]

XAI_MODEL: Model | None = next((model for model in MODELS if model.id == "grok-4-0709"), None)
assert XAI_MODEL is not None


@pytest.mark.integration
def test_xai_provider_get_response_live_call():
    """It invokes the live xAI API and returns text."""
    from utils.llm.model_registry import (
        _get_api_key_for_provider,  # type: ignore[import]
    )
    from utils.llm.providers.xai import XAIProvider  # type: ignore[import]

    # API keys are already configured by the session-scoped fixture
    api_key = _get_api_key_for_provider(XAIProvider)
    assert api_key is not None, "API key should be configured by fixture"
    provider = xai_module.XAIProvider(api_key=api_key)
    assert_capital_of_france(
        lambda prompt: provider.get_response(
            XAI_MODEL,
            prompt,
            temperature=0,
            wait_time=1,
        )
    )


@pytest.mark.integration
def test_xai_structured_output():
    """It returns structured output matching the Pydantic schema."""
    assert_structured_person_extraction(
        lambda prompt, schema, **options: XAI_MODEL.get_structured_response(
            prompt, schema, temperature=0, max_tokens=100, wait_time=1, **options
        )
    )
