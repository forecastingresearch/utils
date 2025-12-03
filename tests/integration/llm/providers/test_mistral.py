"""Integration tests for Mistral model helpers."""

from __future__ import annotations

import pytest

import utils.llm.providers.mistral as mistral_module  # type: ignore[import]
from tests.integration.helpers import (  # type: ignore[import]
    assert_capital_of_france,
    assert_structured_person_extraction,
)
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]

MISTRAL_MODEL: Model | None = next(
    (model for model in MODELS if model.id == "magistral-medium-2506"), None
)
assert MISTRAL_MODEL is not None


@pytest.mark.integration
def test_mistral_provider_get_response_live_call():
    """It invokes the live Mistral API and returns text."""
    from utils.llm.model_registry import (
        _get_api_key_for_provider,  # type: ignore[import]
    )
    from utils.llm.providers.mistral import MistralProvider  # type: ignore[import]

    # API keys are already configured by the session-scoped fixture
    api_key = _get_api_key_for_provider(MistralProvider)
    assert api_key is not None, "API key should be configured by fixture"
    provider = mistral_module.MistralProvider(api_key=api_key)
    assert_capital_of_france(
        lambda prompt: provider.get_response(
            MISTRAL_MODEL,
            prompt,
            temperature=0,
            wait_time=1,
            max_tokens=256,
        )
    )


@pytest.mark.integration
def test_mistral_structured_output():
    """It returns structured output matching the Pydantic schema."""
    assert_structured_person_extraction(
        lambda prompt, schema, **options: MISTRAL_MODEL.get_structured_response(
            prompt, schema, temperature=0, max_tokens=200, wait_time=1, **options
        )
    )
