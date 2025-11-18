"""Integration tests for Google Gemini model helpers."""

from __future__ import annotations

import pytest
import utils.llm.providers.google as google_module  # type: ignore[import]
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]
from utils.tests.integration.helpers import (
    assert_capital_of_france,  # type: ignore[import]
)

GOOGLE_MODEL: Model | None = next(
    (model for model in MODELS if model.id == "gemini-2.5-flash"), None
)
assert GOOGLE_MODEL is not None


@pytest.mark.integration
def test_google_provider_get_response_live_call():
    """It invokes the live Google Gemini API and returns text."""
    from utils.llm.model_registry import (
        _get_api_key_for_provider,  # type: ignore[import]
    )
    from utils.llm.providers.google import GoogleProvider  # type: ignore[import]

    # API keys are already configured by the session-scoped fixture
    api_key = _get_api_key_for_provider(GoogleProvider)
    assert api_key is not None, "API key should be configured by fixture"
    provider = google_module.GoogleProvider(api_key=api_key)
    assert_capital_of_france(
        lambda prompt: provider.get_response(
            GOOGLE_MODEL,
            prompt,
            temperature=0,
            wait_time=1,
        )
    )
