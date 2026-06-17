"""Integration tests for Google Gemini model helpers."""

from __future__ import annotations

import pytest

from tests.integration.helpers import assert_capital_of_france  # type: ignore[import]
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]

GOOGLE_MODEL: Model | None = next(
    (model for model in MODELS if model.model_key == "gemini-2.5-pro"), None
)
assert GOOGLE_MODEL is not None


@pytest.mark.integration
def test_google_provider_get_response_live_call():
    """It invokes the live Google Gemini API and returns text."""
    assert_capital_of_france(
        lambda prompt: GOOGLE_MODEL.get_response(
            prompt,
            options={"temperature": 0},
        )
    )
