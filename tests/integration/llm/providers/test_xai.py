"""Integration tests for xAI model helpers."""

from __future__ import annotations

import pytest

from tests.integration.helpers import assert_capital_of_france  # type: ignore[import]
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]

XAI_MODEL: Model | None = next((model for model in MODELS if model.model_key == "grok-4.3"), None)
assert XAI_MODEL is not None


@pytest.mark.integration
def test_xai_provider_get_response_live_call():
    """It invokes the live xAI API and returns text."""
    assert_capital_of_france(
        lambda prompt: XAI_MODEL.get_response(
            prompt,
            options={"temperature": 0},
        )
    )
