"""Integration tests for Anthropic model helpers."""

from __future__ import annotations

import pytest

from tests.integration.helpers import assert_capital_of_france
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]

ANTHROPIC_MODEL: Model | None = next(
    (model for model in MODELS if model.model_key == "claude-sonnet-4-6"), None
)
assert ANTHROPIC_MODEL is not None


@pytest.mark.integration
def test_anthropic_provider_get_response_live_call():
    """It invokes the live Anthropic API and returns text."""
    assert_capital_of_france(
        lambda prompt: ANTHROPIC_MODEL.get_response(
            prompt,
            options={"temperature": 0, "max_tokens": 16},
        )
    )
