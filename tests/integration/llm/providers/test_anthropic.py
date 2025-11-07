"""Integration tests for Anthropic model helpers."""

from __future__ import annotations

import pytest
import utils.llm.providers.anthropic as anthropic_module  # type: ignore[import]
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]
from utils.tests.integration.helpers import (
    assert_capital_of_france,  # type: ignore[import]
)

ANTHROPIC_MODEL: Model | None = next(
    (model for model in MODELS if model.id == "claude-3-7-sonnet-20250219"), None
)
assert ANTHROPIC_MODEL is not None


@pytest.mark.integration
def test_anthropic_provider_get_response_live_call():
    """It invokes the live Anthropic API and returns text."""
    provider = anthropic_module.AnthropicProvider()
    assert_capital_of_france(
        lambda prompt: provider.get_response(
            ANTHROPIC_MODEL,
            prompt,
            temperature=0,
            max_tokens=16,
            wait_time=1,
        )
    )
