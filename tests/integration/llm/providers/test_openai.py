"""Integration tests for OpenAI model helpers."""

from __future__ import annotations

import pytest

from tests.integration.helpers import assert_capital_of_france  # type: ignore[import]
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]

OPENAI_MODEL: Model | None = next(
    (model for model in MODELS if model.model_key == "gpt-5-mini-2025-08-07"), None
)
assert OPENAI_MODEL is not None


@pytest.mark.integration
def test_openai_provider_get_response_live_call():
    """It invokes the live OpenAI API and returns text."""
    assert_capital_of_france(
        lambda prompt: OPENAI_MODEL.get_response(
            prompt,
            options={"max_output_tokens": 256},
        )
    )
