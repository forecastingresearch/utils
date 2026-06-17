"""Integration tests for Together AI model helpers."""

from __future__ import annotations

import pytest

from tests.integration.helpers import assert_capital_of_france  # type: ignore[import]
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]

TOGETHER_MODEL: Model | None = next(
    (model for model in MODELS if model.model_key == "minimax-m2.7"), None
)
assert TOGETHER_MODEL is not None
assert TOGETHER_MODEL.active is True


@pytest.mark.integration
def test_together_provider_get_response_live_call():
    """It invokes the live Together AI API and returns text."""
    assert_capital_of_france(
        lambda prompt: TOGETHER_MODEL.get_response(
            prompt,
            options={"temperature": 0, "max_tokens": 256},
        )
    )
