"""Integration tests that validate every registry model can be invoked."""

from __future__ import annotations

import pytest

from utils.llm.model_registry import MODELS, Model  # type: ignore[import]

from ..helpers import assert_capital_of_france


@pytest.mark.integration
@pytest.mark.parametrize("model", MODELS, ids=lambda item: item.id)
def test_registered_model_live_call(model: Model):
    """Each model entry should be callable via its registered provider."""
    assert_capital_of_france(
        lambda prompt: model.get_response(
            prompt,
            temperature=0,
            max_tokens=256,
            wait_time=1,
        )
    )
