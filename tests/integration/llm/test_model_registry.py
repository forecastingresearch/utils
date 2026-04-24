"""Integration tests that validate every registry model can be invoked."""

from __future__ import annotations

import pytest

from utils.llm.model_registry import MODELS, Model  # type: ignore[import]
from utils.llm.providers.anthropic import AnthropicProvider  # type: ignore[import]
from utils.llm.providers.google import GoogleProvider  # type: ignore[import]
from utils.llm.providers.openai import OpenAIProvider  # type: ignore[import]
from utils.llm.providers.together import TogetherProvider  # type: ignore[import]
from utils.llm.providers.xai import XAIProvider  # type: ignore[import]

from ..helpers import assert_capital_of_france


def _minimal_options_for_model(model: Model) -> dict:
    if model.provider_cls is AnthropicProvider:
        return {"max_tokens": 16}
    if model.provider_cls is OpenAIProvider:
        return {"max_output_tokens": 16}
    if model.provider_cls in {TogetherProvider, XAIProvider}:
        return {"max_tokens": 16}
    if model.provider_cls is GoogleProvider:
        return {}
    return {}


@pytest.mark.integration
@pytest.mark.parametrize("model", MODELS, ids=lambda item: item.id)
def test_registered_model_live_call(model: Model):
    """Each model entry should be callable via its registered provider."""
    assert_capital_of_france(
        lambda prompt: model.get_response(
            prompt,
            options=_minimal_options_for_model(model),
        )
    )
