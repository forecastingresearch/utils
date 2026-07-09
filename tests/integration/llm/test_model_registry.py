"""Integration tests that validate representative registry models can be invoked."""

from __future__ import annotations

import pytest

from utils.llm.model_registry import (  # type: ignore[import]
    MODELS,
    MODELS_BY_KEY,
    Model,
)
from utils.llm.provider_registry import PROVIDERS  # type: ignore[import]

from ..helpers import assert_capital_of_france

# Curated representatives per provider. Together churns its serverless catalog, so its
# representative is chosen dynamically from active models rather than hardcoded.
SMOKE_TEST_MODEL_KEYS = [
    "gpt-5-mini-2025-08-07",
    "claude-sonnet-4-6",
    "grok-4.3",
    "gemini-2.5-pro",
]

LATEST_ACTIVE_TOGETHER_MODEL: Model = max(
    (model for model in MODELS if model.provider == PROVIDERS["Together"] and model.active),
    key=lambda model: (model.release_date, model.model_key),
)

SMOKE_TEST_MODELS = [MODELS_BY_KEY[model_key] for model_key in SMOKE_TEST_MODEL_KEYS] + [
    LATEST_ACTIVE_TOGETHER_MODEL
]


def _minimal_options_for_model(model: Model) -> dict:
    if model.provider == PROVIDERS["Anthropic"]:
        return {"max_tokens": 16}
    if model.provider == PROVIDERS["OpenAI"]:
        return {"max_output_tokens": 256}
    if model.provider == PROVIDERS["Together"]:
        return {"temperature": 0, "max_tokens": 256}
    if model.provider == PROVIDERS["xAI"]:
        return {"temperature": 0}
    if model.provider == PROVIDERS["Google"]:
        return {}
    return {}


def test_together_smoke_options_leave_room_for_answer_text():
    """Keep routed Together smoke calls aligned with the provider smoke path."""
    assert _minimal_options_for_model(LATEST_ACTIVE_TOGETHER_MODEL) == {
        "temperature": 0,
        "max_tokens": 256,
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    "model",
    SMOKE_TEST_MODELS,
    ids=lambda item: item.model_key,
)
def test_registered_model_live_call(model: Model):
    """Representative active model entries should be callable via their providers."""
    assert model.active is True
    assert_capital_of_france(
        lambda prompt: model.get_response(
            prompt,
            options=_minimal_options_for_model(model),
        )
    )
