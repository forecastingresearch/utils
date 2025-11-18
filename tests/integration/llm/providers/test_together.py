"""Integration tests for Together AI model helpers."""

from __future__ import annotations

import pytest
import utils.llm.providers.together as together_module  # type: ignore[import]
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]
from utils.tests.integration.helpers import (
    assert_capital_of_france,  # type: ignore[import]
)

TOGETHER_MODEL: Model | None = next(
    (model for model in MODELS if model.id == "GLM-4.5-Air-FP8"), None
)
assert TOGETHER_MODEL is not None


@pytest.mark.integration
def test_together_provider_get_response_live_call():
    """It invokes the live Together AI API and returns text."""
    from utils.llm.model_registry import (
        _get_api_key_for_provider,  # type: ignore[import]
    )
    from utils.llm.providers.together import TogetherProvider  # type: ignore[import]

    # API keys are already configured by the session-scoped fixture
    api_key = _get_api_key_for_provider(TogetherProvider)
    assert api_key is not None, "API key should be configured by fixture"
    provider = together_module.TogetherProvider(api_key=api_key)
    assert_capital_of_france(
        lambda prompt: provider.get_response(
            TOGETHER_MODEL,
            prompt,
            temperature=0,
            wait_time=1,
            max_tokens=256,
        )
    )
