"""Unit tests for model registry API key configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from utils.llm.model_registry import (  # type: ignore[import]
    MODELS,
    AnthropicProvider,
    GoogleProvider,
    MistralProvider,
    OpenAIProvider,
    TogetherProvider,
    XAIProvider,
    configure_api_keys,
    validate_provider_keys,
)


def test_configure_api_keys_with_explicit_keys():
    """Test configure_api_keys() with explicitly provided keys."""
    from utils.llm.model_registry import _PROVIDER_API_KEYS  # type: ignore[import]

    # Clear any existing keys first
    _PROVIDER_API_KEYS.clear()

    configure_api_keys(
        openai="sk-test-openai",
        anthropic="sk-ant-test-anthropic",
        google="test-google",
        xai="test-xai",
        together="test-together",
        mistral="test-mistral",
    )

    # Verify keys are stored
    assert _PROVIDER_API_KEYS[OpenAIProvider] == "sk-test-openai"
    assert _PROVIDER_API_KEYS[AnthropicProvider] == "sk-ant-test-anthropic"
    assert _PROVIDER_API_KEYS[GoogleProvider] == "test-google"
    assert _PROVIDER_API_KEYS[XAIProvider] == "test-xai"
    assert _PROVIDER_API_KEYS[TogetherProvider] == "test-together"
    assert _PROVIDER_API_KEYS[MistralProvider] == "test-mistral"

    # Clear for other tests
    _PROVIDER_API_KEYS.clear()


def test_configure_api_keys_partial():
    """Test configure_api_keys() with only some keys provided."""
    from utils.llm.model_registry import _PROVIDER_API_KEYS  # type: ignore[import]

    # Clear any existing keys first
    _PROVIDER_API_KEYS.clear()

    configure_api_keys(openai="sk-test-openai", anthropic="sk-ant-test-anthropic")

    assert _PROVIDER_API_KEYS[OpenAIProvider] == "sk-test-openai"
    assert _PROVIDER_API_KEYS[AnthropicProvider] == "sk-ant-test-anthropic"
    assert GoogleProvider not in _PROVIDER_API_KEYS

    # Clear for other tests
    _PROVIDER_API_KEYS.clear()


@patch("utils.llm.model_registry.get_secret")
def test_configure_api_keys_from_gcp(mock_get_secret):
    """Test configure_api_keys(from_gcp=True) loads keys from GCP."""
    from utils.llm.model_registry import _PROVIDER_API_KEYS  # type: ignore[import]

    # Clear any existing keys first
    _PROVIDER_API_KEYS.clear()

    # Mock GCP secret manager responses
    mock_get_secret.side_effect = {
        "API_KEY_OPENAI": "sk-gcp-openai",
        "API_KEY_ANTHROPIC": "sk-ant-gcp-anthropic",
        "API_KEY_GEMINI": "gcp-google",
        "API_KEY_XAI": "gcp-xai",
        "API_KEY_TOGETHERAI": "gcp-together",
        "API_KEY_MISTRAL": "gcp-mistral",
    }.get

    configure_api_keys(from_gcp=True)

    assert _PROVIDER_API_KEYS[OpenAIProvider] == "sk-gcp-openai"
    assert _PROVIDER_API_KEYS[AnthropicProvider] == "sk-ant-gcp-anthropic"
    assert _PROVIDER_API_KEYS[GoogleProvider] == "gcp-google"
    assert _PROVIDER_API_KEYS[XAIProvider] == "gcp-xai"
    assert _PROVIDER_API_KEYS[TogetherProvider] == "gcp-together"
    assert _PROVIDER_API_KEYS[MistralProvider] == "gcp-mistral"

    # Clear for other tests
    _PROVIDER_API_KEYS.clear()


@patch("utils.llm.model_registry.get_secret")
def test_configure_api_keys_from_gcp_handles_missing_secrets(mock_get_secret):
    """Test configure_api_keys(from_gcp=True) handles missing GCP secrets gracefully."""
    from utils.llm.model_registry import _PROVIDER_API_KEYS  # type: ignore[import]

    # Clear any existing keys first
    _PROVIDER_API_KEYS.clear()

    # Mock some secrets existing, some missing
    def mock_get_secret_side_effect(secret_name: str) -> str:
        if secret_name == "API_KEY_OPENAI":
            return "sk-gcp-openai"
        elif secret_name == "API_KEY_ANTHROPIC":
            raise RuntimeError("GCP not configured")
        else:
            raise RuntimeError("Secret not found")

    mock_get_secret.side_effect = mock_get_secret_side_effect

    configure_api_keys(from_gcp=True)

    # Only OpenAI should be configured
    assert _PROVIDER_API_KEYS[OpenAIProvider] == "sk-gcp-openai"
    assert AnthropicProvider not in _PROVIDER_API_KEYS

    # Clear for other tests
    _PROVIDER_API_KEYS.clear()


def test_configure_api_keys_explicit_overrides_gcp():
    """Test that explicitly provided keys override GCP keys."""
    from utils.llm.model_registry import _PROVIDER_API_KEYS  # type: ignore[import]

    # Clear any existing keys first
    _PROVIDER_API_KEYS.clear()

    with patch("utils.llm.model_registry.get_secret") as mock_get_secret:
        mock_get_secret.return_value = "sk-gcp-openai"

        configure_api_keys(from_gcp=True, openai="sk-explicit-openai")

        # Explicit key should override GCP key
        assert _PROVIDER_API_KEYS[OpenAIProvider] == "sk-explicit-openai"

        # Clear for other tests
        _PROVIDER_API_KEYS.clear()


def test_validate_provider_keys_success():
    """Test validate_provider_keys() succeeds when all keys are configured."""
    from utils.llm.model_registry import _PROVIDER_API_KEYS  # type: ignore[import]

    # Clear any existing keys first
    _PROVIDER_API_KEYS.clear()

    configure_api_keys(
        openai="sk-test-openai",
        anthropic="sk-ant-test-anthropic",
        google="test-google",
        xai="test-xai",
        together="test-together",
        mistral="test-mistral",
    )

    # Should not raise
    validate_provider_keys(MODELS)

    # Clear for other tests
    _PROVIDER_API_KEYS.clear()


def test_validate_provider_keys_raises_on_missing():
    """Test validate_provider_keys() raises when keys are missing."""
    from utils.llm.model_registry import _PROVIDER_API_KEYS  # type: ignore[import]

    _PROVIDER_API_KEYS.clear()

    configure_api_keys(openai="sk-test-openai")

    # Should raise because other providers are missing
    with pytest.raises(ValueError, match="API keys not configured"):
        validate_provider_keys(MODELS)

    # Clear for other tests
    _PROVIDER_API_KEYS.clear()


def test_configure_api_keys_clears_provider_cache():
    """Test that configure_api_keys() clears the provider instance cache."""
    from utils.llm.model_registry import (  # type: ignore[import]
        _PROVIDER_API_KEYS,
        _get_provider_instance,
    )

    # Clear any existing keys first
    _PROVIDER_API_KEYS.clear()

    # Configure initial keys
    configure_api_keys(openai="sk-test-1")

    # Get an instance (will be cached)
    instance1 = _get_provider_instance(OpenAIProvider)

    # Configure different keys
    configure_api_keys(openai="sk-test-2")

    # Get instance again - should be new instance with new key
    instance2 = _get_provider_instance(OpenAIProvider)

    # Instances should be different (cache was cleared)
    assert instance1 is not instance2

    # Clear for other tests
    _PROVIDER_API_KEYS.clear()
