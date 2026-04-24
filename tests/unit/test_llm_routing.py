"""Unit tests for LLM provider routing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from utils.llm.lab_registry import LABS
from utils.llm.provider_registry import PROVIDERS, Provider


def test_labs_have_leaderboard_names():
    """Lab registry should support leaderboard display metadata."""
    assert LABS["DeepSeek"].name == "DeepSeek"
    assert LABS["OpenAI"].leaderboard_name == "OpenAI"
    assert LABS["Moonshot"].name == "Moonshot"
    assert LABS["Moonshot"].leaderboard_name == "Moonshot AI"
    assert "Google" not in LABS
    assert LABS["Google DeepMind"].name == "Google DeepMind"
    assert LABS["Google DeepMind"].leaderboard_name == "Google DeepMind"


def test_provider_registry_contains_supported_api_routes():
    """Provider registry should describe API routes separately from labs."""
    assert PROVIDERS["Together"].name == "Together"
    assert PROVIDERS["Anthropic"].name == "Anthropic"
    assert PROVIDERS["Together"].key_name == "together"


def test_get_response_routes_by_provider_and_preserves_options():
    """The public router should pass provider, model ID, prompt, and options unchanged."""
    from utils.llm import model_registry
    from utils.llm.model_registry import TogetherProvider

    observed: dict[str, Any] = {}
    options = {"temperature": 0, "tools": [{"type": "web_search"}]}

    class FakeProvider:
        def get_response(
            self,
            *,
            model_id: str,
            prompt: str,
            options: dict[str, Any],
        ) -> str:
            observed["model_id"] = model_id
            observed["prompt"] = prompt
            observed["options"] = options
            return "forecast text"

    with patch.object(
        model_registry,
        "_get_provider_instance",
        return_value=FakeProvider(),
    ) as mock_get_provider_instance:
        response = model_registry.get_response(
            PROVIDERS["Together"],
            "deepseek-ai/DeepSeek-V3.1",
            "forecast prompt",
            options,
        )

    assert response == "forecast text"
    mock_get_provider_instance.assert_called_once_with(TogetherProvider)
    assert observed["prompt"] == "forecast prompt"
    assert observed == {
        "model_id": "deepseek-ai/DeepSeek-V3.1",
        "prompt": "forecast prompt",
        "options": options,
    }


def test_model_get_response_routes_full_name_and_options():
    """Model routing should call providers through the public final interface."""
    from utils.llm import model_registry
    from utils.llm.model_registry import Model, OpenAIProvider

    observed: dict[str, Any] = {}
    options = {"temperature": 0, "max_tokens": 128}
    model = Model(
        id="reasoning-model",
        full_name="reasoning-model",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
        reasoning_model=True,
    )

    class FakeProvider:
        def get_response(
            self,
            *,
            model_id: str,
            prompt: str,
            options: dict[str, Any],
        ) -> str:
            observed["model_id"] = model_id
            observed["prompt"] = prompt
            observed["options"] = options
            return "reasoning text"

    with patch.object(model_registry, "_get_provider_instance", return_value=FakeProvider()):
        response = model.get_response("forecast prompt", options=options)

    assert response == "reasoning text"
    assert observed == {
        "model_id": model.full_name,
        "prompt": "forecast prompt",
        "options": options,
    }


def test_validate_provider_keys_uses_providers_not_labs():
    """Provider key validation should not require a key for third-party model-making labs."""
    from utils.llm.model_registry import (
        _PROVIDER_API_KEYS,
        configure_api_keys,
        validate_provider_keys,
    )

    _PROVIDER_API_KEYS.clear()
    configure_api_keys(together="test-together")

    validate_provider_keys([PROVIDERS["Together"]])
    _PROVIDER_API_KEYS.clear()


def test_validate_provider_keys_reports_missing_provider():
    """Missing API provider keys should raise a clear error."""
    from utils.llm.model_registry import (
        _PROVIDER_API_KEYS,
        configure_api_keys,
        validate_provider_keys,
    )

    _PROVIDER_API_KEYS.clear()
    configure_api_keys()

    with pytest.raises(ValueError, match="Together"):
        validate_provider_keys([PROVIDERS["Together"]])


def test_validate_provider_keys_rejects_non_provider_objects():
    """Provider key validation should fail clearly for old model-based inputs."""
    from utils.llm.model_registry import MODELS, validate_provider_keys

    with pytest.raises(TypeError, match="Provider"):
        validate_provider_keys([MODELS[0]])  # type: ignore[list-item]


def test_provider_name_lookup_accepts_provider_objects_only():
    """Provider routing should not accidentally accept labs as providers."""
    from utils.llm.model_registry import get_response

    with pytest.raises(TypeError, match="Provider"):
        get_response(
            provider=LABS["DeepSeek"],  # type: ignore[arg-type]
            model_id="deepseek-ai/DeepSeek-V3.1",
            prompt="forecast prompt",
            options={},
        )


def test_get_response_reports_unsupported_provider():
    """Provider routing should reject providers outside the supported route registry."""
    from utils.llm.model_registry import get_response

    unsupported_provider = Provider(name="Unsupported", key_name="unsupported")

    with pytest.raises(ValueError, match="Unsupported"):
        get_response(
            provider=unsupported_provider,
            model_id="unsupported/model",
            prompt="forecast prompt",
            options={},
        )


def test_openai_provider_forwards_options_without_defaults():
    """Provider should forward only caller-supplied OpenAI options."""
    from utils.llm.providers.openai import OpenAIProvider

    with patch("utils.llm.providers.openai.OpenAI") as mock_openai:
        response = MagicMock(status="completed", output_text=" 42 ")
        mock_client = MagicMock()
        mock_client.responses.create.return_value = response
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        text = provider._call_model(
            model_id="gpt-5-mini-2025-08-07",
            prompt="forecast",
            options={"reasoning": {"effort": "low"}, "tools": [{"type": "web_search"}]},
        )

    assert text == "42"
    mock_client.responses.create.assert_called_once_with(
        model="gpt-5-mini-2025-08-07",
        input="forecast",
        reasoning={"effort": "low"},
        tools=[{"type": "web_search"}],
    )


def test_openai_provider_route_fields_override_reserved_options():
    """Provider should keep OpenAI route-owned fields authoritative."""
    from utils.llm.providers.openai import OpenAIProvider

    with patch("utils.llm.providers.openai.OpenAI") as mock_openai:
        response = MagicMock(status="completed", output_text="forecast")
        mock_client = MagicMock()
        mock_client.responses.create.return_value = response
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        provider._call_model(
            model_id="gpt-5-mini-2025-08-07",
            prompt="forecast",
            options={"model": "wrong-model", "input": "wrong-prompt", "temperature": 0},
        )

    mock_client.responses.create.assert_called_once_with(
        model="gpt-5-mini-2025-08-07",
        input="forecast",
        temperature=0,
    )


def test_anthropic_provider_forwards_options_without_asserting_max_tokens():
    """Anthropic provider should forward caller options without default max tokens."""
    from utils.llm.providers.anthropic import AnthropicProvider

    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.get_final_text.return_value = "forecast"

    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = stream
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        text = provider._call_model(
            model_id="claude-opus-4-6",
            prompt="forecast",
            options={
                "max_tokens": 16000,
                "thinking": {"type": "adaptive"},
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            },
        )

    assert text == "forecast"
    mock_client.messages.stream.assert_called_once_with(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "forecast"}],
        max_tokens=16000,
        thinking={"type": "adaptive"},
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
    )


def test_anthropic_provider_route_fields_override_reserved_options():
    """Anthropic provider should keep route-owned fields authoritative."""
    from utils.llm.providers.anthropic import AnthropicProvider

    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.get_final_text.return_value = "forecast"

    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = stream
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        provider._call_model(
            model_id="claude-opus-4-6",
            prompt="forecast",
            options={
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "wrong-prompt"}],
                "max_tokens": 16000,
            },
        )

    mock_client.messages.stream.assert_called_once_with(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "forecast"}],
        max_tokens=16000,
    )


def test_anthropic_provider_uses_sdk_final_text_helper():
    """Provider should use Anthropic SDK final-text extraction."""
    from utils.llm.providers.anthropic import AnthropicProvider

    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.get_final_text.return_value = "  *0.25\n*0.50  "

    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = stream
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        text = provider._call_model(
            model_id="claude-opus-4-6",
            prompt="forecast",
            options={"max_tokens": 4096},
        )

    assert text == "*0.25\n*0.50"
    stream.get_final_text.assert_called_once_with()
    stream.get_final_message.assert_not_called()


def test_anthropic_provider_logs_raw_response_when_response_has_no_text(caplog):
    """Provider should log the raw provider response when text extraction fails."""
    from utils.llm.providers.anthropic import AnthropicProvider

    final_message = MagicMock()
    final_message.content = [
        SimpleNamespace(type="thinking", thinking="private reasoning"),
        SimpleNamespace(type="server_tool_use", name="web_search"),
    ]
    final_message.model_dump_json.return_value = '{"raw":"provider response"}'
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.get_final_text.side_effect = RuntimeError(
        ".get_final_text() can only be called when the API returns a `text` content block."
    )
    stream.get_final_message.return_value = final_message

    caplog.set_level("ERROR", logger="utils.llm.providers.anthropic")
    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = stream
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        with pytest.raises(RuntimeError, match="content block"):
            provider._call_model(
                model_id="claude-opus-4-6",
                prompt="forecast",
                options={"max_tokens": 4096},
            )

    assert "LLM provider response did not include text content" in caplog.text
    assert "response=" in caplog.text
    assert '{"raw":"provider response"}' in caplog.text


def test_google_provider_forwards_options_as_generate_content_config():
    """Google provider should treat options as GenerateContentConfig fields."""
    from utils.llm.providers.google import GoogleProvider

    with patch("utils.llm.providers.google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "forecast"
        mock_client_cls.return_value = mock_client

        provider = GoogleProvider(api_key="google-test")
        text = provider._call_model(
            model_id="gemini-2.5-pro",
            prompt="forecast",
            options={"temperature": 0, "tools": [{"googleSearch": {}}]},
        )

    assert text == "forecast"
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-pro"
    assert call_kwargs["contents"] == "forecast"
    assert call_kwargs["config"].model_dump(exclude_none=True) == {
        "temperature": 0.0,
        "tools": [{"google_search": {}}],
    }


def test_google_provider_route_fields_are_not_config_options():
    """Google provider should not let route-owned fields become config options."""
    from utils.llm.providers.google import GoogleProvider

    with patch("utils.llm.providers.google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "forecast"
        mock_client_cls.return_value = mock_client

        provider = GoogleProvider(api_key="google-test")
        text = provider._call_model(
            model_id="gemini-2.5-pro",
            prompt="forecast",
            options={"model": "wrong", "contents": "wrong", "temperature": 0},
        )

    assert text == "forecast"
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-pro"
    assert call_kwargs["contents"] == "forecast"
    assert call_kwargs["config"].model_dump(exclude_none=True) == {"temperature": 0.0}


def test_xai_provider_route_fields_override_reserved_options():
    """Provider should keep xAI route-owned fields authoritative."""
    from utils.llm.providers.xai import XAIProvider

    with patch("utils.llm.providers.xai.openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_text = "forecast"
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = XAIProvider(api_key="xai-test")
        text = provider._call_model(
            model_id="grok-4-0709",
            prompt="forecast",
            options={
                "model": "wrong-model",
                "input": "wrong-prompt",
                "temperature": 0,
            },
        )

    assert text == "forecast"
    mock_client.responses.create.assert_called_once_with(
        model="grok-4-0709",
        input="forecast",
        temperature=0,
    )


def test_together_provider_route_fields_override_reserved_options():
    """Together provider should keep route-owned fields authoritative."""
    from utils.llm.providers.together import TogetherProvider

    with patch("utils.llm.providers.together.Together") as mock_together:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "forecast"
        mock_together.return_value = mock_client

        provider = TogetherProvider(api_key="together-test")
        text = provider._call_model(
            model_id="deepseek-ai/DeepSeek-V3.1",
            prompt="forecast",
            options={
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "wrong-prompt"}],
                "temperature": 0,
            },
        )

    assert text == "forecast"
    mock_client.chat.completions.create.assert_called_once_with(
        model="deepseek-ai/DeepSeek-V3.1",
        messages=[{"role": "user", "content": "forecast"}],
        temperature=0,
    )


def test_retry_helper_does_not_return_prompt_rewrite_sentinel():
    """Retry helper should raise provider errors instead of returning sentinels."""
    from utils.llm.utils import get_response_with_retry

    def failing_call() -> str:
        raise RuntimeError("repetitive patterns")

    with pytest.raises(RuntimeError, match="repetitive patterns"):
        get_response_with_retry(
            api_call=failing_call,
            wait_time=0,
            error_msg="provider failed",
            max_retries=1,
        )
