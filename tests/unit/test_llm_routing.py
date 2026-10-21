"""Unit tests for LLM provider routing."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from utils.llm.lab_registry import LABS
from utils.llm.provider_registry import PROVIDERS, Provider


def test_labs_have_display_names():
    """Lab registry names should be display-ready."""
    assert LABS["DeepSeek"].name == "DeepSeek"
    assert LABS["OpenAI"].name == "OpenAI"
    assert LABS["Moonshot"].name == "Moonshot AI"
    assert LABS["MiniMax"].name == "MiniMax"
    assert "Google" not in LABS
    assert LABS["Google DeepMind"].name == "Google DeepMind"


def test_provider_registry_contains_supported_api_routes():
    """Provider registry should describe API routes separately from labs."""
    assert PROVIDERS["Together"].name == "Together"
    assert PROVIDERS["Anthropic"].name == "Anthropic"
    assert PROVIDERS["Moonshot AI"].name == "Moonshot AI"
    assert PROVIDERS["Moonshot AI"].key_name == "moonshot_ai"
    assert PROVIDERS["Meta"].name == "Meta"
    assert PROVIDERS["Meta"].key_name == "meta"


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


def test_model_get_response_routes_provider_model_id_and_options():
    """Model routing should call providers through the public final interface."""
    from utils.llm import model_registry
    from utils.llm.model_registry import Model

    observed: dict[str, Any] = {}
    options = {"temperature": 0, "max_tokens": 128}
    model = Model(
        model_key="reasoning-model",
        provider_model_id="reasoning-model",
        provider=PROVIDERS["OpenAI"],
        lab=LABS["OpenAI"],
        manual_release_date=date(2026, 1, 1),
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
        "model_id": model.provider_model_id,
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
        background=True,
    )


def test_openai_provider_does_not_retry_forbidden_responses():
    """A 403 such as a misalignment policy violation must fail without retrying."""
    import httpx
    from openai import PermissionDeniedError

    from utils.llm.providers.openai import OpenAIProvider

    forbidden = PermissionDeniedError(
        "misalignment_policy_violation",
        response=httpx.Response(403, request=httpx.Request("POST", "https://api.openai.com")),
        body={"code": "misalignment_policy_violation"},
    )

    with (
        patch("utils.llm.providers.openai.OpenAI") as mock_openai,
        patch("utils.llm.utils.time.sleep") as mock_sleep,
    ):
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = forbidden
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        with pytest.raises(PermissionDeniedError):
            provider.get_response(model_id="gpt-6-astra", prompt="forecast", options={})

    assert mock_client.responses.create.call_count == 1
    mock_sleep.assert_not_called()


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
        background=True,
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
                "output_config": {"effort": "max"},
                "thinking": {"type": "adaptive"},
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            },
        )

    assert text == "forecast"
    mock_client.messages.stream.assert_called_once_with(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "forecast"}],
        max_tokens=16000,
        output_config={"effort": "max"},
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


def test_anthropic_provider_uses_beta_messages_for_fallback_options():
    """Anthropic fallback options should route through the beta Messages API."""
    from utils.llm.providers.anthropic import AnthropicProvider

    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.get_final_text.return_value = "forecast"

    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.beta.messages.stream.return_value = stream
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        text = provider._call_model(
            model_id="claude-fable-5",
            prompt="forecast",
            options={
                "max_tokens": 16000,
                "fallbacks": [{"model": "claude-opus-4-8"}],
                "betas": ["server-side-fallback-2026-06-01"],
            },
        )

    assert text == "forecast"
    mock_client.beta.messages.stream.assert_called_once_with(
        model="claude-fable-5",
        messages=[{"role": "user", "content": "forecast"}],
        max_tokens=16000,
        fallbacks=[{"model": "claude-opus-4-8"}],
        betas=["server-side-fallback-2026-06-01"],
    )
    mock_client.messages.stream.assert_not_called()


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
    # The final message is read for its id and usage only, after the text helper ran.
    stream.get_final_message.assert_called_once_with()


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


def test_google_provider_rejects_response_without_text():
    """Google provider should not return None when the SDK response has no text."""
    from utils.llm.providers.google import GoogleProvider

    with patch("utils.llm.providers.google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = None
        mock_client_cls.return_value = mock_client

        provider = GoogleProvider(api_key="google-test")
        with pytest.raises(RuntimeError, match="Google response did not include text"):
            provider._call_model(
                model_id="gemini-2.5-pro",
                prompt="forecast",
                options={},
            )


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


def test_moonshot_ai_provider_streams_openai_chat_completions_route():
    """Moonshot AI provider should stream Kimi through the OpenAI-compatible chat API."""
    from utils.llm.providers.moonshot_ai import MoonshotAIProvider

    def _chunk(content):
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content=content))]
        return chunk

    usage_only_chunk = MagicMock()
    usage_only_chunk.choices = []

    with patch("utils.llm.providers.moonshot_ai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [
            _chunk(None),  # role-only opening chunk carries no text
            _chunk(" fore"),
            _chunk("cast "),
            usage_only_chunk,  # trailing usage chunk has no choices
        ]
        mock_openai.return_value = mock_client

        provider = MoonshotAIProvider(api_key="moonshot-test")
        text = provider._call_model(
            model_id="kimi-k2.6",
            prompt="forecast",
            options={
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "wrong-prompt"}],
                "max_tokens": 16000,
            },
        )

    assert text == "forecast"
    mock_openai.assert_called_once_with(
        api_key="moonshot-test",
        base_url="https://api.moonshot.ai/v1",
    )
    mock_client.chat.completions.create.assert_called_once_with(
        model="kimi-k2.6",
        messages=[{"role": "user", "content": "forecast"}],
        max_tokens=16000,
        stream=True,
    )


def test_meta_provider_uses_openai_responses_route():
    """Meta provider should call the Meta Model API through the OpenAI Responses route."""
    from utils.llm.providers.meta import MetaProvider

    with patch("utils.llm.providers.meta.OpenAI") as mock_openai:
        response = MagicMock(status="completed", output_text=" 42 ")
        mock_client = MagicMock()
        mock_client.responses.create.return_value = response
        mock_openai.return_value = mock_client

        provider = MetaProvider(api_key="meta-test")
        text = provider._call_model(
            model_id="muse-spark-1.3",
            prompt="forecast",
            options={"reasoning": {"effort": "xhigh"}, "tools": [{"type": "web_search"}]},
        )

    assert text == "42"
    mock_openai.assert_called_once_with(api_key="meta-test", base_url="https://api.meta.ai/v1")
    mock_client.responses.create.assert_called_once_with(
        model="muse-spark-1.3",
        input="forecast",
        reasoning={"effort": "xhigh"},
        tools=[{"type": "web_search"}],
    )
    mock_client.chat.completions.create.assert_not_called()


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


def _openai_background_client(statuses: list[str], **final_fields: Any) -> MagicMock:
    """Fake OpenAI client whose background response moves through `statuses` on retrieve."""
    created = MagicMock(id="resp_123", status=statuses[0])
    polled = [MagicMock(id="resp_123", status=status, **final_fields) for status in statuses[1:]]
    mock_client = MagicMock()
    mock_client.responses.create.return_value = created
    mock_client.responses.retrieve.side_effect = polled
    return mock_client


def test_openai_provider_polls_background_response_and_reports_id_and_usage():
    """A background response is polled until completed; its id goes to the caller at once."""
    from utils.llm.providers.openai import OpenAIProvider
    from utils.llm.types import CallContext

    usage = SimpleNamespace(
        input_tokens=71_000,
        output_tokens=2_300,
        output_tokens_details=SimpleNamespace(reasoning_tokens=1_900),
    )
    output = [
        SimpleNamespace(type="web_search_call"),
        SimpleNamespace(type="reasoning"),
        SimpleNamespace(type="web_search_call"),
        SimpleNamespace(type="message"),
    ]
    seen_ids: list[str] = []
    polls_before_id: list[int] = []

    with (
        patch("utils.llm.providers.openai.OpenAI") as mock_openai,
        patch("utils.llm.providers.openai.time.sleep") as mock_sleep,
    ):
        mock_client = _openai_background_client(
            ["queued", "in_progress", "completed"],
            output_text=" 0.42 ",
            usage=usage,
            output=output,
        )
        mock_openai.return_value = mock_client

        def record_id(response_id: str) -> None:
            seen_ids.append(response_id)
            polls_before_id.append(mock_client.responses.retrieve.call_count)

        provider = OpenAIProvider(api_key="sk-test")
        response = provider.get_response_with_metadata(
            model_id="gpt-6-astra",
            prompt="forecast",
            options={"reasoning": {"effort": "high"}},
            context=CallContext(on_response_id=record_id),
        )

    assert response.text == "0.42"
    assert response.provider_response_id == "resp_123"
    assert response.status == "completed"
    assert response.usage.input_tokens == 71_000
    assert response.usage.output_tokens == 2_300
    assert response.usage.reasoning_tokens == 1_900
    assert response.usage.tool_calls == 2
    assert response.attempts == 1
    assert response.duration_s is not None
    assert seen_ids == ["resp_123"]
    assert polls_before_id == [0]
    assert mock_client.responses.retrieve.call_count == 2
    assert mock_sleep.call_count == 2
    mock_client.responses.cancel.assert_not_called()


def test_openai_provider_cancels_background_response_without_retrying():
    """A set cancel event cancels the server-side response and is never retried."""
    import threading

    from utils.llm.providers.openai import OpenAIProvider
    from utils.llm.types import CallContext, LLMCallCancelled

    cancel_event = threading.Event()

    with patch("utils.llm.providers.openai.OpenAI") as mock_openai:
        mock_client = _openai_background_client(["queued", "completed"], output_text="late")
        created = mock_client.responses.create.return_value

        def accept_then_cancel(**kwargs):
            del kwargs
            cancel_event.set()  # the caller gives up right after the request is accepted
            return created

        mock_client.responses.create.side_effect = accept_then_cancel
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        with pytest.raises(LLMCallCancelled):
            provider.get_response_with_metadata(
                model_id="gpt-6-astra",
                prompt="forecast",
                options={},
                context=CallContext(cancel_event=cancel_event),
            )

    mock_client.responses.create.assert_called_once()
    mock_client.responses.cancel.assert_called_once_with("resp_123")
    mock_client.responses.retrieve.assert_not_called()


def test_openai_provider_budget_cancels_overrunning_response():
    """The run's `timeout` option bounds the whole background response, then cancels it."""
    from utils.llm.providers.openai import OpenAIProvider

    with (
        patch("utils.llm.providers.openai.OpenAI") as mock_openai,
        patch("utils.llm.providers.openai.time.sleep"),
        patch("utils.llm.providers.openai.time.monotonic", side_effect=[0.0, 0.0, 1000.0]),
    ):
        mock_client = _openai_background_client(["queued", "in_progress", "in_progress"])
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        with pytest.raises(TimeoutError, match="did not finish within 900s"):
            provider._call_model(
                model_id="gpt-5.6-sol", prompt="forecast", options={"timeout": 900}
            )

    # The budget never reaches the SDK as a per-request timeout.
    assert "timeout" not in mock_client.responses.create.call_args.kwargs
    mock_client.responses.cancel.assert_called_once_with("resp_123")


def test_openai_provider_reports_failed_background_response_with_its_id():
    """A terminal non-completed status raises, naming the response so it can be looked up."""
    from utils.llm.providers.openai import OpenAIProvider

    with (
        patch("utils.llm.providers.openai.OpenAI") as mock_openai,
        patch("utils.llm.providers.openai.time.sleep"),
    ):
        mock_client = _openai_background_client(
            ["queued", "incomplete"], incomplete_details=SimpleNamespace(reason="max_output_tokens")
        )
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        with pytest.raises(RuntimeError, match="status=incomplete, id=resp_123"):
            provider._call_model(model_id="gpt-6-astra", prompt="forecast", options={})


def test_openai_provider_fetches_response_by_id():
    """fetch_response returns a finished response without creating a new one."""
    from utils.llm.providers.openai import OpenAIProvider

    with patch("utils.llm.providers.openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.responses.retrieve.side_effect = [
            MagicMock(id="resp_9", status="in_progress"),
            MagicMock(id="resp_9", status="completed", output_text="0.7", usage=None, output=[]),
        ]
        mock_openai.return_value = mock_client
        provider = OpenAIProvider(api_key="sk-test")

        pending = provider.fetch_response("resp_9")
        finished = provider.fetch_response("resp_9")

    assert (pending.status, pending.text) == ("in_progress", "")
    assert (finished.status, finished.text, finished.provider_response_id) == (
        "completed",
        "0.7",
        "resp_9",
    )
    assert finished.usage.tool_calls == 0
    mock_client.responses.create.assert_not_called()


def test_anthropic_provider_reports_message_id_and_usage():
    """The streamed final message supplies the id, token counts and server tool uses."""
    from utils.llm.providers.anthropic import AnthropicProvider
    from utils.llm.types import CallContext

    message = SimpleNamespace(
        id="msg_abc",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1200, output_tokens=340),
        content=[
            SimpleNamespace(type="server_tool_use"),
            SimpleNamespace(type="web_search_tool_result"),
            SimpleNamespace(type="text"),
        ],
    )
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.get_final_text.return_value = "0.3"
    stream.get_final_message.return_value = message
    seen_ids: list[str] = []

    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = stream
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-ant-test")
        response = provider.get_response_with_metadata(
            model_id="claude-opus-5-5",
            prompt="forecast",
            options={"max_tokens": 4096},
            context=CallContext(on_response_id=seen_ids.append),
        )

    assert response.text == "0.3"
    assert response.provider_response_id == "msg_abc"
    assert response.status == "end_turn"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (1200, 340)
    assert response.usage.tool_calls == 1
    assert seen_ids == ["msg_abc"]


def test_openai_provider_retries_a_failed_poll_without_recreating_the_response():
    """A poll that fails is polled again; the generation is never started a second time."""
    from utils.llm.providers.openai import OpenAIProvider

    with (
        patch("utils.llm.providers.openai.OpenAI") as mock_openai,
        patch("utils.llm.providers.openai.time.sleep"),
    ):
        mock_client = MagicMock()
        mock_client.responses.create.return_value = MagicMock(id="resp_1", status="queued")
        mock_client.responses.retrieve.side_effect = [
            ConnectionError("vpn dropped it"),
            ConnectionError("still dropping"),
            MagicMock(id="resp_1", status="completed", output_text="0.5", usage=None, output=[]),
        ]
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        response = provider.get_response_with_metadata(
            model_id="gpt-6-astra", prompt="forecast", options={}
        )

    assert response.text == "0.5"
    assert mock_client.responses.create.call_count == 1
    assert mock_client.responses.retrieve.call_count == 3
    assert response.attempts == 1


def test_openai_provider_keeps_polling_through_failures_until_the_budget():
    """Failing polls never start a second generation; only the budget ends the wait."""
    import itertools

    from utils.llm.providers.openai import OpenAIProvider

    clock = itertools.count(start=0.0, step=100.0)
    with (
        patch("utils.llm.providers.openai.OpenAI") as mock_openai,
        patch("utils.llm.providers.openai.time.sleep"),
        patch("utils.llm.providers.openai.time.monotonic", side_effect=lambda: next(clock)),
    ):
        mock_client = MagicMock()
        mock_client.responses.create.return_value = MagicMock(id="resp_1", status="queued")
        mock_client.responses.retrieve.side_effect = ConnectionError("down")
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test", default_wait_time=0)
        with pytest.raises(TimeoutError, match="the last 9 polls failed"):
            provider.get_response_with_metadata(
                model_id="gpt-5.6-sol", prompt="forecast", options={"timeout": 900}
            )

    assert mock_client.responses.create.call_count == 1
    assert mock_client.responses.retrieve.call_count == 9
    mock_client.responses.cancel.assert_called_once_with("resp_1")


def test_openai_provider_does_not_retry_an_incomplete_response():
    """An incomplete response (e.g. the output-token cap) is raised, not paid for again."""
    from utils.llm.providers.openai import OpenAIProvider
    from utils.llm.types import LLMResponseIncomplete

    with (
        patch("utils.llm.providers.openai.OpenAI") as mock_openai,
        patch("utils.llm.providers.openai.time.sleep"),
    ):
        mock_client = _openai_background_client(
            ["queued", "incomplete"], incomplete_details=SimpleNamespace(reason="max_output_tokens")
        )
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test", default_wait_time=0)
        with pytest.raises(LLMResponseIncomplete, match="max_output_tokens"):
            provider.get_response_with_metadata(
                model_id="gpt-6-astra", prompt="forecast", options={}
            )

    assert mock_client.responses.create.call_count == 1


def test_openai_provider_retries_a_failed_background_response():
    """A response that failed server-side produced nothing, so the call is retried."""
    from utils.llm.providers.openai import OpenAIProvider

    with (
        patch("utils.llm.providers.openai.OpenAI") as mock_openai,
        patch("utils.llm.providers.openai.time.sleep"),
    ):
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = [
            MagicMock(id="resp_1", status="queued"),
            MagicMock(id="resp_2", status="queued"),
        ]
        mock_client.responses.retrieve.side_effect = [
            MagicMock(id="resp_1", status="failed", incomplete_details=None, error="server_error"),
            MagicMock(id="resp_2", status="completed", output_text="0.5", usage=None, output=[]),
        ]
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test", default_wait_time=0)
        response = provider.get_response_with_metadata(
            model_id="gpt-6-astra", prompt="forecast", options={}
        )

    assert (response.text, response.provider_response_id, response.attempts) == ("0.5", "resp_2", 2)


def test_openai_provider_can_stop_waiting_without_cancelling_the_response():
    """With cancel_server_side=False a cancelled wait leaves the response to be fetched later."""
    import threading

    from utils.llm.providers.openai import OpenAIProvider
    from utils.llm.types import CallContext, LLMCallCancelled

    cancel_event = threading.Event()
    seen_ids: list[str] = []

    with patch("utils.llm.providers.openai.OpenAI") as mock_openai:
        mock_client = _openai_background_client(["queued", "completed"], output_text="late")
        created = mock_client.responses.create.return_value

        def accept_then_cancel(**kwargs):
            del kwargs
            cancel_event.set()
            return created

        mock_client.responses.create.side_effect = accept_then_cancel
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        with pytest.raises(LLMCallCancelled, match="keeps running"):
            provider.get_response_with_metadata(
                model_id="gpt-6-astra",
                prompt="forecast",
                options={},
                context=CallContext(
                    cancel_event=cancel_event,
                    on_response_id=seen_ids.append,
                    cancel_server_side=False,
                ),
            )

    assert seen_ids == ["resp_123"]
    mock_client.responses.create.assert_called_once()
    mock_client.responses.cancel.assert_not_called()


def test_openai_provider_waits_for_an_existing_response_by_id():
    """wait_for_response polls a response another process started; no create call."""
    from utils.llm.providers.openai import OpenAIProvider

    with (
        patch("utils.llm.providers.openai.OpenAI") as mock_openai,
        patch("utils.llm.providers.openai.time.sleep"),
    ):
        mock_client = MagicMock()
        mock_client.responses.retrieve.side_effect = [
            MagicMock(id="resp_old", status="in_progress"),
            MagicMock(id="resp_old", status="in_progress"),
            MagicMock(id="resp_old", status="completed", output_text="0.8", usage=None, output=[]),
        ]
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        response = provider.wait_for_response("resp_old")

    assert (response.text, response.provider_response_id, response.status) == (
        "0.8",
        "resp_old",
        "completed",
    )
    mock_client.responses.create.assert_not_called()
    assert mock_client.responses.retrieve.call_count == 3


def test_failed_attempts_are_retried_per_provider_billing():
    """The unbilled OpenAI failures get five attempts; a billed Anthropic stream gets two."""
    from utils.llm.providers.anthropic import AnthropicProvider
    from utils.llm.providers.openai import OpenAIProvider

    with (
        patch("utils.llm.providers.openai.OpenAI") as mock_openai,
        patch("utils.llm.utils.time.sleep"),
    ):
        mock_openai.return_value.responses.create.side_effect = ConnectionError("create failed")
        provider = OpenAIProvider(api_key="sk-test", default_wait_time=0)
        with pytest.raises(ConnectionError):
            provider.get_response_with_metadata(model_id="gpt-6-astra", prompt="p", options={})
        assert mock_openai.return_value.responses.create.call_count == 5

    with patch("anthropic.Anthropic") as mock_anthropic, patch("utils.llm.utils.time.sleep"):
        mock_anthropic.return_value.messages.stream.side_effect = ConnectionError("stream dropped")
        provider = AnthropicProvider(api_key="sk-ant-test", default_wait_time=0)
        with pytest.raises(ConnectionError):
            provider.get_response_with_metadata(
                model_id="claude-opus-5-5", prompt="p", options={"max_tokens": 16}
            )
        assert mock_anthropic.return_value.messages.stream.call_count == 2
