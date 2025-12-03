"""Central model registry for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, Type

from ..gcp.secret_manager import get_secret
from ..helpers.constants import (
    ANTHROPIC_API_KEY_SECRET_NAME,
    GOOGLE_GEMINI_API_KEY_SECRET_NAME,
    MISTRAL_API_KEY_SECRET_NAME,
    OPENAI_API_KEY_SECRET_NAME,
    TOGETHER_API_KEY_SECRET_NAME,
    XAI_API_KEY_SECRET_NAME,
)
from .lab_registry import LABS, Lab
from .providers.anthropic import AnthropicProvider
from .providers.base import BaseLLMProvider
from .providers.google import GoogleProvider
from .providers.mistral import MistralProvider
from .providers.openai import OpenAIProvider
from .providers.together import TogetherProvider
from .providers.xai import XAIProvider

# Registry for API keys by provider class
_PROVIDER_API_KEYS: dict[Type[BaseLLMProvider], str] = {}

# Mapping from provider name strings to provider classes
_PROVIDER_NAME_TO_CLASS: dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "xai": XAIProvider,
    "together": TogetherProvider,
    "mistral": MistralProvider,
}

# Mapping from provider classes to GCP secret names
_PROVIDER_CLASS_TO_SECRET_NAME: dict[Type[BaseLLMProvider], str] = {
    OpenAIProvider: OPENAI_API_KEY_SECRET_NAME,
    AnthropicProvider: ANTHROPIC_API_KEY_SECRET_NAME,
    GoogleProvider: GOOGLE_GEMINI_API_KEY_SECRET_NAME,
    XAIProvider: XAI_API_KEY_SECRET_NAME,
    TogetherProvider: TOGETHER_API_KEY_SECRET_NAME,
    MistralProvider: MISTRAL_API_KEY_SECRET_NAME,
}


@dataclass(frozen=True, slots=True)
class Model:
    """Registered LLM model metadata."""

    id: str
    full_name: str
    token_limit: int
    provider_cls: Type[BaseLLMProvider]
    lab: Lab
    org: str | None = None
    source: str | None = None
    reasoning_model: bool = False

    def get_response(self, prompt: str, **options: Any) -> str:
        """Request a response from the model's provider."""
        provider = _get_provider_instance(self.provider_cls)
        return provider.get_response(self, prompt, **options)


def _get_api_key_for_provider(provider_cls: Type[BaseLLMProvider]) -> str | None:
    """Look up API key for a provider from the registry configuration.

    Returns:
        API key string if configured, None otherwise.
    """
    return _PROVIDER_API_KEYS.get(provider_cls)


@lru_cache(maxsize=None)
def _get_provider_instance(provider_cls: Type[BaseLLMProvider]) -> BaseLLMProvider:
    """Return a cached provider instance for the given provider class."""
    api_key = _get_api_key_for_provider(provider_cls)
    if api_key is not None:
        return provider_cls(api_key=api_key)
    return provider_cls()


def configure_api_keys(
    *,
    from_gcp: bool = False,
    openai: str | None = None,
    anthropic: str | None = None,
    google: str | None = None,
    xai: str | None = None,
    together: str | None = None,
    mistral: str | None = None,
) -> None:
    """Configure API keys for LLM providers.

    This function allows you to set API keys either explicitly or by loading them
    from GCP Secret Manager. Once configured, these keys will be used automatically
    when providers are instantiated through the model registry.

    Args:
        from_gcp: If True, load all API keys from GCP Secret Manager. If False,
            only the explicitly provided keys will be configured.
        openai: OpenAI API key (e.g., "sk-...")
        anthropic: Anthropic API key (e.g., "sk-ant-...")
        google: Google Gemini API key
        xai: xAI API key
        together: Together AI API key
        mistral: Mistral API key

    Examples:
        # For non-GCP users:
        configure_api_keys(openai="sk-...", anthropic="sk-ant-...")

        # For GCP users:
        configure_api_keys(from_gcp=True)

        # Mixed: some explicit, some from GCP
        configure_api_keys(from_gcp=True, openai="custom-key")
    """
    if from_gcp:
        # Load all keys from GCP Secret Manager
        for provider_cls, secret_name in _PROVIDER_CLASS_TO_SECRET_NAME.items():
            try:
                api_key = get_secret(secret_name)
                _PROVIDER_API_KEYS[provider_cls] = api_key
            except RuntimeError:
                # GCP not configured or secret doesn't exist, skip this provider
                pass

    # Set explicitly provided keys (these override GCP keys if both are set)
    key_mapping = {
        "openai": (OpenAIProvider, openai),
        "anthropic": (AnthropicProvider, anthropic),
        "google": (GoogleProvider, google),
        "xai": (XAIProvider, xai),
        "together": (TogetherProvider, together),
        "mistral": (MistralProvider, mistral),
    }

    for provider_cls, api_key in key_mapping.values():
        if api_key is not None:
            _PROVIDER_API_KEYS[provider_cls] = api_key

    # Clear the provider instance cache since keys have changed
    _get_provider_instance.cache_clear()


def validate_provider_keys(models: list[Model]) -> None:
    """Validate that all providers needed by the given models have API keys configured.

    Args:
        models: List of Model objects to validate.

    Raises:
        ValueError: If any model's provider lacks a configured API key.
    """
    missing_keys = []
    provider_names = {cls: name for name, cls in _PROVIDER_NAME_TO_CLASS.items()}

    for model in models:
        provider_cls = model.provider_cls
        if provider_cls not in _PROVIDER_API_KEYS:
            provider_name = provider_names.get(provider_cls, provider_cls.__name__)
            missing_keys.append(f"{provider_name} (for model {model.id})")

    if missing_keys:
        missing_list = ", ".join(missing_keys)
        raise ValueError(
            f"API keys not configured for the following providers: {missing_list}. "
            "Call configure_api_keys() or configure_api_keys(from_gcp=True) to set them."
        )


MODELS: Final[list[Model]] = [
    Model(
        id="gpt-4.1-mini",
        full_name="gpt-4.1-mini",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
    ),
    Model(
        id="gpt-5-2025-08-07",
        full_name="gpt-5-2025-08-07",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
        reasoning_model=True,
    ),
    Model(
        id="gpt-5-mini-2025-08-07",
        full_name="gpt-5-mini-2025-08-07",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
        reasoning_model=True,
    ),
    Model(
        id="gpt-5-nano-2025-08-07",
        full_name="gpt-5-nano-2025-08-07",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
        reasoning_model=True,
    ),
    Model(
        id="gpt-5.1-2025-11-13",
        full_name="gpt-5.1-2025-11-13",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
        reasoning_model=True,
    ),
    Model(
        id="o3-2025-04-16",
        full_name="o3-2025-04-16",
        token_limit=200_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
        reasoning_model=True,
    ),
    Model(
        id="gpt-4.1-2025-04-14",
        full_name="gpt-4.1-2025-04-14",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
    ),
    Model(
        id="DeepSeek-V3.1",
        full_name="deepseek-ai/DeepSeek-V3.1",
        token_limit=128_000,
        provider_cls=TogetherProvider,
        lab=LABS["DeepSeek"],
    ),
    # NOTE: This model is no longer available in the Together API.
    # Model(
    #     id="Qwen2.5-Coder-32B-Instruct",
    #     full_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    #     token_limit=262_144,
    #     provider_cls=TogetherProvider,
    #     lab=LABS["Qwen"],
    # ),
    Model(
        id="Qwen3-235B-A22B-fp8-tput",
        full_name="Qwen/Qwen3-235B-A22B-fp8-tput",
        token_limit=40_960,
        provider_cls=TogetherProvider,
        lab=LABS["Qwen"],
    ),
    Model(
        id="Qwen3-235B-A22B-Thinking-2507",
        full_name="Qwen/Qwen3-235B-A22B-Thinking-2507",
        token_limit=262_144,
        provider_cls=TogetherProvider,
        lab=LABS["Qwen"],
    ),
    Model(
        id="Kimi-K2-Instruct",
        full_name="moonshotai/Kimi-K2-Instruct",
        token_limit=128_000,
        provider_cls=TogetherProvider,
        lab=LABS["Moonshot"],
    ),
    Model(
        id="Kimi-K2-Instruct-0905",
        full_name="moonshotai/Kimi-K2-Instruct-0905",
        token_limit=262_144,
        provider_cls=TogetherProvider,
        lab=LABS["Moonshot"],
    ),
    Model(
        id="Kimi-K2-Thinking",
        full_name="moonshotai/Kimi-K2-Thinking",
        token_limit=262_144,
        provider_cls=TogetherProvider,
        lab=LABS["Moonshot"],
        reasoning_model=False,
    ),
    Model(
        id="GLM-4.5-Air-FP8",
        full_name="zai-org/GLM-4.5-Air-FP8",
        token_limit=131_072,
        provider_cls=TogetherProvider,
        lab=LABS["Z.ai"],
    ),
    Model(
        id="GLM-4.6",
        full_name="zai-org/GLM-4.6",
        token_limit=202_752,
        provider_cls=TogetherProvider,
        lab=LABS["Z.ai"],
        reasoning_model=False,
    ),
    Model(
        id="claude-sonnet-4-5-20250929",
        full_name="claude-sonnet-4-5-20250929",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="claude-haiku-4-5-20251001",
        full_name="claude-haiku-4-5-20251001",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="claude-opus-4-1-20250805",
        full_name="claude-opus-4-1-20250805",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="claude-sonnet-4-20250514",
        full_name="claude-sonnet-4-20250514",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="claude-3-7-sonnet-20250219",
        full_name="claude-3-7-sonnet-20250219",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="grok-4-fast-reasoning",
        full_name="grok-4-fast-reasoning",
        token_limit=2_000_000,
        provider_cls=XAIProvider,
        lab=LABS["xAI"],
    ),
    Model(
        id="grok-4-fast-non-reasoning",
        full_name="grok-4-fast-non-reasoning",
        token_limit=2_000_000,
        provider_cls=XAIProvider,
        lab=LABS["xAI"],
    ),
    Model(
        id="grok-4-0709",
        full_name="grok-4-0709",
        token_limit=256_000,
        provider_cls=XAIProvider,
        lab=LABS["xAI"],
    ),
    Model(
        id="grok-4-1-fast-reasoning",
        full_name="grok-4-1-fast-reasoning",
        token_limit=2_000_000,
        provider_cls=XAIProvider,
        lab=LABS["xAI"],
        reasoning_model=True,
    ),
    Model(
        id="grok-4-1-fast-non-reasoning",
        full_name="grok-4-1-fast-non-reasoning",
        token_limit=2_000_000,
        provider_cls=XAIProvider,
        lab=LABS["xAI"],
        reasoning_model=False,
    ),
    Model(
        id="gemini-2.5-pro",
        full_name="gemini-2.5-pro",
        token_limit=1_048_576,
        provider_cls=GoogleProvider,
        lab=LABS["Google"],
    ),
    Model(
        id="gemini-2.5-flash",
        full_name="models/gemini-2.5-flash",
        token_limit=1_048_576,
        provider_cls=GoogleProvider,
        lab=LABS["Google"],
    ),
    Model(
        id="gemini-3-pro-preview",
        full_name="gemini-3-pro-preview",
        token_limit=1_048_576,
        provider_cls=GoogleProvider,
        lab=LABS["Google"],
        reasoning_model=False,
    ),
    Model(
        id="mistral-large-2411",
        full_name="mistral-large-2411",
        token_limit=128_000,
        provider_cls=MistralProvider,
        lab=LABS["Mistral"],
    ),
    Model(
        id="magistral-medium-2506",
        full_name="magistral-medium-2506",
        token_limit=40_000,
        provider_cls=MistralProvider,
        lab=LABS["Mistral"],
    ),
]
