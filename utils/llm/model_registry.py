"""Central model registry for LLM providers.

Adding a base model:

1. Look up the model in Models.dev. If present, copy its exact `provider_id` & `model_id` into
   `ModelsDevReference`; the checked-in snapshot is only a generated subset, not the catalog.
   In Models.dev source paths, `provider_id` is the folder under `providers/`, and
   `model_id` is the TOML filename stem under `models/`, e.g.
   `providers/anthropic/models/claude-opus-4-8.toml` -> `anthropic` / `claude-opus-4-8`.

2. Add the model to the provider-specific list below with the provider helper. `model_key` is our
   stable key; set `provider_model_id` only when the provider API ID differs. For routed
   providers like Together, set `lab_key`.

   Example:
   ```
   openai_model(
       model_key="gpt-5.5-2026-04-23",
       models_dev_reference=ModelsDevReference(
           provider_id="openai",
           model_id="gpt-5.5",
       ),
   )
   ```

3. If Models.dev is missing the model, lacks a full release date, or has a date we
   intentionally do not want to use, set `manual_release_date` on the model declaration.
   Do not add a separate release-date override map.

4. Insert the entry where `(release_date, model_key)` stays ascending within that provider
   list. Use `active=False` only for historical routes that should stay registered but
   leave `ACTIVE_MODEL_RUNS`.

5. Add benchmark call configs in `model_runs.py` with explicit `model_run_key` values.

After changing `ModelsDevReference` values, refresh the Models.dev snapshot from the repo's root
directory:
```
python - <<'PY'
from scripts.refresh_models_dev_metadata import write_models_dev_snapshot

write_models_dev_snapshot()
PY
```
Incorrect exact references fail with nearby Models.dev suggestions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any, Final, Type

from google.api_core import exceptions

from ..gcp.secret_manager import get_secret
from ..helpers.constants import (
    ANTHROPIC_API_KEY_SECRET_NAME,
    GOOGLE_GEMINI_API_KEY_SECRET_NAME,
    MOONSHOT_AI_API_KEY_SECRET_NAME,
    OPENAI_API_KEY_SECRET_NAME,
    TOGETHER_API_KEY_SECRET_NAME,
    XAI_API_KEY_SECRET_NAME,
)
from ._identifiers import filename_safe_name, validate_registry_key
from .lab_registry import LABS, Lab
from .metadata.models_dev import ModelsDevModel, load_models_dev_snapshot
from .provider_registry import PROVIDERS, Provider
from .providers.anthropic import AnthropicProvider
from .providers.base import BaseLLMProvider
from .providers.google import GoogleProvider
from .providers.moonshot_ai import MoonshotAIProvider
from .providers.openai import OpenAIProvider
from .providers.together import TogetherProvider
from .providers.xai import XAIProvider

# Registry for API keys by provider class
_PROVIDER_API_KEYS: dict[Type[BaseLLMProvider], str] = {}

# Mapping from provider routes to provider classes
_PROVIDER_TO_CLASS: dict[Provider, Type[BaseLLMProvider]] = {
    PROVIDERS["OpenAI"]: OpenAIProvider,
    PROVIDERS["Anthropic"]: AnthropicProvider,
    PROVIDERS["Google"]: GoogleProvider,
    PROVIDERS["Moonshot AI"]: MoonshotAIProvider,
    PROVIDERS["xAI"]: XAIProvider,
    PROVIDERS["Together"]: TogetherProvider,
}

# Mapping from provider classes to GCP secret names
_PROVIDER_CLASS_TO_SECRET_NAME: dict[Type[BaseLLMProvider], str] = {
    OpenAIProvider: OPENAI_API_KEY_SECRET_NAME,
    AnthropicProvider: ANTHROPIC_API_KEY_SECRET_NAME,
    GoogleProvider: GOOGLE_GEMINI_API_KEY_SECRET_NAME,
    MoonshotAIProvider: MOONSHOT_AI_API_KEY_SECRET_NAME,
    XAIProvider: XAI_API_KEY_SECRET_NAME,
    TogetherProvider: TOGETHER_API_KEY_SECRET_NAME,
}


@dataclass(frozen=True, slots=True)
class ModelsDevReference:
    """Reference to an underlying model entry in Models.dev."""

    # Models.dev provider directory ID, not necessarily this repo's API provider route.
    provider_id: str
    # Models.dev model entry ID used to look up release metadata in the snapshot.
    model_id: str


@dataclass(frozen=True, slots=True)
class Model:
    """Canonical LLM model metadata."""

    # Stable registry key for the base model. Model runs reference this in their keys.
    model_key: str
    # Exact model identifier sent to the provider API. Often equals `model_key`,
    # but routed providers may require provider-specific IDs.
    provider_model_id: str
    # Organization that created the model, used for leaderboard grouping and labels.
    lab: Lab
    # Provider route used for live API calls.
    provider: Provider
    # Optional Models.dev entry used to resolve release metadata from the checked-in snapshot.
    models_dev_reference: ModelsDevReference | None = None
    # Explicit release date source or correction when Models.dev is missing or incorrect.
    manual_release_date: date | None = None
    # False keeps historical metadata registered while preventing live benchmark runs.
    active: bool = True

    def __post_init__(self) -> None:
        """Validate the model declaration against configured metadata."""
        validate_registry_key(self.model_key, field_name="Model model_key")

        if self.models_dev_reference is None:
            if self.manual_release_date is None:
                raise ValueError(f"Model {self.model_key} is missing release date source")
            return

        try:
            metadata = self.models_dev_metadata
        except KeyError as exc:
            reference = self.models_dev_reference
            raise ValueError(
                f"Model {self.model_key} has invalid Models.dev reference "
                f"{reference.provider_id}/{reference.model_id}: {exc}"
            ) from exc

        if self.manual_release_date is not None:
            return
        if metadata.release_date is not None:
            return
        raise ValueError(f"Model {self.model_key} is missing release date metadata")

    @property
    def models_dev_provider_id(self) -> str | None:
        """Return the Models.dev provider ID for compatibility and debugging."""
        if self.models_dev_reference is None:
            return None
        return self.models_dev_reference.provider_id

    @property
    def models_dev_model_id(self) -> str | None:
        """Return the Models.dev model ID for compatibility and debugging."""
        if self.models_dev_reference is None:
            return None
        return self.models_dev_reference.model_id

    @property
    def models_dev_metadata(self) -> ModelsDevModel | None:
        """Return configured Models.dev metadata, or None when no reference is configured."""
        if self.models_dev_reference is None:
            return None
        return load_models_dev_snapshot().get_model(
            provider_id=self.models_dev_reference.provider_id,
            model_id=self.models_dev_reference.model_id,
        )

    @property
    def release_date(self) -> date:
        """Return this model's release date, preferring explicit manual overrides."""
        if self.manual_release_date is not None:
            return self.manual_release_date
        metadata = self.models_dev_metadata
        if metadata is not None and metadata.release_date is not None:
            return metadata.release_date
        raise ValueError(f"Model {self.model_key} is missing release date metadata")

    @property
    def filename_safe_name(self) -> str:
        """Return a filename-safe model name."""
        return filename_safe_name(self.model_key, field_name="Model model_key")

    def get_response(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Request a response from the model's provider."""
        return get_response(
            self.provider,
            self.provider_model_id,
            prompt=prompt,
            options=options,
        )


def provider_model(
    *,
    model_key: str,
    lab_key: str,
    provider_key: str,
    provider_model_id: str | None = None,
    models_dev_reference: ModelsDevReference | None = None,
    manual_release_date: date | None = None,
    active: bool = True,
) -> Model:
    """Create a model declaration for a provider route."""
    return Model(
        model_key=model_key,
        provider_model_id=provider_model_id or model_key,
        lab=LABS[lab_key],
        provider=PROVIDERS[provider_key],
        models_dev_reference=models_dev_reference,
        manual_release_date=manual_release_date,
        active=active,
    )


def openai_model(
    *,
    model_key: str,
    provider_model_id: str | None = None,
    models_dev_reference: ModelsDevReference | None = None,
    manual_release_date: date | None = None,
    active: bool = True,
) -> Model:
    """Create an OpenAI model declaration."""
    return provider_model(
        model_key=model_key,
        provider_model_id=provider_model_id,
        lab_key="OpenAI",
        provider_key="OpenAI",
        models_dev_reference=models_dev_reference,
        manual_release_date=manual_release_date,
        active=active,
    )


def anthropic_model(
    *,
    model_key: str,
    provider_model_id: str | None = None,
    models_dev_reference: ModelsDevReference | None = None,
    manual_release_date: date | None = None,
    active: bool = True,
) -> Model:
    """Create an Anthropic model declaration."""
    return provider_model(
        model_key=model_key,
        provider_model_id=provider_model_id,
        lab_key="Anthropic",
        provider_key="Anthropic",
        models_dev_reference=models_dev_reference,
        manual_release_date=manual_release_date,
        active=active,
    )


def xai_model(
    *,
    model_key: str,
    provider_model_id: str | None = None,
    models_dev_reference: ModelsDevReference | None = None,
    manual_release_date: date | None = None,
    active: bool = True,
) -> Model:
    """Create an xAI model declaration."""
    return provider_model(
        model_key=model_key,
        provider_model_id=provider_model_id,
        lab_key="xAI",
        provider_key="xAI",
        models_dev_reference=models_dev_reference,
        manual_release_date=manual_release_date,
        active=active,
    )


def google_model(
    *,
    model_key: str,
    provider_model_id: str | None = None,
    models_dev_reference: ModelsDevReference | None = None,
    manual_release_date: date | None = None,
    active: bool = True,
) -> Model:
    """Create a Google model declaration."""
    return provider_model(
        model_key=model_key,
        provider_model_id=provider_model_id,
        lab_key="Google DeepMind",
        provider_key="Google",
        models_dev_reference=models_dev_reference,
        manual_release_date=manual_release_date,
        active=active,
    )


def together_model(
    *,
    model_key: str,
    lab_key: str,
    provider_model_id: str | None = None,
    models_dev_reference: ModelsDevReference | None = None,
    manual_release_date: date | None = None,
    active: bool = True,
) -> Model:
    """Create a Together-routed model declaration."""
    return provider_model(
        model_key=model_key,
        provider_model_id=provider_model_id,
        lab_key=lab_key,
        provider_key="Together",
        models_dev_reference=models_dev_reference,
        manual_release_date=manual_release_date,
        active=active,
    )


def moonshot_ai_model(
    *,
    model_key: str,
    provider_model_id: str | None = None,
    models_dev_reference: ModelsDevReference | None = None,
    manual_release_date: date | None = None,
    active: bool = True,
) -> Model:
    """Create a Moonshot AI-routed model declaration."""
    return provider_model(
        model_key=model_key,
        provider_model_id=provider_model_id,
        lab_key="Moonshot",
        provider_key="Moonshot AI",
        models_dev_reference=models_dev_reference,
        manual_release_date=manual_release_date,
        active=active,
    )


@lru_cache(maxsize=None)
def _get_provider_instance(provider_cls: Type[BaseLLMProvider]) -> BaseLLMProvider:
    """Return a cached provider instance for the given provider class."""
    api_key = _PROVIDER_API_KEYS.get(provider_cls)
    if api_key is not None:
        return provider_cls(api_key=api_key)
    return provider_cls()


def configure_api_keys(
    *,
    from_gcp: bool = False,
    openai: str | None = None,
    anthropic: str | None = None,
    google: str | None = None,
    moonshot_ai: str | None = None,
    xai: str | None = None,
    together: str | None = None,
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
        moonshot_ai: Moonshot AI API key
        xai: xAI API key
        together: Together AI API key

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
            except (RuntimeError, exceptions.NotFound):
                # GCP not configured or secret doesn't exist, skip this provider
                pass

    # Set explicitly provided keys (these override GCP keys if both are set)
    key_mapping = {
        PROVIDERS["OpenAI"]: openai,
        PROVIDERS["Anthropic"]: anthropic,
        PROVIDERS["Google"]: google,
        PROVIDERS["Moonshot AI"]: moonshot_ai,
        PROVIDERS["xAI"]: xai,
        PROVIDERS["Together"]: together,
    }

    for provider, api_key in key_mapping.items():
        if api_key is not None:
            provider_cls = _PROVIDER_TO_CLASS[provider]
            _PROVIDER_API_KEYS[provider_cls] = api_key

    # Clear the provider instance cache since keys have changed
    _get_provider_instance.cache_clear()


def get_response(
    provider: Provider,
    model_id: str,
    prompt: str,
    options: dict[str, Any] | None = None,
) -> str:
    """Request text from a model through the requested API provider."""
    provider_cls = _get_provider_class(provider)
    provider_instance = _get_provider_instance(provider_cls)
    return provider_instance.get_response(
        model_id=model_id,
        prompt=prompt,
        options=options if options is not None else {},
    )


def _get_provider_class(provider: Provider) -> Type[BaseLLMProvider]:
    """Return the provider class for a supported API provider route."""
    if not isinstance(provider, Provider):
        raise TypeError(f"provider must be a Provider, got {type(provider).__name__}")
    try:
        return _PROVIDER_TO_CLASS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider.name}") from exc


def validate_provider_keys(providers: list[Provider]) -> None:
    """Validate that all requested API providers have API keys configured."""
    missing_keys = []
    for provider in providers:
        if not isinstance(provider, Provider):
            raise TypeError(
                "validate_provider_keys() expects Provider objects. "
                f"Got {type(provider).__name__}."
            )
        provider_cls = _get_provider_class(provider)
        if provider_cls not in _PROVIDER_API_KEYS:
            missing_keys.append(provider.name)

    if missing_keys:
        missing_list = ", ".join(sorted(set(missing_keys)))
        raise ValueError(
            f"API keys not configured for the following providers: {missing_list}. "
            "Call configure_api_keys() or configure_api_keys(from_gcp=True) to set them."
        )


# OpenAI models: https://developers.openai.com/api/docs/models
OPENAI_MODELS: Final[list[Model]] = [
    openai_model(
        model_key="gpt-4-0613",
        manual_release_date=date(2023, 6, 13),
    ),
    openai_model(
        model_key="gpt-3.5-turbo-0125",
        manual_release_date=date(2024, 1, 25),
    ),
    openai_model(
        model_key="gpt-4-turbo-2024-04-09",
        manual_release_date=date(2024, 4, 9),
    ),
    openai_model(
        model_key="gpt-4o-2024-05-13",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-4o-2024-05-13"),
    ),
    openai_model(
        model_key="gpt-4o-mini-2024-07-18",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-4o-mini"),
    ),
    openai_model(
        model_key="gpt-4o-2024-11-20",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-4o-2024-11-20"),
    ),
    openai_model(
        model_key="o3-mini-2025-01-31",
        manual_release_date=date(2025, 1, 31),
    ),
    openai_model(
        model_key="gpt-4.5-preview-2025-02-27",
        manual_release_date=date(2025, 2, 27),
    ),
    openai_model(
        model_key="gpt-4.1-2025-04-14",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-4.1"),
    ),
    openai_model(
        model_key="o3-2025-04-16",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="o3"),
    ),
    openai_model(
        model_key="o4-mini-2025-04-16",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="o4-mini"),
    ),
    openai_model(
        model_key="gpt-5-2025-08-07",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-5"),
    ),
    openai_model(
        model_key="gpt-5-mini-2025-08-07",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-5-mini"),
    ),
    openai_model(
        model_key="gpt-5-nano-2025-08-07",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-5-nano"),
    ),
    openai_model(
        model_key="gpt-5.1-2025-11-13",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-5.1"),
    ),
    openai_model(
        model_key="gpt-5.2-2025-12-11",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-5.2"),
    ),
    openai_model(
        model_key="gpt-5.4-2026-03-05",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-5.4"),
    ),
    openai_model(
        model_key="gpt-5.4-mini-2026-03-17",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-5.4-mini"),
    ),
    openai_model(
        model_key="gpt-5.4-nano-2026-03-17",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-5.4-nano"),
    ),
    openai_model(
        model_key="gpt-5.5-2026-04-23",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-5.5"),
    ),
    openai_model(
        model_key="gpt-5.6-sol",
        models_dev_reference=ModelsDevReference(provider_id="openai", model_id="gpt-5.6-sol"),
    ),
]

# Together models: https://docs.together.ai/docs/serverless-models
TOGETHER_MODELS: Final[list[Model]] = [
    together_model(
        model_key="llama-2-70b-chat-hf",
        lab_key="Meta",
        manual_release_date=date(2023, 7, 18),
        active=False,
    ),
    together_model(
        model_key="mixtral-8x7b-instruct-v0.1",
        lab_key="Mistral AI",
        manual_release_date=date(2023, 12, 11),
        active=False,
    ),
    together_model(
        model_key="mistral-large-latest",
        lab_key="Mistral AI",
        manual_release_date=date(2024, 2, 26),
        active=False,
    ),
    together_model(
        model_key="mixtral-8x22b-instruct-v0.1",
        lab_key="Mistral AI",
        manual_release_date=date(2024, 4, 17),
        active=False,
    ),
    together_model(
        model_key="llama-3-70b-chat-hf",
        lab_key="Meta",
        manual_release_date=date(2024, 4, 18),
        active=False,
    ),
    together_model(
        model_key="llama-3-8b-chat-hf",
        lab_key="Meta",
        manual_release_date=date(2024, 4, 18),
        active=False,
    ),
    together_model(
        model_key="qwen1.5-110b-chat",
        lab_key="Qwen",
        manual_release_date=date(2024, 4, 25),
        active=False,
    ),
    together_model(
        model_key="meta-llama-3.1-405b-instruct-turbo",
        lab_key="Meta",
        manual_release_date=date(2024, 7, 23),
        active=False,
    ),
    together_model(
        model_key="mistral-large-2407",
        lab_key="Mistral AI",
        manual_release_date=date(2024, 7, 24),
        active=False,
    ),
    together_model(
        model_key="qwen2.5-72b-instruct-turbo",
        lab_key="Qwen",
        manual_release_date=date(2024, 9, 19),
        active=False,
    ),
    together_model(
        model_key="llama-3.2-3b-instruct-turbo",
        lab_key="Meta",
        manual_release_date=date(2024, 9, 25),
        active=False,
    ),
    together_model(
        model_key="mistral-large-2411",
        lab_key="Mistral AI",
        models_dev_reference=ModelsDevReference(
            provider_id="mistral", model_id="mistral-large-2411"
        ),
        active=False,
    ),
    together_model(
        model_key="qwq-32b-preview",
        lab_key="Qwen",
        manual_release_date=date(2024, 11, 28),
        active=False,
    ),
    together_model(
        model_key="llama-3.3-70b-instruct-turbo",
        lab_key="Meta",
        models_dev_reference=ModelsDevReference(
            provider_id="togetherai", model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo"
        ),
        active=False,
    ),
    together_model(
        model_key="deepseek-v3",
        lab_key="DeepSeek",
        models_dev_reference=ModelsDevReference(
            provider_id="togetherai", model_id="deepseek-ai/DeepSeek-V3"
        ),
        active=False,
    ),
    together_model(
        model_key="deepseek-r1",
        lab_key="DeepSeek",
        models_dev_reference=ModelsDevReference(
            provider_id="togetherai", model_id="deepseek-ai/DeepSeek-R1"
        ),
        active=False,
    ),
    together_model(
        model_key="llama-4-maverick-17b-128e-instruct-fp8",
        lab_key="Meta",
        manual_release_date=date(2025, 4, 5),
        active=False,
    ),
    together_model(
        model_key="llama-4-scout-17b-16e-instruct",
        lab_key="Meta",
        manual_release_date=date(2025, 4, 5),
        active=False,
    ),
    together_model(
        model_key="qwen3-235b-a22b-fp8-tput",
        lab_key="Qwen",
        manual_release_date=date(2025, 4, 29),
        active=False,
    ),
    together_model(
        model_key="magistral-medium-2506",
        lab_key="Mistral AI",
        manual_release_date=date(2025, 5, 28),
        active=False,
    ),
    together_model(
        model_key="kimi-k2-instruct",
        lab_key="Moonshot",
        manual_release_date=date(2025, 7, 12),
        active=False,
    ),
    together_model(
        model_key="qwen3-235b-a22b-thinking-2507",
        lab_key="Qwen",
        manual_release_date=date(2025, 7, 25),
        active=False,
    ),
    together_model(
        model_key="glm-4.5-air-fp8",
        lab_key="Z.ai",
        manual_release_date=date(2025, 7, 28),
        active=False,
    ),
    together_model(
        model_key="deepseek-v3.1",
        provider_model_id="deepseek-ai/DeepSeek-V3.1",
        lab_key="DeepSeek",
        models_dev_reference=ModelsDevReference(
            provider_id="togetherai", model_id="deepseek-ai/DeepSeek-V3-1"
        ),
        active=False,
    ),
    together_model(
        model_key="kimi-k2-instruct-0905",
        lab_key="Moonshot",
        manual_release_date=date(2025, 9, 5),
        active=False,
    ),
    together_model(
        model_key="glm-4.6",
        lab_key="Z.ai",
        models_dev_reference=ModelsDevReference(provider_id="zai", model_id="glm-4.6"),
        active=False,
    ),
    together_model(
        model_key="kimi-k2-thinking",
        lab_key="Moonshot",
        models_dev_reference=ModelsDevReference(
            provider_id="moonshotai", model_id="kimi-k2-thinking"
        ),
        active=False,
    ),
    together_model(
        model_key="glm-4.7",
        lab_key="Z.ai",
        models_dev_reference=ModelsDevReference(provider_id="zai", model_id="glm-4.7"),
        active=False,
    ),
    together_model(
        model_key="kimi-k2.5",
        provider_model_id="moonshotai/Kimi-K2.5",
        lab_key="Moonshot",
        models_dev_reference=ModelsDevReference(
            provider_id="togetherai", model_id="moonshotai/Kimi-K2.5"
        ),
        active=False,
    ),
    together_model(
        model_key="glm-5",
        lab_key="Z.ai",
        models_dev_reference=ModelsDevReference(provider_id="zai", model_id="glm-5"),
        active=False,
    ),
    together_model(
        model_key="minimax-m2.5",
        provider_model_id="MiniMaxAI/MiniMax-M2.5",
        lab_key="MiniMax",
        models_dev_reference=ModelsDevReference(provider_id="minimax", model_id="MiniMax-M2.5"),
        active=False,
    ),
    together_model(
        model_key="minimax-m2.7",
        provider_model_id="MiniMaxAI/MiniMax-M2.7",
        lab_key="MiniMax",
        models_dev_reference=ModelsDevReference(provider_id="minimax", model_id="MiniMax-M2.7"),
        active=False,
    ),
    together_model(
        model_key="gemma-4-31b-it",
        provider_model_id="google/gemma-4-31B-it",
        lab_key="Google DeepMind",
        models_dev_reference=ModelsDevReference(provider_id="google", model_id="gemma-4-31b-it"),
    ),
    together_model(
        model_key="glm-5.1",
        provider_model_id="zai-org/GLM-5.1",
        lab_key="Z.ai",
        models_dev_reference=ModelsDevReference(provider_id="zai", model_id="glm-5.1"),
        active=False,
    ),
    together_model(
        model_key="kimi-k2.6",
        provider_model_id="moonshotai/Kimi-K2.6",
        lab_key="Moonshot",
        models_dev_reference=ModelsDevReference(provider_id="moonshotai", model_id="kimi-k2.6"),
    ),
    together_model(
        model_key="deepseek-v4-pro",
        provider_model_id="deepseek-ai/DeepSeek-V4-Pro",
        lab_key="DeepSeek",
        models_dev_reference=ModelsDevReference(provider_id="deepseek", model_id="deepseek-v4-pro"),
    ),
    together_model(
        model_key="minimax-m3",
        provider_model_id="MiniMaxAI/MiniMax-M3",
        lab_key="MiniMax",
        models_dev_reference=ModelsDevReference(provider_id="minimax", model_id="MiniMax-M3"),
    ),
    together_model(
        model_key="glm-5.2",
        provider_model_id="zai-org/GLM-5.2",
        lab_key="Z.ai",
        manual_release_date=date(2026, 6, 13),
    ),
]


# Moonshot AI models: https://platform.moonshot.ai/docs/guide
MOONSHOT_AI_MODELS: Final[list[Model]] = [
    moonshot_ai_model(
        model_key="kimi-k2.5-moonshot-ai",
        provider_model_id="kimi-k2.5",
        manual_release_date=date(2026, 1, 27),
    ),
    moonshot_ai_model(
        model_key="kimi-k2.6-moonshot-ai",
        provider_model_id="kimi-k2.6",
        models_dev_reference=ModelsDevReference(provider_id="moonshotai", model_id="kimi-k2.6"),
    ),
    moonshot_ai_model(
        model_key="kimi-k3",
        models_dev_reference=ModelsDevReference(provider_id="moonshotai", model_id="kimi-k3"),
    ),
]


# Anthropic models: https://platform.claude.com/docs/en/about-claude/models/overview
ANTHROPIC_MODELS: Final[list[Model]] = [
    anthropic_model(
        model_key="claude-2.1",
        manual_release_date=date(2023, 11, 21),
        active=False,
    ),
    anthropic_model(
        model_key="claude-3-opus-20240229",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-3-opus-20240229"
        ),
        active=False,
    ),
    anthropic_model(
        model_key="claude-3-haiku-20240307",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-3-haiku-20240307"
        ),
        active=False,
    ),
    anthropic_model(
        model_key="claude-3-5-sonnet-20240620",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-3-5-sonnet-20240620"
        ),
        active=False,
    ),
    anthropic_model(
        model_key="claude-3-5-sonnet-20241022",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-3-5-sonnet-20241022"
        ),
        active=False,
    ),
    anthropic_model(
        model_key="claude-3-7-sonnet-20250219",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-3-7-sonnet-20250219"
        ),
        active=False,
    ),
    anthropic_model(
        model_key="claude-opus-4-20250514",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-opus-4-20250514"
        ),
        active=False,
    ),
    anthropic_model(
        model_key="claude-sonnet-4-20250514",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-sonnet-4-20250514"
        ),
        active=False,
    ),
    anthropic_model(
        model_key="claude-opus-4-1-20250805",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-opus-4-1-20250805"
        ),
        active=False,
    ),
    anthropic_model(
        model_key="claude-sonnet-4-5-20250929",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-sonnet-4-5-20250929"
        ),
    ),
    anthropic_model(
        model_key="claude-haiku-4-5-20251001",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-haiku-4-5-20251001"
        ),
    ),
    anthropic_model(
        model_key="claude-opus-4-5-20251101",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-opus-4-5-20251101"
        ),
    ),
    anthropic_model(
        model_key="claude-opus-4-6",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-opus-4-6"
        ),
    ),
    anthropic_model(
        model_key="claude-sonnet-4-6",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-sonnet-4-6"
        ),
    ),
    anthropic_model(
        model_key="claude-opus-4-7",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic", model_id="claude-opus-4-7"
        ),
    ),
    anthropic_model(
        model_key="claude-opus-4-8",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic",
            model_id="claude-opus-4-8",
        ),
    ),
    anthropic_model(
        model_key="claude-fable-5",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic",
            model_id="claude-fable-5",
        ),
    ),
    anthropic_model(
        model_key="claude-sonnet-5",
        models_dev_reference=ModelsDevReference(
            provider_id="anthropic",
            model_id="claude-sonnet-5",
        ),
    ),
]

# xAI models: https://console.x.ai/ -> API Models
XAI_MODELS: Final[list[Model]] = [
    xai_model(
        model_key="grok-beta",
        manual_release_date=date(2024, 11, 4),
    ),
    xai_model(
        model_key="grok-4-0709",
        manual_release_date=date(2025, 7, 9),
    ),
    xai_model(
        model_key="grok-4-fast-non-reasoning",
        manual_release_date=date(2025, 9, 19),
    ),
    xai_model(
        model_key="grok-4-fast-reasoning",
        manual_release_date=date(2025, 9, 19),
    ),
    xai_model(
        model_key="grok-4-1-fast-non-reasoning",
        manual_release_date=date(2025, 11, 17),
    ),
    xai_model(
        model_key="grok-4-1-fast-reasoning",
        manual_release_date=date(2025, 11, 17),
    ),
    xai_model(
        model_key="grok-4.20-0309-non-reasoning",
        models_dev_reference=ModelsDevReference(
            provider_id="xai", model_id="grok-4.20-0309-non-reasoning"
        ),
    ),
    xai_model(
        model_key="grok-4.20-0309-reasoning",
        models_dev_reference=ModelsDevReference(
            provider_id="xai", model_id="grok-4.20-0309-reasoning"
        ),
    ),
    xai_model(
        model_key="grok-4.20-beta-0309-non-reasoning",
        provider_model_id="grok-4.20-beta-0309-non-reasoning",
        manual_release_date=date(2026, 3, 9),
        active=False,
    ),
    xai_model(
        model_key="grok-4.20-beta-0309-reasoning",
        provider_model_id="grok-4.20-beta-0309-reasoning",
        manual_release_date=date(2026, 3, 9),
        active=False,
    ),
    xai_model(
        model_key="grok-4.3",
        models_dev_reference=ModelsDevReference(provider_id="xai", model_id="grok-4.3"),
    ),
    xai_model(
        model_key="grok-4.5",
        models_dev_reference=ModelsDevReference(provider_id="xai", model_id="grok-4.5"),
    ),
]

# Google models: https://ai.google.dev/gemini-api/docs/models
GOOGLE_MODELS: Final[list[Model]] = [
    google_model(
        model_key="gemini-1.5-flash",
        manual_release_date=date(2024, 5, 1),
    ),
    google_model(
        model_key="gemini-1.5-pro",
        manual_release_date=date(2024, 5, 1),
    ),
    google_model(
        model_key="gemini-2.0-flash-lite-001",
        manual_release_date=date(2025, 2, 5),
    ),
    google_model(
        model_key="gemini-2.5-pro-exp-03-25",
        manual_release_date=date(2025, 3, 25),
    ),
    google_model(
        model_key="gemini-2.5-pro-preview-03-25",
        manual_release_date=date(2025, 4, 4),
    ),
    google_model(
        model_key="gemini-2.5-flash-preview-04-17",
        manual_release_date=date(2025, 4, 17),
    ),
    google_model(
        model_key="gemini-2.5-flash",
        models_dev_reference=ModelsDevReference(provider_id="google", model_id="gemini-2.5-flash"),
    ),
    google_model(
        model_key="gemini-2.5-pro",
        models_dev_reference=ModelsDevReference(provider_id="google", model_id="gemini-2.5-pro"),
    ),
    google_model(
        model_key="gemini-3-pro-preview",
        models_dev_reference=ModelsDevReference(
            provider_id="google", model_id="gemini-3-pro-preview"
        ),
    ),
    google_model(
        model_key="gemini-3-flash-preview",
        models_dev_reference=ModelsDevReference(
            provider_id="google", model_id="gemini-3-flash-preview"
        ),
    ),
    google_model(
        model_key="gemini-3.1-pro-preview",
        models_dev_reference=ModelsDevReference(
            provider_id="google", model_id="gemini-3.1-pro-preview"
        ),
    ),
    google_model(
        model_key="gemini-3.1-flash-lite-preview",
        models_dev_reference=ModelsDevReference(
            provider_id="google", model_id="gemini-3.1-flash-lite-preview"
        ),
    ),
    google_model(
        model_key="gemini-3.1-flash-lite",
        models_dev_reference=ModelsDevReference(
            provider_id="google", model_id="gemini-3.1-flash-lite"
        ),
    ),
    google_model(
        model_key="gemini-3.5-flash",
        models_dev_reference=ModelsDevReference(provider_id="google", model_id="gemini-3.5-flash"),
    ),
]


def _validate_unique_model_keys(models: Sequence[Model]) -> None:
    """Reject duplicate model keys in a model registry list."""
    seen_model_keys = set()
    for model in models:
        if model.model_key in seen_model_keys:
            raise ValueError(f"Duplicate LLM model_key: {model.model_key}")
        seen_model_keys.add(model.model_key)


def create_models_list(models: Sequence[Model]) -> list[Model]:
    """Create a validated model registry list."""
    _validate_unique_model_keys(models)
    return list(models)


MODELS: Final[list[Model]] = create_models_list(
    [
        *OPENAI_MODELS,
        *TOGETHER_MODELS,
        *MOONSHOT_AI_MODELS,
        *ANTHROPIC_MODELS,
        *XAI_MODELS,
        *GOOGLE_MODELS,
    ]
)
MODELS_BY_KEY: Final[dict[str, Model]] = {model.model_key: model for model in MODELS}


def model_release_dates_by_key() -> dict[str, date]:
    """Return release dates keyed by canonical model_key."""
    return {model.model_key: model.release_date for model in MODELS}
