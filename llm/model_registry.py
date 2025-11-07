"""Central model registry for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, Type

from .lab_registry import LABS, Lab
from .providers.anthropic import AnthropicProvider
from .providers.base import BaseLLMProvider
from .providers.google import GoogleProvider
from .providers.mistral import MistralProvider
from .providers.openai import OpenAIProvider
from .providers.together import TogetherProvider
from .providers.xai import XAIProvider


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


@lru_cache(maxsize=None)
def _get_provider_instance(provider_cls: Type[BaseLLMProvider]) -> BaseLLMProvider:
    """Return a cached provider instance for the given provider class."""
    return provider_cls()


MODELS: Final[list[Model]] = [
    Model(
        id="gpt-4.1-mini",
        full_name="gpt-4.1-mini",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
    ),
    Model(
        id="Qwen2.5-Coder-32B-Instruct",
        full_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        token_limit=262_144,
        provider_cls=TogetherProvider,
        lab=LABS["Qwen"],
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
        id="GLM-4.5-Air-FP8",
        full_name="zai-org/GLM-4.5-Air-FP8",
        token_limit=131_072,
        provider_cls=TogetherProvider,
        lab=LABS["Z.ai"],
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
