"""Base helpers for LLM providers with shared retry semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final

from ..utils import get_response_with_retry

if TYPE_CHECKING:
    from ..model_registry import Model

_DEFAULT_WAIT_TIME_SECONDS: Final[int] = 30


class BaseLLMProvider(ABC):
    """Abstract base provider that wraps provider-specific API calls with retry logic."""

    rate_limit_message: str = "LLM provider request exceeded rate limit."

    def __init__(self, *, default_wait_time: int | None = None) -> None:
        """Initialize the provider with an optional custom backoff interval."""
        self._default_wait_time = default_wait_time or _DEFAULT_WAIT_TIME_SECONDS

    def get_response(self, model: "Model", prompt: str, **options: Any) -> str:
        """Return the provider response for a prompt, retrying on rate limits."""
        wait_time = options.pop("wait_time", self._default_wait_time)

        def api_call() -> str:
            return self._call_model(model, prompt, **options)

        return get_response_with_retry(api_call, wait_time, self.rate_limit_message)

    @abstractmethod
    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        """Execute a request against the underlying provider."""
