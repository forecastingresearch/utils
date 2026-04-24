"""Base helpers for LLM providers with shared retry semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final

from ..utils import get_response_with_retry

_DEFAULT_WAIT_TIME_SECONDS: Final[int] = 30


class BaseLLMProvider(ABC):
    """Abstract base provider that wraps provider-specific API calls with retry logic."""

    retry_message: str = "LLM provider request failed."

    def __init__(self, *, default_wait_time: int | None = None) -> None:
        """Initialize the provider with an optional custom backoff interval."""
        self._default_wait_time = default_wait_time or _DEFAULT_WAIT_TIME_SECONDS

    def get_response(
        self,
        *,
        model_id: str,
        prompt: str,
        options: dict[str, Any],
    ) -> str:
        """Return the provider response for a prompt, retrying on provider errors."""

        def api_call() -> str:
            return self._call_model(model_id=model_id, prompt=prompt, options=options)

        return get_response_with_retry(
            api_call,
            self._default_wait_time,
            self.retry_message,
        )

    @abstractmethod
    def _call_model(self, *, model_id: str, prompt: str, options: dict[str, Any]) -> str:
        """Execute a request against the underlying provider."""
