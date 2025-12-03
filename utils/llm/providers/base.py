"""Base helpers for LLM providers with shared retry semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel

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

    def get_structured_response(
        self,
        model: "Model",
        prompt: str,
        response_schema: type[BaseModel],
        **options: Any,
    ) -> BaseModel:
        """Return structured output from the provider, validated against a Pydantic schema.

        Args:
            model: The model to use for the request.
            prompt: The prompt to send to the model.
            response_schema: A Pydantic BaseModel class defining the expected response structure.
            **options: Additional options to pass to the provider (temperature, max_tokens, etc.).

        Returns:
            An instance of response_schema with validated data from the model response.

        Raises:
            ValueError: If the response cannot be parsed or validated against the schema.
        """
        wait_time = options.pop("wait_time", self._default_wait_time)

        def api_call() -> BaseModel:
            return self._call_model_structured(model, prompt, response_schema, **options)

        return get_response_with_retry(api_call, wait_time, self.rate_limit_message)

    @abstractmethod
    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        """Execute a request against the underlying provider."""

    @abstractmethod
    def _call_model_structured(
        self,
        model: "Model",
        prompt: str,
        response_schema: type[BaseModel],
        **options: Any,
    ) -> BaseModel:
        """Execute a structured output request against the underlying provider.

        Args:
            model: The model to use for the request.
            prompt: The prompt to send to the model.
            response_schema: A Pydantic BaseModel class defining the expected response structure.
            **options: Additional options to pass to the provider.

        Returns:
            An instance of response_schema with validated data.

        Raises:
            ValueError: If the response cannot be parsed or validated against the schema.
        """
