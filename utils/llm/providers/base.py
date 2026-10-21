"""Base helpers for LLM providers with shared retry semantics."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Final

from ..types import CallContext, LLMCallCancelled, LLMResponse, LLMResponseIncomplete
from ..utils import get_response_with_retry

_DEFAULT_WAIT_TIME_SECONDS: Final[int] = 30


class BaseLLMProvider(ABC):
    """Abstract base provider that wraps provider-specific API calls with retry logic."""

    retry_message: str = "LLM provider request failed."
    # Provider errors that a retry cannot fix, such as a request the provider refused.
    non_retryable_exceptions: tuple[type[BaseException], ...] = ()

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
        """Return the provider response text for a prompt, retrying on provider errors."""
        return self.get_response_with_metadata(
            model_id=model_id, prompt=prompt, options=options
        ).text

    def get_response_with_metadata(
        self,
        *,
        model_id: str,
        prompt: str,
        options: dict[str, Any],
        context: CallContext | None = None,
    ) -> LLMResponse:
        """Return the provider response with its metadata, retrying on provider errors.

        `context` carries the caller's cancellation event and response-id hook; both
        are optional and a plain call behaves exactly like `get_response`.
        """
        call_context = context if context is not None else CallContext()
        attempts = 0
        started = time.monotonic()

        def api_call() -> LLMResponse:
            nonlocal attempts
            attempts += 1
            return self._call_model_with_metadata(
                model_id=model_id, prompt=prompt, options=options, context=call_context
            )

        response = get_response_with_retry(
            api_call,
            self._default_wait_time,
            self.retry_message,
            non_retryable=(
                *self.non_retryable_exceptions,
                LLMCallCancelled,
                LLMResponseIncomplete,
            ),
            cancel_event=call_context.cancel_event,
        )
        return replace(response, attempts=attempts, duration_s=time.monotonic() - started)

    def fetch_response(self, provider_response_id: str) -> LLMResponse:
        """Return a previously started response by its provider id, without a new call.

        Only providers whose API keeps responses addressable support this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot fetch responses by id; the provider API has no "
            "retrieve-by-id route for this call type."
        )

    def wait_for_response(
        self,
        provider_response_id: str,
        *,
        budget: float | None = None,
        context: CallContext | None = None,
    ) -> LLMResponse:
        """Wait for a response started earlier, by its provider id, and return it.

        Only providers whose API keeps responses addressable support this; the
        default refuses, like `fetch_response`.
        """
        del budget, context
        return self.fetch_response(provider_response_id)

    def _call_model_with_metadata(
        self,
        *,
        model_id: str,
        prompt: str,
        options: dict[str, Any],
        context: CallContext,
    ) -> LLMResponse:
        """Execute one request and return the text plus whatever metadata the provider reports.

        The default wraps `_call_model`, so a provider that reports no metadata needs
        no change. Providers override this to attach ids, usage, or cancellation.
        """
        del context
        return LLMResponse(text=self._call_model(model_id=model_id, prompt=prompt, options=options))

    @abstractmethod
    def _call_model(self, *, model_id: str, prompt: str, options: dict[str, Any]) -> str:
        """Execute a request against the underlying provider."""
