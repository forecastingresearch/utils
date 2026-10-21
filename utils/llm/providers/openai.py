"""Helpers for invoking OpenAI models with retry support."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Final

from openai import OpenAI, PermissionDeniedError  # type: ignore[import]

from ..types import CallContext, LLMCallCancelled, LLMResponse, LLMUsage, int_or_none
from .base import BaseLLMProvider

# A background response is polled, starting often and backing off, so a short
# answer is picked up quickly and a long one does not hammer the API.
_POLL_INTERVAL_START_SECONDS: Final[float] = 2.0
_POLL_INTERVAL_MAX_SECONDS: Final[float] = 10.0
# How long a background response may run before it is cancelled and the call
# fails, when the run options carry no `timeout`.
_DEFAULT_RESPONSE_BUDGET_SECONDS: Final[float] = 1800.0

_PENDING_STATUSES: Final[frozenset[str]] = frozenset({"queued", "in_progress"})
# Consecutive failed polls tolerated before the error propagates. A poll is a short
# request; a failed one must never restart the (billed) generation it watches.
_POLL_FAILURE_LIMIT: Final[int] = 10

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """LLM provider that wraps the OpenAI Responses API.

    Responses are requested in background mode: the request returns at once with a
    response id, the model runs server-side, and the result is polled with short
    requests. No connection stays idle while the model works, so a network path
    that drops idle connections cannot lose a finished (and billed) response; the
    id is reported to the caller as soon as it exists, so a response can be
    cancelled or fetched later. Background mode is transport, not a model option,
    which is why it is not part of any run's declared options.
    """

    retry_message = "OpenAI API request failed."
    # HTTP 403, including misalignment_policy_violation, which OpenAI says not to retry.
    # A cancelled call and a call that outran its budget are the caller's decision.
    non_retryable_exceptions = (PermissionDeniedError, LLMCallCancelled, TimeoutError)
    # Subclasses for OpenAI-compatible APIs without background mode turn this off and
    # get the synchronous `responses.create` call instead.
    background_responses: bool = True

    def __init__(self, *, api_key: str | None = None, default_wait_time: int | None = None) -> None:
        """Instantiate the OpenAI client using the provided API key.

        Args:
            api_key: OpenAI API key (e.g., "sk-..."). If None, an error will be raised.
            default_wait_time: Optional custom backoff interval.

        Raises:
            ValueError: If api_key is None.
        """
        super().__init__(default_wait_time=default_wait_time)
        if api_key is None:
            raise ValueError(
                "API key required for OpenAIProvider. "
                "Call configure_api_keys() or provide api_key parameter."
            )
        self._openai_client = self._create_client(api_key)

    def _create_client(self, api_key: str) -> OpenAI:
        """Return the OpenAI SDK client for this provider's API."""
        return OpenAI(api_key=api_key)

    def _call_model(self, *, model_id: str, prompt: str, options: dict[str, Any]) -> str:
        return self._call_model_with_metadata(
            model_id=model_id, prompt=prompt, options=options, context=CallContext()
        ).text

    def _call_model_with_metadata(
        self,
        *,
        model_id: str,
        prompt: str,
        options: dict[str, Any],
        context: CallContext,
    ) -> LLMResponse:
        request_payload: Dict[str, Any] = {
            **options,
            "model": model_id,
            "input": prompt,
        }

        if not self.background_responses:
            response = self._openai_client.responses.create(**request_payload)
            context.report_response_id(getattr(response, "id", None))
            return _finished_response(response)

        # A run's `timeout` option bounds the whole response here, not one HTTP
        # request: the create and retrieve calls are short and keep the SDK default.
        budget = request_payload.pop("timeout", None) or _DEFAULT_RESPONSE_BUDGET_SECONDS
        response = self._openai_client.responses.create(**request_payload, background=True)
        context.report_response_id(getattr(response, "id", None))
        response = self._wait_for_response(response, budget=float(budget), context=context)
        return _finished_response(response)

    def fetch_response(self, provider_response_id: str) -> LLMResponse:
        """Return a background response by id.

        A response that is still running comes back with empty text and its
        pending status; a finished one is returned like a fresh call's result.
        """
        response = self._openai_client.responses.retrieve(provider_response_id)
        status = getattr(response, "status", None)
        if status in _PENDING_STATUSES:
            return LLMResponse(text="", provider_response_id=provider_response_id, status=status)
        return _finished_response(response)

    def wait_for_response(
        self,
        provider_response_id: str,
        *,
        budget: float | None = None,
        context: CallContext | None = None,
    ) -> LLMResponse:
        """Wait for a background response started earlier, by id, and return it.

        Same polling, cancellation and budget behaviour as a fresh call, but no new
        generation is ever created: this is how a caller picks up a response whose
        first process died while it was still running.
        """
        response = self._openai_client.responses.retrieve(provider_response_id)
        response = self._wait_for_response(
            response,
            budget=float(budget) if budget else _DEFAULT_RESPONSE_BUDGET_SECONDS,
            context=context if context is not None else CallContext(),
        )
        return _finished_response(response)

    def _wait_for_response(self, response: Any, *, budget: float, context: CallContext) -> Any:
        """Poll a background response until it leaves the pending statuses.

        Cancellation and the budget both cancel the response server-side first, so
        no billed work outlives the caller's interest in it. A failed poll is retried
        in place (up to `_POLL_FAILURE_LIMIT` times in a row): the generation it
        watches is already paid for and must not be started again because a short
        status request failed.
        """
        response_id = getattr(response, "id", None)
        started = time.monotonic()
        interval = _POLL_INTERVAL_START_SECONDS
        failed_polls = 0
        while getattr(response, "status", None) in _PENDING_STATUSES:
            if context.cancelled():
                self._cancel_quietly(response_id)
                raise LLMCallCancelled(f"OpenAI response {response_id} cancelled by the caller.")
            if time.monotonic() - started > budget:
                self._cancel_quietly(response_id)
                raise TimeoutError(
                    f"OpenAI response {response_id} did not finish within {budget:.0f}s "
                    "and was cancelled."
                )
            if context.cancel_event is not None:
                context.cancel_event.wait(interval)
            else:
                time.sleep(interval)
            interval = min(interval * 2, _POLL_INTERVAL_MAX_SECONDS)
            try:
                response = self._openai_client.responses.retrieve(response_id)
            except Exception as exc:  # noqa: BLE001 - a poll failure is retried, not escalated
                failed_polls += 1
                if failed_polls >= _POLL_FAILURE_LIMIT:
                    raise
                logger.warning(
                    "Polling OpenAI response %s failed (%d/%d in a row): %s: %s; polling again",
                    response_id,
                    failed_polls,
                    _POLL_FAILURE_LIMIT,
                    type(exc).__name__,
                    exc,
                )
                continue
            failed_polls = 0
        return response

    def _cancel_quietly(self, response_id: str | None) -> None:
        """Cancel a background response; a failed cancel must not mask the real error."""
        if not response_id:
            return
        try:
            self._openai_client.responses.cancel(response_id)
        except Exception:  # noqa: BLE001 - best effort, the caller is already giving up
            pass


def _finished_response(response: Any) -> LLMResponse:
    """Turn a terminal Responses API object into an `LLMResponse`, or raise if it failed."""
    status = getattr(response, "status", None)
    response_id = getattr(response, "id", None)
    if status != "completed":
        reason = getattr(response, "incomplete_details", None) or getattr(response, "error", None)
        status_text = f"OpenAI response incomplete (status={status}, id={response_id})"
        if reason:
            status_text += f", reason={reason}"
        raise RuntimeError(status_text)

    # output_text is the text of the response in reasoning models
    output_text = getattr(response, "output_text", "")
    text = output_text.strip() if isinstance(output_text, str) else str(output_text)
    return LLMResponse(
        text=text,
        provider_response_id=response_id if isinstance(response_id, str) else None,
        status=status,
        usage=_usage_from_response(response),
    )


def _usage_from_response(response: Any) -> LLMUsage:
    """Read token counts and the number of built-in tool calls from a Responses object."""
    usage = getattr(response, "usage", None)
    output_details = getattr(usage, "output_tokens_details", None)
    output_items = getattr(response, "output", None)
    tool_calls = None
    if isinstance(output_items, (list, tuple)):
        tool_calls = sum(
            1 for item in output_items if str(getattr(item, "type", "")).endswith("_call")
        )
    return LLMUsage(
        input_tokens=int_or_none(getattr(usage, "input_tokens", None)),
        output_tokens=int_or_none(getattr(usage, "output_tokens", None)),
        reasoning_tokens=int_or_none(getattr(output_details, "reasoning_tokens", None)),
        tool_calls=tool_calls,
    )
