"""Shared helpers for working with LLM providers."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any, Callable

import anthropic
import httpx
import openai
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)


# Total attempts, including the first. The OpenAI and Anthropic SDK clients already retry
# HTTP-level 408/409/429/5xx and connection errors up to twice with backoff before raising, so this
# layer mainly covers failures inside a stream, and every retry repeats the full cost of a long
# web-search call. Two retries absorb a double fault without multiplying spend.
_DEFAULT_MAX_RETRIES: int = 3


class TransientProviderError(RuntimeError):
    """A provider reported a failure it may not repeat.

    Raised by providers so the retry loop treats it as transient, for example a Responses API
    response whose status is `failed` with a server-side error code.
    """


# Failures where the same request can reasonably succeed later: the connection dropped, the call
# timed out, the provider rate-limited us, or the provider's side failed. Everything else (bad
# requests, auth, refused requests, truncation, our own parsing errors) raises on the first attempt.
# APITimeoutError subclasses APIConnectionError in both SDKs. Each SDK's `_make_status_error` maps
# 429 to RateLimitError, 409 to ConflictError and other 5xx to InternalServerError; the Anthropic
# SDK maps 529 overloaded to its own OverloadedError, which is not an InternalServerError. The
# OpenAI SDK also serves xAI, Z.ai, Moonshot and Meta.
# Both SDKs wrap httpx errors in APIConnectionError only around the initial send; a streamed body
# is read outside those wrappers, so a timeout or dropped connection mid-stream surfaces as a raw
# httpx error: a TimeoutException (ReadTimeout), a NetworkError (ReadError) or a
# RemoteProtocolError (peer closed the connection). Their sibling httpx.TransportError subclasses
# (UnsupportedProtocol, LocalProtocolError, ProxyError) are client-side mistakes and are not
# listed. google-genai does not wrap httpx errors at all. The retired Together provider's
# `together` SDK exceptions are deliberately not classified here. Providers raise
# TransientProviderError themselves for a failure the SDK reports inside a successful HTTP
# response, such as a Responses API `failed` status.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.ConflictError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    anthropic.ConflictError,
    genai_errors.ServerError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    TransientProviderError,
)

# A failure that starts after the stream has opened arrives as an SSE `error` event on the 200
# response, not as an HTTP status, so the SDK's status-code ladder cannot name it. The Anthropic
# SDK raises a plain APIStatusError with `status_code == 200` and the whole event as `body`:
# `{"type": "error", "error": {"type": "overloaded_error", "message": ...}}`. The OpenAI SDK
# (also serving xAI, Z.ai, Moonshot and Meta) raises a bare APIError whose `body` is the event's
# inner error object: `{"type": "server_error", "message": ...}`. Only these error types mean the
# provider's side failed or throttled us; anything else in that position is the request's fault.
_TRANSIENT_ERROR_BODY_TYPES: frozenset[str] = frozenset(
    {"overloaded_error", "api_error", "rate_limit_error", "server_error"}
)


def _error_body_type(exc: BaseException) -> str | None:
    """Return the provider error type carried in an SDK exception body, if any."""
    body = getattr(exc, "body", None)
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    if isinstance(error, Mapping):
        return error.get("type")
    return body.get("type")


def is_transient(exc: BaseException) -> bool:
    """Return whether a provider failure is worth retrying with the same request."""
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True
    if isinstance(exc, (anthropic.APIStatusError, openai.APIError)):
        return _error_body_type(exc) in _TRANSIENT_ERROR_BODY_TYPES
    # google-genai reports rate limits as a ClientError carrying code 429.
    return isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429


def response_to_plain_text(response: Any) -> str:
    """Return a readable plain-text representation of a provider response."""
    for method_name in ("model_dump_json", "json", "to_json"):
        method = getattr(response, method_name, None)
        if callable(method):
            try:
                return str(method(indent=2))
            except TypeError:
                try:
                    return str(method())
                except Exception:  # noqa: BLE001 - best-effort debug formatting
                    continue
            except Exception:  # noqa: BLE001 - best-effort debug formatting
                continue
    return str(response)


def get_response_with_retry(
    api_call: Callable[[], str | Any],
    wait_time: int,
    error_msg: str,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retryable: Callable[[BaseException], bool] = is_transient,
) -> str | Any:
    """Execute an API call, retrying with a delay only when the failure can succeed later.

    Args:
        api_call (Callable): The API call to execute.
        wait_time (int): Seconds to wait between retries.
        error_msg (str): Message prefix for retry log entries.
        max_retries (int): Maximum number of attempts before raising.
        retryable (Callable): Decides whether a failure gets another attempt. Anything it
            rejects propagates after the first attempt.
    """
    for attempt in range(max_retries):
        try:
            return api_call()
        except Exception as exc:  # noqa: BLE001 - classification happens in `retryable`
            if not retryable(exc):
                raise
            logger.info(
                "%s (attempt %d/%d): %s: %s",
                error_msg,
                attempt + 1,
                max_retries,
                type(exc).__name__,
                exc,
            )

            if attempt + 1 >= max_retries:
                raise

            logger.info("Waiting for %s seconds before retrying...", wait_time)
            time.sleep(wait_time)
