"""Call-level types shared by the LLM providers and their callers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable


class LLMCallCancelled(Exception):
    """Raised when a call is abandoned because its `CallContext` was cancelled."""


@dataclass(frozen=True)
class LLMUsage:
    """Token and tool-call counts reported by the provider for one call."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    tool_calls: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    """One model response with the metadata the provider reported alongside the text."""

    text: str
    provider_response_id: str | None = None
    status: str | None = None
    usage: LLMUsage | None = None
    # Attempts made by the retry wrapper, including the one that produced this response.
    attempts: int = 1
    # Wall-clock seconds spent in the retry wrapper, waits between attempts included.
    duration_s: float | None = None


@dataclass
class CallContext:
    """Caller-side hooks for one call: cancellation and the provider response id.

    `cancel_event`, when set, makes the provider abandon the call at its next
    checkpoint (between retries, between polls) and cancel server-side work it can.
    `on_response_id` is invoked with the provider's response id as soon as it is
    known, before the response has finished, so the caller can record it.
    """

    cancel_event: threading.Event | None = None
    on_response_id: Callable[[str], None] | None = None

    def cancelled(self) -> bool:
        """Return whether the caller asked for the call to be abandoned."""
        return self.cancel_event is not None and self.cancel_event.is_set()

    def report_response_id(self, provider_response_id: str | None) -> None:
        """Hand the provider response id to the caller, if there is one to hand over."""
        if provider_response_id and self.on_response_id is not None:
            self.on_response_id(provider_response_id)


def int_or_none(value: object) -> int | None:
    """Return `value` as an int when it is a real integer count, else None.

    Provider SDK objects expose counts as ints; anything else (a missing field, a
    mock, a float placeholder) is reported as unknown rather than guessed.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
