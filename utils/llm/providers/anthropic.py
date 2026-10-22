"""Helpers for invoking Anthropic models with retry support."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from ..utils import response_to_plain_text
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

_BETA_MESSAGES_OPTION_KEYS = frozenset({"betas", "fallbacks", "fallback_credit_token"})
# Server-side tools (web search, web fetch) can pause a long turn with `stop_reason`
# "pause_turn". The turn is resumed by sending the assistant content back as-is. This bounds
# how many times one call is resumed before giving up.
MAX_PAUSE_TURN_CONTINUATIONS = 10


def _uses_beta_messages_api(options: dict[str, Any]) -> bool:
    return bool(_BETA_MESSAGES_OPTION_KEYS & options.keys())


class AnthropicProvider(BaseLLMProvider):
    """LLM provider that communicates with the Anthropic Messages API."""

    retry_message = "Anthropic API request failed."

    def __init__(self, *, api_key: str | None = None, default_wait_time: int | None = None) -> None:
        """Instantiate the Anthropic client using the provided API key.

        Args:
            api_key: Anthropic API key (e.g., "sk-ant-..."). If None, an error will be raised.
            default_wait_time: Optional custom backoff interval.

        Raises:
            ValueError: If api_key is None.
        """
        super().__init__(default_wait_time=default_wait_time)
        if api_key is None:
            raise ValueError(
                "API key required for AnthropicProvider. "
                "Call configure_api_keys() or provide api_key parameter."
            )
        self._anthropic_console = anthropic.Anthropic(api_key=api_key)

    def _call_model(self, *, model_id: str, prompt: str, options: dict[str, Any]) -> str:
        call_args: dict[str, Any] = {
            **options,
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        messages = (
            self._anthropic_console.beta.messages
            if _uses_beta_messages_api(options)
            else self._anthropic_console.messages
        )

        # One logical turn may span several API calls when server-side tools pause it; the
        # text of every segment is part of the model's answer.
        # If a later segment fails with a transient error, the retry layer re-runs `_call_model`
        # from the first segment, so completed segments are discarded and paid for again.
        # Accepted for simplicity.
        text_segments: list[str] = []
        for _ in range(MAX_PAUSE_TURN_CONTINUATIONS + 1):
            with messages.stream(**call_args) as stream:
                stream.until_done()
                final_message = stream.get_final_message()
                stop_reason = final_message.stop_reason
                # `end_turn` and `stop_sequence`: the model finished its answer. `pause_turn`:
                # the API paused a server-tool loop and the turn is resumed below. Anything else
                # ends the call: max_tokens and model_context_window_exceeded mean a truncated
                # answer, refusal means no answer, and an unknown reason must not pass as a
                # finished answer.
                paused = stop_reason == "pause_turn"
                if stop_reason not in ("end_turn", "stop_sequence") and not paused:
                    raise RuntimeError(
                        f"Anthropic stopped {model_id} with stop_reason={stop_reason!r}; "
                        "the turn did not finish, so there is no complete answer to return."
                    )

                try:
                    text_segments.append(stream.get_final_text())
                except RuntimeError:
                    if not paused:
                        logger.error(
                            "LLM provider response did not include text content. "
                            f"response={response_to_plain_text(final_message)}"
                        )
                        raise

            if not paused:
                return "\n".join(segment for segment in text_segments if segment).strip()

            logger.info(f"Anthropic paused the turn for {model_id}; continuing it.")
            call_args["messages"] = [
                *call_args["messages"],
                {"role": "assistant", "content": final_message.content},
            ]

        raise RuntimeError(
            f"Anthropic kept returning stop_reason=pause_turn for {model_id} after "
            f"{MAX_PAUSE_TURN_CONTINUATIONS} continuations."
        )
