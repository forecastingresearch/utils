"""Helpers for invoking Anthropic models with retry support."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from ..types import CallContext, LLMResponse, LLMUsage, int_or_none
from ..utils import response_to_plain_text
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

_BETA_MESSAGES_OPTION_KEYS = frozenset({"betas", "fallbacks", "fallback_credit_token"})


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
        # The Messages API has no cancel or retrieve-by-id for a streamed call, so the
        # context's cancellation is honoured between retries only (see BaseLLMProvider).
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

        with messages.stream(**call_args) as stream:
            stream.until_done()

            try:
                text = stream.get_final_text().strip()
            except RuntimeError:
                logger.error(
                    "LLM provider response did not include text content. response=%s",
                    response_to_plain_text(stream.get_final_message()),
                )
                raise
            message = stream.get_final_message()

        message_id = getattr(message, "id", None)
        context.report_response_id(message_id if isinstance(message_id, str) else None)
        return LLMResponse(
            text=text,
            provider_response_id=message_id if isinstance(message_id, str) else None,
            status=_str_or_none(getattr(message, "stop_reason", None)),
            usage=_usage_from_message(message),
        )


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _usage_from_message(message: Any) -> LLMUsage:
    """Read token counts and server-side tool uses from a final Messages API message."""
    usage = getattr(message, "usage", None)
    content = getattr(message, "content", None)
    tool_calls = None
    if isinstance(content, (list, tuple)):
        tool_calls = sum(
            1 for block in content if getattr(block, "type", None) == "server_tool_use"
        )
    return LLMUsage(
        input_tokens=int_or_none(getattr(usage, "input_tokens", None)),
        output_tokens=int_or_none(getattr(usage, "output_tokens", None)),
        reasoning_tokens=None,
        tool_calls=tool_calls,
    )
