"""Shared base for providers that stream through an OpenAI-compatible chat completions API."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

from openai import OpenAI  # type: ignore[import]

from .base import BaseLLMProvider


class OpenAICompatibleChatProvider(BaseLLMProvider):
    """LLM provider that streams chat completions from an OpenAI-compatible API.

    Subclasses set `provider_name` (used in error messages), `api_base_url`, and
    `retry_message`.
    """

    provider_name: ClassVar[str]
    api_base_url: ClassVar[str]

    def __init__(self, *, api_key: str | None = None, default_wait_time: int | None = None) -> None:
        """Instantiate the OpenAI SDK client against this provider's base URL.

        Args:
            api_key: Provider API key. If None, an error will be raised.
            default_wait_time: Optional custom backoff interval.

        Raises:
            ValueError: If api_key is None.
        """
        super().__init__(default_wait_time=default_wait_time)
        if api_key is None:
            raise ValueError(
                f"API key required for {type(self).__name__}. "
                "Call configure_api_keys() or provide api_key parameter."
            )
        self._chat_client = OpenAI(api_key=api_key, base_url=self.api_base_url)

    def _call_model(self, *, model_id: str, prompt: str, options: dict[str, Any]) -> str:
        # Stream so that long thinking phases keep bytes flowing on the connection: a
        # non-streamed request buffers the whole completion server-side, and intermediaries
        # drop the idle connection before it arrives. Server-side tools (for example Z.ai's
        # built-in `web_search`) feed the same stream, so no client tool loop is needed.
        request_payload: Dict[str, Any] = {
            **options,
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }

        stream = self._chat_client.chat.completions.create(**request_payload)
        content_parts = []
        finish_reason = None
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
            if choice.delta.content:
                content_parts.append(choice.delta.content)

        # Only an explicit reason other than `stop` fails the call: `length` is a truncated
        # answer, and Z.ai also reports `sensitive`, `model_context_window_exceeded` and
        # `network_error`. A stream with no reason at all still returns its text, as before;
        # whether Moonshot always sends one has not been checked against the live API.
        if finish_reason not in (None, "stop"):
            raise RuntimeError(
                f"{self.provider_name} stopped {model_id} with finish_reason={finish_reason!r}; "
                "the answer is not complete."
            )

        content = "".join(content_parts).strip()
        if not content:
            raise RuntimeError(f"{self.provider_name} response did not include text content.")
        return content
