"""Helpers for invoking Together AI models with retry support."""

from __future__ import annotations

from typing import Any, Dict, Final, Iterable, Union

from together import Together  # type: ignore[import]

from .base import BaseLLMProvider

TOGETHER_REQUEST_TIMEOUT_SECONDS: Final[int] = 180


def _flatten_content(content: Any) -> str:
    """Return text content from Together responses as a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if isinstance(content, Iterable) and not isinstance(content, (dict,)):
        parts = [_flatten_content(item) for item in content]
        return "".join(parts)
    if isinstance(content, dict):
        # Common structure: {"type": "output_text", "text": "..."}
        text = content.get("text")
        if text is not None:
            return _flatten_content(text)
    return str(content)


class TogetherProvider(BaseLLMProvider):
    """LLM provider that wraps the Together AI chat completion API."""

    retry_message = "Together AI API request failed."

    def __init__(self, *, api_key: str | None = None, default_wait_time: int | None = None) -> None:
        """Instantiate the Together AI client using the provided API key.

        Args:
            api_key: Together AI API key. If None, an error will be raised.
            default_wait_time: Optional custom backoff interval.

        Raises:
            ValueError: If api_key is None.
        """
        super().__init__(default_wait_time=default_wait_time)
        if api_key is None:
            raise ValueError(
                "API key required for TogetherProvider. "
                "Call configure_api_keys() or provide api_key parameter."
            )
        self._together_client = Together(
            api_key=api_key,
            timeout=TOGETHER_REQUEST_TIMEOUT_SECONDS,
        )

    def _call_model(self, *, model_id: str, prompt: str, options: dict[str, Any]) -> str:
        request_payload: Dict[str, Any] = {
            **options,
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        response = self._together_client.chat.completions.create(**request_payload)

        message_content: Union[str, Iterable[Any]] = response.choices[0].message.content

        return _flatten_content(message_content).strip()
