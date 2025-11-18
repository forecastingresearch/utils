"""Helpers for invoking Together AI models with retry support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Union

from together import Together  # type: ignore[import]

from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model


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

    rate_limit_message = "Together AI API request exceeded rate limit."

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
        self._together_client = Together(api_key=api_key)

    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        temperature = options.get("temperature")
        max_tokens = options.get("max_tokens")
        model_name = model.full_name

        request_payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        if temperature is not None:
            request_payload["temperature"] = temperature
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens

        response = self._together_client.chat.completions.create(**request_payload)

        message_content: Union[str, Iterable[Any]] = response.choices[0].message.content

        return _flatten_content(message_content).strip()
