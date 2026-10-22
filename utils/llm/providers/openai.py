"""Helpers for invoking OpenAI models with retry support."""

from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI  # type: ignore[import]

from ..utils import TransientProviderError
from .base import BaseLLMProvider

# Error codes on a `failed` Responses API response that the provider's side caused, so the same
# request can succeed later. Every other code (for example `invalid_prompt`) is the request's fault.
_TRANSIENT_RESPONSE_ERROR_CODES = frozenset({"server_error", "rate_limit_exceeded"})


def raise_unless_completed(response: Any, *, provider_name: str, model_id: str) -> None:
    """Raise when a Responses API response did not finish with status `completed`.

    A `failed` response with a server-side error code raises `TransientProviderError` so the
    retry loop tries again. An `incomplete` response (the model stopped early, for example at
    `max_output_tokens` or a content filter), a `failed` response with any other error code, and
    any unknown status raise a plain `RuntimeError`, which is terminal.
    """
    status = getattr(response, "status", None)
    if status == "completed":
        return
    error = getattr(response, "error", None)
    error_code = getattr(error, "code", None)
    error_message = getattr(error, "message", None)
    if status == "failed" and error_code in _TRANSIENT_RESPONSE_ERROR_CODES:
        raise TransientProviderError(
            f"{provider_name} response for {model_id} failed (status={status}, "
            f"error={error_code}: {error_message})"
        )
    status_text = f"{provider_name} response for {model_id} not completed (status={status})"
    incomplete_details = getattr(response, "incomplete_details", None)
    if incomplete_details:
        status_text += f", incomplete_details={incomplete_details}"
    if error:
        status_text += f", error={error}"
    raise RuntimeError(status_text)


class OpenAIProvider(BaseLLMProvider):
    """LLM provider that wraps the OpenAI Responses API."""

    retry_message = "OpenAI API request failed."

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
        request_payload: Dict[str, Any] = {
            **options,
            "model": model_id,
            "input": prompt,
        }

        response = self._openai_client.responses.create(**request_payload)

        raise_unless_completed(response, provider_name="OpenAI", model_id=model_id)

        # output_text is the text of the response in reasoning models
        output_text = getattr(response, "output_text", "")
        if isinstance(output_text, str):
            return output_text.strip()

        return str(output_text)
