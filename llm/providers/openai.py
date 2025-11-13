"""Helpers for invoking OpenAI models with retry support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from openai import OpenAI  # type: ignore[import]

from ...helpers.constants import OPENAI_API_KEY_SECRET_NAME
from ...keys.secrets import get_secret
from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model


class OpenAIProvider(BaseLLMProvider):
    """LLM provider that wraps the OpenAI Responses API."""

    rate_limit_message = "OpenAI API request exceeded rate limit."

    def __init__(self) -> None:
        """Instantiate the OpenAI client using the configured API secret."""
        super().__init__()
        api_key = get_secret(OPENAI_API_KEY_SECRET_NAME)
        self._openai_client = OpenAI(api_key=api_key)

    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        temperature = options.get("temperature", 0.8)
        max_tokens = options.get("max_tokens")
        model_name = model.full_name

        # OpenAI doesn't support temperature for reasoning models
        if model.reasoning_model:
            request_payload: Dict[str, Any] = {
                "model": model_name,
                "input": prompt,
            }
        else:
            request_payload = {
                "model": model_name,
                "input": prompt,
                "temperature": temperature,
            }

        if max_tokens is not None:
            request_payload["max_output_tokens"] = max_tokens

        response = self._openai_client.responses.create(**request_payload)

        # Get status text (this is useful for catching errors in reasoning models)
        status = getattr(response, "status", None)
        if status != "completed":
            reason = getattr(response, "incomplete_details", None)
            status_text = f"OpenAI response incomplete (status={status})"
            if reason:
                status_text += f", reason={reason}"
            raise RuntimeError(status_text)

        # output_text is the text of the response in reasoning models
        output_text = getattr(response, "output_text", "")
        if isinstance(output_text, str):
            return output_text.strip()

        return str(output_text)
