"""Helpers for invoking xAI models with retry support."""

from __future__ import annotations

from typing import Any, Dict

import openai

from .base import BaseLLMProvider


class XAIProvider(BaseLLMProvider):
    """LLM provider that wraps calls to the xAI Responses API."""

    retry_message = "xAI API request failed."

    def __init__(self, *, api_key: str | None = None, default_wait_time: int | None = None) -> None:
        """Instantiate the xAI client using the provided API key.

        Args:
            api_key: xAI API key. If None, an error will be raised.
            default_wait_time: Optional custom backoff interval.

        Raises:
            ValueError: If api_key is None.
        """
        super().__init__(default_wait_time=default_wait_time)
        if api_key is None:
            raise ValueError(
                "API key required for XAIProvider. "
                "Call configure_api_keys() or provide api_key parameter."
            )
        self._xai_client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    def _call_model(self, *, model_id: str, prompt: str, options: dict[str, Any]) -> str:
        request_payload: Dict[str, Any] = {
            **options,
            "model": model_id,
            "input": prompt,
        }

        response = self._xai_client.responses.create(**request_payload)

        # Get status text (this is useful for catching errors in reasoning models)
        status = getattr(response, "status", None)
        if status != "completed":
            reason = getattr(response, "incomplete_details", None)
            status_text = f"xAI response incomplete (status={status})"
            if reason:
                status_text += f", reason={reason}"
            raise RuntimeError(status_text)

        # output_text is the text of the response in reasoning models
        output_text = getattr(response, "output_text", "")
        if isinstance(output_text, str):
            return output_text.strip()

        return str(output_text)
