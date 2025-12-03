"""Helpers for invoking OpenAI models with retry support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from openai import OpenAI  # type: ignore[import]
from pydantic import BaseModel

from ..utils import create_json_prompt, extract_json_from_text, parse_and_validate_json
from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model


class OpenAIProvider(BaseLLMProvider):
    """LLM provider that wraps the OpenAI Responses API."""

    rate_limit_message = "OpenAI API request exceeded rate limit."

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

    def _call_model_structured(
        self,
        model: "Model",
        prompt: str,
        response_schema: type[BaseModel],
        **options: Any,
    ) -> BaseModel:
        """Execute a structured output request using JSON parsing fallback.

        Note: OpenAI's Responses API doesn't support structured outputs natively.
        This implementation uses prompt enhancement to request JSON output,
        which is then parsed and validated.
        """
        temperature = options.get("temperature", 0.8)
        max_tokens = options.get("max_tokens")
        model_name = model.full_name

        # Convert Pydantic schema to JSON schema and enhance prompt
        json_schema = response_schema.model_json_schema()
        enhanced_prompt = create_json_prompt(prompt, json_schema)

        # OpenAI doesn't support temperature for reasoning models
        if model.reasoning_model:
            request_payload: Dict[str, Any] = {
                "model": model_name,
                "input": enhanced_prompt,
            }
        else:
            request_payload = {
                "model": model_name,
                "input": enhanced_prompt,
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

        # Extract text from response
        output_text = getattr(response, "output_text", "")
        if not isinstance(output_text, str):
            output_text = str(output_text)
        output_text = output_text.strip()

        # Extract and validate JSON
        json_text = extract_json_from_text(output_text, provider_name="OpenAI")
        return parse_and_validate_json(json_text, response_schema, "OpenAI", output_text)
