"""Helpers for invoking Mistral models with retry support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union

from mistralai import Mistral, UserMessage  # type: ignore[import]
from pydantic import BaseModel

from ..utils import create_json_prompt, extract_json_from_text, parse_and_validate_json
from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..model_registry import Model


class MistralProvider(BaseLLMProvider):
    """LLM provider that calls the Mistral chat completion API."""

    rate_limit_message = "Mistral API request exceeded rate limit."

    def __init__(self, *, api_key: str | None = None, default_wait_time: int | None = None) -> None:
        """Instantiate the Mistral client using the provided API key.

        Args:
            api_key: Mistral API key. If None, an error will be raised.
            default_wait_time: Optional custom backoff interval.

        Raises:
            ValueError: If api_key is None.
        """
        super().__init__(default_wait_time=default_wait_time)
        if api_key is None:
            raise ValueError(
                "API key required for MistralProvider. "
                "Call configure_api_keys() or provide api_key parameter."
            )
        self._mistral_client = Mistral(api_key=api_key)

    def _call_model(self, model: "Model", prompt: str, **options: Any) -> str:
        max_tokens = options.get("max_tokens")
        temperature = options.get("temperature")
        model_name = model.full_name

        messages: List[Union[Dict[str, str], UserMessage]] = [
            {"role": "user", "content": prompt},
        ]

        request_payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        if temperature is not None:
            request_payload["temperature"] = temperature
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens

        chat_response = self._mistral_client.chat.complete(**request_payload)

        content = chat_response.choices[0].message.content

        # Handle different content types: None, str, or list
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Content can be a list of content blocks, extract text from each
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                else:
                    text_parts.append(str(item))
            return "".join(text_parts)

        return str(content)

    def _call_model_structured(
        self,
        model: "Model",
        prompt: str,
        response_schema: type[BaseModel],
        **options: Any,
    ) -> BaseModel:
        """Execute a structured output request using JSON parsing fallback."""
        max_tokens = options.get("max_tokens")
        temperature = options.get("temperature")
        model_name = model.full_name

        # Convert Pydantic schema to JSON schema and enhance prompt
        json_schema = response_schema.model_json_schema()
        enhanced_prompt = create_json_prompt(prompt, json_schema)

        messages: List[Union[Dict[str, str], UserMessage]] = [
            {"role": "user", "content": enhanced_prompt},
        ]

        request_payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        if temperature is not None:
            request_payload["temperature"] = temperature
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens

        chat_response = self._mistral_client.chat.complete(**request_payload)

        message = chat_response.choices[0].message
        content = message.content

        # Handle different content types: None, str, or list
        # Mistral may return TextChunk objects with thinking attributes for reasoning models
        response_text = ""
        if content is None:
            response_text = ""
        elif isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            # Content can be a list of content blocks, extract text from each
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    # Dictionary with text key
                    if "text" in item:
                        text_parts.append(item["text"])
                elif hasattr(item, "text"):
                    # TextChunk or similar object - extract text attribute
                    item_text = getattr(item, "text", None)
                    if item_text:
                        text_parts.append(str(item_text))
            response_text = "".join(text_parts)
        elif hasattr(content, "text"):
            # Handle TextChunk or similar objects directly
            response_text = str(getattr(content, "text", ""))
        else:
            # Check if content stringifies to thinking=... (reasoning content)
            # This happens when content is an object with a thinking attribute
            content_str = str(content)
            if content_str.startswith("thinking="):
                # For reasoning models, extract text from TextChunk objects in thinking attribute
                if hasattr(content, "thinking"):
                    thinking = getattr(content, "thinking", None)
                    if isinstance(thinking, list):
                        # Extract text from TextChunk objects in thinking list
                        text_parts = []
                        for item in thinking:
                            if hasattr(item, "text"):
                                item_text = getattr(item, "text", None)
                                if item_text:
                                    text_parts.append(str(item_text))
                        response_text = "".join(text_parts)
                    elif hasattr(thinking, "text"):
                        # Single TextChunk object
                        response_text = str(getattr(thinking, "text", ""))
                # Also check for a separate text attribute on content
                if not response_text and hasattr(content, "text"):
                    text_attr = getattr(content, "text", None)
                    if text_attr:
                        response_text = str(text_attr)
                # Fallback: check message for text attribute
                if not response_text and hasattr(message, "text") and message.text:
                    response_text = str(message.text)
            else:
                response_text = content_str

        response_text = response_text.strip()

        # Extract and validate JSON
        json_text = extract_json_from_text(response_text, provider_name="Mistral")
        return parse_and_validate_json(json_text, response_schema, "Mistral", response_text)
