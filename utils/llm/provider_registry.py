"""Registry of API providers used to call LLMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Provider:
    """Metadata for an API provider route."""

    name: str
    key_name: str


PROVIDERS: Final[dict[str, Provider]] = {
    "OpenAI": Provider(name="OpenAI", key_name="openai"),
    "Anthropic": Provider(name="Anthropic", key_name="anthropic"),
    "Google": Provider(name="Google", key_name="google"),
    "xAI": Provider(name="xAI", key_name="xai"),
    "Together": Provider(name="Together", key_name="together"),
}
