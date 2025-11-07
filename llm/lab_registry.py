"""Registry of labs responsible for published LLMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Lab:
    """Metadata describing an LLM research lab."""

    name: str


LABS: Final[dict[str, Lab]] = {
    "Anthropic": Lab(name="Anthropic"),
    "DeepSeek": Lab(name="DeepSeek"),
    "Moonshot": Lab(name="Moonshot"),
    "Google": Lab(name="Google"),
    "Meta": Lab(name="Meta"),
    "Mistral AI": Lab(name="Mistral AI"),
    "Mistral": Lab(name="Mistral"),
    "OpenAI": Lab(name="OpenAI"),
    "Qwen": Lab(name="Qwen"),
    "xAI": Lab(name="xAI"),
    "Z.ai": Lab(name="Z.ai"),
}
