"""Registry of labs responsible for published LLMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Lab:
    """Metadata describing an LLM research lab."""

    name: str
    display_name: str | None = None

    @property
    def leaderboard_name(self) -> str:
        """Return the name to show on leaderboards."""
        return self.display_name or self.name


LABS: Final[dict[str, Lab]] = {
    "Anthropic": Lab(name="Anthropic"),
    "DeepSeek": Lab(name="DeepSeek"),
    "Moonshot": Lab(name="Moonshot", display_name="Moonshot AI"),
    "Google DeepMind": Lab(name="Google DeepMind"),
    "Meta": Lab(name="Meta"),
    "OpenAI": Lab(name="OpenAI"),
    "Qwen": Lab(name="Qwen"),
    "xAI": Lab(name="xAI"),
    "Z.ai": Lab(name="Z.ai"),
}
