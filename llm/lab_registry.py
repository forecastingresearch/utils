"""Registry of labs responsible for published LLMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Lab:
    """Metadata describing an LLM research lab."""

    name: str
    logo: str


LABS: Final[dict[str, Lab]] = {
    "ForecastBench": Lab(name="ForecastBench", logo="fri.png"),
    "Anthropic": Lab(name="Anthropic", logo="anthropic.svg"),
    "DeepSeek": Lab(name="DeepSeek", logo="deepseek.svg"),
    "Moonshot": Lab(name="Moonshot", logo="moonshot.svg"),
    "Google": Lab(name="Google", logo="deepmind.svg"),
    "Meta": Lab(name="Meta", logo="meta.svg"),
    "Mistral AI": Lab(name="Mistral AI", logo="mistral.svg"),
    "Mistral": Lab(name="Mistral", logo="mistral.svg"),
    "OpenAI": Lab(name="OpenAI", logo="openai.svg"),
    "Qwen": Lab(name="Qwen", logo="qwen.svg"),
    "xAI": Lab(name="xAI", logo="xai.svg"),
    "Z.ai": Lab(name="Z.ai", logo="zai.svg"),
}
