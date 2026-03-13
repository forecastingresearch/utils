"""Utilities package for various helper functions and classes."""

__all__ = [
    "archiving",
    "gcp",
    "helpers",
    "llm",
]
from . import archiving, gcp, helpers


def __getattr__(name: str):
    """Lazily import heavy submodules so lightweight consumers avoid pulling in LLM deps."""
    if name == "llm":
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
