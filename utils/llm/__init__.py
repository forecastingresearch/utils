"""Utilities for working with large language models."""

from importlib import import_module

__all__ = ["lab_registry", "model_registry", "provider_registry", "providers"]


def __getattr__(name: str):
    """Lazily import LLM submodules on attribute access."""
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
