"""Utilities for working with large language models."""

from importlib import import_module

_MODEL_RUN_EXPORTS = {
    "ACTIVE_MODEL_RUNS",
    "ACTIVE_MODEL_RUNS_BY_KEY",
    "ACTIVE_MODEL_RUNS_BY_SLUG",
    "MODEL_RUNS",
    "MODEL_RUNS_BY_KEY",
    "MODEL_RUNS_BY_SLUG",
    "ModelRun",
    "get_model_run",
    "get_model_run_by_slug",
    "select_model_runs",
}

__all__ = [
    "ACTIVE_MODEL_RUNS",
    "ACTIVE_MODEL_RUNS_BY_KEY",
    "ACTIVE_MODEL_RUNS_BY_SLUG",
    "MODEL_RUNS",
    "MODEL_RUNS_BY_KEY",
    "MODEL_RUNS_BY_SLUG",
    "ModelRun",
    "get_model_run",
    "get_model_run_by_slug",
    "lab_registry",
    "model_registry",
    "model_runs",
    "provider_registry",
    "providers",
    "select_model_runs",
]


def __getattr__(name: str):
    """Lazily import LLM submodules on attribute access."""
    if name in _MODEL_RUN_EXPORTS:
        module = import_module(f"{__name__}.model_runs")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {
        "lab_registry",
        "model_registry",
        "model_runs",
        "provider_registry",
        "providers",
    }:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
