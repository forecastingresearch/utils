"""Artificial Analysis-backed model-run declarations."""

from collections.abc import Callable
from typing import Any

# Every declaration here is intended to be part of MODEL_RUNS. Add a run here
# only after its provider-callable options are ready for benchmark selection.
ARTIFICIAL_ANALYSIS_MODEL_RUN_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "model_run_key": "claude-opus-4-7-aa-run-variant-01",
        "slug": "claude-opus-4-7-high-16384",
        "model_key": "claude-opus-4-7",
        "options": {
            "max_tokens": 16384,
            "output_config": {"effort": "high"},
        },
        "artificial_analysis_id": "2fa8e143-77a8-4d05-bfa8-d3b54634c00f",
    },
    {
        "model_run_key": "claude-opus-4-7-aa-run-variant-02",
        "slug": "claude-opus-4-7-adaptive-thinking-max-128000",
        "model_key": "claude-opus-4-7",
        "options": {
            "max_tokens": 128000,
            "output_config": {"effort": "max"},
            "thinking": {"type": "adaptive"},
        },
        "artificial_analysis_id": "e9a09db3-8fd6-41dd-ba2f-20e0a2bff7f2",
    },
)


def create_artificial_analysis_model_runs(
    model_run_factory: Callable[..., Any],
) -> list[Any]:
    """Build AA-backed model runs using the main registry's factory."""
    return [
        model_run_factory(**declaration)
        for declaration in ARTIFICIAL_ANALYSIS_MODEL_RUN_DECLARATIONS
    ]
