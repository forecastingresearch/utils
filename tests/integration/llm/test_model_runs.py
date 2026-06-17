"""Integration smoke tests for shared LLM model runs."""

import os

import pytest

from utils.llm import model_runs

from .helpers import assert_capital_of_france


def _latest_active_model_run_keys_by_provider() -> tuple[str, ...]:
    """Return one latest active model-run key for each API provider route."""
    latest_by_provider: dict[str, model_runs.ModelRun] = {}
    for run in model_runs.ACTIVE_MODEL_RUNS:
        provider_name = run.provider.name
        previous = latest_by_provider.get(provider_name)
        if previous is None or (
            run.release_date,
            run.model_key,
            run.model_run_key,
        ) > (
            previous.release_date,
            previous.model_key,
            previous.model_run_key,
        ):
            latest_by_provider[provider_name] = run
    return tuple(
        latest_by_provider[provider_name].model_run_key
        for provider_name in sorted(latest_by_provider)
    )


DEFAULT_SMOKE_MODEL_RUN_KEYS = _latest_active_model_run_keys_by_provider()


def _selected_model_run_keys() -> tuple[str, ...]:
    """Return model-run keys selected for live smoke testing."""
    raw_keys = os.getenv("LLM_MODEL_RUN_KEYS")
    if not raw_keys:
        return DEFAULT_SMOKE_MODEL_RUN_KEYS
    selected_keys = tuple(key.strip() for key in raw_keys.split(",") if key.strip())
    if not selected_keys:
        return DEFAULT_SMOKE_MODEL_RUN_KEYS
    return selected_keys


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_run",
    model_runs.select_model_runs(_selected_model_run_keys()),
    ids=lambda run: run.slug,
)
def test_model_run_live_call(model_run: model_runs.ModelRun):
    """A shared model run should be callable with its declared provider options."""
    assert_capital_of_france(model_run.get_response)
