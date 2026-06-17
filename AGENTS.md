# Repository Instructions

This repo is the shared `fri-utils` package. It contains general-purpose utilities as well as the shared LLM registry.

## Package Baseline

This package targets Python 3.10+. Black is configured with
`target-version = ["py310"]`; do not broaden `requires-python` without first
checking that formatted code remains valid for the older target.

## Local Development Setup

Use Python 3.10 or newer for local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` delegates to `.[dev]`; it installs this package and the dev
tools from `pyproject.toml` without editable mode.

When another repo needs local utils changes during development, use that repo's
virtual environment and install utils explicitly in editable mode, for example:

```bash
python -m pip install -e ../utils
```

Do not add local relative paths to another repo's requirements files. Those
files should use the deployed git pin when ready to deploy.

## Shared LLM Registry

The shared LLM registry has two layers:

- `utils.llm.model_registry.MODELS` contains canonical provider-callable base models.
- `utils.llm.model_runs.MODEL_RUNS` contains exact benchmarkable model-plus-options runs.

Users of this registry should choose from `MODEL_RUNS` by `model_run_key`; that key is the immutable benchmark identity for the exact model and options that were used. Use `slug` only as a human-readable convenience name.

Any new option or set of options to a model will require a new model run entry.

### Adding a base model

- Add provider/lab registry entries first only if the provider or lab is missing.
- Look up the model in Models.dev. Prefer a `ModelsDevReference` when Models.dev
  has the provider/model entry.
- Verify the release date that Models.dev provides.
- In Models.dev source paths, `provider_id` is the folder under `providers/`,
  and `model_id` is the TOML filename stem under `models/`, for example
  `providers/anthropic/models/claude-opus-4-8.toml` maps to `anthropic` /
  `claude-opus-4-8`.
- The checked-in Models.dev snapshot is not a catalog; it contains only
  registry-referenced models and only `id`, `name`, and `release_date`.
- Use exact Models.dev `provider_id`/`model_id` values. If a reference is wrong,
  refreshing the snapshot should fail and suggest nearby Models.dev entries.
- Use `manual_release_date` when the model is missing from Models.dev, when the
  Models.dev entry lacks a usable full release date, when the Models.dev entry is incorrect,
  or for deliberate historical/manual entries.
- Put the model in the provider-specific list in `utils/llm/model_registry.py`:
  `OPENAI_MODELS`, `TOGETHER_MODELS`, `ANTHROPIC_MODELS`, `XAI_MODELS`, or `GOOGLE_MODELS`.
- Insert the model where `(release_date, model_key)` stays ascending within its
  provider-specific list.
- Use `filename_safe_name` when a model key needs to become part of a filename; do not hand-roll lossy replacements for characters such as `/` or `*`.
- Use `provider_model_id` for the exact string sent to the provider API. It may differ from `model_key`, especially for routed providers like Together.
- Set `active=False` when a provider stops supporting a model route.
- Do not add duplicate `model_key`'s. `MODELS = create_models_list(...)` validates uniqueness.

After changing `ModelsDevReference` values, refresh the Models.dev snapshot from the utils repo:
```bash
python - <<'PY'
from scripts.refresh_models_dev_metadata import write_models_dev_snapshot

write_models_dev_snapshot()
PY
```

### Adding a model run

- Add it to `utils/llm/model_runs.py` with `_model_run(model_run_key=..., slug=..., model_key=..., options=...)`.
- Write `model_run_key` explicitly as the stable benchmark identifier. Regular runs
  use `model_key` plus `-run-variant-XX`; Artificial Analysis-backed runs use
  `model_key` plus `-aa-run-variant-XX`. In both cases, `XX` is the next two-digit
  variant number for that `model_key`, and the key must never change after publication.
- Add every new `model_run_key` to `HISTORICAL_MODEL_RUN_KEYS` in `tests/unit/test_llm_model_runs.py`; the test ledger is the explicit list of all model-run keys ever created.
- Write `slug` explicitly as the descriptive human-readable identifier. Slugs must be unique, but may be changed before or after publication when naming conventions improve.
- Use `filename_safe_name` when a model-run key needs to become part of a filename; do not hand-roll lossy replacements for characters such as `/` or `*`.
- Put every runtime call option in the `ModelRun` declaration; do not add hidden defaults elsewhere.
- Use exact provider option names and values as they are passed to `get_response`.
- If an option affects performance and should appear in human-readable slugs, filenames, or forecast labels, make the handwritten `slug` reflect it.
- Unit tests validate declared top-level provider option names against installed provider SDK
  signatures and check selected Models.dev capabilities such as temperature support,
  reasoning support, and output-token limits. Run those tests and the live model-run
  integration smoke test when adding or changing options.
- Do not add duplicate `model_run_key`s, duplicate `slug`s, or duplicate model-plus-options fingerprints. `MODEL_RUNS = create_model_runs_list(...)` validates uniqueness.
- `MODEL_RUNS` is the historical registry. `ACTIVE_MODEL_RUNS` is derived from
  it by dropping runs whose base `Model` has `active=False`.
- Add unit tests for new naming behavior, registry inclusion, and routed provider options when relevant.
- Do not make pointless tests; e.g. do _not_ add tests that only assert old attributes/constants are absent or that new attributes/constants exist.
- Do not add tests that merely restate the implementation data being changed. A
  test must verify behavior, a reusable invariant, or a contract whose expected
  value comes from an independent source of truth.

#### Adding an Artificial Analysis-backed model run

- Use the checked-in Artificial Analysis snapshot as the source for stable AA model IDs and AA names.
- Refresh the snapshot from the AA endpoint; do not hand-edit individual AA models into the JSON file.
- The official AA API key is `API_KEY_ARTIFICIAL_ANALYSIS` in GCP Secret Manager.
- Do not hard-code an AA display name in a `ModelRun`; `ModelRun.display_name` currently returns the base `model_key`.
- A non-null `artificial_analysis_id` is the marker that a run is AA-backed.
- AA-backed model-run keys must use `model_key` plus `-aa-run-variant-XX` so they
  are not confused with regular runs for the same base model.
- Add or update the canonical base `Model` only if the provider-callable model is missing from `utils.llm.model_registry`.
- Add the callable model-plus-options declaration to
  `ARTIFICIAL_ANALYSIS_MODEL_RUN_DECLARATIONS` in
  `utils/llm/artificial_analysis_model_runs.py`. Every declaration there is
  automatically included in `utils.llm.model_runs.MODEL_RUNS`; do not add the
  same AA run manually to `MODEL_RUNS`.
- Use the exact provider option names that are passed at runtime. Token suffixes in slugs must reflect the actual token cap option used for the call.

Artificial Analysis token caps should be encoded in the run options this way:

- Non-reasoning models: use `16384` output tokens, adjusted downward if the model has a smaller context window or a lower maximum output-token cap.
- Reasoning models: use the maximum output tokens allowed by the model creator for that reasoning configuration.
- If the correct cap is not clear from provider/model documentation or the AA metadata, stop and confirm rather than guessing.

After adding an AA model run:

- Add or update unit tests that prove the AA ID resolves from the snapshot and that the declared options match the intended provider call.
- Add or update shared registry coverage tests for the new selectable model-run key.
- Run the focused model-run and AA metadata tests, then run the full lint/test suite before committing.

### Live Model-Run Smoke Tests

Integration tests that hit real LLM APIs require provider API keys.

- `tests/conftest.py` loads `.env`, then `configure_api_keys(from_gcp=True)` when pytest is run with `--integration`.
- `configure_api_keys(from_gcp=True)` reads provider keys from GCP Secret Manager using the secret names in `utils/helpers/constants.py`.
- The standard LLM secret names are `API_KEY_OPENAI`, `API_KEY_ANTHROPIC`, `API_KEY_GEMINI`, `API_KEY_XAI`, and `API_KEY_TOGETHERAI`.
- To test a specific shared model run, set `LLM_MODEL_RUN_KEYS` to one or more comma-separated immutable `model_run_key`s and run `pytest --integration tests/integration/llm/test_model_runs.py`.
- The model-run integration test calls `model_run.get_response`, so it uses the run's declared provider route, provider model ID, and options.
- For a newly added model run, prefer running its exact smoke test before assuming the provider accepts the declared options.

## Validation

- Run `make lint` before committing. It runs `isort .`, `black .`, `flake8 .`,
  and `pydocstyle .`.
- Run `make test` before committing code changes. Use `PYTEST_ARGS=...` for a
  focused test pass while iterating.
- Run `make test-integration` or `make test-integration-parallel` only when the
  relevant provider/GCP credentials are available.
