# utils

This package provides utilities for use across the Forecasting Research Institute's codebase.

# Install

Install the package using pip:

```bash
pip install git+https://github.com/forecastingresearch/utils.git
```

Or add it to your `requirements.txt`:

```
fri-utils
```

If you're using `uv`:

```bash
uv add fri-utils
```

# Quickstart

## LLM model-run quickstart

```python
from utils.llm.model_runs import get_model_run
from utils.llm.model_registry import configure_api_keys

configure_api_keys(from_gcp=True)

model_runs = [
    get_model_run("gpt-5-mini-2025-08-07-run-variant-02"),
    get_model_run("claude-sonnet-4-6-run-variant-01"),
]

for model_run in model_runs:
    response = model_run.get_response("What is the capital of France?")
    print(model_run.slug, response)
```

Shared `ModelRun` objects are the primary surface for LLM calls. A model run is
an exact base model plus the provider options used for benchmarking. The example
above selects two runs by immutable `model_run_key`; the first has the
human-readable slug `gpt-5-mini-2025-08-07-1024`.

Use immutable `model_run_key` values for durable references. Human-readable
slugs are available for display and convenience lookups, but they are not the
stable integration contract.

# Methods

## Configuring LLMs

Benchmark callers should choose shared model-run configurations by immutable
`model_run_key` through `get_model_run`.

`model_run.get_response()` accepts provider-native request options as keyword
arguments.
For example:

```python
model_run.get_response(
    'What is the capital of France?',
    temperature=0,
)
```

Use option names supported by the respective provider (`utils/llm/providers`).

If you don’t see an option you need, feel free to open a GitHub issue!

### Third-party metadata

The shared LLM registry includes normalized metadata from Models.dev and
Artificial Analysis. See `THIRD_PARTY_NOTICES.md` for Models.dev license terms
and Artificial Analysis attribution.


### Configuring keys from GCP Secret Manager

In some cases, your project may have API keys set in a Google Cloud Project.

If so, you can use the `from_gcp=True` shortcut to set your keys for all model providers:

```python
from utils.llm.model_runs import get_model_run
from utils.llm.model_registry import configure_api_keys

configure_api_keys(from_gcp=True) # Configure all provider keys from GCP.
model_run = get_model_run("gpt-5-mini-2025-08-07-run-variant-02")
response = model_run.get_response("Hello")
```

If you're setting up a Google Cloud Project, the API keys must be stored in Secret Manager with the following names:
- `API_KEY_ANTHROPIC` for Anthropic
- `API_KEY_GEMINI` for Google/Gemini
- `API_KEY_OPENAI` for OpenAI
- `API_KEY_XAI` for xAI
- `API_KEY_TOGETHERAI` for Together AI
- `API_KEY_ARTIFICIAL_ANALYSIS` for refreshing the Artificial Analysis metadata snapshot

You can also check `utils/helpers/constants.py` for the complete list of secret names.

## Other utilities

To import other utilities from this package, use:

```python
from utils import archiving  # tar.gz compression & extraction
from utils import gcp  # Google Cloud Storage utilities
```

For example:

```python
from utils.gcp.storage import list_files, upload_file, download_file
from utils.archiving.tar_gz import compress_directory, extract_archive
```

# Development

## Install

First, install dependencies. We recommend using a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

If you want to run the integration tests, make sure you're authenticated with Google Cloud. You'll need [the `gcloud` CLI](https://docs.cloud.google.com/sdk/docs/install-sdk).

```
gcloud auth application-default login
```

After authenticating, you'll see a message like:

```
Credentials saved to file: [/home/yourusername/.config/gcloud/application_default_credentials.json]
```

Copy `sample.env` to `.env` and replace the `GOOGLE_APPLICATION_CREDENTIALS` with this path. (Reach out to a team member to check that you have the right values for the other variables in this file.)

## Test

To run unit tests:

```
make test
```

To run integration tests:

```
make test-integration-parallel
```

## Contributing

Be sure to lint your contribution before creating a pull request:

```
make lint
```

Check testing coverage:

```
make coverage
```
