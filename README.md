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

## LLM Text Output

Shared `ModelRun` objects are the primary surface for LLM calls. A model run is an exact base model
plus the provider options used for benchmarking. It's identified by an immutable `model_run_key` or
by a more-easily-interpreted `slug`.

To list available model run `slug` and `model_run_key` values:
```python
from utils.llm import ACTIVE_MODEL_RUNS

width = max(len(run.slug) for run in ACTIVE_MODEL_RUNS)
for run in ACTIVE_MODEL_RUNS:
    print(f"{run.slug:<{width}}  {run.model_run_key}")
```

You can call a model run using the `get_response(prompt)` method as shown below:
```python
from utils.llm.model_runs import get_model_run, get_model_run_by_slug
from utils.llm.model_registry import configure_api_keys

configure_api_keys(from_gcp=True)
# configure_api_keys(openai="...", anthropic="...")  # if not using GCP, pass keys explicitly

model_runs = [
    get_model_run("gpt-5-mini-2025-08-07-run-variant-02"),
    get_model_run_by_slug("claude-sonnet-4-6-1024"),
]

for model_run in model_runs:
    response = model_run.get_response("What is the capital of France?")
    print(model_run.slug, response)
```

The example above selects the first run by immutable `model_run_key` and the second by its
human-readable slug `claude-sonnet-4-6-1024`. Use immutable `model_run_key` values for durable
references. Human-readable slugs are available for display and convenience lookups but while the
`model_run_key` should be used for stable lookups

## LLM Structured Output

`get_response` returns the model's text. When you ask a provider for structured output, that
text is the JSON, so you parse it yourself with `json.loads`.

Structured output needs no special support in this package: the options you pass to
`get_response` are forwarded to the provider unchanged. That means you use each provider's
own option names, which differ. Every example below continues from this shared setup, and
they all request the same quantile forecast:

```python
import json

from pydantic import BaseModel

from utils.llm.model_registry import configure_api_keys
from utils.llm.model_runs import get_model_run

configure_api_keys(from_gcp=True)
# configure_api_keys(openai="...", anthropic="...")  # if not using GCP, pass keys explicitly


class Quantile(BaseModel):
    """One point of a predictive distribution."""

    value: float
    rationale: str


class QuantileForecast(BaseModel):
    """Five-point quantile forecast for a numeric quantity."""

    p10: Quantile
    p25: Quantile
    p50: Quantile
    p75: Quantile
    p90: Quantile


PROMPT = (
    "Forecast the global average surface temperature anomaly in 2030, in degrees "
    "Celsius above the 1850-1900 pre-industrial baseline. Give the 10th, 25th, 50th, "
    "75th, and 90th percentiles, and a one-sentence rationale for each."
)
```

Not every model supports structured output. Check the `structured_output` flag on the model's
Models.dev metadata (`model_run.model.models_dev_metadata.raw`) before relying on it.

Output handling is the same everywhere — `json.loads` the response, then validate it into
your model. Only the option you send differs.

### Anthropic

Anthropic takes the Pydantic class directly as `output_format` and derives the JSON schema
itself:

```python
model_run = get_model_run("claude-haiku-4-5-20251001-run-variant-02")
response = model_run.get_response(PROMPT, output_format=QuantileForecast)

forecast = QuantileForecast.model_validate(json.loads(response))
print(forecast.p50.value, forecast.p50.rationale)
```

### Gemini

Gemini's SDK likewise accepts the class, as `response_schema`, and converts it before
sending. It additionally requires a matching `response_mime_type`:

```python
model_run = get_model_run("gemini-3.1-flash-lite-run-variant-01")
response = model_run.get_response(
    PROMPT,
    response_schema=QuantileForecast,
    response_mime_type="application/json",
)

forecast = QuantileForecast.model_validate(json.loads(response))
print(forecast.p50.value, forecast.p50.rationale)
```

### OpenAI

OpenAI cannot take the class, because the Pydantic-aware parameter belongs to the SDK's
`responses.parse()` while this package calls `responses.create()`. You pass a JSON schema
dict under `text` instead, and derive it from the class with `model_json_schema()`:

```python
# OpenAI requires every object, nested ones included, to forbid extra properties and to
# list all of its properties as required. Pydantic emits neither.
schema = QuantileForecast.model_json_schema()
for obj in [schema, *schema.get("$defs", {}).values()]:
    obj["additionalProperties"] = False
    obj["required"] = list(obj["properties"])

model_run = get_model_run("gpt-5.4-nano-2026-03-17-run-variant-01")
response = model_run.get_response(
    PROMPT,
    text={
        "format": {
            "type": "json_schema",
            "name": "quantile_forecast",
            "schema": schema,
            "strict": True,
        }
    },
)

forecast = QuantileForecast.model_validate(json.loads(response))
print(forecast.p50.value, forecast.p50.rationale)
```

### Moonshot AI

Moonshot uses an OpenAI-compatible `chat.completions` endpoint rather than the Responses
API, so the option is `response_format` and the schema sits one level deeper, under
`json_schema`:

```python
# Same requirement as OpenAI: every object, nested ones included, must forbid extra
# properties and list all of its properties as required.
schema = QuantileForecast.model_json_schema()
for obj in [schema, *schema.get("$defs", {}).values()]:
    obj["additionalProperties"] = False
    obj["required"] = list(obj["properties"])

model_run = get_model_run("kimi-k3-run-variant-01")
response = model_run.get_response(
    PROMPT,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "quantile_forecast",
            "schema": schema,
            "strict": True,
        },
    },
)

forecast = QuantileForecast.model_validate(json.loads(response))
print(forecast.p50.value, forecast.p50.rationale)
```

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
