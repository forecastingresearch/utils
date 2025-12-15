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


```
from utils.llm.model_registry import configure_api_keys, MODELS

# Input the API key for any model provider you like!
configure_api_keys(
    openai="...",
    anthropic="...",
    google="...",
    xai="...",
    together="...",
    mistral="...",
)

# Call any model we support!
# See the full list of supported models in `utils/llm/model_registry.py`
model = next(m for m in MODELS if m.id == "gemini-2.5-flash")
model.get_response("Hello")
# > "Hello! How can I help you?"
```
# Methods

## Configuring LLMs

`model.get_response()` accepts any optional argument your model accepts.
For example:

```
model.get_response(
    'What is the capital of France?',
    temperature=0,
    max_tokens=256,
)
```

You can check whether an option is supported by looking at the code for the respective model provider (`utils/llm/providers`). 

If you don’t see an option you need, feel free to open a GitHub issue!


### Configuring keys from GCP Secret Manager

In some cases, your project may have a API keys set in a Google Cloud Project. 

If so, you can use the `from_gcp=True` shortcut to set your keys for all model providers:

```
configure_api_keys(from_gcp=True) # Configure all provider keys from GCP.
model = next(m for m in MODELS if m.id == "gpt-4.1-mini")
response = model.get_response("Hello")
```

If you're setting up a Google Cloud Project, the API keys must be stored in Secret Manager with the following names:
- `API_KEY_ANTHROPIC` for Anthropic
- `API_KEY_GEMINI` for Google/Gemini
- `API_KEY_MISTRAL` for Mistral
- `API_KEY_OPENAI` for OpenAI
- `API_KEY_XAI` for xAI
- `API_KEY_TOGETHERAI` for Together AI

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
