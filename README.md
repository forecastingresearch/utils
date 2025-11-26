# utils

This package provides utilities for use across the Forecasting Research Institute's codebase.

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

## Using GCP

In some cases, your project may have a API keys set in a Google Cloud Project. 

If so, you can use the `gcp=True` shortcut to set your keys for all model providers:

```
configure_api_keys(gcp=True) # Configure all provider keys from GCP.
model = next(m for m in MODELS if m.id == "gpt-4.1-mini")
response = model.get_response("Hello")
```

## Other methods

- `archiving/` - Utility functions for tar.gz compression & extraction.
- `gcp/` - Utilities for interacting with Google Cloud Storage.

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