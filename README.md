# utils

This package provides utilities for use across the Forecasting Research Institute's codebase.

# Structure

- `archiving/` - Utility functions for tar.gz compression & extraction.
- `gcp/` - Utilities for interacting with Google Cloud Storage.
- `keys/` - Utilities for interacting with keys.
- `llm/` - Utilities for calling various model providers.
    - `llm/model_registry.py` - Exposes available models.
    - `Model.get_response(prompt, **options)` - Gets response from a given prompt.

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