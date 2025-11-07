# utils

This package provides utilities for use across the Forecasting Research Institute's codebase.

# Structure

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

## Test

```
make test
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