"""Export the shared LLM model-run registry to NDJSON for the data warehouse.

Use this when the warehouse needs the current model-run dimension table. It
writes one denormalized record per `MODEL_RUNS` entry, one JSON object per
line, with the base-model fields resolved so the warehouse does not need the
Python registry logic. Newline-delimited JSON is what `bq load` and BigQuery
external tables read directly.

Run from the utils repo root. Without `--output` the NDJSON goes to stdout so
it can be streamed straight into GCS without touching disk:

    python -m scripts.export_model_runs | gcloud storage cp - gs://<bucket>/llm-forecasters.ndjson
    python -m scripts.export_model_runs --output llm-forecasters.ndjson

The GitHub Actions workflow `.github/workflows/export-model-runs.yml` runs the
first form on every push to `main` that touches the registry and overwrites the
single object `gs://github-ci-export-for-dwh-ingestion/llm-forecasters.ndjson`.
The file is always a full snapshot: `model_run_key` values are immutable, but
`release_date` and the `models_dev_*` fields can change when the Models.dev
snapshot is refreshed, so the warehouse should reload the whole file, not
append to it.

The `models_dev_*` fields come from the checked-in Models.dev snapshot and are
null for models declared without a `ModelsDevReference`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from utils.llm import model_registry, model_runs

MODELS_DEV_FIELDS = (
    "models_dev_provider_id",
    "models_dev_model_id",
    "models_dev_name",
    "models_dev_context_limit",
    "models_dev_output_limit",
    "models_dev_reasoning",
    "models_dev_structured_output",
    "models_dev_temperature",
    "models_dev_tool_call",
)


def _models_dev_fields(model: model_registry.Model) -> dict[str, Any]:
    """Return the Models.dev identifier and capability fields for a base model."""
    fields: dict[str, Any] = dict.fromkeys(MODELS_DEV_FIELDS)
    metadata = model.models_dev_metadata
    if metadata is None:
        return fields

    limit = metadata.raw.get("limit") or {}
    fields.update(
        {
            "models_dev_provider_id": model.models_dev_provider_id,
            "models_dev_model_id": model.models_dev_model_id,
            "models_dev_name": metadata.name,
            "models_dev_context_limit": limit.get("context"),
            "models_dev_output_limit": limit.get("output"),
            "models_dev_reasoning": metadata.raw.get("reasoning"),
            "models_dev_structured_output": metadata.raw.get("structured_output"),
            "models_dev_temperature": metadata.raw.get("temperature"),
            "models_dev_tool_call": metadata.raw.get("tool_call"),
        }
    )
    return fields


def _model_run_record(run: model_runs.ModelRun) -> dict[str, Any]:
    """Return one denormalized warehouse record for a model run."""
    release_date_source = "manual" if run.model.manual_release_date is not None else "models_dev"
    return {
        "model_run_key": run.model_run_key,
        "slug": run.slug,
        "model_key": run.model_key,
        "provider_model_id": run.provider_model_id,
        "lab": run.lab.name,
        "provider": run.provider.name,
        "release_date": run.release_date.isoformat(),
        "release_date_source": release_date_source,
        "active": run.model.active,
        "options": run.options,
        "artificial_analysis_id": run.artificial_analysis_id,
        **_models_dev_fields(run.model),
    }


def model_run_records() -> list[dict[str, Any]]:
    """Return one record per registry model run, in registry order."""
    return [_model_run_record(run) for run in model_runs.MODEL_RUNS]


def model_runs_ndjson() -> str:
    """Return the model-run records as newline-delimited JSON, one record per line.

    The whole document is built in memory before anything is written, so a
    failure while resolving a record never produces a partial export.
    """
    lines = [json.dumps(record, sort_keys=False) for record in model_run_records()]
    return "".join(f"{line}\n" for line in lines)


def write_model_runs_ndjson(output_path: Path) -> Path:
    """Write the model-run records as NDJSON to `output_path` and return that path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(model_runs_ndjson())
    return output_path


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and write the export to stdout or to `--output`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the NDJSON to this path instead of stdout.",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        sys.stdout.write(model_runs_ndjson())
        return
    written_path = write_model_runs_ndjson(args.output)
    print(f"Wrote {len(model_run_records())} model runs to {written_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
