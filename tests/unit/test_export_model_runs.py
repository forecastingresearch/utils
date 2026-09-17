"""Tests for the warehouse model-run NDJSON export."""

import json

from scripts import export_model_runs
from utils.llm import model_runs


def _record_by_key(model_run_key: str) -> dict:
    return next(
        record
        for record in export_model_runs.model_run_records()
        if record["model_run_key"] == model_run_key
    )


def test_ndjson_has_one_parseable_object_per_model_run() -> None:
    """Every registry run becomes exactly one JSON object line, in registry order."""
    lines = export_model_runs.model_runs_ndjson().splitlines()

    assert len(lines) == len(model_runs.MODEL_RUNS)
    parsed = [json.loads(line) for line in lines]
    assert [record["model_run_key"] for record in parsed] == [
        run.model_run_key for run in model_runs.MODEL_RUNS
    ]


def test_models_dev_fields_follow_the_snapshot() -> None:
    """Models.dev fields are copied from the referenced snapshot entry."""
    run = next(run for run in model_runs.MODEL_RUNS if run.model.models_dev_reference is not None)
    record = _record_by_key(run.model_run_key)
    raw = run.model.models_dev_metadata.raw

    assert record["models_dev_provider_id"] == run.model.models_dev_reference.provider_id
    assert record["models_dev_model_id"] == run.model.models_dev_reference.model_id
    assert record["models_dev_name"] == raw["name"]
    assert record["models_dev_context_limit"] == raw["limit"]["context"]
    assert record["models_dev_output_limit"] == raw["limit"]["output"]
    for flag in ("reasoning", "structured_output", "temperature", "tool_call"):
        assert record[f"models_dev_{flag}"] == raw[flag]


def test_models_dev_fields_are_null_without_reference() -> None:
    """Runs whose model has no Models.dev reference still carry every field, as null."""
    run = next(run for run in model_runs.MODEL_RUNS if run.model.models_dev_reference is None)
    record = _record_by_key(run.model_run_key)

    assert record["release_date_source"] == "manual"
    assert all(record[field] is None for field in export_model_runs.MODELS_DEV_FIELDS)


def test_main_streams_ndjson_to_stdout_by_default(capsys) -> None:
    """Without --output the NDJSON goes to stdout and nothing else is printed there."""
    export_model_runs.main([])

    captured = capsys.readouterr()
    assert captured.out == export_model_runs.model_runs_ndjson()


def test_main_writes_to_requested_path(tmp_path, capsys) -> None:
    """With --output the writer creates parent directories and keeps stdout clean."""
    output_path = tmp_path / "nested" / "llm-forecasters.ndjson"

    export_model_runs.main(["--output", str(output_path)])

    assert output_path.read_text() == export_model_runs.model_runs_ndjson()
    assert capsys.readouterr().out == ""
