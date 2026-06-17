"""Tests for the checked-in Artificial Analysis metadata snapshot."""

from pathlib import Path

import pytest

from scripts import refresh_models_dev_metadata
from utils.llm.metadata import artificial_analysis

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT_DIR = Path(__file__).resolve().parents[2]


def test_load_artificial_analysis_snapshot_exposes_model_metadata():
    """Load the snapshot and expose normalized model fields."""
    snapshot = artificial_analysis.load_artificial_analysis_snapshot()

    model = snapshot.get_model("2dad8957-4c16-4e74-bf2d-8b21514e0ae9")
    opus_adaptive = snapshot.get_model("e9a09db3-8fd6-41dd-ba2f-20e0a2bff7f2")
    opus_non_reasoning = snapshot.get_model("2fa8e143-77a8-4d05-bfa8-d3b54634c00f")

    assert snapshot.source == refresh_models_dev_metadata.ARTIFICIAL_ANALYSIS_URL
    assert snapshot.prompt_options == {"parallel_queries": 1, "prompt_length": 1000}
    assert model.id == "2dad8957-4c16-4e74-bf2d-8b21514e0ae9"
    assert model.name == "o3-mini"
    assert opus_adaptive.name == "Claude Opus 4.7 (Adaptive Reasoning, Max Effort)"
    assert opus_non_reasoning.name == "Claude Opus 4.7 (Non-reasoning, High Effort)"


def test_artificial_analysis_snapshot_rejects_unknown_model():
    """Raise clear lookup errors for unknown Artificial Analysis model IDs."""
    snapshot = artificial_analysis.load_artificial_analysis_snapshot()

    with pytest.raises(KeyError, match="Unknown Artificial Analysis model_id missing-model"):
        snapshot.get_model("missing-model")


def test_load_artificial_analysis_snapshot_supports_endpoint_dump_shape(tmp_path):
    """Load AA model names from a generated full endpoint snapshot."""
    snapshot_path = tmp_path / "artificial_analysis_snapshot.json"
    snapshot_path.write_text("""
{
  "data": [
    {
      "id": "opus-aa-id",
      "name": "Claude Opus 4.7 (Adaptive Reasoning, Max Effort)",
      "slug": "claude-opus-4-7"
    }
  ],
  "prompt_options": {
    "parallel_queries": 1,
    "prompt_length": "medium"
  },
  "source": "https://artificialanalysis.ai/api/v2/data/llms/models",
  "status": 200
}
""".strip())

    snapshot = artificial_analysis.load_artificial_analysis_snapshot(snapshot_path)

    assert snapshot.get_model("opus-aa-id").name == (
        "Claude Opus 4.7 (Adaptive Reasoning, Max Effort)"
    )


def test_normalize_artificial_analysis_api_response_keeps_only_runtime_fields():
    """Keep only AA fields needed for stable IDs, display names, and attribution."""
    api_response = {
        "status": 200,
        "prompt_options": {"prompt_length": "medium", "parallel_queries": 1},
        "data": [
            {
                "id": "z-model",
                "name": "Z Model",
                "slug": "z-model",
                "model_creator": {"slug": "z-lab", "name": "Z Lab", "id": "z"},
                "evaluations": {"score_b": 2, "score_a": 1},
                "pricing": {"output": 2.5, "input": 1.5},
                "median_output_tokens_per_second": 12.5,
                "median_time_to_first_token_seconds": 3.5,
                "median_time_to_first_answer_token": 3.5,
                "ignored": "drop me",
            },
            {
                "id": "a-model",
                "name": "A Model",
                "slug": "a-model",
                "model_creator": {"id": "a", "name": "A Lab", "slug": "a-lab"},
                "evaluations": {},
                "pricing": {},
            },
        ],
    }

    normalized = refresh_models_dev_metadata.normalize_artificial_analysis_api_response(
        api_response
    )

    assert normalized["source"] == refresh_models_dev_metadata.ARTIFICIAL_ANALYSIS_URL
    assert normalized["prompt_options"] == {"parallel_queries": 1, "prompt_length": "medium"}
    assert [model["id"] for model in normalized["data"]] == ["a-model", "z-model"]
    assert normalized["data"][1] == {"id": "z-model", "name": "Z Model"}


def test_checked_in_artificial_analysis_snapshot_is_minimal():
    """Do not redistribute the full AA endpoint response in package data."""
    snapshot = refresh_models_dev_metadata.json.loads(artificial_analysis.SNAPSHOT_PATH.read_text())

    assert set(snapshot) == {"data", "prompt_options", "source"}
    assert all(set(model) == {"id", "name"} for model in snapshot["data"])


def test_artificial_analysis_snapshot_refresh_uses_gcp_secret(monkeypatch):
    """Use the official GCP Secret Manager key for AA metadata refreshes."""
    monkeypatch.setattr(
        refresh_models_dev_metadata,
        "get_secret",
        lambda secret_name: "gcp-aa-key",
    )

    request_headers = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_urlopen(request, timeout):
        request_headers.update(request.headers)
        assert timeout == 30
        return FakeResponse()

    monkeypatch.setattr(refresh_models_dev_metadata.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(refresh_models_dev_metadata.json, "load", lambda response: {"data": []})

    assert refresh_models_dev_metadata.fetch_artificial_analysis_api_response() == {"data": []}
    assert request_headers["X-api-key"] == "gcp-aa-key"


def test_artificial_analysis_snapshot_refresh_requires_api_key(monkeypatch):
    """Require the GCP Secret Manager key when it is unavailable."""
    monkeypatch.setattr(
        refresh_models_dev_metadata,
        "get_secret",
        lambda secret_name: (_ for _ in ()).throw(RuntimeError("GCP unavailable")),
    )

    with pytest.raises(RuntimeError, match="API_KEY_ARTIFICIAL_ANALYSIS"):
        refresh_models_dev_metadata.fetch_artificial_analysis_api_response()


def test_write_snapshots_updates_models_dev_and_artificial_analysis(monkeypatch, tmp_path):
    """Write both LLM metadata snapshots from the shared refresh entrypoint."""
    models_dev_output = tmp_path / "models_dev_snapshot.json"
    artificial_analysis_output = tmp_path / "artificial_analysis_snapshot.json"
    monkeypatch.setattr(
        refresh_models_dev_metadata,
        "DEFAULT_MODELS_DEV_OUTPUT_PATH",
        models_dev_output,
    )
    monkeypatch.setattr(
        refresh_models_dev_metadata,
        "DEFAULT_ARTIFICIAL_ANALYSIS_OUTPUT_PATH",
        artificial_analysis_output,
    )
    monkeypatch.setattr(
        refresh_models_dev_metadata,
        "fetch_models_dev_api_response",
        lambda: {
            "openai": {
                "id": "openai",
                "name": "OpenAI",
                "models": {
                    "gpt-test": {
                        "id": "gpt-test",
                        "name": "GPT Test",
                    }
                },
            }
        },
    )
    monkeypatch.setattr(
        refresh_models_dev_metadata,
        "read_models_dev_references_from_model_registry",
        lambda: frozenset(
            {
                refresh_models_dev_metadata.ModelsDevReference(
                    provider_id="openai",
                    model_id="gpt-test",
                )
            }
        ),
    )
    monkeypatch.setattr(
        refresh_models_dev_metadata,
        "fetch_artificial_analysis_api_response",
        lambda: {
            "status": 200,
            "data": [
                {
                    "id": "aa-test",
                    "name": "AA Test",
                    "slug": "aa-test",
                    "model_creator": {"id": "creator", "name": "Creator"},
                }
            ],
        },
    )

    refresh_models_dev_metadata.write_snapshots()

    assert models_dev_output.exists()
    assert artificial_analysis_output.exists()
    assert "gpt-test" in models_dev_output.read_text()
    assert "aa-test" in artificial_analysis_output.read_text()


def test_artificial_analysis_snapshot_is_included_as_package_data():
    """Include the JSON snapshot when utils is installed as a package."""
    pyproject = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text())

    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert package_data["utils.llm.metadata"] == ["*.json"]
