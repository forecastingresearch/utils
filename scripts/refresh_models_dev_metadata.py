"""Refresh the checked-in LLM metadata snapshots."""

import ast
import difflib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.api_core import exceptions

from utils.gcp.secret_manager import get_secret
from utils.helpers.constants import ARTIFICIAL_ANALYSIS_API_KEY_SECRET_NAME

MODELS_DEV_URL = "https://models.dev/api.json"
ARTIFICIAL_ANALYSIS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
DEFAULT_MODELS_DEV_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "utils" / "llm" / "metadata" / "models_dev_snapshot.json"
)
DEFAULT_MODEL_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "utils" / "llm" / "model_registry.py"
)
DEFAULT_ARTIFICIAL_ANALYSIS_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "llm"
    / "metadata"
    / "artificial_analysis_snapshot.json"
)

MODEL_FIELDS = (
    "id",
    "limit",
    "name",
    "reasoning",
    "release_date",
    "structured_output",
    "temperature",
    "tool_call",
)


@dataclass(frozen=True, slots=True, order=True)
class ModelsDevReference:
    """A provider/model reference into Models.dev."""

    provider_id: str
    model_id: str


def _sorted_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a dictionary with keys sorted recursively."""
    return {key: _sort_json_value(value) for key, value in sorted(data.items())}


def _sort_json_value(value: Any) -> Any:
    """Return JSON-like data with dictionaries sorted recursively."""
    if isinstance(value, dict):
        return _sorted_dict(value)
    if isinstance(value, list):
        return [_sort_json_value(item) for item in value]
    return value


def _literal_string_keyword(call: ast.Call, keyword_name: str) -> str | None:
    """Return a string literal keyword argument from an AST call, if present."""
    for keyword in call.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                return value
    return None


def read_models_dev_references_from_model_registry(
    model_registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> frozenset[ModelsDevReference]:
    """Read Models.dev references from the model registry without importing it."""
    tree = ast.parse(model_registry_path.read_text(), filename=str(model_registry_path))
    references = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        else:
            continue
        if function_name != "ModelsDevReference":
            continue

        provider_id = _literal_string_keyword(node, "provider_id")
        model_id = _literal_string_keyword(node, "model_id")
        if provider_id is None or model_id is None:
            raise ValueError(
                "ModelsDevReference calls in model_registry.py must use literal "
                "provider_id and model_id keyword arguments."
            )
        references.add(ModelsDevReference(provider_id=provider_id, model_id=model_id))
    return frozenset(references)


def _format_model_suggestions(
    *,
    provider_id: str,
    provider_data: dict[str, Any],
    missing_model_id: str,
) -> str:
    """Format nearby Models.dev model suggestions for an incorrect reference."""
    models = provider_data.get("models", {})
    candidates_by_key = {
        model_id: f'{provider_id}/{model_id} name="{model_data.get("name", "")}"'
        for model_id, model_data in models.items()
    }
    search_space = list(candidates_by_key)
    search_space.extend(
        model_data.get("name", "") for model_data in models.values() if model_data.get("name")
    )
    close_values = difflib.get_close_matches(
        missing_model_id,
        search_space,
        n=5,
        cutoff=0.35,
    )
    suggestions = []
    for value in close_values:
        if value in candidates_by_key:
            suggestions.append(candidates_by_key[value])
            continue
        for model_id, model_data in models.items():
            if model_data.get("name") == value:
                suggestions.append(candidates_by_key[model_id])
                break

    # Preserve order while de-duplicating suggestions found by ID and display name.
    suggestions = list(dict.fromkeys(suggestions))
    if not suggestions:
        return f"No nearby model IDs found for provider {provider_id}."
    return "Possible matches:\n  " + "\n  ".join(suggestions)


def _raise_missing_models_dev_reference(
    *,
    api_response: dict[str, Any],
    reference: ModelsDevReference,
) -> None:
    """Raise a targeted error for a missing Models.dev reference."""
    provider_data = api_response.get(reference.provider_id)
    if provider_data is None:
        provider_suggestions = difflib.get_close_matches(
            reference.provider_id,
            list(api_response),
            n=5,
            cutoff=0.35,
        )
        suffix = (
            "Possible provider IDs:\n  " + "\n  ".join(provider_suggestions)
            if provider_suggestions
            else "No nearby provider IDs found."
        )
        raise ValueError(
            f"Missing Models.dev provider reference: {reference.provider_id}\n{suffix}"
        )

    suggestions = _format_model_suggestions(
        provider_id=reference.provider_id,
        provider_data=provider_data,
        missing_model_id=reference.model_id,
    )
    raise ValueError(
        f"Missing Models.dev reference: {reference.provider_id}/{reference.model_id}\n"
        f"{suggestions}"
    )


def normalize_models_dev_api_response(
    api_response: dict[str, Any],
    *,
    models_dev_references: frozenset[ModelsDevReference],
) -> dict[str, Any]:
    """Normalize a Models.dev API response into the checked-in snapshot shape."""
    providers = {}
    references_by_provider: dict[str, list[ModelsDevReference]] = {}
    for reference in sorted(models_dev_references):
        references_by_provider.setdefault(reference.provider_id, []).append(reference)

    for provider_id, references in sorted(references_by_provider.items()):
        provider_data = api_response.get(provider_id)
        if provider_data is None:
            _raise_missing_models_dev_reference(
                api_response=api_response,
                reference=references[0],
            )

        models = {}
        provider_models = provider_data.get("models", {})
        for reference in sorted(references):
            model_data = provider_models.get(reference.model_id)
            if model_data is None:
                _raise_missing_models_dev_reference(
                    api_response=api_response,
                    reference=reference,
                )

            normalized_model = {}
            for field in MODEL_FIELDS:
                if field in model_data:
                    value = model_data[field]
                    normalized_model[field] = (
                        _sorted_dict(value) if isinstance(value, dict) else value
                    )
            models[reference.model_id] = normalized_model
        providers[provider_id] = {
            "id": provider_data["id"],
            "name": provider_data["name"],
            "models": models,
        }
    return {
        "source": MODELS_DEV_URL,
        "providers": providers,
    }


def normalize_artificial_analysis_api_response(api_response: dict[str, Any]) -> dict[str, Any]:
    """Normalize Artificial Analysis response fields needed at runtime."""
    return {
        "source": ARTIFICIAL_ANALYSIS_URL,
        "prompt_options": _sorted_dict(api_response.get("prompt_options") or {}),
        "data": [
            {"id": model_data["id"], "name": model_data["name"]}
            for model_data in sorted(api_response.get("data", []), key=lambda item: item["id"])
        ],
    }


def fetch_models_dev_api_response() -> dict[str, Any]:
    """Fetch the current Models.dev API response."""
    request = urllib.request.Request(MODELS_DEV_URL, headers={"User-Agent": "fri-utils"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_artificial_analysis_api_response() -> dict[str, Any]:
    """Fetch the current Artificial Analysis LLM models API response."""
    try:
        api_key = get_secret(ARTIFICIAL_ANALYSIS_API_KEY_SECRET_NAME)
    except (RuntimeError, exceptions.NotFound):
        api_key = None
    if not api_key:
        raise RuntimeError(
            f"Configure {ARTIFICIAL_ANALYSIS_API_KEY_SECRET_NAME} in GCP Secret Manager "
            "to refresh the Artificial Analysis snapshot."
        )
    request = urllib.request.Request(
        ARTIFICIAL_ANALYSIS_URL,
        headers={
            "User-Agent": "fri-utils",
            "x-api-key": api_key,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def write_json_snapshot(snapshot: dict[str, Any], output_path: Path) -> None:
    """Write a deterministic JSON snapshot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")


def write_models_dev_snapshot(output_path: Path = DEFAULT_MODELS_DEV_OUTPUT_PATH) -> None:
    """Fetch, normalize, and write the Models.dev snapshot."""
    snapshot = normalize_models_dev_api_response(
        fetch_models_dev_api_response(),
        models_dev_references=read_models_dev_references_from_model_registry(),
    )
    write_json_snapshot(snapshot, output_path)


def write_artificial_analysis_snapshot(
    output_path: Path = DEFAULT_ARTIFICIAL_ANALYSIS_OUTPUT_PATH,
) -> None:
    """Fetch, normalize, and write the Artificial Analysis snapshot."""
    snapshot = normalize_artificial_analysis_api_response(fetch_artificial_analysis_api_response())
    write_json_snapshot(snapshot, output_path)


def write_snapshots() -> None:
    """Fetch, normalize, and write all checked-in LLM metadata snapshots."""
    models_dev_snapshot = normalize_models_dev_api_response(
        fetch_models_dev_api_response(),
        models_dev_references=read_models_dev_references_from_model_registry(),
    )
    artificial_analysis_snapshot = normalize_artificial_analysis_api_response(
        fetch_artificial_analysis_api_response()
    )
    write_json_snapshot(models_dev_snapshot, DEFAULT_MODELS_DEV_OUTPUT_PATH)
    write_json_snapshot(artificial_analysis_snapshot, DEFAULT_ARTIFICIAL_ANALYSIS_OUTPUT_PATH)


if __name__ == "__main__":
    write_snapshots()
