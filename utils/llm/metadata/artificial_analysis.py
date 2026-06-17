"""Loader for the checked-in Artificial Analysis metadata snapshot."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

SNAPSHOT_PATH = Path(__file__).with_name("artificial_analysis_snapshot.json")


@dataclass(frozen=True, slots=True)
class ArtificialAnalysisModel:
    """Normalized metadata for one Artificial Analysis LLM model entry."""

    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ArtificialAnalysisSnapshot:
    """Loaded Artificial Analysis metadata indexed by stable model ID."""

    models: dict[str, ArtificialAnalysisModel]
    source: str
    prompt_options: dict[str, Any]

    def get_model(self, model_id: str) -> ArtificialAnalysisModel:
        """Return a model by Artificial Analysis stable model ID."""
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown Artificial Analysis model_id {model_id}") from exc


def _model_from_json(data: dict[str, Any]) -> ArtificialAnalysisModel:
    """Build normalized model metadata from snapshot JSON."""
    return ArtificialAnalysisModel(
        id=data["id"],
        name=data["name"],
    )


def _models_from_snapshot_json(data: dict[str, Any]) -> dict[str, ArtificialAnalysisModel]:
    """Build model metadata from current or legacy snapshot JSON."""
    if "data" in data:
        return {model_data["id"]: _model_from_json(model_data) for model_data in data["data"]}
    return {
        model_id: _model_from_json(model_data) for model_id, model_data in data["models"].items()
    }


@lru_cache(maxsize=1)
def load_artificial_analysis_snapshot(
    path: Path = SNAPSHOT_PATH,
) -> ArtificialAnalysisSnapshot:
    """Load the checked-in Artificial Analysis metadata snapshot."""
    data = json.loads(path.read_text())
    return ArtificialAnalysisSnapshot(
        models=_models_from_snapshot_json(data),
        source=data.get("source", ""),
        prompt_options=data.get("prompt_options") or {},
    )
