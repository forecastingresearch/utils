"""Loader for the checked-in Models.dev metadata snapshot."""

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

SNAPSHOT_PATH = Path(__file__).with_name("models_dev_snapshot.json")


@dataclass(frozen=True, slots=True)
class ModelsDevModel:
    """Normalized metadata for one Models.dev model."""

    id: str
    name: str
    release_date: date | None
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelsDevProvider:
    """Normalized metadata for one Models.dev provider."""

    id: str
    name: str
    models: dict[str, ModelsDevModel]


@dataclass(frozen=True, slots=True)
class ModelsDevSnapshot:
    """Loaded Models.dev metadata indexed by provider and model ID."""

    providers: dict[str, ModelsDevProvider]

    def get_model(self, *, provider_id: str, model_id: str) -> ModelsDevModel:
        """Return a model by Models.dev provider and model IDs."""
        try:
            provider = self.providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"Unknown Models.dev provider_id {provider_id}") from exc
        try:
            return provider.models[model_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Models.dev model_id {model_id} for provider_id {provider_id}"
            ) from exc


def _parse_date(value: str | None) -> date | None:
    """Parse an ISO date value from the snapshot."""
    if value is None:
        return None
    if len(value) != len("YYYY-MM-DD"):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid Models.dev release_date: {value}") from exc


def _model_from_json(data: dict[str, Any]) -> ModelsDevModel:
    """Build normalized model metadata from snapshot JSON."""
    return ModelsDevModel(
        id=data["id"],
        name=data["name"],
        release_date=_parse_date(data.get("release_date")),
        raw=data,
    )


@lru_cache(maxsize=1)
def load_models_dev_snapshot(path: Path = SNAPSHOT_PATH) -> ModelsDevSnapshot:
    """Load the checked-in Models.dev metadata snapshot."""
    data = json.loads(path.read_text())
    providers = {}
    for provider_id, provider_data in data["providers"].items():
        models = {
            model_id: _model_from_json(model_data)
            for model_id, model_data in provider_data["models"].items()
        }
        providers[provider_id] = ModelsDevProvider(
            id=provider_data["id"],
            name=provider_data["name"],
            models=models,
        )
    return ModelsDevSnapshot(providers=providers)
