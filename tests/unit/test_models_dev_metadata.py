"""Tests for the checked-in Models.dev metadata snapshot."""

import json
from datetime import date
from pathlib import Path

import pytest

from scripts import refresh_models_dev_metadata
from utils.llm.metadata import models_dev

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT_DIR = Path(__file__).resolve().parents[2]


def test_load_models_dev_snapshot_exposes_provider_and_model_metadata():
    """Load the snapshot and expose normalized provider and model fields."""
    snapshot = models_dev.load_models_dev_snapshot()

    openai = snapshot.providers["openai"]
    assert openai.name == "OpenAI"

    gpt_4o = openai.models["gpt-4o-2024-05-13"]
    assert gpt_4o.id == "gpt-4o-2024-05-13"
    assert gpt_4o.name == "GPT-4o (2024-05-13)"
    assert gpt_4o.release_date == date(2024, 5, 13)
    assert {
        "id",
        "limit",
        "name",
        "reasoning",
        "release_date",
        "structured_output",
        "temperature",
        "tool_call",
    } <= set(gpt_4o.raw)


def test_models_dev_snapshot_can_lookup_model_by_provider_and_model_id():
    """Look up a normalized model by Models.dev provider and model IDs."""
    snapshot = models_dev.load_models_dev_snapshot()

    model = snapshot.get_model(provider_id="anthropic", model_id="claude-3-haiku-20240307")

    assert model.name == "Claude Haiku 3"
    assert model.release_date == date(2024, 3, 13)


def test_models_dev_snapshot_rejects_unknown_provider_or_model():
    """Raise clear lookup errors for unknown provider or model IDs."""
    snapshot = models_dev.load_models_dev_snapshot()

    try:
        snapshot.get_model(provider_id="missing", model_id="gpt-4o-2024-05-13")
    except KeyError as exc:
        assert "Unknown Models.dev provider_id missing" in str(exc)
    else:
        raise AssertionError("Expected missing provider lookup to fail")

    try:
        snapshot.get_model(provider_id="openai", model_id="missing")
    except KeyError as exc:
        assert "Unknown Models.dev model_id missing for provider_id openai" in str(exc)
    else:
        raise AssertionError("Expected missing model lookup to fail")


def test_models_dev_snapshot_preserves_raw_partial_release_dates(tmp_path):
    """Preserve month-only source dates while omitting typed date values."""
    snapshot_path = tmp_path / "models_dev_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "source": refresh_models_dev_metadata.MODELS_DEV_URL,
                "providers": {
                    "abacus": {
                        "id": "abacus",
                        "name": "Abacus",
                        "models": {
                            "kimi-k2.5": {
                                "id": "kimi-k2.5",
                                "name": "Kimi K2.5",
                                "release_date": "2026-01",
                            }
                        },
                    }
                },
            }
        )
    )
    snapshot = models_dev.load_models_dev_snapshot(snapshot_path)

    model = snapshot.get_model(provider_id="abacus", model_id="kimi-k2.5")

    assert model.release_date is None
    assert model.raw["release_date"] == "2026-01"


def test_models_dev_snapshot_rejects_invalid_full_release_dates(tmp_path):
    """Reject malformed full source dates instead of silently dropping them."""
    snapshot_path = tmp_path / "models_dev_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "source": refresh_models_dev_metadata.MODELS_DEV_URL,
                "providers": {
                    "scaleway": {
                        "id": "scaleway",
                        "name": "Scaleway",
                        "models": {
                            "qwen3-embedding-8b": {
                                "id": "qwen3-embedding-8b",
                                "name": "Qwen3 Embedding 8B",
                                "release_date": "2025-25-11",
                            }
                        },
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Invalid Models.dev release_date: 2025-25-11"):
        models_dev.load_models_dev_snapshot(snapshot_path)


def test_read_models_dev_references_from_model_registry_uses_ast(tmp_path):
    """Discover Models.dev references from declarations without importing the registry."""
    registry_path = tmp_path / "model_registry.py"
    registry_path.write_text("""
openai_model(
    model_key="gpt-test",
    models_dev_reference=ModelsDevReference(
        provider_id="openai",
        model_id="gpt-test",
    ),
)
together_model(
    model_key="manual-only",
    manual_release_date=date(2026, 1, 1),
)
anthropic_model(
    model_key="claude-test",
    models_dev_reference=ModelsDevReference(provider_id="anthropic", model_id="claude-test"),
)
""")

    references = refresh_models_dev_metadata.read_models_dev_references_from_model_registry(
        registry_path
    )

    assert references == frozenset(
        {
            refresh_models_dev_metadata.ModelsDevReference("anthropic", "claude-test"),
            refresh_models_dev_metadata.ModelsDevReference("openai", "gpt-test"),
        }
    )


def test_normalize_models_dev_api_response_keeps_expected_fields_sorted():
    """Normalize referenced model validation fields and sort providers and models."""
    api_response = {
        "openai": {
            "id": "openai",
            "name": "OpenAI",
            "models": {
                "z-model": {
                    "id": "z-model",
                    "name": "Z Model",
                    "release_date": "2026-01-02",
                    "last_updated": "2026-01-03",
                    "limit": {"output": 2, "context": 1},
                    "cost": {"input": 1.5},
                    "reasoning": True,
                    "temperature": False,
                    "tool_call": True,
                    "structured_output": True,
                    "ignored": "drop me",
                },
                "a-model": {
                    "id": "a-model",
                    "name": "A Model",
                    "release_date": None,
                    "last_updated": None,
                    "limit": {},
                    "cost": None,
                    "reasoning": False,
                    "temperature": True,
                    "tool_call": False,
                    "structured_output": False,
                },
                "unused-model": {
                    "id": "unused-model",
                    "name": "Unused Model",
                    "release_date": "2026-01-04",
                },
            },
        },
        "unused-provider": {
            "id": "unused-provider",
            "name": "Unused Provider",
            "models": {
                "unused": {
                    "id": "unused",
                    "name": "Unused",
                    "release_date": "2026-01-05",
                }
            },
        },
    }

    normalized = refresh_models_dev_metadata.normalize_models_dev_api_response(
        api_response,
        models_dev_references=frozenset(
            {
                refresh_models_dev_metadata.ModelsDevReference("openai", "z-model"),
                refresh_models_dev_metadata.ModelsDevReference("openai", "a-model"),
            }
        ),
    )

    assert list(normalized["providers"]) == ["openai"]
    assert list(normalized["providers"]["openai"]["models"]) == ["a-model", "z-model"]
    assert normalized["providers"]["openai"]["models"]["z-model"] == {
        "id": "z-model",
        "limit": {"context": 1, "output": 2},
        "name": "Z Model",
        "reasoning": True,
        "release_date": "2026-01-02",
        "structured_output": True,
        "temperature": False,
        "tool_call": True,
    }


def test_normalize_models_dev_api_response_errors_with_reference_suggestions():
    """Reject incorrect exact references and suggest nearby provider-local candidates."""
    api_response = {
        "openai": {
            "id": "openai",
            "name": "OpenAI",
            "models": {
                "gpt-5.6-2026-06-01": {
                    "id": "gpt-5.6-2026-06-01",
                    "name": "GPT-5.6",
                    "release_date": "2026-06-01",
                },
                "gpt-5.6-chat": {
                    "id": "gpt-5.6-chat",
                    "name": "GPT-5.6 Chat",
                    "release_date": "2026-06-01",
                },
            },
        }
    }

    with pytest.raises(ValueError) as excinfo:
        refresh_models_dev_metadata.normalize_models_dev_api_response(
            api_response,
            models_dev_references=frozenset(
                {refresh_models_dev_metadata.ModelsDevReference("openai", "gpt-5.6")}
            ),
        )

    message = str(excinfo.value)
    assert "Missing Models.dev reference: openai/gpt-5.6" in message
    assert 'openai/gpt-5.6-2026-06-01 name="GPT-5.6"' in message
    assert 'openai/gpt-5.6-chat name="GPT-5.6 Chat"' in message


def test_checked_in_models_dev_snapshot_contains_only_registry_references():
    """Keep the Models.dev snapshot scoped to the registry references that use it."""
    references = refresh_models_dev_metadata.read_models_dev_references_from_model_registry()
    snapshot = json.loads(models_dev.SNAPSHOT_PATH.read_text())
    snapshot_references = frozenset(
        refresh_models_dev_metadata.ModelsDevReference(provider_id, model_id)
        for provider_id, provider_data in snapshot["providers"].items()
        for model_id in provider_data["models"]
    )

    assert snapshot_references == references
    assert all(
        set(model_data)
        <= {
            "id",
            "limit",
            "name",
            "reasoning",
            "release_date",
            "structured_output",
            "temperature",
            "tool_call",
        }
        for provider_data in snapshot["providers"].values()
        for model_data in provider_data["models"].values()
    )


def test_models_dev_snapshot_is_included_as_package_data():
    """Include the JSON snapshot when utils is installed as a package."""
    pyproject = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text())

    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert package_data["utils.llm.metadata"] == ["*.json"]
