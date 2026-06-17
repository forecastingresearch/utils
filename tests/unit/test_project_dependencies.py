"""Tests for package dependency ownership."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]

SHARED_RUNTIME_DEPENDENCIES = {
    "anthropic",
    "google-cloud-secret-manager",
    "google-cloud-storage",
    "google-genai",
    "openai",
    "python-dotenv",
    "together",
}


def _requirement_name(requirement: str) -> str:
    return (
        requirement.split("==", maxsplit=1)[0]
        .split(">=", maxsplit=1)[0]
        .split("<", maxsplit=1)[0]
        .strip()
    )


def test_shared_runtime_dependencies_are_declared_in_pyproject():
    """Keep shared LLM runtime dependencies owned by pyproject metadata."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependency_names = {
        _requirement_name(dependency) for dependency in pyproject["project"]["dependencies"]
    }

    assert dependency_names == SHARED_RUNTIME_DEPENDENCIES


def test_requirements_txt_delegates_to_pyproject_dev_extra():
    """Keep local requirements install behavior delegated to the dev extra."""
    requirements = (ROOT / "requirements.txt").read_text().splitlines()

    assert requirements == [".[dev]"]


def test_dev_extra_dependencies_are_pinned():
    """Keep dev extra dependencies pinned for reproducible local installs."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]

    assert all("==" in dependency for dependency in dev_dependencies)


def test_third_party_notices_cover_metadata_sources():
    """Preserve license and attribution notices for checked-in metadata snapshots."""
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert "Models.dev" in notices
    assert "https://github.com/anomalyco/models.dev" in notices
    assert "MIT License" in notices
    assert "Copyright (c) 2025 models.dev" in notices
    assert "Artificial Analysis" in notices
    assert "https://artificialanalysis.ai/" in notices
    assert "THIRD_PARTY_NOTICES.md" in pyproject["tool"]["setuptools"]["license-files"]
