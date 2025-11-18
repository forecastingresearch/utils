"""Pytest configuration for selective integration test runs."""

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[1] / ".env",
    override=False,
)


@pytest.fixture(scope="session", autouse=True)
def configure_llm_api_keys(request):
    """Auto-configure LLM API keys from GCP for integration tests.

    Uses session scope to cache API keys across all integration tests,
    avoiding redundant GCP Secret Manager calls.
    """
    # Only configure if we're running integration tests
    # Check if any test has the integration marker (check config instead of individual test)
    config = request.config
    if config.getoption("--integration"):
        try:
            from utils.llm.model_registry import (
                configure_api_keys,  # type: ignore[import]
            )

            configure_api_keys(from_gcp=True)
        except Exception:
            # If GCP is not configured, skip this fixture
            pass


def pytest_addoption(parser):
    """Register the custom --integration flag."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run tests marked as integration",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --integration is provided."""
    if config.getoption("--integration"):
        return

    skip_integration = pytest.mark.skip(reason="use --integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
