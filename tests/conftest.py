"""Pytest configuration for selective integration test runs."""

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[1] / ".env",
    override=False,
)


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
