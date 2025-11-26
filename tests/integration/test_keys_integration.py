"""Integration tests for secret manager helpers."""

import os
import uuid
from importlib import import_module

import pytest
from google.api_core import exceptions as gcloud_exceptions

from utils.helpers.constants import (
    ANTHROPIC_API_KEY_SECRET_NAME,
    GOOGLE_APPLICATION_CREDENTIALS_ENV_VAR,
    GOOGLE_CLOUD_PROJECT_ENV_VAR,
)

unique_secret_name = f"integration-test-secret-{uuid.uuid4()}"


@pytest.mark.integration
def test_get_secret_integration(monkeypatch):
    """Integration check using real Secret Manager access.

    Requires an API_KEY_ANTHROPIC secret to exist in the project.
    """
    project_id = os.environ.get(GOOGLE_CLOUD_PROJECT_ENV_VAR)
    assert project_id is not None
    credentials_path = os.environ.get(GOOGLE_APPLICATION_CREDENTIALS_ENV_VAR)
    assert credentials_path is not None

    secrets_module = import_module("utils.gcp.secret_manager")

    value = secrets_module.get_secret(ANTHROPIC_API_KEY_SECRET_NAME)
    assert value is not None and value != ""

    with pytest.raises(gcloud_exceptions.NotFound):
        secrets_module.get_secret(unique_secret_name)
