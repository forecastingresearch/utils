"""Unit tests for secret manager helpers."""

from importlib import import_module, reload
from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as gcloud_exceptions
from utils.helpers.constants import GOOGLE_CLOUD_PROJECT_ENV_VAR


def _reload_secrets(monkeypatch):
    """Reload the secrets module with the desired environment."""
    monkeypatch.setenv(GOOGLE_CLOUD_PROJECT_ENV_VAR, "test-project")
    secrets_module = import_module("utils.keys.secrets")

    reloaded_secrets = reload(secrets_module)
    return reloaded_secrets


@patch("utils.keys.secrets.secretmanager.SecretManagerServiceClient")
def test_get_secret_returns_decoded_payload(mock_client_class, monkeypatch):
    """It returns the decoded payload when Secret Manager responds with bytes."""
    secrets_module = _reload_secrets(monkeypatch)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.payload.data = b"super-secret"
    mock_client.access_secret_version.return_value = mock_response
    mock_client_class.return_value = mock_client

    result = secrets_module.get_secret("api-key")

    assert result == "super-secret"
    mock_client.access_secret_version.assert_called_once_with(
        request={"name": "projects/test-project/secrets/api-key/versions/latest"}
    )


@patch("utils.keys.secrets.secretmanager.SecretManagerServiceClient")
def test_get_secret_raises_type_error_for_non_bytes_payload(mock_client_class, monkeypatch):
    """It raises TypeError if Secret Manager returns a non-bytes payload."""
    secrets_module = _reload_secrets(monkeypatch)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.payload.data = "not-bytes"
    mock_client.access_secret_version.return_value = mock_response
    mock_client_class.return_value = mock_client

    with pytest.raises(TypeError):
        secrets_module.get_secret("api-key")


@patch("utils.keys.secrets.secretmanager.SecretManagerServiceClient")
def test_get_secret_that_may_not_exist_returns_none_when_missing(mock_client_class, monkeypatch):
    """It returns None when the secret version is missing."""
    secrets_module = _reload_secrets(monkeypatch)

    mock_client = MagicMock()
    mock_client.access_secret_version.side_effect = gcloud_exceptions.NotFound("missing")
    mock_client_class.return_value = mock_client

    result = secrets_module.get_secret_that_may_not_exist("missing-secret")

    assert result is None
