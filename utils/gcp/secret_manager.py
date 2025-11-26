"""Helpers for reading Google Secret Manager secrets."""

from __future__ import annotations

import os

from google.cloud import secretmanager

from ..helpers.constants import GOOGLE_CLOUD_PROJECT_ENV_VAR


def get_project_id() -> str:
    """Return the configured Google Cloud project ID.

    Raises:
        RuntimeError: If the expected environment variable is not set.
    """
    project_id = os.environ.get(GOOGLE_CLOUD_PROJECT_ENV_VAR)
    if not project_id:
        raise RuntimeError(
            f"{GOOGLE_CLOUD_PROJECT_ENV_VAR} environment variable must be set to read secrets."
        )
    return project_id


def _build_resource_name(secret_name: str, version_id: str) -> str:
    project_id = get_project_id()
    return f"projects/{project_id}/secrets/{secret_name}/versions/{version_id}"


def get_secret(secret_name: str, version_id: str = "latest") -> str:
    """Retrieve the UTF-8 decoded payload for a secret version."""
    client = secretmanager.SecretManagerServiceClient()
    name = _build_resource_name(secret_name, version_id)
    response = client.access_secret_version(request={"name": name})
    payload_bytes = response.payload.data
    if not isinstance(payload_bytes, (bytes, bytearray)):
        raise TypeError("Secret payload data must be bytes.")
    return bytes(payload_bytes).decode("utf-8")
