"""Helpers for LLM registry identifiers."""

import re

FILENAME_SAFE_BYTE_PATTERN = re.compile(rb"[a-z0-9-]")


def validate_registry_key(value: str, *, field_name: str) -> None:
    """Reject invalid registry keys."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def filename_safe_name(value: str, *, field_name: str) -> str:
    """Return an injective filename-safe encoding of a registry key.

    Example: ``opus/4.3`` becomes ``k-opus~2f4~2e3``.
    """
    validate_registry_key(value, field_name=field_name)
    encoded_parts = []
    for byte in value.encode("utf-8"):
        byte_value = bytes([byte])
        if FILENAME_SAFE_BYTE_PATTERN.fullmatch(byte_value):
            encoded_parts.append(chr(byte))
        else:
            encoded_parts.append(f"~{byte:02x}")
    return "k-" + "".join(encoded_parts)
