"""Google Cloud Platform related utilities."""

from . import storage
from .secret_manager import get_secret

__all__ = [
    "get_secret",
    "storage",
]
