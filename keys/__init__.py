"""Secret access helpers."""

from .secrets import get_secret, get_secret_that_may_not_exist

__all__ = [
    "get_secret",
    "get_secret_that_may_not_exist",
]
