from __future__ import annotations


class MissingOptionalDependency(ImportError):
    """Raised when an optional dependency group is required but not installed."""
