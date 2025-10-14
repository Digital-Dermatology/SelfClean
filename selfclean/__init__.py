"""
SelfClean.

A holistic self-supervised data cleaning strategy to detect off-topic samples, near duplicates and label errors.
"""

__author__ = "Fabian Groeger"


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "SelfClean":
        from .cleaner.selfclean import SelfClean

        return SelfClean
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
