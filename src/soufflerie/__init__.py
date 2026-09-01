"""Public package root for Soufflerie."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("soufflerie")
except PackageNotFoundError:  # pragma: no cover - only for an unpackaged source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
