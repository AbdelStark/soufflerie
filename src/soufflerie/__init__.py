"""Public package root for Soufflerie."""

from importlib.metadata import PackageNotFoundError, version

from soufflerie.errors import ArtifactIntegrityError, SchemaVersionError, SoufflerieError
from soufflerie.schemas import (
    ArrayDescriptor,
    ArtifactRef,
    CaseConfig,
    FlowFields,
    GridSpec,
    Provenance,
    ShapeParams,
    SolverDiagnostics,
    SolverResult,
    canonical_json,
    canonical_sha256,
)

try:
    __version__ = version("soufflerie")
except PackageNotFoundError:  # pragma: no cover - only for an unpackaged source tree
    __version__ = "0.1.0"

__all__ = [
    "ArrayDescriptor",
    "ArtifactIntegrityError",
    "ArtifactRef",
    "CaseConfig",
    "FlowFields",
    "GridSpec",
    "Provenance",
    "SchemaVersionError",
    "ShapeParams",
    "SolverDiagnostics",
    "SolverResult",
    "SoufflerieError",
    "__version__",
    "canonical_json",
    "canonical_sha256",
]
