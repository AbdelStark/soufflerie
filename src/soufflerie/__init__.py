"""Public package root for Soufflerie."""

from importlib.metadata import PackageNotFoundError, version

from soufflerie.config import (
    Range,
    RunSchedule,
    ServiceConfig,
    SweepConfig,
    TrainingConfig,
    ValidationConfig,
    config_digest,
    load_config,
    parse_config,
)
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
    "Range",
    "RunSchedule",
    "SchemaVersionError",
    "ServiceConfig",
    "ShapeParams",
    "SolverDiagnostics",
    "SolverResult",
    "SoufflerieError",
    "SweepConfig",
    "TrainingConfig",
    "ValidationConfig",
    "__version__",
    "canonical_json",
    "canonical_sha256",
    "config_digest",
    "load_config",
    "parse_config",
]
