"""Public package root for Soufflerie."""

from importlib.metadata import PackageNotFoundError, version

from soufflerie.artifacts import (
    ArtifactEnvelope,
    LineageNode,
    ReaderLimits,
    safe_read_json,
    safe_read_npz,
    safe_read_parquet,
    safe_read_tensors,
    validate_release_provenance,
    verify_consumer_identities,
    verify_lineage,
)
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
from soufflerie.geometry import (
    GeometryDiagnostics,
    ellipse_sdf,
    normalized_sdf_input,
    obstacle_mask,
    reference_diameter_lu,
    validate_geometry,
)
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
from soufflerie.solver import (
    CompletedLatticeRun,
    DerivedLatticeConfig,
    LatticeConfig,
    LatticeState,
    SolverConvergenceFailure,
    SolverStabilityFailure,
    WarpKernelAdapter,
    derive_lattice,
    run_lifecycle,
)

try:
    __version__ = version("soufflerie")
except PackageNotFoundError:  # pragma: no cover - only for an unpackaged source tree
    __version__ = "0.1.0"

__all__ = [
    "ArrayDescriptor",
    "ArtifactEnvelope",
    "ArtifactIntegrityError",
    "ArtifactRef",
    "CaseConfig",
    "CompletedLatticeRun",
    "DerivedLatticeConfig",
    "FlowFields",
    "GeometryDiagnostics",
    "GridSpec",
    "LatticeConfig",
    "LatticeState",
    "LineageNode",
    "Provenance",
    "Range",
    "ReaderLimits",
    "RunSchedule",
    "SchemaVersionError",
    "ServiceConfig",
    "ShapeParams",
    "SolverConvergenceFailure",
    "SolverDiagnostics",
    "SolverResult",
    "SolverStabilityFailure",
    "SoufflerieError",
    "SweepConfig",
    "TrainingConfig",
    "ValidationConfig",
    "WarpKernelAdapter",
    "__version__",
    "canonical_json",
    "canonical_sha256",
    "config_digest",
    "derive_lattice",
    "ellipse_sdf",
    "load_config",
    "normalized_sdf_input",
    "obstacle_mask",
    "parse_config",
    "reference_diameter_lu",
    "run_lifecycle",
    "safe_read_json",
    "safe_read_npz",
    "safe_read_parquet",
    "safe_read_tensors",
    "validate_geometry",
    "validate_release_provenance",
    "verify_consumer_identities",
    "verify_lineage",
]
