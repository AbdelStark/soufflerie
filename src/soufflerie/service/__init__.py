"""HTTP application and solve-job lifecycle adapters."""

from typing import TYPE_CHECKING, Any

from soufflerie.service.contracts import (
    MAX_REQUEST_BODY_BYTES,
    MAX_RESPONSE_BODY_BYTES,
    ConsistencyFlags,
    EncodedArtifact,
    GateStatus,
    HealthResponse,
    JobState,
    PredictionRequest,
    PredictionResponse,
    PublicError,
    PublicErrorCode,
    PublicErrorResponse,
    ReadinessProbe,
    ReadinessStatus,
    ShapeRequest,
    SolveAccepted,
    SolveComparison,
    SolveEvent,
    SolveEventData,
    SolveEventName,
    SolveResultResponse,
    SolveStatus,
    ValidationStatus,
    assess_health,
)
from soufflerie.service.jobs import (
    DEFAULT_HEARTBEAT_SECONDS,
    TERMINAL_RETENTION_SECONDS,
    ProgressCallback,
    SolveExecutor,
    SolveJobBackend,
    SolveJobManager,
    validate_job_transition,
)

if TYPE_CHECKING:
    from soufflerie.service.app import EventStreamResponse, RequestBoundaryMiddleware, create_app


def __getattr__(name: str) -> Any:
    if name in {"EventStreamResponse", "RequestBoundaryMiddleware", "create_app"}:
        from soufflerie.service import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_HEARTBEAT_SECONDS",
    "MAX_REQUEST_BODY_BYTES",
    "MAX_RESPONSE_BODY_BYTES",
    "TERMINAL_RETENTION_SECONDS",
    "ConsistencyFlags",
    "EncodedArtifact",
    "EventStreamResponse",
    "GateStatus",
    "HealthResponse",
    "JobState",
    "PredictionRequest",
    "PredictionResponse",
    "ProgressCallback",
    "PublicError",
    "PublicErrorCode",
    "PublicErrorResponse",
    "ReadinessProbe",
    "ReadinessStatus",
    "RequestBoundaryMiddleware",
    "ShapeRequest",
    "SolveAccepted",
    "SolveComparison",
    "SolveEvent",
    "SolveEventData",
    "SolveEventName",
    "SolveExecutor",
    "SolveJobBackend",
    "SolveJobManager",
    "SolveResultResponse",
    "SolveStatus",
    "ValidationStatus",
    "assess_health",
    "create_app",
    "validate_job_transition",
]
