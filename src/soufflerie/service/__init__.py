"""HTTP application and solve-job lifecycle adapters."""

from typing import TYPE_CHECKING, Any

from soufflerie.service.admission import (
    CLIENT_HMAC_KEY_BYTES,
    MAX_CLIENT_STATES,
    AdmissionController,
    AdmissionSettings,
    AdmissionSnapshot,
    ServiceReadiness,
    evaluate_service_readiness,
    load_admission_settings,
)
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
    "CLIENT_HMAC_KEY_BYTES",
    "DEFAULT_HEARTBEAT_SECONDS",
    "MAX_CLIENT_STATES",
    "MAX_REQUEST_BODY_BYTES",
    "MAX_RESPONSE_BODY_BYTES",
    "TERMINAL_RETENTION_SECONDS",
    "AdmissionController",
    "AdmissionSettings",
    "AdmissionSnapshot",
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
    "ServiceReadiness",
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
    "evaluate_service_readiness",
    "load_admission_settings",
    "validate_job_transition",
]
