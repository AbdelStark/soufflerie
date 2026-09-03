"""Strict, framework-independent HTTP service contracts."""

from __future__ import annotations

import base64
import binascii
import builtins
import hmac
from datetime import UTC, datetime
from typing import Annotated, Literal, Self, TypeAlias
from uuid import UUID

from pydantic import Field, StringConstraints, field_validator, model_validator

from soufflerie.config import ServiceConfig
from soufflerie.observability import CorrelationId
from soufflerie.schemas import ContentId, Sha256, StrictFrozenModel, VersionedModel, sha256_bytes

MAX_REQUEST_BODY_BYTES = 16 * 1024
MAX_RESPONSE_BODY_BYTES = 4 * 1024 * 1024
MAX_BASE64_CHARACTERS = ((MAX_RESPONSE_BODY_BYTES + 2) // 3) * 4

ValidationStatus: TypeAlias = Literal["green", "red"]
GateStatus: TypeAlias = Literal["green", "red"]
JobState: TypeAlias = Literal["queued", "running", "succeeded", "failed", "expired"]
SolveEventName: TypeAlias = Literal["queued", "running", "progress", "completed", "failed"]
ReadinessStatus: TypeAlias = Literal["ready", "not_ready"]
Uuid7 = Annotated[
    str,
    StringConstraints(
        min_length=36,
        max_length=36,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    ),
]
BoundedMessage = Annotated[str, StringConstraints(min_length=1, max_length=256)]
SafeDisplayToken = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$"),
]
FiniteNonnegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]

PublicErrorCode: TypeAlias = Literal[
    "SOUFFLERIE_ERROR",
    "CONFIG_INVALID",
    "CASE_OUT_OF_DOMAIN",
    "SOLVER_UNSTABLE",
    "SOLVER_NOT_CONVERGED",
    "ARTIFACT_INTEGRITY",
    "SCHEMA_UNSUPPORTED",
    "DEPENDENCY_UNAVAILABLE",
    "DEVICE_UNAVAILABLE",
    "CAPACITY_EXHAUSTED",
    "REMOTE_EXECUTION",
    "VALIDATION_RED",
    "INTERNAL_INVARIANT",
    "REQUEST_INVALID",
    "REQUEST_TOO_LARGE",
    "UNSUPPORTED_MEDIA_TYPE",
    "JOB_NOT_FOUND",
    "METHOD_NOT_ALLOWED",
    "NOT_FOUND",
    "INTERNAL_ERROR",
]


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_uuid7(value: str, *, label: str) -> str:
    identifier = UUID(value)
    if identifier.version != 7:
        raise ValueError(f"{label} must be UUIDv7")
    return str(identifier)


def _validate_state_payload(
    *,
    state: JobState,
    progress: float,
    result: SolveResultResponse | None,
    error: PublicError | None,
    label: str,
) -> None:
    if state == "queued":
        if progress != 0.0:
            raise ValueError(f"queued {label} progress must be zero")
        if result is not None or error is not None:
            raise ValueError(f"non-terminal {label} cannot contain result or error")
    elif state == "running":
        if progress >= 1.0:
            raise ValueError(f"running {label} progress must be less than one")
        if result is not None or error is not None:
            raise ValueError(f"non-terminal {label} cannot contain result or error")
    elif state == "succeeded":
        if progress != 1.0 or result is None or error is not None:
            raise ValueError(
                f"succeeded terminal {label} requires progress one and result without error"
            )
    elif state == "failed":
        if result is not None or error is None:
            raise ValueError(f"failed {label} requires error without result")
    elif result is not None or error is not None:
        raise ValueError(f"expired {label} cannot contain result or error")


class ShapeRequest(StrictFrozenModel):
    """One public ellipse design in the frozen supported domain."""

    aspect_ratio: float = Field(ge=0.5, le=1.0, allow_inf_nan=False)
    rotation_deg: float = Field(ge=0.0, le=30.0, allow_inf_nan=False)
    scale: float = Field(ge=0.75, le=1.25, allow_inf_nan=False)


class PredictionRequest(VersionedModel):
    """Closed request shared by synchronous prediction and reference solve."""

    shape: ShapeRequest
    reynolds: float = Field(ge=40.0, le=300.0, allow_inf_nan=False)


class EncodedArtifact(StrictFrozenModel):
    """Self-verifying bounded artifact embedded in a JSON response."""

    media_type: Literal["image/png", "application/x-npz"]
    encoding: Literal["base64"]
    data: Annotated[str, StringConstraints(min_length=4, max_length=MAX_BASE64_CHARACTERS)]
    sha256: Sha256
    bytes: int = Field(gt=0, le=MAX_RESPONSE_BODY_BYTES)

    @model_validator(mode="after")
    def _payload_matches_metadata(self) -> Self:
        try:
            decoded = base64.b64decode(self.data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("data must be canonical base64") from error
        if base64.b64encode(decoded).decode("ascii") != self.data:
            raise ValueError("data must use canonical padded base64")
        if len(decoded) != self.bytes:
            raise ValueError("bytes must equal the decoded payload length")
        if not hmac.compare_digest(sha256_bytes(decoded), self.sha256):
            raise ValueError("sha256 must match the decoded payload")
        return self

    @property
    def decoded_bytes(self) -> builtins.bytes:
        return base64.b64decode(self.data, validate=True)


class ConsistencyFlags(StrictFrozenModel):
    """Per-case physics checks whose labels are derived from frozen gates."""

    head_field_gap_pct: FiniteNonnegative
    head_field_gap: GateStatus
    divergence_ratio_to_solver_baseline: FiniteNonnegative
    divergence: GateStatus
    obstacle_velocity_ratio: FiniteNonnegative
    obstacle_compliance: GateStatus
    ood: Literal[False]

    @model_validator(mode="after")
    def _statuses_match_values(self) -> Self:
        expected = (
            (
                "head_field_gap",
                self.head_field_gap,
                "green" if self.head_field_gap_pct <= 10.0 else "red",
            ),
            (
                "divergence",
                self.divergence,
                "green" if self.divergence_ratio_to_solver_baseline < 3.0 else "red",
            ),
            (
                "obstacle_compliance",
                self.obstacle_compliance,
                "green" if self.obstacle_velocity_ratio < 0.01 else "red",
            ),
        )
        for name, actual, required in expected:
            if actual != required:
                raise ValueError(f"{name} status must be {required} for its measured value")
        return self


class PredictionResponse(VersionedModel):
    """Identity-rich, bounded synchronous prediction response."""

    correlation_id: CorrelationId
    case_id: ContentId
    fields_png: EncodedArtifact
    fields_npz: EncodedArtifact
    cd_head: float = Field(allow_inf_nan=False)
    cd_field: float = Field(allow_inf_nan=False)
    consistency: ConsistencyFlags
    validation_status: ValidationStatus
    model_id: ContentId
    dataset_id: ContentId
    report_id: ContentId
    inference_ms: FiniteNonnegative
    request_ms: FiniteNonnegative

    @field_validator("correlation_id")
    @classmethod
    def _correlation_is_uuid7(cls, value: str) -> str:
        return _canonical_uuid7(value, label="correlation_id")

    @model_validator(mode="after")
    def _response_is_coherent(self) -> Self:
        if self.fields_png.media_type != "image/png":
            raise ValueError("fields_png must use image/png")
        if self.fields_npz.media_type != "application/x-npz":
            raise ValueError("fields_npz must use application/x-npz")
        if self.request_ms < self.inference_ms:
            raise ValueError("request_ms must include inference_ms")
        return self


class PublicError(StrictFrozenModel):
    """Stable sanitized error detail returned at the HTTP boundary."""

    code: PublicErrorCode
    message: BoundedMessage
    retryable: bool
    correlation_id: CorrelationId

    @field_validator("correlation_id")
    @classmethod
    def _correlation_is_uuid7(cls, value: str) -> str:
        return _canonical_uuid7(value, label="correlation_id")


class PublicErrorResponse(VersionedModel):
    """Envelope shared by every non-success HTTP response."""

    error: PublicError


class SolveAccepted(VersionedModel):
    """Accepted reference-solve job and its two retrieval paths."""

    job_id: Uuid7
    case_id: ContentId
    state: Literal["queued"]
    status_url: Annotated[str, StringConstraints(min_length=43, max_length=43)]
    events_url: Annotated[str, StringConstraints(min_length=50, max_length=50)]
    expires_at: datetime

    @field_validator("job_id")
    @classmethod
    def _job_is_uuid7(cls, value: str) -> str:
        return _canonical_uuid7(value, label="job_id")

    @field_validator("expires_at")
    @classmethod
    def _expiry_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="expires_at")

    @model_validator(mode="after")
    def _urls_match_job(self) -> Self:
        if self.status_url != f"/solve/{self.job_id}":
            raise ValueError("status_url must identify job_id")
        if self.events_url != f"/solve/{self.job_id}/events":
            raise ValueError("events_url must identify job_id")
        return self


class SolveComparison(StrictFrozenModel):
    """Prediction-to-reference errors without duplicating prediction fields."""

    model_id: ContentId
    dataset_id: ContentId
    report_id: ContentId
    cd_head: float = Field(allow_inf_nan=False)
    cd_field: float = Field(allow_inf_nan=False)
    cd_head_error_pct: FiniteNonnegative
    cd_field_error_pct: FiniteNonnegative
    velocity_rel_l2: FiniteNonnegative


class SolveResultResponse(VersionedModel):
    """Terminal reference result plus its immutable prediction comparison."""

    correlation_id: CorrelationId
    job_id: Uuid7
    case_id: ContentId
    reference_fields_png: EncodedArtifact
    reference_fields_npz: EncodedArtifact
    cd: float = Field(allow_inf_nan=False)
    cl_mean: float = Field(allow_inf_nan=False)
    strouhal: FiniteNonnegative | None
    comparison: SolveComparison
    solver_artifact_id: ContentId
    provenance_sha256: Sha256
    solver_ms: FiniteNonnegative
    request_ms: FiniteNonnegative

    @field_validator("correlation_id")
    @classmethod
    def _correlation_is_uuid7(cls, value: str) -> str:
        return _canonical_uuid7(value, label="correlation_id")

    @field_validator("job_id")
    @classmethod
    def _job_is_uuid7(cls, value: str) -> str:
        return _canonical_uuid7(value, label="job_id")

    @model_validator(mode="after")
    def _result_is_coherent(self) -> Self:
        if self.reference_fields_png.media_type != "image/png":
            raise ValueError("reference_fields_png must use image/png")
        if self.reference_fields_npz.media_type != "application/x-npz":
            raise ValueError("reference_fields_npz must use application/x-npz")
        if self.request_ms < self.solver_ms:
            raise ValueError("request_ms must include solver_ms")
        return self


class SolveStatus(VersionedModel):
    """Polling snapshot for one ephemeral solve job."""

    job_id: Uuid7
    case_id: ContentId
    state: JobState
    progress: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    result: SolveResultResponse | None
    error: PublicError | None
    sequence: int = Field(ge=1)

    @field_validator("job_id")
    @classmethod
    def _job_is_uuid7(cls, value: str) -> str:
        return _canonical_uuid7(value, label="job_id")

    @model_validator(mode="after")
    def _state_is_coherent(self) -> Self:
        _validate_state_payload(
            state=self.state,
            progress=self.progress,
            result=self.result,
            error=self.error,
            label="status",
        )
        if self.result is not None and (
            self.result.job_id != self.job_id or self.result.case_id != self.case_id
        ):
            raise ValueError("terminal result identities must match solve status")
        return self


class SolveEventData(StrictFrozenModel):
    """Typed state payload transported inside one SSE event."""

    case_id: ContentId
    state: JobState
    progress: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    result: SolveResultResponse | None
    error: PublicError | None

    @model_validator(mode="after")
    def _state_is_coherent(self) -> Self:
        _validate_state_payload(
            state=self.state,
            progress=self.progress,
            result=self.result,
            error=self.error,
            label="event data",
        )
        return self


class SolveEvent(VersionedModel):
    """Monotonic replayable solve state event."""

    sequence: int = Field(ge=1)
    job_id: Uuid7
    timestamp: datetime
    event: SolveEventName
    data: SolveEventData

    @field_validator("job_id")
    @classmethod
    def _job_is_uuid7(cls, value: str) -> str:
        return _canonical_uuid7(value, label="job_id")

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="timestamp")

    @model_validator(mode="after")
    def _event_matches_state(self) -> Self:
        expected: dict[SolveEventName, JobState] = {
            "queued": "queued",
            "running": "running",
            "progress": "running",
            "completed": "succeeded",
            "failed": "failed",
        }
        if self.data.state != expected[self.event]:
            raise ValueError("event does not match data state")
        if self.data.result is not None and self.data.result.job_id != self.job_id:
            raise ValueError("event result job_id must match event job_id")
        return self


class ReadinessProbe(StrictFrozenModel):
    """Private startup evidence used to derive an allowlisted health response."""

    model_id: ContentId
    model_dataset_id: ContentId
    report_id: ContentId
    report_model_id: ContentId
    report_dataset_id: ContentId
    validation_status: ValidationStatus
    device_class: SafeDisplayToken
    model_integrity_verified: bool
    report_integrity_verified: bool
    device_available: bool
    warmup_complete: bool
    checked_at: datetime
    last_successful_readiness_check: datetime | None

    @field_validator("checked_at")
    @classmethod
    def _checked_at_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="checked_at")

    @field_validator("last_successful_readiness_check")
    @classmethod
    def _last_success_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware_utc(value, label="last_successful_readiness_check")

    @model_validator(mode="after")
    def _last_success_precedes_probe(self) -> Self:
        if (
            self.last_successful_readiness_check is not None
            and self.last_successful_readiness_check > self.checked_at
        ):
            raise ValueError("last successful readiness check cannot follow checked_at")
        return self


class HealthResponse(VersionedModel):
    """Exact public liveness/readiness allowlist."""

    liveness: Literal["ok"]
    readiness: ReadinessStatus
    package_version: SafeDisplayToken
    model_id: ContentId
    dataset_id: ContentId
    report_id: ContentId
    validation_status: ValidationStatus
    device_class: SafeDisplayToken
    last_successful_readiness_check: datetime | None

    @field_validator("last_successful_readiness_check")
    @classmethod
    def _last_success_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware_utc(value, label="last_successful_readiness_check")


def assess_health(
    config: ServiceConfig,
    probe: ReadinessProbe,
    *,
    package_version: str,
) -> HealthResponse:
    """Derive readiness without exposing mismatch or infrastructure details."""

    identities_match = (
        probe.model_id == config.model_id
        and probe.model_dataset_id == config.dataset_id
        and probe.report_id == config.report_id
        and probe.report_model_id == config.model_id
        and probe.report_dataset_id == config.dataset_id
    )
    ready = (
        identities_match
        and probe.model_integrity_verified
        and probe.report_integrity_verified
        and probe.device_available
        and probe.warmup_complete
    )
    return HealthResponse(
        liveness="ok",
        readiness="ready" if ready else "not_ready",
        package_version=package_version,
        model_id=config.model_id,
        dataset_id=config.dataset_id,
        report_id=config.report_id,
        validation_status=probe.validation_status,
        device_class=probe.device_class,
        last_successful_readiness_check=(
            probe.checked_at if ready else probe.last_successful_readiness_check
        ),
    )


__all__ = [
    "MAX_REQUEST_BODY_BYTES",
    "MAX_RESPONSE_BODY_BYTES",
    "ConsistencyFlags",
    "EncodedArtifact",
    "GateStatus",
    "HealthResponse",
    "JobState",
    "PredictionRequest",
    "PredictionResponse",
    "PublicError",
    "PublicErrorCode",
    "PublicErrorResponse",
    "ReadinessProbe",
    "ReadinessStatus",
    "ShapeRequest",
    "SolveAccepted",
    "SolveComparison",
    "SolveEvent",
    "SolveEventData",
    "SolveEventName",
    "SolveResultResponse",
    "SolveStatus",
    "ValidationStatus",
    "assess_health",
]
