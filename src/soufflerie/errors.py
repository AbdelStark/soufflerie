"""Stable public exception taxonomy."""

from __future__ import annotations

from typing import ClassVar


class SoufflerieError(Exception):
    """Base class for typed, automation-safe failures."""

    code: ClassVar[str] = "SOUFFLERIE_ERROR"
    default_retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, retryable: bool | None = None) -> None:
        super().__init__(message)
        self.retryable = self.default_retryable if retryable is None else retryable


class ConfigurationError(SoufflerieError):
    code = "CONFIG_INVALID"


class DomainError(SoufflerieError):
    code = "CASE_OUT_OF_DOMAIN"


class NumericalStabilityError(SoufflerieError):
    code = "SOLVER_UNSTABLE"


class NonConvergenceError(SoufflerieError):
    code = "SOLVER_NOT_CONVERGED"


class ArtifactIntegrityError(SoufflerieError):
    code = "ARTIFACT_INTEGRITY"


class SchemaVersionError(SoufflerieError):
    code = "SCHEMA_UNSUPPORTED"

    def __init__(self, version: object, *, supported: tuple[int, ...] = (1,)) -> None:
        self.version = version
        self.supported = supported
        rendered = ", ".join(str(item) for item in supported)
        super().__init__(f"unsupported schema version {version!r}; supported versions: {rendered}")


class DependencyUnavailableError(SoufflerieError):
    code = "DEPENDENCY_UNAVAILABLE"


class DeviceUnavailableError(SoufflerieError):
    code = "DEVICE_UNAVAILABLE"


class CapacityError(SoufflerieError):
    code = "CAPACITY_EXHAUSTED"
    default_retryable = True


class JobNotFoundError(SoufflerieError):
    code = "JOB_NOT_FOUND"


class EventCursorError(SoufflerieError):
    code = "EVENT_CURSOR_INVALID"


class IdempotencyConflictError(SoufflerieError):
    code = "IDEMPOTENCY_CONFLICT"


class RemoteExecutionError(SoufflerieError):
    code = "REMOTE_EXECUTION"
    default_retryable = True


class ValidationGateError(SoufflerieError):
    code = "VALIDATION_RED"


class InternalInvariantError(SoufflerieError):
    code = "INTERNAL_INVARIANT"


__all__ = [
    "ArtifactIntegrityError",
    "CapacityError",
    "ConfigurationError",
    "DependencyUnavailableError",
    "DeviceUnavailableError",
    "DomainError",
    "EventCursorError",
    "IdempotencyConflictError",
    "InternalInvariantError",
    "JobNotFoundError",
    "NonConvergenceError",
    "NumericalStabilityError",
    "RemoteExecutionError",
    "SchemaVersionError",
    "SoufflerieError",
    "ValidationGateError",
]
