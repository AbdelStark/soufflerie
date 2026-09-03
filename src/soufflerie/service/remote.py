"""Provider-neutral reference-solve orchestration and comparison assembly."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from soufflerie.config import ServiceConfig
from soufflerie.errors import ArtifactIntegrityError, InternalInvariantError
from soufflerie.schemas import canonical_sha256
from soufflerie.service.contracts import (
    EncodedArtifact,
    PredictionRequest,
    PredictionResponse,
    SolveComparison,
    SolveResultResponse,
)
from soufflerie.validation.metrics import MetricObservation, evaluate_case_metrics

Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
ProgressCallback = Callable[[float], Awaitable[None]]
MonotonicClock = Callable[[], float]

_CONTENT_ID_PATTERN = re.compile(r"^[0-9a-f]{20}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _immutable_float32(value: object, *, name: str) -> Float32Array:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or value.dtype != np.dtype(np.float32)
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise ArtifactIntegrityError(
            f"remote comparison {name} must be finite contiguous float32 with two dimensions"
        )
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    result.flags.writeable = False
    return result


def _immutable_mask(value: object, *, name: str) -> BoolArray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or value.dtype != np.dtype(np.bool_)
        or not value.flags.c_contiguous
    ):
        raise ArtifactIntegrityError(
            f"remote comparison {name} must be contiguous bool with two dimensions"
        )
    result = np.array(value, dtype=np.bool_, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class PredictionForComparison:
    """One verified public prediction plus the raw velocity fields it encoded."""

    response: PredictionResponse
    u: Float32Array
    v: Float32Array

    def __post_init__(self) -> None:
        if not isinstance(self.response, PredictionResponse):
            raise ArtifactIntegrityError("comparison prediction must contain a typed response")
        u = _immutable_float32(self.u, name="prediction u")
        v = _immutable_float32(self.v, name="prediction v")
        if u.shape != v.shape:
            raise ArtifactIntegrityError("comparison prediction velocity shapes differ")
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)


@dataclass(frozen=True, slots=True)
class ReferenceProjection:
    """Bounded public projection derived from one verified immutable solver run."""

    fields_png: EncodedArtifact
    fields_npz: EncodedArtifact
    u: Float32Array
    v: Float32Array
    obstacle_mask: BoolArray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.fields_png, EncodedArtifact)
            or self.fields_png.media_type != "image/png"
        ):
            raise ArtifactIntegrityError("reference projection PNG has the wrong media type")
        if (
            not isinstance(self.fields_npz, EncodedArtifact)
            or self.fields_npz.media_type != "application/x-npz"
        ):
            raise ArtifactIntegrityError("reference projection NPZ has the wrong media type")
        u = _immutable_float32(self.u, name="reference u")
        v = _immutable_float32(self.v, name="reference v")
        mask = _immutable_mask(self.obstacle_mask, name="reference obstacle mask")
        if u.shape != v.shape or u.shape != mask.shape:
            raise ArtifactIntegrityError("reference projection field shapes differ")
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "obstacle_mask", mask)


@dataclass(frozen=True, slots=True)
class ReferenceSolve:
    """Verified remote result with all identities needed for public comparison."""

    public_request_sha256: str
    solver_case_id: str
    solver_artifact_sha256: str
    provenance_sha256: str
    projection: ReferenceProjection
    cd: float
    cl_mean: float
    strouhal: float | None
    inlet_velocity_lu: float
    solver_ms: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.public_request_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.public_request_sha256) is None
        ):
            raise ArtifactIntegrityError("reference result has an invalid public request digest")
        if (
            not isinstance(self.solver_case_id, str)
            or _CONTENT_ID_PATTERN.fullmatch(self.solver_case_id) is None
        ):
            raise ArtifactIntegrityError("reference result has an invalid solver case identity")
        if (
            not isinstance(self.solver_artifact_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.solver_artifact_sha256) is None
        ):
            raise ArtifactIntegrityError("reference result has an invalid artifact digest")
        if (
            not isinstance(self.provenance_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.provenance_sha256) is None
        ):
            raise ArtifactIntegrityError("reference result has an invalid provenance digest")
        if not isinstance(self.projection, ReferenceProjection):
            raise ArtifactIntegrityError("reference result must contain a verified projection")
        scalars = (self.cd, self.cl_mean, self.inlet_velocity_lu, self.solver_ms)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in scalars
        ):
            raise ArtifactIntegrityError("reference result contains a non-finite scalar")
        if self.solver_ms < 0.0:
            raise ArtifactIntegrityError("reference solver time must be nonnegative")
        if self.inlet_velocity_lu <= 0.0:
            raise ArtifactIntegrityError("reference inlet velocity must be positive")
        if self.strouhal is not None and (
            isinstance(self.strouhal, bool)
            or not isinstance(self.strouhal, (int, float))
            or not math.isfinite(self.strouhal)
            or self.strouhal < 0.0
        ):
            raise ArtifactIntegrityError("reference Strouhal value must be finite and nonnegative")

    @property
    def solver_artifact_id(self) -> str:
        return self.solver_artifact_sha256[:20]


class ComparisonPredictor(Protocol):
    """Prediction boundary implemented by the warm service runtime."""

    async def predict(
        self,
        request: PredictionRequest,
        *,
        correlation_id: str,
    ) -> PredictionForComparison: ...


class ReferenceSolveBackend(Protocol):
    """Infrastructure boundary that returns one verified remote solve."""

    async def solve(
        self,
        request: PredictionRequest,
        *,
        job_id: str,
        correlation_id: str,
    ) -> ReferenceSolve: ...


def _metric_value(observation: MetricObservation, *, label: str) -> float:
    if observation.status != "valid" or observation.value is None:
        raise ArtifactIntegrityError(f"remote comparison {label} is invalid")
    return observation.value


class RemoteSolveExecutor:
    """Join prediction and one remote artifact into a terminal solve response."""

    def __init__(
        self,
        *,
        config: ServiceConfig,
        predictor: ComparisonPredictor,
        backend: ReferenceSolveBackend,
        monotonic: MonotonicClock = time.perf_counter,
    ) -> None:
        if not config.solve_enabled:
            raise InternalInvariantError("remote solve executor requires enabled solve service")
        self._config = config
        self._predictor = predictor
        self._backend = backend
        self._monotonic = monotonic

    async def execute(
        self,
        request: PredictionRequest,
        *,
        job_id: str,
        case_id: str,
        correlation_id: str,
        progress: ProgressCallback,
    ) -> SolveResultResponse:
        started = self._clock()
        expected_request_sha256 = canonical_sha256(request)
        if case_id != expected_request_sha256[:20]:
            raise InternalInvariantError("admitted case identity does not match its request")

        await progress(0.05)
        prediction = await self._predictor.predict(request, correlation_id=correlation_id)
        self._validate_prediction(prediction, case_id=case_id, correlation_id=correlation_id)
        await progress(0.15)

        reference = await self._backend.solve(
            request,
            job_id=job_id,
            correlation_id=correlation_id,
        )
        if reference.public_request_sha256 != expected_request_sha256:
            raise ArtifactIntegrityError("remote result is bound to another public request")
        if prediction.u.shape != reference.projection.u.shape:
            raise ArtifactIntegrityError("prediction and reference field shapes differ")

        metrics = evaluate_case_metrics(
            case_id=case_id,
            prediction_u=prediction.u,
            prediction_v=prediction.v,
            solver_u=reference.projection.u,
            solver_v=reference.projection.v,
            fluid_mask=np.ascontiguousarray(~reference.projection.obstacle_mask),
            obstacle_mask=reference.projection.obstacle_mask,
            cd_head=prediction.response.cd_head,
            cd_field=prediction.response.cd_field,
            cd_solver=reference.cd,
            inlet_velocity_lu=reference.inlet_velocity_lu,
        )
        comparison = SolveComparison(
            model_id=prediction.response.model_id,
            dataset_id=prediction.response.dataset_id,
            report_id=prediction.response.report_id,
            cd_head=prediction.response.cd_head,
            cd_field=prediction.response.cd_field,
            cd_head_error_pct=_metric_value(metrics.cd_head_pct, label="head drag error"),
            cd_field_error_pct=_metric_value(metrics.cd_field_pct, label="field drag error"),
            velocity_rel_l2=_metric_value(metrics.velocity_rel_l2, label="velocity error"),
        )
        await progress(0.95)
        completed = self._clock()
        if completed < started:
            raise InternalInvariantError("remote solve monotonic clock regressed")
        elapsed_ms = (completed - started) * 1_000.0
        request_ms = max(elapsed_ms, reference.solver_ms)
        return SolveResultResponse(
            correlation_id=correlation_id,
            job_id=job_id,
            case_id=case_id,
            reference_fields_png=reference.projection.fields_png,
            reference_fields_npz=reference.projection.fields_npz,
            cd=reference.cd,
            cl_mean=reference.cl_mean,
            strouhal=reference.strouhal,
            comparison=comparison,
            solver_artifact_id=reference.solver_artifact_id,
            provenance_sha256=reference.provenance_sha256,
            solver_ms=reference.solver_ms,
            request_ms=request_ms,
        )

    def _validate_prediction(
        self,
        prediction: PredictionForComparison,
        *,
        case_id: str,
        correlation_id: str,
    ) -> None:
        response = prediction.response
        if response.correlation_id != correlation_id or response.case_id != case_id:
            raise InternalInvariantError("comparison prediction identities do not match the job")
        identities = (response.model_id, response.dataset_id, response.report_id)
        expected = (self._config.model_id, self._config.dataset_id, self._config.report_id)
        if identities != expected:
            raise ArtifactIntegrityError("comparison prediction artifact identities diverged")

    def _clock(self) -> float:
        value = self._monotonic()
        if not math.isfinite(value):
            raise InternalInvariantError("remote solve monotonic clock must be finite")
        return value


__all__ = [
    "ComparisonPredictor",
    "PredictionForComparison",
    "ReferenceProjection",
    "ReferenceSolve",
    "ReferenceSolveBackend",
    "RemoteSolveExecutor",
]
