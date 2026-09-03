"""Mounted-volume adapter for cancellable public reference solves."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, Literal, Protocol

from infra.policy import CheckoutState
from infra.remote_execution import (
    DeviceClass,
    RemoteSolveRequest,
    encode_remote_model,
)
from soufflerie.datagen.run_artifact import RunArtifact, RunArtifactStore
from soufflerie.errors import (
    ArtifactIntegrityError,
    ConfigurationError,
    RemoteExecutionError,
    SoufflerieError,
)
from soufflerie.schemas import ArtifactRef, CaseConfig, ShapeParams, canonical_sha256
from soufflerie.service.contracts import PredictionRequest
from soufflerie.service.remote import ReferenceProjection, ReferenceSolve

PUBLIC_SOLVE_NX: Literal[512] = 512
PUBLIC_SOLVE_NY: Literal[640] = 640
PUBLIC_SOLVE_STEPS: Literal[20000] = 20_000
PUBLIC_SOLVE_WARMUP_STEPS: Literal[10000] = 10_000
PUBLIC_SOLVE_INLET_VELOCITY_LU = 0.05
PUBLIC_SOLVE_SEED: Literal[20260901] = 20_260_901

RemoteInvoker = Callable[[bytes, str], Awaitable[ArtifactRef]]
ReloadVolume = Callable[[], None]
MonotonicClock = Callable[[], float]


class ReferenceRunProjector(Protocol):
    """Turn one already-verified run into bounded public field artifacts."""

    def project(self, run: RunArtifact) -> ReferenceProjection: ...


def public_solve_case(request: PredictionRequest) -> CaseConfig:
    """Bind a public physical request to the frozen v0.1 numerical policy."""

    if not isinstance(request, PredictionRequest):
        raise TypeError("public solve input must be a PredictionRequest")
    return CaseConfig(
        shape=ShapeParams.model_validate(request.shape.model_dump(mode="python")),
        reynolds=request.reynolds,
        nx=PUBLIC_SOLVE_NX,
        ny=PUBLIC_SOLVE_NY,
        steps=PUBLIC_SOLVE_STEPS,
        warmup_steps=PUBLIC_SOLVE_WARMUP_STEPS,
        inlet_velocity_lu=PUBLIC_SOLVE_INLET_VELOCITY_LU,
        seed=PUBLIC_SOLVE_SEED,
    )


def public_design_id(request: PredictionRequest) -> str:
    """Return the physical design identity shared with standalone solves."""

    if not isinstance(request, PredictionRequest):
        raise TypeError("public solve input must be a PredictionRequest")
    return canonical_sha256(
        {
            "schema_version": 1,
            "design_kind": "standalone-solve-v1",
            "shape": request.shape.model_dump(mode="json"),
            "reynolds": request.reynolds,
        }
    )[:20]


def build_public_remote_request(
    request: PredictionRequest,
    *,
    job_id: str,
    checkout: CheckoutState,
    device_class: DeviceClass,
) -> RemoteSolveRequest:
    """Create the one canonical remote request whose attempt token is the job ID."""

    if checkout.source_dirty:
        raise ConfigurationError("public remote solve requires a clean source revision")
    try:
        parsed_job_id = uuid.UUID(job_id)
    except (AttributeError, ValueError) as error:
        raise ConfigurationError("public remote solve job_id must be a canonical UUIDv7") from error
    if parsed_job_id.version != 7 or str(parsed_job_id) != job_id:
        raise ConfigurationError("public remote solve job_id must be a canonical UUIDv7")
    case = public_solve_case(request)
    design_id = public_design_id(request)
    operation_digest = canonical_sha256(
        {
            "schema_version": 1,
            "operation_kind": "public-service-solve-v1",
            "job_id": job_id,
            "public_request_sha256": canonical_sha256(request),
            "solver_case_sha256": case.sha256,
            "design_id": design_id,
            "source_revision": checkout.source_revision,
            "lock_sha256": checkout.lock_sha256,
            "requested_device_class": device_class,
        }
    )
    return RemoteSolveRequest.create(
        operation_kind="single",
        sweep_digest=operation_digest,
        design_id=design_id,
        split="test",
        case=case,
        requested_device_class=device_class,
        source_revision=checkout.source_revision,
        lock_sha256=checkout.lock_sha256,
        attempt_id=f"service-{job_id}",
    )


async def invoke_modal_solve(
    remote_function: Any,
    request_bytes: bytes,
    correlation_id: str,
) -> ArtifactRef:
    """Spawn one Modal call and terminate its worker if the service deadline cancels it."""

    try:
        function_call = await remote_function.spawn.aio(request_bytes, correlation_id)
    except Exception as error:
        raise RemoteExecutionError("remote solve submission failed") from error
    try:
        result = await function_call.get.aio()
    except asyncio.CancelledError:
        with suppress(Exception):
            await asyncio.shield(function_call.cancel.aio(terminate_containers=True))
        raise
    except SoufflerieError:
        raise
    except Exception as error:
        raise RemoteExecutionError("remote solve invocation failed") from error
    try:
        return ArtifactRef.model_validate(result)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            "remote solve returned an invalid artifact reference"
        ) from error


class MountedVolumeRemoteSolveBackend:
    """Build, invoke, reload, verify, and project one public remote solve."""

    def __init__(
        self,
        *,
        checkout: CheckoutState,
        device_class: DeviceClass,
        invoke: RemoteInvoker,
        reload_volume: ReloadVolume,
        run_store: RunArtifactStore,
        projector: ReferenceRunProjector,
        monotonic: MonotonicClock = time.perf_counter,
    ) -> None:
        if checkout.source_dirty:
            raise ConfigurationError("public remote solve requires a clean source revision")
        self._checkout = checkout
        self._device_class = device_class
        self._invoke = invoke
        self._reload_volume = reload_volume
        self._run_store = run_store
        self._projector = projector
        self._monotonic = monotonic

    async def solve(
        self,
        request: PredictionRequest,
        *,
        job_id: str,
        correlation_id: str,
    ) -> ReferenceSolve:
        remote_request = build_public_remote_request(
            request,
            job_id=job_id,
            checkout=self._checkout,
            device_class=self._device_class,
        )
        started = self._clock()
        try:
            reference = await self._invoke(encode_remote_model(remote_request), correlation_id)
            await asyncio.to_thread(self._reload_volume)
            run = await asyncio.to_thread(self._run_store.open_run, reference)
            self._verify_run(run, remote_request=remote_request, reference=reference)
            projection = await asyncio.to_thread(self._projector.project, run)
        except asyncio.CancelledError:
            raise
        except SoufflerieError:
            raise
        except Exception as error:
            raise RemoteExecutionError("remote solve result assembly failed") from error
        elapsed = self._clock() - started
        if elapsed < 0.0:
            raise RemoteExecutionError("remote solve monotonic clock regressed", retryable=False)
        provenance = run.metadata.provenance
        solver_ms = (provenance.completed_at - provenance.started_at).total_seconds() * 1_000.0
        return ReferenceSolve(
            public_request_sha256=canonical_sha256(request),
            solver_case_id=run.metadata.case_id,
            solver_artifact_sha256=run.reference.sha256,
            provenance_sha256=canonical_sha256(provenance),
            projection=projection,
            cd=run.metadata.cd,
            cl_mean=run.metadata.cl_mean,
            strouhal=run.metadata.strouhal,
            inlet_velocity_lu=remote_request.case.inlet_velocity_lu,
            solver_ms=solver_ms,
        )

    def _verify_run(
        self,
        run: RunArtifact,
        *,
        remote_request: RemoteSolveRequest,
        reference: ArtifactRef,
    ) -> None:
        metadata = run.metadata
        provenance = metadata.provenance
        if run.reference != reference or reference.artifact_type != "run":
            raise ArtifactIntegrityError("remote run reference changed during verification")
        if metadata.case != remote_request.case or metadata.case_id != remote_request.case.case_id:
            raise ArtifactIntegrityError("remote run case identity does not match the request")
        if metadata.design_id != remote_request.design_id or metadata.split != "test":
            raise ArtifactIntegrityError("remote run design identity does not match the request")
        if (
            provenance.source_dirty
            or provenance.source_revision != self._checkout.source_revision
            or provenance.lock_sha256 != self._checkout.lock_sha256
            or provenance.device_class != self._device_class
            or provenance.config_sha256 != remote_request.case.sha256
            or provenance.seeds != (PUBLIC_SOLVE_SEED,)
        ):
            raise ArtifactIntegrityError("remote run provenance does not match the service build")

    def _clock(self) -> float:
        value = self._monotonic()
        if not math.isfinite(value):
            raise RemoteExecutionError(
                "remote solve monotonic clock must be finite", retryable=False
            )
        return value


__all__ = [
    "PUBLIC_SOLVE_INLET_VELOCITY_LU",
    "PUBLIC_SOLVE_NX",
    "PUBLIC_SOLVE_NY",
    "PUBLIC_SOLVE_SEED",
    "PUBLIC_SOLVE_STEPS",
    "PUBLIC_SOLVE_WARMUP_STEPS",
    "MountedVolumeRemoteSolveBackend",
    "ReferenceRunProjector",
    "build_public_remote_request",
    "invoke_modal_solve",
    "public_design_id",
    "public_solve_case",
]
