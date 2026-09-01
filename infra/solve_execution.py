"""Provider-neutral transaction and numerical assembly for one remote solve."""

from __future__ import annotations

import importlib
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from infra.remote_execution import (
    RemoteSolveRequest,
    parse_remote_model,
    validate_correlation_id,
)
from infra.runtime_manifest import RuntimeBuildManifest
from soufflerie.datagen.run_artifact import LocalRunArtifactStore
from soufflerie.datagen.sweep_state import LocalSweepStateStore
from soufflerie.errors import InternalInvariantError, RemoteExecutionError, SoufflerieError
from soufflerie.geometry import ellipse_sdf, obstacle_mask, validate_geometry
from soufflerie.schemas import ArtifactRef, FlowFields, Provenance, SolverResult
from soufflerie.solver import (
    WarpObstacleStepper,
    derive_lattice,
    estimate_strouhal,
    mean_force_coefficients,
    run_lifecycle,
)


class _WarpDevice(Protocol):
    is_cuda: bool
    name: str


class _WarpModule(Protocol):
    def get_device(self, device: str) -> _WarpDevice: ...


def assert_solve_request_matches_build(
    request: RemoteSolveRequest,
    build: RuntimeBuildManifest,
) -> None:
    """Reject work whose immutable identity does not match the running image."""

    if build.source_dirty:
        raise RuntimeError("remote solve requires an image built from clean source")
    if request.source_revision != build.source_revision:
        raise RuntimeError("remote solve request source revision does not match the image")
    if request.lock_sha256 != build.lock_sha256:
        raise RuntimeError("remote solve request lock digest does not match the image")


def run_solver_case(
    request: RemoteSolveRequest,
    build: RuntimeBuildManifest,
) -> SolverResult:
    """Execute one complete CUDA solve and assemble its publication boundary."""

    wp = cast(_WarpModule, importlib.import_module("warp"))

    assert_solve_request_matches_build(request, build)
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("remote solve resolved a non-CUDA device")
    if request.requested_device_class.casefold() not in device.name.casefold():
        raise RuntimeError("resolved GPU does not match the explicitly requested device class")

    case = request.case
    validate_geometry(case.shape, case.grid)
    sdf = ellipse_sdf(case.shape, case.grid)
    mask = obstacle_mask(sdf)
    derived = derive_lattice(case)
    stepper = WarpObstacleStepper("cuda:0")
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    completed = run_lifecycle(derived, mask, stepper=stepper)
    history = stepper.force_history()
    coefficients = mean_force_coefficients(history)
    strouhal = estimate_strouhal(history, derived)
    elapsed = time.perf_counter() - started
    completed_at = datetime.now(UTC)
    provenance = Provenance(
        source_revision=build.source_revision,
        source_dirty=build.source_dirty,
        python_version=build.python_version,
        lock_sha256=build.lock_sha256,
        packages=build.packages,
        os=platform.system().casefold(),
        architecture=platform.machine(),
        device_class=request.requested_device_class,
        dtype_policy="fp32-lbm-fp64-reduction",
        config_sha256=case.sha256,
        parent_sha256={},
        seeds=(case.seed,),
        deterministic=True,
        started_at=started_at,
        completed_at=completed_at,
        gpu_seconds=elapsed,
    )
    return SolverResult(
        case_id=case.case_id,
        fields=FlowFields(
            u=completed.mean_fields.u,
            v=completed.mean_fields.v,
            rho=completed.mean_fields.rho,
            sdf=sdf,
            obstacle_mask=mask,
        ),
        cd=coefficients.cd,
        cl_mean=coefficients.cl,
        strouhal=strouhal.strouhal,
        force_steps=history.steps,
        cd_history=history.cd,
        cl_history=history.cl,
        diagnostics=completed.diagnostics,
        provenance=provenance,
    )


def _fail_owned_attempt(
    *,
    state_store: LocalSweepStateStore,
    request: RemoteSolveRequest,
    correlation_id: str,
    error: SoufflerieError,
    commit_volume: Callable[[], None],
) -> None:
    state_store.fail_case(
        request.case.case_id,
        attempt_id=request.attempt_id,
        lease_owner=correlation_id,
        error=error,
        now=datetime.now(UTC),
    )
    commit_volume()


def execute_solve_request(
    case_json: bytes,
    correlation_id: str,
    *,
    root: Path,
    build: RuntimeBuildManifest,
    reload_volume: Callable[[], None],
    commit_volume: Callable[[], None],
    solve_case: Callable[[RemoteSolveRequest, RuntimeBuildManifest], SolverResult],
) -> ArtifactRef:
    """Run the exact solve transaction behind the provider decorator."""

    request = parse_remote_model(case_json, RemoteSolveRequest)
    owner = validate_correlation_id(correlation_id)
    assert_solve_request_matches_build(request, build)
    reload_volume()
    artifact_store = LocalRunArtifactStore(root)
    state_store = LocalSweepStateStore(root, sweep_digest=request.sweep_digest)
    now = datetime.now(UTC)
    state_store.initialize_case(request.case.case_id, now=now)
    claimed = state_store.claim_case(
        request.case.case_id,
        attempt_id=request.attempt_id,
        lease_owner=owner,
        now=now,
    )
    if claimed is None:
        state = state_store.read_case(request.case.case_id)
        if state.state == "succeeded" and state.run_digest is not None:
            return artifact_store.verify_run(
                case_id=request.case.case_id,
                run_digest=state.run_digest,
            )
        raise RemoteExecutionError(
            f"case {request.case.case_id} is not claimable in state {state.state}"
        )
    commit_volume()

    if request.force_retryable_failure and claimed.attempt == 1:
        forced = RemoteExecutionError("forced smoke preemption before numerical execution")
        _fail_owned_attempt(
            state_store=state_store,
            request=request,
            correlation_id=owner,
            error=forced,
            commit_volume=commit_volume,
        )
        raise forced

    try:
        result = solve_case(request, build)
        reference = artifact_store.publish_run(
            attempt_id=request.attempt_id,
            design_id=request.design_id,
            split=request.split,
            case=request.case,
            result=result,
        )
    except SoufflerieError as error:
        _fail_owned_attempt(
            state_store=state_store,
            request=request,
            correlation_id=owner,
            error=error,
            commit_volume=commit_volume,
        )
        raise
    except Exception as error:
        invariant = InternalInvariantError("unexpected remote solver failure")
        _fail_owned_attempt(
            state_store=state_store,
            request=request,
            correlation_id=owner,
            error=invariant,
            commit_volume=commit_volume,
        )
        raise invariant from error
    commit_volume()
    try:
        state_store.succeed_case(
            request.case.case_id,
            attempt_id=request.attempt_id,
            lease_owner=owner,
            run_digest=reference.sha256,
            artifact_store=artifact_store,
            now=datetime.now(UTC),
        )
    except SoufflerieError as error:
        _fail_owned_attempt(
            state_store=state_store,
            request=request,
            correlation_id=owner,
            error=error,
            commit_volume=commit_volume,
        )
        raise
    commit_volume()
    return reference


__all__ = [
    "assert_solve_request_matches_build",
    "execute_solve_request",
    "run_solver_case",
]
