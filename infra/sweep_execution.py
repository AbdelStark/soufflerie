"""Provider-neutral resumable orchestration for the remote smoke sweep."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from infra.remote_execution import (
    RemoteSolveRequest,
    RemoteSweepRequest,
    SmokeDesignPoint,
    SweepSummary,
    encode_remote_model,
    load_remote_request,
    smoke_design,
)
from infra.runtime_manifest import RuntimeBuildManifest
from soufflerie.datagen.run_artifact import LocalRunArtifactStore
from soufflerie.datagen.sweep_state import (
    MAX_SWEEP_ATTEMPTS,
    LocalSweepStateStore,
    ResumePlan,
    build_resume_plan,
)
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ArtifactRef


def assert_sweep_request_matches_build(
    request: RemoteSweepRequest,
    build: RuntimeBuildManifest,
) -> None:
    """Reject work whose immutable identity does not match the running image."""

    if build.source_dirty:
        raise RuntimeError("remote sweep requires an image built from clean source")
    if request.source_revision != build.source_revision:
        raise RuntimeError("remote sweep request source revision does not match the image")
    if request.lock_sha256 != build.lock_sha256:
        raise RuntimeError("remote sweep request lock digest does not match the image")


def _fresh_plan(
    *,
    points: tuple[SmokeDesignPoint, ...],
    state_store: LocalSweepStateStore,
    artifact_store: LocalRunArtifactStore,
) -> ResumePlan:
    return build_resume_plan(
        case_ids=(point.case.case_id for point in points),
        state_store=state_store,
        artifact_store=artifact_store,
        now=datetime.now(UTC),
    )


def _assert_preserved_successes(previous: ResumePlan, current: ResumePlan) -> None:
    before = {run.case_id: run.reference.sha256 for run in previous.succeeded_runs}
    after = {run.case_id: run.reference.sha256 for run in current.succeeded_runs}
    for case_id, digest in before.items():
        if after.get(case_id) != digest:
            raise ArtifactIntegrityError(
                "successful run digest changed while resuming the remote sweep"
            )


def _round_inputs(
    *,
    request: RemoteSweepRequest,
    points_by_case: dict[str, SmokeDesignPoint],
    state_store: LocalSweepStateStore,
    case_ids: tuple[str, ...],
    forced_case_id: str | None,
) -> tuple[tuple[bytes, ...], tuple[str, ...]]:
    payloads: list[bytes] = []
    owners: list[str] = []
    for case_id in case_ids:
        point = points_by_case[case_id]
        state = state_store.read_case(case_id)
        next_attempt = state.attempt + 1
        attempt_id = f"s{next_attempt}-{case_id}"
        owner = f"sweep.{request.request_digest[:16]}.{next_attempt}.{case_id}"
        solve_request = RemoteSolveRequest.create(
            operation_kind="smoke-sweep",
            sweep_digest=request.request_digest,
            design_id=point.design_id,
            split=point.split,
            case=point.case,
            requested_device_class=request.requested_device_class,
            source_revision=request.source_revision,
            lock_sha256=request.lock_sha256,
            attempt_id=attempt_id,
            force_retryable_failure=case_id == forced_case_id,
        )
        payloads.append(encode_remote_model(solve_request))
        owners.append(owner)
    return tuple(payloads), tuple(owners)


def orchestrate_smoke_sweep(
    config_ref: ArtifactRef,
    *,
    root: Path,
    build: RuntimeBuildManifest,
    reload_volume: Callable[[], None],
    commit_volume: Callable[[], None],
    submit: Callable[[tuple[bytes, ...], tuple[str, ...]], None],
) -> SweepSummary:
    """Run the exact sweep state machine behind the provider decorator."""

    started = time.perf_counter()
    reference = ArtifactRef.model_validate_json(config_ref.model_dump_json())
    reload_volume()
    request = load_remote_request(root, reference)
    assert_sweep_request_matches_build(request, build)
    points = smoke_design(request)
    points_by_case = {point.case.case_id: point for point in points}
    artifact_store = LocalRunArtifactStore(root)
    state_store = LocalSweepStateStore(root, sweep_digest=request.request_digest)
    now = datetime.now(UTC)
    for point in points:
        state_store.initialize_case(point.case.case_id, now=now)
    commit_volume()
    reload_volume()

    plan = _fresh_plan(
        points=points,
        state_store=state_store,
        artifact_store=artifact_store,
    )
    skipped = tuple(run.case_id for run in plan.succeeded_runs)
    initial = plan.claimable_case_ids
    resumed: set[str] = {
        case_id for case_id in initial if state_store.read_case(case_id).attempt > 0
    }
    attempts = 0
    forced_case_id = (
        points[0].case.case_id
        if request.force_failure_once and points[0].case.case_id in initial
        else None
    )

    for round_index in range(MAX_SWEEP_ATTEMPTS):
        claimable = plan.claimable_case_ids
        if not claimable:
            break
        if round_index > 0:
            resumed.update(claimable)
        payloads, owners = _round_inputs(
            request=request,
            points_by_case=points_by_case,
            state_store=state_store,
            case_ids=claimable,
            forced_case_id=forced_case_id if round_index == 0 else None,
        )
        submit(payloads, owners)
        attempts += len(claimable)
        reload_volume()
        next_plan = _fresh_plan(
            points=points,
            state_store=state_store,
            artifact_store=artifact_store,
        )
        _assert_preserved_successes(plan, next_plan)
        plan = next_plan

    reload_volume()
    final_plan = _fresh_plan(
        points=points,
        state_store=state_store,
        artifact_store=artifact_store,
    )
    _assert_preserved_successes(plan, final_plan)
    states = [state_store.read_case(point.case.case_id) for point in points]
    references = tuple(
        run.reference for run in sorted(final_plan.succeeded_runs, key=lambda run: run.case_id)
    )
    gpu_seconds = sum(
        artifact_store.open_run(reference_value).metadata.provenance.gpu_seconds
        for reference_value in references
    )
    counts = {
        state_name: sum(state.state == state_name for state in states)
        for state_name in ("pending", "running", "succeeded", "failed")
    }
    elapsed = time.perf_counter() - started
    return SweepSummary.create(
        sweep_digest=request.request_digest,
        config_digest=request.config.config_digest,
        design_kind=request.design_kind,
        requested_device_class=request.requested_device_class,
        source_revision=request.source_revision,
        case_count=request.sample_count,
        pending_count=counts["pending"],
        running_count=counts["running"],
        succeeded_count=counts["succeeded"],
        failed_count=counts["failed"],
        initial_submitted_case_ids=initial,
        resumed_case_ids=tuple(sorted(resumed)),
        skipped_case_ids=tuple(sorted(skipped)),
        run_references=references,
        attempt_count=attempts,
        retry_count=sum(max(0, state.attempt - 1) for state in states),
        estimated_bytes=sum(reference_value.size_bytes for reference_value in references),
        wall_seconds=elapsed,
        gpu_seconds=gpu_seconds,
        final_state="succeeded" if len(references) == request.sample_count else "incomplete",
    )


__all__ = ["assert_sweep_request_matches_build", "orchestrate_smoke_sweep"]
