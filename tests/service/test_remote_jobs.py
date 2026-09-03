from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest

from infra.policy import CheckoutState
from infra.remote_execution import RemoteSolveRequest, parse_remote_model
from infra.service_execution import (
    MountedVolumeRemoteSolveBackend,
    build_public_remote_request,
    invoke_modal_solve,
    public_design_id,
    public_solve_case,
)
from soufflerie.datagen.run_artifact import LocalRunArtifactStore, RunArtifact
from soufflerie.errors import ArtifactIntegrityError, ConfigurationError, RemoteExecutionError
from soufflerie.observability import new_correlation_id
from soufflerie.schemas import (
    ArtifactRef,
    CaseConfig,
    SolverResult,
    canonical_sha256,
    sha256_bytes,
)
from soufflerie.service import (
    ConsistencyFlags,
    EncodedArtifact,
    PredictionForComparison,
    PredictionRequest,
    PredictionResponse,
    ReferenceProjection,
    RemoteSolveExecutor,
    SolveJobManager,
    SolveStatus,
)
from tests.service.helpers import NOW, IdFactory, prediction_request, service_config

SOURCE_REVISION = "a" * 40
LOCK_SHA256 = "b" * 64


def _encoded(
    media_type: Literal["image/png", "application/x-npz"], content: bytes
) -> EncodedArtifact:
    return EncodedArtifact(
        media_type=media_type,
        encoding="base64",
        data=base64.b64encode(content).decode("ascii"),
        sha256=sha256_bytes(content),
        bytes=len(content),
    )


class StaticProjector:
    def __init__(self) -> None:
        self.calls = 0

    def project(self, run: RunArtifact) -> ReferenceProjection:
        self.calls += 1
        return ReferenceProjection(
            fields_png=_encoded("image/png", b"reference-png"),
            fields_npz=_encoded("application/x-npz", b"reference-npz"),
            u=np.ascontiguousarray(run.fields.u_mean, dtype=np.float32),
            v=np.ascontiguousarray(run.fields.v_mean, dtype=np.float32),
            obstacle_mask=np.ascontiguousarray(run.fields.obstacle_mask, dtype=np.bool_),
        )


class StaticPredictor:
    def __init__(self, reference: ReferenceProjection) -> None:
        self.reference = reference
        self.calls = 0

    async def predict(
        self,
        request: PredictionRequest,
        *,
        correlation_id: str,
    ) -> PredictionForComparison:
        assert request == prediction_request()
        self.calls += 1
        response = PredictionResponse(
            correlation_id=correlation_id,
            case_id=canonical_sha256(request)[:20],
            fields_png=_encoded("image/png", b"prediction-png"),
            fields_npz=_encoded("application/x-npz", b"prediction-npz"),
            cd_head=1.365,
            cd_field=1.17,
            consistency=ConsistencyFlags(
                head_field_gap_pct=15.0,
                head_field_gap="red",
                divergence_ratio_to_solver_baseline=1.0,
                divergence="green",
                obstacle_velocity_ratio=0.0,
                obstacle_compliance="green",
                ood=False,
            ),
            validation_status="red",
            model_id="1" * 20,
            dataset_id="2" * 20,
            report_id="3" * 20,
            inference_ms=5.0,
            request_ms=6.0,
        )
        return PredictionForComparison(
            response=response,
            u=np.ascontiguousarray(self.reference.u * np.float32(1.1)),
            v=np.ascontiguousarray(self.reference.v * np.float32(1.1)),
        )


class CapturingInvoker:
    def __init__(self, reference: ArtifactRef) -> None:
        self.reference = reference
        self.calls: list[tuple[bytes, str]] = []

    async def __call__(self, content: bytes, correlation_id: str) -> ArtifactRef:
        self.calls.append((content, correlation_id))
        return self.reference


def _published_reference(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> tuple[LocalRunArtifactStore, ArtifactRef]:
    provenance = solver_result.provenance.model_copy(
        update={"device_class": "L40S", "gpu_seconds": 2.0}
    )
    remote_result = replace(solver_result, provenance=provenance)
    store = LocalRunArtifactStore(tmp_path)
    reference = store.publish_run(
        attempt_id="service-fixture",
        design_id=public_design_id(prediction_request()),
        split="test",
        case=run_case,
        result=remote_result,
    )
    return store, reference


async def _wait_for_terminal(manager: SolveJobManager, job_id: str) -> SolveStatus:
    async with asyncio.timeout(2.0):
        while True:
            status = await manager.status(job_id)
            if status.state in {"succeeded", "failed"}:
                return status
            await asyncio.sleep(0.001)


@pytest.mark.anyio
async def test_remote_job_is_idempotent_identity_bound_and_publishes_comparison(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    request = prediction_request()
    assert public_solve_case(request) == run_case
    store, reference = _published_reference(tmp_path, run_case, solver_result)
    projector = StaticProjector()
    verified_projection = projector.project(store.open_run(reference))
    projector.calls = 0
    invoker = CapturingInvoker(reference)
    reloads: list[str] = []
    gateway = MountedVolumeRemoteSolveBackend(
        checkout=CheckoutState(
            source_revision=SOURCE_REVISION,
            source_dirty=False,
            lock_sha256=LOCK_SHA256,
        ),
        device_class="L40S",
        invoke=invoker,
        reload_volume=lambda: reloads.append("reload"),
        run_store=store,
        projector=projector,
    )
    predictor = StaticPredictor(verified_projection)
    manager = SolveJobManager(
        config=service_config(),
        executor=RemoteSolveExecutor(
            config=service_config(),
            predictor=predictor,
            backend=gateway,
        ),
        id_factory=IdFactory(),
    )
    correlation_id = new_correlation_id(timestamp=NOW)
    try:
        accepted = await manager.submit(
            request,
            correlation_id=correlation_id,
            idempotency_key="same-remote-work",
        )
        duplicate = await manager.submit(
            request,
            correlation_id=correlation_id,
            idempotency_key="same-remote-work",
        )
        assert duplicate == accepted
        status = await _wait_for_terminal(manager, accepted.job_id)
    finally:
        await manager.aclose()

    assert status.state == "succeeded"
    assert status.result is not None
    result = status.result
    assert len(invoker.calls) == predictor.calls == projector.calls == 1
    assert reloads == ["reload"]
    payload, propagated_correlation_id = invoker.calls[0]
    remote_request = parse_remote_model(payload, RemoteSolveRequest)
    assert propagated_correlation_id == correlation_id
    assert remote_request.attempt_id == f"service-{accepted.job_id}"
    assert remote_request.case == run_case
    assert remote_request.request_digest == canonical_sha256(remote_request.logical_identity())
    assert result.case_id == canonical_sha256(request)[:20]
    assert result.solver_artifact_id == reference.sha256[:20]
    verified_run = store.open_run(reference)
    assert result.provenance_sha256 == canonical_sha256(verified_run.metadata.provenance)
    assert result.comparison.model_id == service_config().model_id
    assert result.comparison.cd_head_error_pct == pytest.approx(5.0)
    assert result.comparison.cd_field_error_pct == pytest.approx(10.0)
    assert result.comparison.velocity_rel_l2 == pytest.approx(0.1, abs=2e-5)
    assert result.solver_ms == 2_000.0
    assert result.request_ms >= result.solver_ms


@pytest.mark.anyio
async def test_mounted_backend_rejects_swapped_provenance_and_types_provider_failures(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    store = LocalRunArtifactStore(tmp_path)
    reference = store.publish_run(
        attempt_id="wrong-device",
        design_id=public_design_id(prediction_request()),
        split="test",
        case=run_case,
        result=solver_result,
    )
    checkout = CheckoutState(
        source_revision=SOURCE_REVISION,
        source_dirty=False,
        lock_sha256=LOCK_SHA256,
    )
    backend = MountedVolumeRemoteSolveBackend(
        checkout=checkout,
        device_class="L40S",
        invoke=CapturingInvoker(reference),
        reload_volume=lambda: None,
        run_store=store,
        projector=StaticProjector(),
    )
    with pytest.raises(ArtifactIntegrityError, match="provenance"):
        await backend.solve(
            prediction_request(),
            job_id=IdFactory()(),
            correlation_id=new_correlation_id(timestamp=NOW),
        )

    async def unavailable(_content: bytes, _correlation_id: str) -> ArtifactRef:
        raise RuntimeError("provider account details must not escape")

    failing = MountedVolumeRemoteSolveBackend(
        checkout=checkout,
        device_class="L40S",
        invoke=unavailable,
        reload_volume=lambda: None,
        run_store=store,
        projector=StaticProjector(),
    )
    with pytest.raises(RemoteExecutionError, match="result assembly") as captured:
        await failing.solve(
            prediction_request(),
            job_id=IdFactory()(),
            correlation_id=new_correlation_id(timestamp=NOW),
        )
    assert "account" not in str(captured.value)


@pytest.mark.anyio
async def test_modal_invocation_terminates_remote_container_when_cancelled() -> None:
    entered = asyncio.Event()
    cancelled: list[bool] = []

    class FakeCall:
        def __init__(self) -> None:
            self.get = SimpleNamespace(aio=self._get)
            self.cancel = SimpleNamespace(aio=self._cancel)

        async def _get(self) -> object:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def _cancel(self, *, terminate_containers: bool) -> None:
            cancelled.append(terminate_containers)

    call = FakeCall()

    class FakeFunction:
        def __init__(self) -> None:
            self.spawn = SimpleNamespace(aio=self._spawn)

        async def _spawn(self, content: bytes, correlation_id: str) -> FakeCall:
            assert content == b"request"
            assert correlation_id == "0199a1b2-c3d4-7e5f-8a9b-000000000001"
            return call

    task = asyncio.create_task(
        invoke_modal_solve(
            FakeFunction(),
            b"request",
            "0199a1b2-c3d4-7e5f-8a9b-000000000001",
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled == [True]


def test_public_remote_request_rejects_dirty_build_identity() -> None:
    request = prediction_request()
    dirty = CheckoutState(
        source_revision=SOURCE_REVISION,
        source_dirty=True,
        lock_sha256=LOCK_SHA256,
    )
    with pytest.raises(ConfigurationError, match="clean source revision"):
        build_public_remote_request(
            request,
            job_id=IdFactory()(),
            checkout=dirty,
            device_class="L40S",
        )

    clean = replace(dirty, source_dirty=False)
    for job_id in ("not-a-job", "0199A1B2-C3D4-7E5F-8A9B-000000000001"):
        with pytest.raises(ConfigurationError, match="canonical UUIDv7"):
            build_public_remote_request(
                request,
                job_id=job_id,
                checkout=clean,
                device_class="L40S",
            )
