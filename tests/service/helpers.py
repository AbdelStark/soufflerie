from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime, timedelta

from soufflerie.config import ServiceConfig
from soufflerie.schemas import sha256_bytes
from soufflerie.service import (
    EncodedArtifact,
    PredictionRequest,
    ReadinessProbe,
    ShapeRequest,
    SolveComparison,
    SolveResultResponse,
)
from soufflerie.service.jobs import ProgressCallback

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
MODEL_ID = "1" * 20
DATASET_ID = "2" * 20
REPORT_ID = "3" * 20
JOB_IDS = tuple(f"0199a1b2-c3d4-7e5f-8a9b-{index:012x}" for index in range(1, 20))


class FakeClock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, *, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class IdFactory:
    def __init__(self) -> None:
        self._values = iter(JOB_IDS)

    def __call__(self) -> str:
        return next(self._values)


class ManualExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.entered: asyncio.Queue[str] = asyncio.Queue()
        self._gates: dict[str, asyncio.Event] = {}
        self._failures: dict[str, Exception] = {}

    async def execute(
        self,
        request: PredictionRequest,
        *,
        job_id: str,
        case_id: str,
        correlation_id: str,
        progress: ProgressCallback,
    ) -> SolveResultResponse:
        del request
        self.calls.append(job_id)
        gate = asyncio.Event()
        self._gates[job_id] = gate
        await progress(0.25)
        await self.entered.put(job_id)
        await gate.wait()
        if failure := self._failures.get(job_id):
            raise failure
        await progress(0.75)
        return solve_result(job_id=job_id, case_id=case_id, correlation_id=correlation_id)

    def finish(self, job_id: str) -> None:
        self._gates[job_id].set()

    def fail(self, job_id: str, error: Exception) -> None:
        self._failures[job_id] = error
        self.finish(job_id)


class NeverExecutor:
    async def execute(
        self,
        request: PredictionRequest,
        *,
        job_id: str,
        case_id: str,
        correlation_id: str,
        progress: ProgressCallback,
    ) -> SolveResultResponse:
        del request, job_id, case_id, correlation_id, progress
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def service_config(*, queue_capacity: int = 1) -> ServiceConfig:
    return ServiceConfig(
        model_id=MODEL_ID,
        dataset_id=DATASET_ID,
        report_id=REPORT_ID,
        solve_enabled=True,
        solve_concurrency=1,
        solve_queue_capacity=queue_capacity,
        solve_timeout_seconds=180,
    )


def readiness_probe() -> ReadinessProbe:
    return ReadinessProbe(
        model_id=MODEL_ID,
        model_dataset_id=DATASET_ID,
        report_id=REPORT_ID,
        report_model_id=MODEL_ID,
        report_dataset_id=DATASET_ID,
        validation_status="red",
        device_class="cpu",
        model_integrity_verified=True,
        report_integrity_verified=True,
        device_available=True,
        warmup_complete=True,
        checked_at=NOW,
        last_successful_readiness_check=None,
    )


def prediction_request(*, reynolds: float = 100.0) -> PredictionRequest:
    return PredictionRequest(
        shape=ShapeRequest(aspect_ratio=0.75, rotation_deg=12.0, scale=1.0),
        reynolds=reynolds,
    )


def solve_result(*, job_id: str, case_id: str, correlation_id: str) -> SolveResultResponse:
    png = b"png"
    npz = b"npz"
    return SolveResultResponse(
        correlation_id=correlation_id,
        job_id=job_id,
        case_id=case_id,
        reference_fields_png=EncodedArtifact(
            media_type="image/png",
            encoding="base64",
            data=base64.b64encode(png).decode("ascii"),
            sha256=sha256_bytes(png),
            bytes=len(png),
        ),
        reference_fields_npz=EncodedArtifact(
            media_type="application/x-npz",
            encoding="base64",
            data=base64.b64encode(npz).decode("ascii"),
            sha256=sha256_bytes(npz),
            bytes=len(npz),
        ),
        cd=1.0,
        cl_mean=0.0,
        strouhal=0.16,
        comparison=SolveComparison(
            model_id=MODEL_ID,
            dataset_id=DATASET_ID,
            report_id=REPORT_ID,
            cd_head=1.01,
            cd_field=0.99,
            cd_head_error_pct=1.0,
            cd_field_error_pct=1.0,
            velocity_rel_l2=0.02,
        ),
        solver_artifact_id="4" * 20,
        provenance_sha256="5" * 64,
        solver_ms=50.0,
        request_ms=55.0,
    )
