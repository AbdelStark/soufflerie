from __future__ import annotations

import asyncio

import httpx
import pytest

from soufflerie.config import ServiceConfig
from soufflerie.service import create_app
from soufflerie.service.admission import AdmissionController, AdmissionSettings
from soufflerie.service.contracts import PredictionRequest, SolveResultResponse
from soufflerie.service.jobs import ProgressCallback, SolveJobManager
from tests.service.helpers import readiness_probe, service_config, solve_result

REQUEST = {
    "schema_version": 1,
    "shape": {"aspect_ratio": 0.75, "rotation_deg": 12.0, "scale": 1.0},
    "reynolds": 100.0,
}


class CountingRemoteExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

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
        await progress(0.5)
        return solve_result(job_id=job_id, case_id=case_id, correlation_id=correlation_id)


def _one_solve_config() -> ServiceConfig:
    base = service_config()
    return ServiceConfig.model_validate(
        {
            **base.model_dump(mode="python"),
            "solves_per_day_global": 1,
            "solve_gpu_seconds_per_day": 180.0,
        }
    )


@pytest.mark.anyio
async def test_idempotent_replay_launches_one_remote_call_and_reserves_one_budget() -> None:
    config = _one_solve_config()
    executor = CountingRemoteExecutor()
    manager = SolveJobManager(config=config, executor=executor)
    admission = AdmissionController(
        config=config,
        settings=AdmissionSettings(client_hmac_key=b"a" * 32),
    )
    app = create_app(
        config=config,
        readiness=readiness_probe(),
        package_version="0.1.0",
        solve_jobs=manager,
        admission=admission,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("192.0.2.20", 1234)),
            base_url="http://test",
        ) as client:
            first = await client.post(
                "/solve",
                json=REQUEST,
                headers={"Idempotency-Key": "one-logical-solve"},
            )
            duplicate = await client.post(
                "/solve",
                json=REQUEST,
                headers={"Idempotency-Key": "one-logical-solve"},
            )
            assert first.status_code == duplicate.status_code == 202
            assert first.json() == duplicate.json()

            async with asyncio.timeout(2.0):
                while True:
                    terminal = await client.get(first.json()["status_url"])
                    if terminal.json()["state"] == "succeeded":
                        break
                    await asyncio.sleep(0.001)

            exhausted = await client.post(
                "/solve",
                json={**REQUEST, "reynolds": 101.0},
                headers={"Idempotency-Key": "different-solve"},
            )
        assert exhausted.status_code == 429
        assert exhausted.json()["error"]["code"] == "BUDGET_EXHAUSTED"
        assert int(exhausted.headers["Retry-After"]) >= 1
        assert len(executor.calls) == 1
        assert admission.snapshot().solves_admitted_today == 1
        assert admission.snapshot().gpu_seconds_reserved_today == 180.0
    finally:
        await manager.aclose()
