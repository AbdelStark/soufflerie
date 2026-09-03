from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from soufflerie.config import ServiceConfig
from soufflerie.service import (
    PredictionRequest,
    ReadinessProbe,
    SolveAccepted,
    SolveStatus,
    create_app,
)
from soufflerie.service.admission import (
    AdmissionController,
    AdmissionSettings,
    evaluate_service_readiness,
)
from soufflerie.service.jobs import AdmissionCheck, SolveEventStream, SolveJobManager
from tests.service.helpers import NeverExecutor

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
REQUEST = {
    "schema_version": 1,
    "shape": {"aspect_ratio": 0.75, "rotation_deg": 12.0, "scale": 1.0},
    "reynolds": 100.0,
}


def _config(**changes: object) -> ServiceConfig:
    values: dict[str, object] = {
        "model_id": "1" * 20,
        "dataset_id": "2" * 20,
        "report_id": "3" * 20,
        "solve_enabled": True,
        "solve_concurrency": 1,
        "solve_queue_capacity": 1,
        "solve_timeout_seconds": 180,
        "predictions_per_minute_client": 1,
        "solves_per_hour_client": 2,
        "solves_per_day_global": 20,
        "solve_gpu_seconds_per_day": 3_600.0,
    }
    values.update(changes)
    return ServiceConfig.model_validate(values)


def _probe(config: ServiceConfig, **changes: object) -> ReadinessProbe:
    values: dict[str, object] = {
        "model_id": config.model_id,
        "model_dataset_id": config.dataset_id,
        "report_id": config.report_id,
        "report_model_id": config.model_id,
        "report_dataset_id": config.dataset_id,
        "validation_status": "green",
        "device_class": "cpu",
        "model_integrity_verified": True,
        "report_integrity_verified": True,
        "device_available": True,
        "warmup_complete": True,
        "checked_at": NOW,
        "last_successful_readiness_check": None,
    }
    values.update(changes)
    return ReadinessProbe.model_validate(values)


def test_readiness_precedence_keeps_red_and_budget_exhaustion_prediction_ready() -> None:
    config = _config()

    artifact = evaluate_service_readiness(
        config,
        _probe(config, report_integrity_verified=False, device_available=False),
        solve_enabled=True,
        solve_budget_available=False,
    )
    assert (artifact.prediction_ready, artifact.solve_ready, artifact.reason) == (
        False,
        False,
        "artifact_invalid",
    )

    runtime = evaluate_service_readiness(
        config,
        _probe(config, warmup_complete=False),
        solve_enabled=True,
        solve_budget_available=False,
    )
    assert runtime.reason == "runtime_unavailable"

    budget = evaluate_service_readiness(
        config,
        _probe(config, validation_status="red"),
        solve_enabled=True,
        solve_budget_available=False,
    )
    assert (budget.prediction_ready, budget.solve_ready, budget.reason) == (
        True,
        False,
        "solve_budget_exhausted",
    )

    red = evaluate_service_readiness(
        config,
        _probe(config, validation_status="red"),
        solve_enabled=True,
        solve_budget_available=True,
    )
    assert (red.prediction_ready, red.solve_ready, red.reason) == (
        True,
        True,
        "validation_red",
    )

    disabled = evaluate_service_readiness(
        config,
        _probe(config, validation_status="red"),
        solve_enabled=False,
        solve_budget_available=True,
    )
    assert (disabled.prediction_ready, disabled.solve_ready, disabled.reason) == (
        True,
        False,
        "solve_disabled",
    )


class NeverBackend:
    def __init__(self) -> None:
        self.submit_calls = 0

    async def submit(
        self,
        request: PredictionRequest,
        *,
        correlation_id: str,
        idempotency_key: str | None,
        admission_check: AdmissionCheck | None = None,
    ) -> SolveAccepted:
        del request, correlation_id, idempotency_key
        if admission_check is not None:
            admission_check()
        self.submit_calls += 1
        raise AssertionError("admission must fail before the solve backend")

    async def status(self, job_id: str) -> SolveStatus:
        del job_id
        raise AssertionError("not used")

    async def open_events(self, job_id: str, *, after: int) -> SolveEventStream:
        del job_id, after
        raise AssertionError("not used")


@pytest.mark.anyio
async def test_http_rate_limit_and_kill_switch_fail_before_service_work() -> None:
    config = _config()
    probe = _probe(config, validation_status="red")
    prediction_admission = AdmissionController(
        config=config,
        settings=AdmissionSettings(client_hmac_key=b"p" * 32),
    )
    prediction_app = create_app(
        config=config,
        readiness=probe,
        package_version="0.1.0",
        admission=prediction_admission,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=prediction_app, client=("192.0.2.1", 1234)),
        base_url="http://test",
    ) as client:
        assert (await client.get("/health")).json()["readiness"] == "ready"
        first = await client.post("/predict", json=REQUEST)
        limited = await client.post("/predict", json=REQUEST)
    assert first.status_code == 503
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"
    assert limited.json()["error"]["code"] == "RATE_LIMITED"

    backend = NeverBackend()
    solve_admission = AdmissionController(
        config=config,
        settings=AdmissionSettings(
            client_hmac_key=b"s" * 32,
            solve_enabled_override=False,
        ),
    )
    solve_app = create_app(
        config=config,
        readiness=probe,
        package_version="0.1.0",
        solve_jobs=backend,
        admission=solve_admission,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=solve_app, client=("192.0.2.2", 1234)),
        base_url="http://test",
    ) as client:
        health = await client.get("/health")
        disabled = await client.post("/solve", json=REQUEST)
    assert health.status_code == 200
    assert health.json()["readiness"] == "ready"
    assert disabled.status_code == 503
    assert disabled.json()["error"]["code"] == "SOLVE_DISABLED"
    assert "retry-after" not in disabled.headers
    assert backend.submit_calls == 0


@pytest.mark.anyio
async def test_daily_gpu_budget_closes_only_new_solve_work_with_retry_boundary() -> None:
    config = _config(solve_gpu_seconds_per_day=180.0)
    probe = _probe(config, validation_status="red")
    admission = AdmissionController(
        config=config,
        settings=AdmissionSettings(client_hmac_key=b"b" * 32),
    )
    manager = SolveJobManager(config=config, executor=NeverExecutor())
    app = create_app(
        config=config,
        readiness=probe,
        package_version="0.1.0",
        solve_jobs=manager,
        admission=admission,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("192.0.2.3", 1234)),
            base_url="http://test",
        ) as client:
            admitted = await client.post("/solve", json=REQUEST)
            exhausted = await client.post("/solve", json=REQUEST)
            health = await client.get("/health")
        assert admitted.status_code == 202
        assert exhausted.status_code == 429
        assert exhausted.json()["error"]["code"] == "BUDGET_EXHAUSTED"
        assert 1 <= int(exhausted.headers["retry-after"]) <= 86_400
        assert health.json()["readiness"] == "ready"
    finally:
        await manager.aclose()
