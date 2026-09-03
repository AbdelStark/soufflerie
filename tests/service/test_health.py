from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from soufflerie.config import ServiceConfig, load_config
from soufflerie.errors import DomainError
from soufflerie.service import ReadinessProbe, assess_health, create_app
from soufflerie.validation import load_validation_report

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
PROJECT_ROOT = Path(__file__).parents[2]
MODEL_ID = "1" * 20
DATASET_ID = "2" * 20
REPORT_ID = "3" * 20


def _config() -> ServiceConfig:
    return ServiceConfig(
        model_id=MODEL_ID,
        dataset_id=DATASET_ID,
        report_id=REPORT_ID,
        solve_enabled=False,
        solve_concurrency=0,
        solve_queue_capacity=0,
    )


def _probe(**changes: object) -> ReadinessProbe:
    values: dict[str, object] = {
        "model_id": MODEL_ID,
        "model_dataset_id": DATASET_ID,
        "report_id": REPORT_ID,
        "report_model_id": MODEL_ID,
        "report_dataset_id": DATASET_ID,
        "validation_status": "red",
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


def test_red_validation_is_ready_when_integrity_and_identities_match() -> None:
    health = assess_health(_config(), _probe(), package_version="0.1.0")
    assert health.readiness == "ready"
    assert health.validation_status == "red"
    assert health.last_successful_readiness_check == NOW
    assert set(health.model_dump()) == {
        "schema_version",
        "liveness",
        "readiness",
        "package_version",
        "model_id",
        "dataset_id",
        "report_id",
        "validation_status",
        "device_class",
        "last_successful_readiness_check",
    }


def test_any_integrity_identity_device_or_warmup_failure_is_not_ready() -> None:
    previous = NOW - timedelta(minutes=5)
    changes = (
        {"model_id": "a" * 20},
        {"model_dataset_id": "a" * 20},
        {"report_id": "a" * 20},
        {"report_model_id": "a" * 20},
        {"report_dataset_id": "a" * 20},
        {"model_integrity_verified": False},
        {"report_integrity_verified": False},
        {"device_available": False},
        {"warmup_complete": False},
    )
    for change in changes:
        health = assess_health(
            _config(),
            _probe(**change, last_successful_readiness_check=previous),
            package_version="0.1.0",
        )
        assert health.readiness == "not_ready"
        assert health.last_successful_readiness_check == previous


def test_demo_config_binds_the_checked_canonical_red_report() -> None:
    config = load_config(PROJECT_ROOT / "configs/service/demo-v1.yaml", ServiceConfig)
    report = load_validation_report(PROJECT_ROOT / "reports/validation.json")
    health = assess_health(
        config,
        ReadinessProbe(
            model_id=report.selected_model_id,
            model_dataset_id=report.dataset_id,
            report_id=report.report_id,
            report_model_id=report.selected_model_id,
            report_dataset_id=report.dataset_id,
            validation_status=report.overall_status,
            device_class="cpu",
            model_integrity_verified=True,
            report_integrity_verified=True,
            device_available=True,
            warmup_complete=True,
            checked_at=NOW,
            last_successful_readiness_check=None,
        ),
        package_version="0.1.0",
    )
    assert health.readiness == "ready"
    assert health.validation_status == "red"
    assert (health.model_id, health.dataset_id, health.report_id) == (
        report.selected_model_id,
        report.dataset_id,
        report.report_id,
    )


@pytest.mark.anyio
async def test_health_endpoint_is_allowlisted_and_unready_prediction_is_typed() -> None:
    app = create_app(
        config=_config(),
        readiness=_probe(report_integrity_verified=False),
        package_version="0.1.0",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        health = await client.get("/health")
        predict = await client.post(
            "/predict",
            json={
                "schema_version": 1,
                "shape": {"aspect_ratio": 0.75, "rotation_deg": 12.0, "scale": 1.0},
                "reynolds": 100.0,
            },
        )

    assert health.status_code == 200
    assert health.json()["readiness"] == "not_ready"
    assert predict.status_code == 503
    assert predict.json()["error"]["code"] == "DEPENDENCY_UNAVAILABLE"
    assert predict.headers["x-correlation-id"] == predict.json()["error"]["correlation_id"]
    rendered = str(health.json()).casefold()
    for forbidden in ("host", "path", "account", "credential", "secret"):
        assert forbidden not in rendered


@pytest.mark.anyio
async def test_boundary_maps_typed_and_unexpected_errors_without_leaking_messages() -> None:
    app = create_app(config=_config(), readiness=_probe(), package_version="0.1.0")

    @app.get("/_typed-error", include_in_schema=False)
    async def typed_error() -> None:
        raise DomainError("sentinel-secret /private/runtime/path")

    @app.get("/_unexpected-error", include_in_schema=False)
    async def unexpected_error() -> None:
        raise RuntimeError("sentinel-secret /private/runtime/path")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        typed = await client.get("/_typed-error")
        unexpected = await client.get("/_unexpected-error")
        method = await client.post("/health")

    assert typed.status_code == 422
    assert typed.json()["error"]["code"] == "CASE_OUT_OF_DOMAIN"
    assert unexpected.status_code == 500
    assert unexpected.json()["error"]["code"] == "INTERNAL_ERROR"
    assert method.status_code == 405
    assert method.json()["error"]["code"] == "METHOD_NOT_ALLOWED"
    for response in (typed, unexpected, method):
        assert "sentinel-secret" not in response.text
        assert "/private/runtime/path" not in response.text
        assert response.headers["x-correlation-id"] == response.json()["error"]["correlation_id"]
