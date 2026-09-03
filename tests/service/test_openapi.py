from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from soufflerie.config import ServiceConfig
from soufflerie.service import MAX_REQUEST_BODY_BYTES, ReadinessProbe, create_app
from soufflerie.service.schema_registry import service_openapi_document

PROJECT_ROOT = Path(__file__).parents[2]
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
JOB_ID = "0199a1b2-c3d4-7e5f-8a9b-0123456789ab"
VALID_REQUEST = {
    "schema_version": 1,
    "shape": {"aspect_ratio": 0.75, "rotation_deg": 12.0, "scale": 1.0},
    "reynolds": 100.0,
}


class _OversizedStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"x" * MAX_REQUEST_BODY_BYTES
        yield b"x"


def _app() -> FastAPI:
    config = ServiceConfig(
        model_id="1" * 20,
        dataset_id="2" * 20,
        report_id="3" * 20,
        solve_enabled=False,
        solve_concurrency=0,
        solve_queue_capacity=0,
    )
    readiness = ReadinessProbe(
        model_id=config.model_id,
        model_dataset_id=config.dataset_id,
        report_id=config.report_id,
        report_model_id=config.model_id,
        report_dataset_id=config.dataset_id,
        validation_status="red",
        device_class="cpu",
        model_integrity_verified=True,
        report_integrity_verified=True,
        device_available=True,
        warmup_complete=True,
        checked_at=NOW,
        last_successful_readiness_check=None,
    )
    return create_app(config=config, readiness=readiness, package_version="0.1.0")


def _assert_component(payload: object, component: str) -> None:
    document = service_openapi_document()
    validator = Draft202012Validator({**document, "$ref": f"#/components/schemas/{component}"})
    validator.validate(payload)


def test_openapi_is_checked_in_stable_and_closed() -> None:
    expected = json.loads((PROJECT_ROOT / "schemas/v1/openapi.json").read_text(encoding="utf-8"))
    actual = service_openapi_document()
    assert actual == expected
    Draft202012Validator.check_schema(actual["components"]["schemas"]["PredictionRequest"])
    assert set(actual["paths"]) == {
        "/health",
        "/predict",
        "/solve",
        "/solve/{job_id}",
        "/solve/{job_id}/events",
    }
    operation_ids = {
        operation["operationId"] for path in actual["paths"].values() for operation in path.values()
    }
    assert operation_ids == {
        "health_v1",
        "predict_v1",
        "solve_submit_v1",
        "solve_status_v1",
        "solve_events_v1",
    }
    event_responses = actual["paths"]["/solve/{job_id}/events"]["get"]["responses"]
    assert event_responses["200"]["content"]["text/event-stream"]["schema"][
        "x-event-data-schema"
    ] == {"$ref": "#/components/schemas/SolveEvent"}
    for status in ("404", "422", "500", "503"):
        assert set(event_responses[status]["content"]) == {"application/json"}


@pytest.mark.parametrize(
    ("content", "content_type", "expected_status", "expected_code"),
    (
        (json.dumps(VALID_REQUEST), "text/plain", 415, "UNSUPPORTED_MEDIA_TYPE"),
        ("x" * (MAX_REQUEST_BODY_BYTES + 1), "application/json", 413, "REQUEST_TOO_LARGE"),
        (json.dumps({**VALID_REQUEST, "unknown": 1}), "application/json", 422, "REQUEST_INVALID"),
        (
            json.dumps({**VALID_REQUEST, "reynolds": True}),
            "application/json",
            422,
            "REQUEST_INVALID",
        ),
        (
            '{"schema_version":1,"shape":{"aspect_ratio":0.75,"rotation_deg":12.0,'
            '"scale":1.0},"reynolds":NaN}',
            "application/json",
            422,
            "CASE_OUT_OF_DOMAIN",
        ),
        (
            json.dumps({**VALID_REQUEST, "reynolds": 301.0}),
            "application/json",
            422,
            "CASE_OUT_OF_DOMAIN",
        ),
    ),
)
@pytest.mark.anyio
async def test_request_boundaries_fail_before_service_work(
    content: str,
    content_type: str,
    expected_status: int,
    expected_code: str,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()), base_url="http://test"
    ) as client:
        response = await client.post(
            "/predict", content=content, headers={"content-type": content_type}
        )
    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    _assert_component(response.json(), "PublicErrorResponse")


@pytest.mark.anyio
async def test_chunked_body_is_bounded_without_content_length() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()), base_url="http://test"
    ) as client:
        response = await client.post(
            "/predict",
            content=_OversizedStream(),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


@pytest.mark.anyio
async def test_declared_endpoint_responses_conform_to_golden_components() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()), base_url="http://test"
    ) as client:
        responses = (
            (await client.get("/health"), "HealthResponse", 200),
            (await client.post("/predict", json=VALID_REQUEST), "PublicErrorResponse", 503),
            (await client.post("/solve", json=VALID_REQUEST), "PublicErrorResponse", 503),
            (await client.get(f"/solve/{JOB_ID}"), "PublicErrorResponse", 404),
            (await client.get(f"/solve/{JOB_ID}/events"), "PublicErrorResponse", 404),
            (await client.get("/not-a-route"), "PublicErrorResponse", 404),
        )
    for response, component, status in responses:
        assert response.status_code == status
        _assert_component(response.json(), component)
        assert response.headers["x-correlation-id"]
