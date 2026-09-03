"""Deterministic OpenAPI generation for the public service boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

from soufflerie.config import ServiceConfig
from soufflerie.service.app import create_app
from soufflerie.service.contracts import ReadinessProbe, SolveEvent

_SCHEMA_TIME = datetime(2026, 9, 1, tzinfo=UTC)


def service_openapi_document() -> dict[str, Any]:
    """Generate the identity-independent OpenAPI 3.1 contract."""

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
        checked_at=_SCHEMA_TIME,
        last_successful_readiness_check=None,
    )
    return create_app(config=config, readiness=readiness, package_version="0.1.0").openapi()


def solve_event_schema_document() -> dict[str, object]:
    """Generate the standalone JSON Schema carried by solve SSE data fields."""

    document = cast(dict[str, object], SolveEvent.model_json_schema(mode="validation"))
    document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    document["$id"] = "https://github.com/AbdelStark/soufflerie/schemas/v1/solve-event.json"
    return document


def rendered_service_schema_documents() -> dict[str, str]:
    return {
        "openapi.json": json.dumps(
            service_openapi_document(), indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        "solve-event.json": json.dumps(
            solve_event_schema_document(), indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
    }


__all__ = [
    "rendered_service_schema_documents",
    "service_openapi_document",
    "solve_event_schema_document",
]
