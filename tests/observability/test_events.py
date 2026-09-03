from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from soufflerie.errors import CapacityError
from soufflerie.observability import (
    EVENT_REGISTRY,
    METRIC_REGISTRY,
    NOOP_EVENT_SINK,
    NOOP_METRIC_SINK,
    ArtifactIdentities,
    Event,
    InMemoryEventSink,
    InMemoryMetricSink,
    JsonlEventSink,
    MetricRecord,
    OperationRecorder,
    event_json,
    new_correlation_id,
    observability_schema_documents,
    operation_context,
)

NOW = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
REQUIRED_EVENTS = {
    "case_started",
    "case_completed",
    "case_failed",
    "artifact_published",
    "sweep_progress",
    "epoch_completed",
    "checkpoint_selected",
    "gate_evaluated",
    "prediction_completed",
    "solve_admitted",
    "solve_terminal",
    "redaction_applied",
}


def _event(**changes: object) -> Event:
    values: dict[str, object] = {
        "timestamp": NOW,
        "level": "info",
        "event": "case_started",
        "correlation_id": new_correlation_id(timestamp=NOW),
        "component": "solver",
        "case_id": "case-123",
        "fields": {"reynolds": 100.0, "shape": [0.75, 10.0, 1.0]},
    }
    values.update(changes)
    return Event.model_validate(values)


def test_event_registry_and_schema_are_strict_stable_and_bounded() -> None:
    assert isinstance(EVENT_REGISTRY, MappingProxyType)
    assert EVENT_REGISTRY.keys() >= REQUIRED_EVENTS
    event = _event()
    assert event.schema_version == 1
    assert event.correlation_id.startswith("01")
    assert json.loads(event_json(event))["case_id"] == "case-123"

    with pytest.raises(TypeError):
        EVENT_REGISTRY["case_started"] = EVENT_REGISTRY["case_started"]  # type: ignore[index]
    with pytest.raises(ValidationError):
        _event(timestamp=datetime(2026, 9, 1, 10, 0))
    with pytest.raises(ValidationError):
        _event(correlation_id="00000000-0000-4000-8000-000000000000")
    with pytest.raises(ValidationError):
        _event(event="free_form_event")
    with pytest.raises(ValidationError):
        _event(fields={"nested": {"arbitrary": "object"}})
    for forbidden_key in ("request_headers", "client_key", "client_hash", "rate_limit_key"):
        with pytest.raises(ValidationError, match="forbidden event fields"):
            _event(fields={forbidden_key: "not allowed"})
    with pytest.raises(ValidationError):
        _event(fields={"samples": list(range(65))})
    with pytest.raises(ValidationError):
        _event(fields={"value": float("nan")})


@given(
    component=st.sampled_from(("solver", "datagen", "training", "validation", "service", "infra")),
    value=st.floats(allow_nan=False, allow_infinity=False, width=32),
)
def test_every_component_can_emit_without_global_logging(component: str, value: float) -> None:
    recorder = OperationRecorder(
        correlation_id=new_correlation_id(timestamp=NOW),
        component=component,  # type: ignore[arg-type]
        utc_clock=lambda: NOW,
    )
    event = recorder.emit("redaction_applied", fields={"count": value})
    assert event.component == component

    NOOP_EVENT_SINK.emit(event)
    NOOP_METRIC_SINK.emit(
        MetricRecord(
            timestamp=NOW,
            metric="sweep_cases_total",
            value=1.0,
            unit="count",
            labels={"state": "succeeded"},
        )
    )


def test_metric_registry_enforces_units_labels_scope_and_finite_values() -> None:
    assert isinstance(METRIC_REGISTRY, MappingProxyType)
    sample = MetricRecord(
        timestamp=NOW,
        metric="solver_run_seconds",
        value=2.5,
        unit="seconds",
        labels={"outcome": "succeeded", "device_class": "NVIDIA L40S"},
    )
    assert sample.scope == "aggregate"

    with pytest.raises(ValidationError, match="uses unit"):
        MetricRecord(
            timestamp=NOW,
            metric="solver_run_seconds",
            value=2.5,
            unit="milliseconds",
            labels={"device_class": "cpu", "outcome": "succeeded"},
        )
    with pytest.raises(ValidationError, match="requires exactly these labels"):
        MetricRecord(
            timestamp=NOW,
            metric="solver_run_seconds",
            value=2.5,
            unit="seconds",
            labels={"device_class": "cpu", "outcome": "succeeded", "case_id": "case-1"},
        )
    with pytest.raises(ValidationError, match="offline report metric"):
        MetricRecord(
            timestamp=NOW,
            metric="solver_mass_drift_ratio",
            value=0.001,
            unit="ratio",
            labels={"case_id": "case-1"},
        )
    offline = MetricRecord(
        timestamp=NOW,
        metric="solver_mass_drift_ratio",
        value=0.001,
        unit="ratio",
        labels={"case_id": "case-1"},
        scope="offline",
    )
    assert offline.scope == "offline"
    with pytest.raises(ValidationError):
        MetricRecord(
            timestamp=NOW,
            metric="sweep_cases_total",
            value=float("inf"),
            unit="count",
            labels={"state": "running"},
        )


def test_operation_context_propagates_identity_and_emits_terminal_outcomes() -> None:
    event_sink = InMemoryEventSink()
    metric_sink = InMemoryMetricSink()
    correlation_id = new_correlation_id(timestamp=NOW)
    identities = ArtifactIdentities(case_id="case-1", job_id="job-1")

    with operation_context(
        "case_started",
        correlation_id=correlation_id,
        identities=identities,
        event_sink=event_sink,
        metric_sink=metric_sink,
        utc_clock=lambda: NOW,
    ) as operation:
        operation.annotate(steps_completed=100)
        operation.metric(
            "solver_run_seconds",
            1.5,
            unit="seconds",
            labels={"device_class": "cpu", "outcome": "succeeded"},
        )

    assert [event.event for event in event_sink.events] == ["case_started", "case_completed"]
    assert {event.correlation_id for event in event_sink.events} == {correlation_id}
    assert event_sink.events[-1].case_id == "case-1"
    assert event_sink.events[-1].fields["outcome"] == "succeeded"
    assert metric_sink.metrics[0].correlation_id == correlation_id

    failure_sink = InMemoryEventSink()
    with (
        pytest.raises(CapacityError, match="busy"),
        operation_context("solve_admitted", event_sink=failure_sink, utc_clock=lambda: NOW),
    ):
        raise CapacityError("busy", retryable=True)
    assert [event.event for event in failure_sink.events] == ["solve_admitted", "solve_terminal"]
    terminal = failure_sink.events[-1]
    assert terminal.level == "error"
    assert terminal.fields["error_code"] == "CAPACITY_EXHAUSTED"
    assert terminal.fields["retryable"] is True


def test_jsonl_sink_writes_one_schema_valid_record_per_line(tmp_path: Path) -> None:
    output = tmp_path / "events.jsonl"
    sink = JsonlEventSink(output)
    sink.emit(_event(fields={"iteration": 1}))
    sink.emit(_event(fields={"iteration": 2}))

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [record["fields"]["iteration"] for record in records] == [1, 2]
    assert all(record["schema_version"] == 1 for record in records)


def test_observability_json_schemas_are_versioned_and_closed() -> None:
    documents = observability_schema_documents()
    assert documents.keys() == {"event", "metric"}
    assert documents["event"]["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert documents["event"]["additionalProperties"] is False
    event_properties = documents["event"]["properties"]
    assert isinstance(event_properties, dict)
    fields_schema = event_properties["fields"]
    assert isinstance(fields_schema, dict)
    assert fields_schema["additionalProperties"] is False
    assert "headers" in json.dumps(fields_schema["propertyNames"])
    assert '"maxItems": 64' in json.dumps(fields_schema, sort_keys=True)
    metric_id = documents["metric"]["$id"]
    assert isinstance(metric_id, str)
    assert metric_id.endswith("/schemas/v1/metric.json")
    metric_conditions = documents["metric"]["allOf"]
    assert isinstance(metric_conditions, list)
    assert len(metric_conditions) == len(METRIC_REGISTRY)
    assert '"const": "seconds"' in json.dumps(metric_conditions, sort_keys=True)
    assert '"const": "offline"' in json.dumps(metric_conditions, sort_keys=True)
