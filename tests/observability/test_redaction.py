from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from soufflerie.errors import ConfigurationError
from soufflerie.observability import (
    EXTERNAL_PATH,
    REDACTED,
    ArtifactIdentities,
    Event,
    InMemoryEventSink,
    InMemoryMetricSink,
    JsonlEventSink,
    JsonlMetricSink,
    MetricRecord,
    Redactor,
    event_json,
    metric_json,
    new_correlation_id,
    operation_context,
    redact,
    safe_exception_fields,
)
from soufflerie.schemas import JsonValue

NOW = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
SENTINEL = "sentinel-credential-4fce8307"


def _event(fields: dict[str, object]) -> Event:
    return Event.model_validate(
        {
            "timestamp": NOW,
            "level": "error",
            "event": "operation_failed",
            "correlation_id": new_correlation_id(timestamp=NOW),
            "component": "infra",
            "fields": fields,
        }
    )


def test_recursive_key_value_url_and_configured_extension_redaction() -> None:
    policy = Redactor(
        sensitive_key_extensions=("session",),
        secret_values=(SENTINEL,),
    )
    value: JsonValue = {
        "safe": ["prefix-" + SENTINEL, {"Api-Key": SENTINEL}],
        "authorization_header": "Bearer anything",
        "session_identifier": "session-value",
        "url": f"https://user:{SENTINEL}@example.test/artifact/result?token={SENTINEL}#debug",
    }

    redacted = cast(dict[str, JsonValue], policy.redact(value))
    rendered = json.dumps(redacted)
    assert SENTINEL not in rendered
    assert redacted["safe"] == ["prefix-[REDACTED]", {"Api-Key": REDACTED}]
    assert redacted["authorization_header"] == REDACTED
    assert redacted["session_identifier"] == REDACTED
    assert redacted["url"] == "https://example.test/artifact/result"
    assert redact({"request_headers": {"X-Debug": SENTINEL}}) == {"request_headers": REDACTED}


@given(prefix=st.text(max_size=80), suffix=st.text(max_size=80))
def test_configured_secret_never_survives_any_string_position(prefix: str, suffix: str) -> None:
    value = f"{prefix}{SENTINEL}{suffix}"
    result = redact(value, secret_values=(SENTINEL,))
    assert isinstance(result, str)
    assert SENTINEL not in result


def test_urls_embedded_in_exceptions_drop_userinfo_query_and_fragment() -> None:
    error = ConfigurationError(
        f"fetch https://alice:{SENTINEL}@example.test/private/data?api_key={SENTINEL}#trace failed"
    )
    fields = safe_exception_fields(error, redactor=Redactor(secret_values=(SENTINEL,)))

    assert fields["error_code"] == "CONFIG_INVALID"
    assert fields["retryable"] is False
    assert fields["exception_type"] == "ConfigurationError"
    assert fields["exception_message"] == "fetch https://example.test/private/data failed"
    assert SENTINEL not in json.dumps(fields)
    assert redact("file:///private/tmp/credential.txt") == "file://[EXTERNAL_PATH]"


def test_paths_are_artifact_relative_and_external_paths_are_hidden(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    inside = artifact_root / "dataset" / "manifest.json"
    outside = tmp_path / "settings" / "credentials.json"
    policy = Redactor(artifact_root=artifact_root)

    assert policy.redact(str(inside), key="artifact_path") == "dataset/manifest.json"
    assert policy.redact(str(outside), key="file_path") == EXTERNAL_PATH
    assert policy.redact("../escape/token.txt", key="path") == EXTERNAL_PATH
    assert policy.redact("reports/summary.json", key="path") == "reports/summary.json"
    assert policy.redact("failed at /private/tmp/run/output.json") == f"failed at {EXTERNAL_PATH}"


def test_every_capture_boundary_redacts_before_retaining_or_serializing(tmp_path: Path) -> None:
    event = _event(
        {
            "access_token": SENTINEL,
            "message": f"could not use {SENTINEL}",
            "breadcrumbs": ["safe", SENTINEL],
        }
    )
    policy = Redactor(secret_values=(SENTINEL,))
    memory_sink = InMemoryEventSink(redactor=policy)
    memory_sink.emit(event)
    output = tmp_path / "events.jsonl"
    JsonlEventSink(output, redactor=policy).emit(event)

    captured = memory_sink.events[0]
    assert captured.fields["access_token"] == REDACTED
    assert SENTINEL not in captured.model_dump_json()
    assert SENTINEL not in output.read_text(encoding="utf-8")
    assert SENTINEL not in event_json(event, redactor=policy)
    # Capture is copy-on-write: instrumentation cannot mutate the caller's record.
    assert event.fields["access_token"] == SENTINEL

    metric = MetricRecord(
        timestamp=NOW,
        metric="solver_run_seconds",
        value=1.0,
        unit="seconds",
        labels={"device_class": SENTINEL, "outcome": "succeeded"},
    )
    metric_memory = InMemoryMetricSink(redactor=policy)
    metric_memory.emit(metric)
    metric_output = tmp_path / "metrics.jsonl"
    JsonlMetricSink(metric_output, redactor=policy).emit(metric)
    assert SENTINEL not in metric_memory.metrics[0].model_dump_json()
    assert SENTINEL not in metric_output.read_text(encoding="utf-8")
    assert SENTINEL not in metric_json(metric, redactor=policy)

    recorder_sink = InMemoryEventSink()
    with operation_context(
        "operation_started",
        identities=ArtifactIdentities(job_id=SENTINEL),
        event_sink=recorder_sink,
        redactor=policy,
        utc_clock=lambda: NOW,
    ) as operation:
        operation.annotate(message=f"credential={SENTINEL}")
    assert SENTINEL not in "".join(item.model_dump_json() for item in recorder_sink.events)


def test_redaction_rejects_unbounded_or_invalid_policy_inputs() -> None:
    with pytest.raises(ValueError, match="maximum depth"):
        Redactor(max_depth=1).redact({"outer": {"inner": {"value": "x"}}})
    with pytest.raises(ValueError, match="maximum size"):
        Redactor(max_items=2).redact(["one", "two"])
    with pytest.raises(ValueError, match="non-empty"):
        Redactor(secret_values=("",))
    with pytest.raises(ValueError, match="non-empty"):
        Redactor(sensitive_key_extensions=("",))


def test_redactor_never_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> str:
        raise AssertionError(f"environment access is forbidden: {args!r} {kwargs!r}")

    monkeypatch.setattr("os.getenv", fail_if_called)
    monkeypatch.setattr("os.environ.get", fail_if_called)
    assert redact({"safe": "value", "password": SENTINEL}) == {
        "safe": "value",
        "password": REDACTED,
    }
