# Structured observability

Soufflerie emits vendor-neutral, schema-v1 events and metrics through explicit
dependencies. Domain code does not configure Python logging or a monitoring
client. The no-op sinks are the defaults, so instrumentation calls remain safe
in local libraries, tests, command-line processes, and remote workers.

The checked-in [`event.json`](../../schemas/v1/event.json) and
[`metric.json`](../../schemas/v1/metric.json) files are the machine-readable
contracts. Event names, metric names, units, and labels are immutable registries
in `soufflerie.observability`; adding or changing one requires a coordinated
schema/RFC revision.

## Instrument an operation

Create a UUIDv7 correlation ID at the first local command or HTTP admission
boundary, then propagate that string explicitly to remote calls. Artifact IDs
do not derive from the correlation ID.

```python
from soufflerie.observability import (
    ArtifactIdentities,
    JsonlEventSink,
    JsonlMetricSink,
    operation_context,
)

with operation_context(
    "case_started",
    correlation_id=incoming_correlation_id,
    identities=ArtifactIdentities(case_id=case.case_id, job_id=job_id),
    event_sink=JsonlEventSink(operation_root / "events.jsonl"),
    metric_sink=JsonlMetricSink(operation_root / "metrics.jsonl"),
) as operation:
    with operation.timer.segment("queue"):
        admitted_case = queue.get()
    with operation.timer.segment("compute"):
        result = solve(admitted_case)
    with operation.timer.segment("io"):
        publish(result)
    operation.annotate(steps_completed=result.diagnostics.steps_completed)
    operation.metric(
        "solver_run_seconds",
        result.provenance.gpu_seconds,
        unit="seconds",
        labels={"device_class": "NVIDIA L40S", "outcome": "succeeded"},
    )
```

`case_started` automatically terminates as `case_completed` or `case_failed`.
`solve_admitted` terminates as `solve_terminal` with an explicit outcome. A
registered event without a domain-specific lifecycle uses
`operation_completed` or `operation_failed`. Exceptions are re-raised after a
terminal event captures only their type, stable public code, retryability, and
sanitized bounded message; stack frames are never serialized by this module.

Use `OperationTimer(synchronize=...)` for GPU work. The callback runs before
and after each compute segment, so pending asynchronous kernels are included in
compute time. Queue and I/O spans do not synchronize the GPU. Segments cannot
overlap, and the sum of compute, queue, and I/O cannot exceed monotonic wall
time. Pass the relevant framework's device synchronization callback rather than
a global device fallback.

## Sinks and capture behavior

`NOOP_EVENT_SINK` and `NOOP_METRIC_SINK` avoid conditional instrumentation.
`InMemoryEventSink` and `InMemoryMetricSink` provide bounded, thread-safe local
capture. `JsonlEventSink` and `JsonlMetricSink` append one canonical UTF-8 JSON
record and flush on every call; the caller owns the output directory and its
retention policy. Sink capacity exhaustion and filesystem errors are explicit
and are not silently dropped.

Serialization is the final trust boundary. Every retaining or JSONL sink makes
a redacted copy before capture, even if a caller constructed an unredacted
schema-valid record. The caller's immutable record is unchanged.

## Redaction policy

`Redactor` recursively processes bounded JSON-native values without reading the
environment:

- keys containing `token`, `secret`, `password`, `authorization`, `cookie`, or
  `api_key`, plus configured extensions, become `[REDACTED]`;
- configured exact secret values are replaced wherever they occur in strings;
- URLs lose user information, query strings, and fragments;
- paths beneath a configured artifact root become artifact-relative, while
  other absolute or traversing paths become `[EXTERNAL_PATH]`;
- nested arbitrary event objects, headers, source/IP addresses, request bodies,
  environment dumps, and remote account/workspace identifiers are rejected by
  the event schema.

Configure known secret values from the already-loaded settings object. Do not
ask observability code to enumerate process environment variables. When adding
a sink, call `event_json` or `metric_json` at the capture boundary instead of
serializing model dictionaries directly.

## Metrics and cardinality

Aggregate metric labels must exactly match the central registry. Case, dataset,
model, and job identities belong in events. `solver_mass_drift_ratio` is the
single declared exception: it is accepted only with `scope="offline"`, where
`case_id` is a report-table key and never an online monitoring label.

`validation_metric` carries the metric's declared unit; every other metric has
the fixed unit in [`05-observability.md`](../spec/05-observability.md#metrics).
Operational logs follow provider retention and are not release evidence.
Checked-in validation summaries, benchmark summaries, and release provenance
remain the durable records.
