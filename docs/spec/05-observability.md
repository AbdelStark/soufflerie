# Observability

<a id="objectives"></a>
## Objectives

Observability must answer: which inputs and artifacts produced a result, whether numerical and validation gates passed, where time and GPU budget were spent, and why a job failed. It must not expose credentials, infrastructure identity, user network identifiers, or raw serialized requests.

<a id="structured-events"></a>
## Structured events

Application events are one-line JSON with:

```python
class Event(BaseModel):
    schema_version: Literal[1] = 1
    timestamp: datetime
    level: Literal["debug", "info", "warning", "error"]
    event: str
    correlation_id: str
    component: Literal["solver", "datagen", "training", "validation", "service", "infra"]
    case_id: str | None = None
    dataset_id: str | None = None
    model_id: str | None = None
    job_id: str | None = None
    fields: dict[str, JsonScalar]
```

Allowed event names are defined centrally. At minimum: `case_started`, `case_completed`, `case_failed`, `artifact_published`, `sweep_progress`, `epoch_completed`, `checkpoint_selected`, `gate_evaluated`, `prediction_completed`, `solve_admitted`, `solve_terminal`, and `redaction_applied`.

<a id="metrics"></a>
## Metrics

Metric names and units are stable:

| Metric | Unit | Labels |
|---|---|---|
| `solver_run_seconds` | seconds | device_class, outcome |
| `solver_lups` | lattice updates/second | device_class, grid |
| `solver_mass_drift_ratio` | ratio | case_id |
| `sweep_cases_total` | count | state |
| `remote_gpu_seconds_total` | seconds | milestone, device_class, operation |
| `training_epoch_seconds` | seconds | seed |
| `training_loss` | ratio | seed, split, term |
| `validation_metric` | declared per metric | metric, gate_status |
| `prediction_duration_ms` | milliseconds | device_class, validation_status |
| `solve_queue_depth` | count | state |
| `http_requests_total` | count | route, method, status_class |

High-cardinality IDs appear in events/traces, not aggregate metric labels, except offline report metrics where `case_id` is a table key rather than a monitoring label.

<a id="tracing"></a>
## Tracing

One correlation ID spans HTTP admission, prediction or solve execution, artifact publication, and terminal response. Remote invocations propagate it explicitly. Trace spans cover validation, preprocessing, model forward, consistency calculations, encoding, queue wait, solver kernel loop, and upload. Tracing is optional locally but timing fields remain mandatory.

<a id="provenance"></a>
## Provenance

Every solver result, dataset, model, and report records:

- source revision and dirty flag (release artifacts require `false`);
- Python version, dependency-lock digest, framework versions, OS, architecture, device class, and dtype;
- canonical config and digest;
- seed set and determinism mode;
- parent artifact identities and full SHA-256 digests;
- started/completed UTC timestamps and measured GPU seconds where applicable.

The README cost table is generated from milestone-tagged `remote_gpu_seconds_total` records and includes device class and measurement window. Currency estimates are dated and explicitly separate measured usage from provider pricing.

<a id="redaction"></a>
## Redaction

Keys matching `token`, `secret`, `password`, `authorization`, `cookie`, `api_key`, or configured extensions are replaced before serialization. URLs drop query strings and userinfo. Paths are reduced to artifact-relative paths. Request bodies are represented only by validated case fields and their `case_id`; headers, IP addresses, remote account/workspace IDs, and environment dumps are forbidden. Redaction tests use sentinel secrets and fail if any sentinel reaches captured output.

<a id="retention"></a>
## Retention

Checked-in validation reports, benchmark summaries, and release provenance are durable. Remote operational logs follow provider retention and are not a release dependency. Ephemeral solve job events remain available for 60 minutes after terminal state. Dataset and model artifacts are retained through the v0.1 support window; deletion requires confirming no published manifest or release references them.
