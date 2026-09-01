# RFC-0009: Inference and solve API

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

The service exposes strict synchronous prediction, bounded asynchronous reference-solve jobs with server-sent status events, and identity-rich health through five HTTP paths. Prediction remains available for a red but integrity-valid model and carries validation and per-case consistency state in every response.

## Motivation

The PRD requires sub-100 ms prediction, raw and rendered fields, drag, consistency flags, real solver comparison, streamed status, and model/validation health. A long-running solver cannot safely share the synchronous prediction request lifecycle, and a red validation state must survive every API boundary.

## Goals

- Define closed request/response schemas and stable error/status semantics.
- Keep warm prediction latency separate from remote solve lifecycle.
- Bound queues, payloads, execution, retention, and retries.
- Propagate model, dataset, report, artifact, and correlation identities.
- Make reconnecting solve progress and terminal results deterministic for clients.

## Non-Goals

- User accounts, API keys, durable job history, arbitrary file upload, or batch endpoints.
- Supported prediction outside the training domain.
- WebSocket bidirectional control.
- Engineering-ground-truth language for solver output.

## Proposed Design

Pydantic v2 models use strict types, `extra="forbid"`, finite floats, and schema version 1:

```python
class ShapeRequest(BaseModel):
    aspect_ratio: Annotated[float, Field(ge=0.5, le=1.0, allow_inf_nan=False)]
    rotation_deg: Annotated[float, Field(ge=0.0, le=30.0, allow_inf_nan=False)]
    scale: Annotated[float, Field(ge=0.75, le=1.25, allow_inf_nan=False)]

class PredictionRequest(BaseModel):
    schema_version: Literal[1] = 1
    shape: ShapeRequest
    reynolds: Annotated[float, Field(ge=40.0, le=300.0, allow_inf_nan=False)]

class EncodedArtifact(BaseModel):
    media_type: Literal["image/png", "application/x-npz"]
    encoding: Literal["base64"]
    data: str
    sha256: str
    bytes: int

class ConsistencyFlags(BaseModel):
    head_field_gap_pct: float
    head_field_gap: Literal["green", "red"]
    divergence_ratio_to_solver_baseline: float
    divergence: Literal["green", "red"]
    obstacle_velocity_ratio: float
    obstacle_compliance: Literal["green", "red"]
    ood: Literal[False]

class PredictionResponse(BaseModel):
    schema_version: Literal[1]
    correlation_id: str
    case_id: str
    fields_png: EncodedArtifact
    fields_npz: EncodedArtifact
    cd_head: float
    cd_field: float
    consistency: ConsistencyFlags
    validation_status: Literal["green", "red"]
    model_id: str
    dataset_id: str
    report_id: str
    inference_ms: float
    request_ms: float
```

The NPZ contains float32 `u`, `v`, `rho`, float32 normalized SDF, and uint8 mask at `[320,256]` with `allow_pickle=False`. The PNG is the canonical three-panel velocity magnitude, pressure proxy, and vorticity rendering from RFC-0010. Payload encoding refuses a combined response over 4 MiB.

Prediction flow is validate -> derive geometry -> queue admission -> preprocess -> forward -> de-normalize -> field Cd/metrics -> deterministic render/encode -> response. One loaded `ModelBundle` instance is read-only. A bounded worker semaphore serializes GPU forward unless profiling proves safe configured concurrency; queue capacity defaults to eight. Queue wait contributes to `request_ms` but not `inference_ms`. No automatic device fallback occurs after startup.

```python
class SolveAccepted(BaseModel):
    schema_version: Literal[1]
    job_id: str
    case_id: str
    state: Literal["queued"]
    status_url: str
    events_url: str
    expires_at: datetime

class SolveStatus(BaseModel):
    schema_version: Literal[1]
    job_id: str
    case_id: str
    state: Literal["queued", "running", "succeeded", "failed", "expired"]
    progress: Annotated[float, Field(ge=0, le=1)]
    result: SolveResultResponse | None
    error: PublicError | None
    sequence: int
```

`POST /solve` reuses `PredictionRequest`, enforces security quotas, creates a server-generated UUIDv7 job ID, stores bounded status, and submits exactly one remote solver call using `job_id` as an idempotency key. It returns `202`; capacity rejection returns `429` or `503` without a job. Solver result includes reference PNG/NPZ, Cd/Cl/St, prediction comparison/errors, provenance digest, and timings.

SSE events serialize `id: <sequence>`, `event: <state-change>`, and JSON `data`. Sequence begins at 1 and increases for each persisted transition/progress snapshot. `Last-Event-ID` replays retained later events; an impossible future cursor returns `409`. Heartbeat comments every 15 seconds contain no data. Disconnect never cancels a job. Terminal state is immutable and kept 60 minutes, then status becomes `expired` without deleting content-addressed solver artifacts under normal retention.

The job manager interface isolates infrastructure:

```python
class SolveJobBackend(Protocol):
    async def submit(self, job_id: str, case: CaseConfig) -> None: ...
    async def status(self, job_id: str) -> SolveStatus: ...
    async def events(self, job_id: str, after: int) -> AsyncIterator[SolveEvent]: ...
```

`GET /health` returns HTTP `200` with `liveness="ok"`; readiness may be `ready` or `not_ready`. Prediction requests receive `503` when bundle/report integrity, device, or warmup fails. Red validation alone is ready. Health returns package version, schema version, model/dataset/report IDs, validation status, device class, and last successful readiness check; it excludes account, host, paths, and credentials.

All error responses follow [`04-error-model.md#http-errors`](../spec/04-error-model.md#http-errors). Unexpected exceptions are logged after redaction and produce a generic body. OpenAPI JSON is a checked-in/golden-tested public artifact.

## Alternatives Considered

### Synchronous `/solve`

It is simpler but exceeds normal HTTP deadlines, couples disconnect to work, and makes bounded status opaque. Asynchronous jobs isolate expensive execution and enable comparison progress.

### WebSockets

They support bidirectional communication but the client only needs ordered server events. SSE reconnect semantics and standard HTTP tooling are sufficient.

### Object-store URLs instead of embedded payloads

They reduce response size but require signed URL lifecycle/CORS and expose an additional trust boundary. v0.1 bounded fields fit the 4 MiB response budget.

### Reject serving a red model

That hides the exact failure behavior the project exists to demonstrate. Integrity failures block readiness; validation failures remain visible in successful responses.

## Drawbacks

- Base64 increases field payload size by roughly one third.
- Ephemeral job status can disappear after service restart, although content-addressed result artifacts remain.
- Public asynchronous solves add budget-abuse controls and operational complexity.
- Unversioned paths rely on explicit response schema versions for v0.1.

## Migration / Rollout

1. Implement strict schemas, error mapping, and health with a fake predictor/job backend.
2. Add prediction pipeline, encoding limits, consistency computation, and latency instrumentation.
3. Add job state machine, SSE replay, timeout, and retention tests against a local fake.
4. Connect the remote backend and run one reference-solve integration.
5. Mount the UI and deploy behind configured limits and kill-switch.

Breaking schemas add new paths or schema versions with the deprecation policy in `09-release-and-versioning.md`.

## Testing Strategy

- Generate and diff OpenAPI schema; validate every response against it.
- Test each numeric boundary, forbidden extra, bad media type, oversize request/response, and non-finite encoding.
- Assert timing inclusion/exclusion using a fake monotonic clock.
- Load-test queue admission, prediction isolation from solves, and exact capacity status codes.
- Model every legal/illegal job transition, idempotent submission, reconnect replay, heartbeat, timeout, terminal immutability, and expiry.
- Inject bundle/report mismatches, red reports, unavailable devices, and unexpected exceptions.
- Assert all errors/logs are redacted and health returns only allowlisted keys.

## Open Questions

None for v0.1. Durable multi-instance job state or authenticated users require a new service/security RFC owned by the service maintainer.

## References

- [`prd.md#65-api-ui`](../../prd.md#65-api-ui)
- [`02-public-api.md#http-api`](../spec/02-public-api.md#http-api)
- [`06-security.md#service-controls`](../spec/06-security.md#service-controls)
- [RFC-0008](RFC-0008-validation-and-release-gates.md)
- [RFC-0010](RFC-0010-interactive-demo-and-visualization.md)
- [RFC-0011](RFC-0011-remote-execution-and-persistence.md)
