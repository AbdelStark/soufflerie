# RFC-0012: Observability, provenance, and security

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

Soufflerie uses schema-versioned JSON events, stable offline/online metrics, end-to-end correlation IDs, content-linked provenance, centralized redaction, safe artifact readers, and explicit public GPU-budget controls. Integrity failures close readiness; validation failures remain observable and serviceable.

## Motivation

Numerical and ML evidence crosses local commands, remote workers, artifacts, service responses, UI, and release docs. Without consistent identity and telemetry, a passing report can be detached from its actual dataset/model. Public solve work also introduces credential, cost, and denial-of-budget risks identified in [`06-security.md#threat-model`](../spec/06-security.md#threat-model).

## Goals

- Correlate operations and artifacts without logging sensitive request/infrastructure data.
- Make time, GPU use, numerical diagnostics, training, validation, and service outcomes measurable.
- Enforce lineage and integrity at every artifact load/startup boundary.
- Prevent untrusted serialization, path, request, and public-budget abuse.
- Define safe degraded behavior for red validation and infrastructure failures.

## Non-Goals

- A hosted monitoring vendor requirement, user analytics, or long-term client tracking.
- Cryptographic signing infrastructure beyond release provenance/checksums in v0.1.
- User authentication or confidential result storage.
- Automatic remediation of numerical or validation failures.

## Proposed Design

`soufflerie.observability` owns JSON serialization, event-name registry, correlation context, metric definitions, timer helpers, and redaction. All components emit the `Event` schema from [`05-observability.md#structured-events`](../spec/05-observability.md#structured-events). Unknown fields under `fields` must be JSON scalars or bounded lists; nested arbitrary objects are rejected to keep redaction auditable.

```python
class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...

@contextmanager
def operation_context(
    event: str,
    *,
    correlation_id: str | None = None,
    identities: ArtifactIdentities = ArtifactIdentities(),
) -> Iterator[OperationRecorder]: ...

def redact(value: JsonValue, *, key: str | None = None) -> JsonValue: ...
```

IDs originate at local CLI command or HTTP admission as UUIDv7 and propagate explicitly to remote calls. They do not participate in artifact identity. Domain functions receive an `EventSink`/recorder dependency with a no-op default; they never configure global logging or monitoring clients.

Metrics follow [`05-observability.md#metrics`](../spec/05-observability.md#metrics). Offline workers write schema-versioned metric JSONL alongside operation summaries; the deployed service exposes only an internal/non-public metrics integration configured by the operator. High-cardinality artifact/case/job IDs stay in events. Timer helpers synchronize GPU work when measuring kernel/forward durations and separately capture queue/I/O.

Provenance uses:

```python
class Provenance(BaseModel):
    schema_version: Literal[1] = 1
    source_revision: str
    source_dirty: bool
    python_version: str
    lock_sha256: str
    packages: dict[str, str]
    os: str
    architecture: str
    device_class: str
    dtype_policy: str
    config_sha256: str
    parent_sha256: dict[str, str]
    seeds: tuple[int, ...]
    deterministic: bool
    started_at: datetime
    completed_at: datetime
    gpu_seconds: float
```

Release artifacts require clean source, full 40-character revision, lock digest, allowlisted package versions, all direct parents, and nonnegative timings. Artifact validators build a directed acyclic lineage graph and reject missing parents, cycles, type-incompatible parents, or report/model/dataset mismatch. Content digests cover canonical metadata and member digests, excluding only self-digest and commit timestamp.

Artifact readers use allowlisted formats and schemas. JSON has byte/depth/key limits; NPZ disables pickle and validates central-directory/member sizes before extraction; Parquet projects fixed columns and row limit; model weights use safe tensor loading with declared shape/byte caps. Resolved local paths must remain beneath the artifact root. No request accepts artifact path, URI, URL, archive, or checkpoint.

Settings read named environment variables through strict configuration. Redaction recursively replaces values for case-insensitive sensitive keys and exact configured secret values, strips URL query/userinfo, and normalizes filesystem paths. Exception logging uses type/code plus sanitized message; stack traces are retained only in access-controlled operational logs after filtering and never returned over HTTP.

Public admission implements token buckets using privacy-preserving per-client keys plus global counters: 60 prediction requests/minute/client, two solve requests/hour/client, 20 solves/day global, and concurrency/queue caps from RFC-0009. State is bounded and expires. Reverse-proxy client address trust is disabled unless an allowlisted proxy chain is configured. A `SOUFFLERIE_SOLVE_ENABLED=false` kill-switch and daily GPU-second ceiling close solve admission. They do not alter prediction validation status.

Readiness state follows a fixed precedence:

1. artifact/schema/digest/lineage mismatch -> not ready;
2. dependency/device/model warmup failure -> not ready;
3. daily solve budget exceeded -> prediction ready, solve closed;
4. validation red -> ready with `validation_status=red`;
5. all checks pass -> ready with actual validation status.

Security and performance events never include source address, raw token-bucket key, headers, cookies, environment variables, account/workspace ID, or absolute artifact paths. The public health schema is allowlisted independently of internal metrics.

## Alternatives Considered

### Standard free-form Python logging only

It is convenient but cannot reliably join artifacts, validate fields, or enforce redaction. Structured schema events can still bridge to standard logging sinks.

### Use filenames as provenance

Paths are mutable and environment-specific. Content digests and explicit parent identities are portable and verifiable.

### Reject all service requests when validation is red

It confuses model-integrity failure with honest model-performance failure and defeats the educational behavior. Red remains ready but unavoidable in responses/UI.

### Store raw client IP for rate limiting

It simplifies debugging but retains unnecessary personal data. A short-lived keyed hash meets bounded abuse control with less exposure.

## Drawbacks

- Strict schemas add instrumentation code and can reject ad hoc diagnostics.
- Content hashing and safe preflight add I/O and startup latency.
- In-memory public rate limits are per deployment instance and reset on restart.
- No artifact signatures means repository/provider account compromise remains outside the v0.1 trust model.

## Migration / Rollout

1. Land event/provenance/redaction schemas with property and sentinel tests.
2. Integrate solver/datagen/training/validation timers and artifact parent identities.
3. Add safe readers and lineage verifier before remote artifact consumption.
4. Add service rate/budget limits, readiness precedence, and kill-switch.
5. Add secret/dependency scans and release evidence validation.

Schema changes increment event/provenance/artifact versions and update all producers/consumers together.

## Testing Strategy

- Schema/property-test all event/metric/provenance types and timestamp/timing invariants.
- Plant sentinel secrets in nested structures, URLs, exceptions, settings, and headers; assert no sink/output contains them.
- Tamper every artifact member/parent edge and assert load/startup failure.
- Fuzz JSON/NPZ/Parquet/safe-weight readers with oversize, truncation, traversal, bad schema, and executable-payload attempts.
- Use fake clocks/clients to test token refill, expiry, proxy handling, global daily budget, concurrency, and kill-switch.
- Enumerate readiness precedence combinations including red-validation-ready.
- Validate cost summary aggregation against hand-computed operation records.

## Open Questions

None for v0.1. Multi-instance rate limiting, signatures, or user identity require a future security RFC owned by the security/service maintainer.

## References

- [`05-observability.md`](../spec/05-observability.md)
- [`06-security.md`](../spec/06-security.md)
- [`04-error-model.md#failure-matrix`](../spec/04-error-model.md#failure-matrix)
- [RFC-0005](RFC-0005-dataset-artifacts-and-sweep-lifecycle.md)
- [RFC-0008](RFC-0008-validation-and-release-gates.md)
- [RFC-0009](RFC-0009-inference-and-solve-api.md)
