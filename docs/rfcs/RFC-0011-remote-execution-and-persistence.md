# RFC-0011: Remote execution and persistence

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

All CUDA work runs through one version-pinned Modal application, shared image policy, and persistent volume adapter. Five importable operational entrypoints submit bounded, idempotent domain operations; local tests use CPU or fakes and continuous integration never calls the remote service.

## Motivation

The PRD forbids a local GPU dependency and explicitly requires infrastructure as code, one shared image, a persistent volume, five runnable entrypoints, GPU fallback, fan-out, and recorded GPU usage. Remote infrastructure must remain an adapter rather than the owner of numerical or ML semantics.

## Goals

- Define one remote app/image/volume and GPU-selection policy.
- Make solve, sweep, train, validate, and serve entrypoints importable and runnable.
- Preserve idempotency, artifact integrity, timeouts, and typed error propagation across remote calls.
- Separate functional fallback from performance claims.
- Record exact runtime/dependency/device provenance and GPU seconds.

## Non-Goals

- Provider portability in v0.1, local CUDA setup, distributed training, or multi-region failover.
- CI-triggered paid GPU workloads.
- Provider console configuration as canonical infrastructure.
- A general-purpose remote workflow scheduler.

## Proposed Design

`infra/app.py` is the only file that creates shared infrastructure:

```python
APP_NAME: Final = "soufflerie"
VOLUME_NAME: Final = "soufflerie-data"
VOLUME_MOUNT: Final = "/data"
PRIMARY_GPU: Final = "L40S"
FALLBACK_GPUS: Final = ("A10G",)

app: modal.App
image: modal.Image
volume: modal.Volume
```

The image starts from Debian slim, installs `uv`, copies `pyproject.toml` and `uv.lock`, performs a frozen full-profile install, then copies the package/infra source. The build records OS base digest, Python patch version, lock digest, package versions, and source revision. No mutable latest tags or unpinned `pip install` commands are allowed. Application name, volume name, mount, GPU policy, concurrency, timeouts, and secret references are code-reviewed constants/settings; secrets values remain provider-managed.

Entry points are thin:

```python
# infra/solve.py
@app.function(image=image, gpu=PRIMARY_GPU, volumes={VOLUME_MOUNT: volume}, timeout=180)
def solve_remote(case_json: bytes, correlation_id: str) -> ArtifactRef: ...

# equivalent typed boundaries
def sweep_remote(config_ref: ArtifactRef) -> SweepSummary: ...
def train_remote(config_ref: ArtifactRef, seed: int) -> ModelBundleRef: ...
def validate_remote(config_ref: ArtifactRef) -> ValidationReportRef: ...
def serve_app() -> ASGIApp: ...
```

Serialized input is canonical JSON bytes with a 16 KiB cap, schema version, and digest. Domain functions parse again inside the worker. Return values are small typed references/summaries, never large arrays. Large artifacts use RFC-0005 storage. Workers commit the volume after atomic artifact publication; readers reload/synchronize before expecting another worker’s commit according to the provider API.

GPU selection is centralized. A user may explicitly select the A10G functional fallback when L40S capacity is unavailable. Automatic retry may move an idempotent fresh operation to A10G only before any committed output and records the device change. A resumed training experiment cannot switch device class because determinism identity would change. Performance gates are reported only for the L40S reference; A10G must satisfy memory/functional gates.

Timeout/concurrency defaults:

| Operation | Timeout | Concurrency |
|---|---:|---:|
| single solve | 180 s | 100 sweep workers or 2 public jobs by distinct functions/policy |
| sweep orchestrator | 2 h | 1 orchestrator; task fan-out 50–100 |
| train seed | 75 min | up to 3 independent seeds, account permitting |
| validation | 30 min | 1 |
| service request | platform max with app-level API deadlines | 1 GPU worker; bounded request queue |

Remote retries follow [`04-error-model.md#retry-policy`](../spec/04-error-model.md#retry-policy). Function-level retries are disabled for non-idempotent orchestration unless the domain idempotency key/state machine makes them safe. Preemption, capacity, timeout, application error, and artifact-integrity failure map to distinct codes.

Operational commands are:

```text
modal run infra/solve.py --config configs/cases/smoke.yaml
modal run infra/sweep.py --config configs/sweeps/mvp-v1.yaml [--n 8]
modal run infra/train.py --config configs/training/fno-v1.yaml
modal run infra/validate.py --config configs/validation/v1.yaml
modal deploy infra/serve.py
```

`--n` is smoke-only and generates a distinct dataset identity; it cannot overwrite or satisfy the 1,000-case release dataset. Every command prints artifact IDs, source revision, device class, wall time, GPU seconds, and final state. `solve.py --smoke` executes a real small Warp kernel and returns its digest.

The remote volume contains no source secrets. Artifact paths follow RFC-0005. Dataset/checkpoint/report readers verify digests and schema before use. Release evidence references full digests, not mutable volume paths alone. Backups/retention are documented; deletion checks published references.

CI tests import entrypoint modules through a stubbed remote SDK and assert definitions/policies, but never call remote functions. Authenticated remote smoke/acceptance is manually dispatched from a clean revision and produces reviewed evidence.

## Alternatives Considered

### Local Docker/CUDA as the primary environment

It would improve portability for GPU owners but violates the explicit no-local-GPU constraint and adds host driver setup. A locked remote image is the v0.1 execution contract.

### Separate image per operation

It can reduce serving or solver image size but multiplies builds and compatibility combinations. One image ensures solver, training, validation, and service share the lock; future measured cold-start pressure may justify separation.

### Implicit GPU fallback list on every function

It improves scheduling but can change determinism and invalidate reference performance claims. Central explicit fallback with operation-specific rules preserves evidence.

### Store large outputs as function return values

It couples retention and size limits to RPC behavior. Persistent content-addressed artifacts with small references are more robust.

## Drawbacks

- One full image is large and may have slow cold builds/starts.
- The implementation initially depends on one provider API.
- Manual remote acceptance can lag pull-request CI.
- Volume consistency/atomicity semantics require adapter-specific testing.

## Migration / Rollout

1. Define the app/image/volume policy and stub-import tests.
2. Build the image and run a small kernel smoke; record the environment manifest.
3. Add solve and 8-case sweep entrypoints with artifact publication.
4. Add train/validate, then service deployment with secrets/limits.
5. Execute and retain full release acceptance from a clean source revision.

Provider SDK or image changes update the lock and environment provenance. Moving providers requires a new RFC and `ExecutionBackend`/`ArtifactStore` adapters; logical artifacts stay unchanged.

## Testing Strategy

- Static/import-test that exactly one app/image/volume policy exists and entrypoints share it.
- Inspect image build definition for frozen lock, no mutable install, no embedded secret, and recorded manifest.
- Use local fakes for serialization caps, typed errors, retries, idempotency keys, timeouts, concurrency, and volume commit/reload.
- Run remote kernel smoke twice and verify functional/digest behavior.
- Run 8-case preemption/resume smoke, including one forced failure.
- Verify A10G fallback is named in provenance and cannot resume an L40S training identity.
- Measure remote GPU seconds and end-to-end times for every release operation.
- Confirm CI workflows contain no remote credentials or invocations.

## Open Questions

None for v0.1. Store atomicity proof and account concurrency are verified during smoke by the infrastructure maintainer; the fixed fallback is serialized coordination/lower fan-out, not reduced scope.

## References

- [`prd.md#60-infra-modal-execution-layer`](../../prd.md#60-infra-modal-execution-layer)
- [`01-architecture.md#system-context`](../spec/01-architecture.md#system-context)
- [`08-performance-budget.md#reference-workloads`](../spec/08-performance-budget.md#reference-workloads)
- [RFC-0001](RFC-0001-package-and-runtime-boundaries.md)
- [RFC-0005](RFC-0005-dataset-artifacts-and-sweep-lifecycle.md)
