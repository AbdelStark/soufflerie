# RFC-0005: Dataset artifacts and resumable sweep lifecycle

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

Each accepted solver case becomes one checksum-verified NPZ run archive plus JSON metadata. A lease-based idempotent sweep state machine publishes these artifacts atomically under deterministic keys, and a Parquet manifest becomes the only immutable training index for exactly 1,000 design points.

## Motivation

Remote fan-out can preempt workers, duplicate attempts, or leave partial writes. Training directly from directory listings would make dataset membership mutable and irreproducible. The PRD requires resumability, shared persistent storage, a Parquet manifest, a sub-2 GiB dataset, and split-by-design-point integrity.

## Goals

- Define safe, versioned run, state, and manifest schemas.
- Resume or retry interrupted work without duplicating or silently replacing cases.
- Admit only numerically valid, digest-matching artifacts to the dataset.
- Preserve exact design-point split membership and dataset lineage.
- Support local filesystem and remote-volume stores through one artifact contract.

## Non-Goals

- A general database, distributed transaction service, or streaming dataset.
- Snapshot-level samples or arbitrary external data ingestion.
- Public distribution of the full dataset inside the wheel.
- Silent partial datasets when the 1,000-case release requirement is unmet.

## Proposed Design

The artifact root uses content- and role-specific paths:

```text
soufflerie/v1/
  sweeps/<config_digest>/
    design.json
    state/<case_id>.json
    attempts/<case_id>/<attempt_id>/diagnostics.json
  runs/<case_id>/<run_digest>/
    fields.npz
    metadata.json
    COMMITTED
  datasets/<dataset_id>/
    manifest.parquet
    metadata.json
    statistics.json
    COMMITTED
```

`COMMITTED` is written last and contains the metadata digest. Readers ignore any root without a valid commit marker. Publication stages to an attempt-specific temporary directory, fsyncs where supported, verifies members, and atomically renames or uses store-native commit semantics. Keys derive only from validated lowercase identifiers.

`fields.npz` is uncompressed or ZIP-deflated NumPy data with `allow_pickle=False` and fixed members:

```text
u_mean        float16[320,256]
v_mean        float16[320,256]
rho_mean      float16[320,256]
sdf           float16[320,256]
obstacle_mask uint8[320,256]
force_steps   int64[N]
cd_history    float32[N]
cl_history    float32[N]
```

Downsampling from canonical solver fields is deterministic fp64 `2x2` area averaging followed by one fp32 cast for continuous fields. Mask nearest-center indices use round-to-nearest-even over an endpoint-preserving source-grid linspace. SDF is recomputed from geometry at output resolution rather than averaged. Scalar labels and all provenance remain in metadata. Dataset field casts to fp16 occur only after numerical gates use fp32 solver output; per-member max and mean absolute round-trip quantization errors are recorded.

```python
class RunMetadata(BaseModel):
    schema_version: Literal[1] = 1
    case_id: str
    design_id: str
    split: Split
    case: CaseConfig
    cd: float
    cl_mean: float
    strouhal: float | None
    diagnostics: SolverDiagnostics
    field_members: dict[str, ArrayDescriptor]
    quantization: dict[str, QuantizationStatistic]
    provenance: Provenance
    fields_sha256: str
    artifact_digest: str

class ManifestRow(BaseModel):
    schema_version: Literal[1] = 1
    dataset_id: str
    case_id: str
    design_id: str
    split: Split
    aspect_ratio: float
    rotation_deg: float
    scale: float
    reynolds: float
    run_uri: str
    run_digest: str
    bytes: int
    cd: float
    cl_mean: float
    strouhal: float | None
    solver_valid: Literal[True]
```

The run digest covers fields bytes, logical metadata, and stable provenance. Physical paths, attempt IDs, start/completion timestamps, and GPU timing remain evidence but are excluded from logical identity, so deterministic duplicate work can be recognized across attempts. Parquet columns have explicit Arrow types and no inferred object columns. Rows sort by `design_id`; row group size is fixed at 256; writer library/version and schema fingerprint are metadata. `run_uri` is artifact-root relative, never a local absolute path or credential-bearing URL.

Sweep state uses:

```python
class CaseState(BaseModel):
    schema_version: Literal[1] = 1
    sweep_digest: str
    case_id: str
    state: Literal["pending", "running", "succeeded", "failed"]
    revision: int
    attempt: int
    attempt_id: str | None
    lease_owner: str | None
    lease_expires_at: datetime | None
    run_digest: str | None
    error_code: str | None
    updated_at: datetime
```

Workers claim pending or expired-running cases with compare-before-write semantics and a 10-minute renewable lease. `revision` is monotonic compare-before-write evidence, while `attempt_id` fences stale workers from renewal or completion. Each attempt has a unique random ID used only for lease/publication isolation, never dataset identity. Transient remote errors retry up to three attempts; deterministic configuration, stability, integrity, or invariant failures become terminal. A successful state is immutable and is committed only after the deterministic run root verifies. If another successful artifact exists, matching full digests make the attempt a verified no-op; mismatches raise `ArtifactIntegrityError` and stop the sweep.

The orchestrator maps cases with concurrency configured between 50 and 100, but reduces batch size on provider capacity errors. It never reduces design count. A sweep summary reports counts by state/error, retries, wall time, GPU seconds, and estimated bytes. Resume reads states and verifies every `succeeded` artifact before skipping it.

Manifest publication requires all 1,000 intended design points in successful state; exactly `600/200/200` rows; no duplicate case/design/run digest; all expected schema/dtype/shape/gates; total payload `<2 GiB`; and consistent config/solver revisions. Dataset ID is computed before writing rows using canonical logical row content, excluding physical remote URIs and timestamps. Manifest updates create a new dataset ID.

The builder receives exactly 1,000 explicit `ArtifactRef` values from the
successful sweep plan and opens every parent through the run artifact store.
It never lists `runs/` to infer membership. Standalone Parquet validation
recomputes schema, row, split, size, and logical identity gates; continued
availability of external parent archives is a separate store-level check.

## Alternatives Considered

### One large Zarr/HDF5 dataset

Chunked access would be efficient, but many concurrent writers require additional synchronization and corruption recovery. Per-run immutable files make fan-out and retry ownership explicit; manifest-driven reading provides sufficient scale for 1,000 samples.

### One NPZ shard containing many runs

It reduces file count but makes a failed worker rewrite or coordinate a shared shard. One run per archive aligns atomicity with the shippable work unit.

### Directory listing as dataset discovery

It is convenient but mutable, unordered, and cannot guarantee split or provenance integrity. Only the published manifest is authoritative.

### Provider-native function result storage

It couples data identity/retention to execution semantics and does not satisfy versioned release artifacts. The volume-backed artifact protocol remains portable to local tests.

## Drawbacks

- One thousand small roots and state files add metadata overhead.
- Cross-process compare-before-write depends on the store adapter's documented atomic primitive.
- NPZ offers limited partial reads; each curated run is small enough for v0.1.
- Requiring complete success can delay dataset publication for a few pathological cases.

## Migration / Rollout

1. Implement local `ArtifactStore`, schema codecs, safe NPZ reader/writer, and integrity tests.
2. Implement state transitions, leases, retries, resume, and fault-injection tests locally.
3. Add remote-volume adapter and miniature 8-case smoke sweep.
4. Execute canonical 1,000-case sweep, validate every run, and publish manifest/statistics.
5. Freeze dataset metadata/digest before any model training.

Run or manifest schema changes create new artifact roots and readers; no in-place mutation is allowed.

## Testing Strategy

- Round-trip every array/member and reject pickle, missing/extra member, wrong dtype/shape, non-finite value, and digest mismatch.
- Fault-inject failures before/after each publication step and prove readers see either old committed state or new committed state, never partial state.
- Simulate lease expiry, duplicate attempts, matching/divergent duplicate outputs, capacity backoff, and terminal errors.
- Verify resume re-hashes succeeded artifacts and submits only incomplete cases.
- Validate manifest Arrow schema, stable row order, exact split counts, uniqueness, total bytes, and relative URI policy.
- Compare dataset IDs across physical roots and timestamps; logical equality must produce the same ID.
- Run the 8-case remote smoke before the full sweep and retain summary evidence.

## Open Questions

None for v0.1. Store-specific atomicity must be proven by the infrastructure maintainer before full fan-out; failure selects a serialized coordinator adapter without changing logical schemas.

## References

- [`prd.md#62-datagen-sweep-runner`](../../prd.md#62-datagen-sweep-runner)
- [`03-data-model.md#data-invariants`](../spec/03-data-model.md#data-invariants)
- [`06-security.md#artifact-safety`](../spec/06-security.md#artifact-safety)
- [RFC-0004](RFC-0004-experiment-config-and-design.md)
- [RFC-0011](RFC-0011-remote-execution-and-persistence.md)
