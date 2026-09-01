# Run artifacts and local publication

Every numerically accepted solver result is curated into one immutable run root before it can
enter sweep state or a dataset manifest. `soufflerie.datagen.LocalRunArtifactStore` implements
the local reference adapter for RFC-0005; the remote-volume adapter must preserve the same
logical bytes, identities, and commit-marker behavior.

## Curation boundary

`curate_solver_result(case, result)` accepts only the canonical `256 x 512` fp32 solver grid and
a matching `case_id`. It produces the fixed `128 x 256` output members:

```text
u_mean        float16[128,256]  lattice_velocity
v_mean        float16[128,256]  lattice_velocity
rho_mean      float16[128,256]  lattice_density
sdf           float16[128,256]  lattice_distance
obstacle_mask uint8[128,256]    dimensionless, values 0 or 1
force_steps   int64[N]          step
cd_history    float32[N]        dimensionless
cl_history    float32[N]        dimensionless
```

Continuous solver fields are accumulated as fp64 means over each exact `2 x 2` area and cast
once to fp32. SDF is recomputed from the shared geometry contract directly on the output grid;
it is never averaged. The mask uses nearest-center samples at indices obtained by
round-to-nearest-even over an endpoint-preserving source-grid linspace. This convention avoids
an unstated half-cell tie rule and remains deterministic.

Only after those fp32 operations are complete are `u_mean`, `v_mean`, `rho_mean`, and `sdf`
cast to fp16. Metadata records max and mean absolute round-trip error for each cast. A cast that
produces a non-finite value is rejected. Histories retain their solver dtypes.

## Deterministic codec and identity

`fields.npz` contains exactly the eight members above, in that order. Each NPY member is
C-contiguous and pickle-free. The ZIP container is uncompressed and uses fixed names,
permissions, ordering, and timestamps, so identical curated arrays produce identical bytes.
The bounded reader validates the central directory and every NPY header before NumPy allocates
an array, enforces the run-specific 16 MiB archive ceiling even when generic reader limits are
larger, then returns read-only arrays.

`RunMetadata` binds the case/design/split, scalar labels, diagnostics, exact member descriptors,
quantization evidence, full provenance, and `fields.npz` SHA-256. `artifact_digest` hashes that
logical content plus stable provenance. Attempt IDs, physical paths, start/completion timestamps,
and GPU timing are excluded from logical identity but retained in the metadata file. Thus an
identical deterministic rerun has the same run digest, while any field, label, configuration,
source, dependency, seed, or device-policy change creates a different digest.

The checked-in JSON Schema is [`run-metadata.json`](../../schemas/v1/run-metadata.json). Readers
reject unsupported schema versions, unknown fields, wrong member descriptors, digest mismatch,
truncation, oversized files, object arrays, traversal names, and non-finite content.

## Atomic local publication

Committed roots use:

```text
runs/<case_id>/<artifact_digest>/
  fields.npz
  metadata.json
  COMMITTED
```

Publication writes into an attempt-specific directory beneath `.staging/`, fsyncs each member,
reopens both payloads through the bounded readers, and writes `COMMITTED` last. The marker is one
lowercase SHA-256 digest of the exact metadata bytes. Only then is the complete directory renamed
under `runs/` and its parent fsynced. Failures before rename leave no visible run; a failure after
rename leaves a complete recoverable run. Matching publication is a verified no-op.
Fixed store prefixes and validated identifier directories must be real directories; symlinked or
non-directory components are rejected before publication.

Consumers must open a run through its `ArtifactRef`. `open_run` requires the marker, verifies the
metadata digest, checks case/path/logical identities, verifies the fields digest and exact NPZ
contract, and checks the reference byte count. Directory listing is never an admission signal.

Run the focused contract and fault-injection suite with:

```bash
uv run pytest tests/artifacts/test_run_codec.py tests/artifacts/test_local_store.py
```

The lower-level hostile-input guarantees are documented in
[`docs/security/artifacts.md`](../security/artifacts.md).
