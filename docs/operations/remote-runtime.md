# Locked remote runtime

Soufflerie uses one Modal application (`soufflerie`), one persistent volume
(`soufflerie-data` mounted at `/data`), and one image definition in
`infra/app.py`. Operational entrypoints import those objects; they must not
create provider resources independently.

The image starts from the amd64 Python 3.11.14 slim-bookworm manifest pinned in
`infra/policy.py`. Modal copies uv 0.12.8, installs the `solver`, `ml`, `remote`,
`serve`, and `viz` profiles from `uv.lock` with frozen resolution, then installs
the local package without resolving dependencies. Source, configuration,
schemas, and infrastructure code are copied into the image only after the
third-party dependency layer.

During the build, `runtime-build.json` records the exact base-image reference,
Python and uv versions, full lock digest, full source revision, dirty flag, and
every installed distribution/version. The manifest validates its own SHA-256
digest. Remote evidence must come from `source_dirty=false`; rebuilding after a
commit changes the image identity and recorded revision.

## Authentication and secrets

Authenticate with a local Modal profile. `MODAL_ENVIRONMENT` and
`SOUFFLERIE_REMOTE_GPU` are the only local runtime selectors in `.env.example`;
neither is a credential. Secret values stay in the provider secret store. The
reviewed future secret reference is `soufflerie-runtime`, but the kernel smoke
does not attach it because no secret is needed.

The default and performance-reference device is L40S. A10G can be selected
explicitly as a functional fallback:

```bash
SOUFFLERIE_REMOTE_GPU=A10G \
  uv run --extra remote modal run infra/solve.py --smoke
```

A10G output must identify A10G and cannot satisfy L40S timing gates or resume a
device-bound training run. There is no automatic provider retry or implicit
device fallback.

## Authenticated kernel smoke

Run the acceptance smoke only from a clean commit:

```bash
uv run --extra remote modal run infra/solve.py --smoke
```

The command builds or reuses the one locked image, mounts the shared volume,
executes two independent two-step fp32 Warp D2Q9 runs on `cuda:0`, and emits a
schema-v1 JSON record. The two state digests and build-manifest digests must be
identical. Each run reports requested and resolved device identities, source,
lock, image, volume, artifact digest, wall time, and GPU seconds.

The kernel smoke does not publish a solver artifact. Use the domain entrypoints
below when a durable run is required.

## Idempotent solve

Run one checked-in case from a clean commit:

```bash
uv run --extra remote modal run infra/solve.py \
  --config configs/cases/cylinder-re100-v1.yaml
```

The local adapter validates the YAML and submits a canonical schema-v1 JSON
envelope capped at 16 KiB. The worker reparses that envelope, checks its source
revision and lock digest against the image manifest, reloads the volume, and
claims a ten-minute fenced lease. Provider retries remain zero. A successful
worker atomically publishes and verifies the run under
`/data/soufflerie/v1/runs/<case_id>/<run_digest>`, commits the artifact, records
success, commits the state, and returns only an `ArtifactRef`.

Standalone solves use the `standalone-solve-v1` namespace and a non-release
test role. Their design identity is deliberately unable to masquerade as a
point from the canonical 1,000-case design.

## Eight-case resume smoke

The authenticated sweep acceptance command is:

```bash
uv run --extra remote modal run infra/sweep.py \
  --config configs/sweeps/mvp-v1.yaml \
  --n 8
```

`--n 8` means the fixed `remote-smoke-v1` stratified design. It has its own
request/sweep digest bound to the canonical config, source revision, lock
digest, and selected device class. It is not the RFC-0004 maximin LHS design,
cannot create a release dataset, and cannot satisfy the 1,000-case gate.

On a fresh identity, the command deliberately records one retryable failure
before numerical execution. The orchestrator verifies the other successful
artifacts, submits only the missing case on its next fenced attempt, and checks
that every earlier full run digest is unchanged. The terminal `SweepSummary`
reports state counts, initial and resumed case IDs, attempts, retries, artifact
references, bytes, wall time, aggregate GPU seconds, device, source revision,
and an evidence digest. Rerunning the same command after success executes zero
cases and verifies all eight committed artifacts before skipping them.

If a real provider preemption leaves a live lease, the command reports an
incomplete summary instead of duplicating work. Rerun after the ten-minute
lease expires; at most three domain attempts are allowed. Changing the source
revision, lock, config, or device creates a different smoke identity rather
than resuming incompatible work.

The release command without `--n 8` intentionally fails until RFC-0004's
frozen 1,000-point design is implemented. No reduced run may silently become a
release dataset.

## Persistence and retention

The shared volume contains immutable requests, fenced sweep state, and
content-addressed run artifacts; it contains no source credentials. Readers
reload before consuming another worker's commit and verify schema, byte count,
full digest, commit marker, and fixed NPZ members. Published run roots must not
be edited in place. Back up `/data/soufflerie/v1` using provider-managed volume
snapshots or an integrity-preserving export before a release. Do not delete a
run referenced by sweep state or a published dataset manifest; v0.1 has no
automatic garbage collector.

## Failure and cost controls

- Solve functions time out after 180 seconds, permit at most 100 workers, and
  have provider retries disabled. Domain state permits only the two named
  retryable classes: capacity exhaustion and remote execution failure.
- The kernel smoke, staging function, and sweep orchestrator each use one
  container. The orchestrator times out after two hours. Other operation limits
  remain centralized as reviewed constants in `infra/policy.py`.
- CI uses a stubbed Modal module and never authenticates or calls the provider.
- A missing profile, unavailable GPU, dirty source image, non-CUDA resolution,
  manifest mismatch, serialization violation, artifact mismatch, or kernel
  mismatch fails explicitly.
