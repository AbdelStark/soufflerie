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

## Cylinder acceptance and idempotent solve

Run one checked-in case from a clean commit:

```bash
uv run --extra remote modal run infra/solve.py \
  --config configs/cases/cylinder-re100.yaml
```

That exact canonical path runs the three-grid Re=100 study plus an independent
canonical repeat, then writes the typed JSON and rendered Markdown evidence in
`reports/solver/`. Other checked-in case paths retain the single-solve behavior.
Each worker receives a canonical schema-v1 JSON envelope capped at 16 KiB,
checks its source revision and lock digest against the image manifest, reloads
the volume, and claims a ten-minute fenced lease. Provider retries remain zero.
A successful worker atomically publishes and verifies the run under
`/data/soufflerie/v1/runs/<case_id>/<run_digest>`, commits the artifact, records
success, commits the state, and returns only an `ArtifactRef`.

Cylinder acceptance uses distinct operation identities for coarse, canonical,
fine, and repeat runs, so the determinism gate cannot be satisfied by replaying
one successful state record. Standalone solves use the `standalone-solve-v1`
namespace. Neither identity can masquerade as a point from the canonical
1,000-case design.

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

## Canonical 1,000-case sweep

From a clean commit, omitting `--n` selects only the frozen RFC-0004 maximin
LHS design:

```bash
uv run --extra remote modal run infra/sweep.py \
  --config configs/sweeps/mvp-v1.yaml \
  --output /tmp/soufflerie-sweep-summary.json
```

The request binds all 1,000 exact cases, source revision, lock digest, device,
and configuration under the distinct `canonical-lhs-v1` identity. Modal admits
at most 100 solve workers, so provider capacity may reduce live concurrency but
never the design. Every successful state and run archive is reloaded and
checksum-verified before it is skipped or admitted.

Only a terminal `1000/1000` plan invokes `build_manifest`. That builder opens
all 1,000 explicit run references again, enforces the design, split,
provenance, and sub-2 GiB gates, and atomically publishes
`datasets/<dataset_id>/`. An incomplete or terminally failed sweep returns no
dataset reference. Rerunning the same clean identity resumes only eligible
cases; no invalid sample is replaced.

The digest-bound `SweepSummary` records invocation submissions and cumulative
claimed attempts,
ordered failure-code counts retained by case state, retries, run references,
payload bytes, wall/GPU seconds, dataset/manifest/statistics digests, and final
state. `--output` writes the same JSON outside the repository for evidence
capture without dirtying a resumable checkout.

## Training and validation

The same image and volume back the identity-checked training and validation
entrypoints. Their exact runbooks are [`training.md`](training.md) and
[`validation.md`](validation.md). Training admits at most three independent
seed workers for 75 minutes each; validation admits one worker for 30 minutes.
Both disable provider retries, exchange only bounded typed requests/receipts,
and commit large checkpoints, models, and reports through the shared volume.

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
- Training staging/workers use the reviewed 75-minute/concurrency-three policy;
  validation staging/workers use 30 minutes/concurrency one. Neither may change
  a resumed experiment's device class or accept mismatched report parents.
- CI uses a stubbed Modal module and never authenticates or calls the provider.
- A missing profile, unavailable GPU, dirty source image, non-CUDA resolution,
  manifest mismatch, serialization violation, artifact mismatch, or kernel
  mismatch fails explicitly.
