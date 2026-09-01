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

The smoke does not publish a solver artifact or implement the domain solve
entrypoint; those idempotency, serialization, and volume-commit semantics are
owned by issue #42.

## Failure and cost controls

- Solve functions time out after 180 seconds and have provider retries disabled.
- The smoke uses one container; sweep, train, validation, and service limits are
  centralized as reviewed constants in `infra/policy.py`.
- CI uses a stubbed Modal module and never authenticates or calls the provider.
- A missing profile, unavailable GPU, dirty source image, non-CUDA resolution,
  manifest mismatch, or kernel mismatch fails explicitly.
