# Bundled CPU smoke model

Soufflerie ships one self-contained `fno2d-v1` fixture so a fresh installed
wheel can prove the safe-bundle and real PhysicsNeMo CPU inference path. The
fixture is deliberately synthetic and untrained. Its fixed output biases test
plumbing and serialization only; they are not flow-accuracy, validation, or
release-quality evidence.

## Package resources

The exact resources are rooted at the packaged
[`resource.json`](../../src/soufflerie/resources/model/resource.json) descriptor:

```text
resource.json
bundle.json
preprocessing.json
architecture.json
model-card.md
model.safetensors.gz
```

`resource.json` ships the immutable model `ArtifactRef`, bundle-metadata
digest, transparent synthetic-parent record and digest, compressed weight
SHA-256 and byte count, decoded byte count, and expected field/drag output
digests. The wheel's `RECORD` additionally authenticates each installed
resource against the signed distribution artifact.

The 28 float32 tensors occupy 151,126,144 bytes in canonical safetensors form.
They contain zero kernels plus fixed field biases `(0.25, -0.5, 0.75)` and drag
bias `0.125`. Deterministic gzip reduces the committed resource to less than
256 KiB without quantization or dtype changes. Materialization expands at most
the bundle's 192 MiB cap; the raw checkpoint never enters Git or the wheel.

## Trust and import sequence

`materialize_bundled_cpu_model(root)` performs these checks before returning a
model reference:

1. Require the exact six-member, regular-file resource layout.
2. Strictly validate `resource.json` and its synthetic-parent identity.
3. Verify bundle metadata, preprocessing, architecture, card, and compressed
   weight digests and byte counts.
4. Decompress through a 192 MiB ceiling, then construct `ModelBundle`, which
   verifies the decoded safetensors digest and every logical identity.
5. Publish through `LocalModelBundleStore` staging, reopen the untrusted bundle,
   write its commit marker, and atomically rename it into place.

This path needs only the base installation and imports neither Torch nor
PhysicsNeMo. `run_bundled_cpu_smoke(root)` then explicitly loads the locked ML
runtime, constructs the fixed FNO on CPU, strictly installs the verified state,
runs one normalized zero-input batch, and compares output hashes with the
packaged descriptor. Training, remote, service, demo, and visualization modules
remain outside both paths.

## Regeneration and acceptance

The generator owns every resource byte, including a canonical sorted
safetensors header and deterministic gzip stream:

```bash
uv run python scripts/generate_bundled_model.py --check
uv build
uv run pytest tests/package/test_bundled_model.py
```

The test builds the wheel, installs it with locked dependencies in fresh base
and ML environments, proves base materialization and import isolation, runs the
real CPU prediction, validates schema-v1 finite output and exact digests, and
checks package membership. The fresh ML install is marked `remote` because it
may download the locked optional runtime; normal pull-request gates remain
network-free and still exercise resource regeneration, materialization,
tamper rejection, wheel contents, and base import isolation.

Changing any tensor byte, bundle field, source or lock digest, synthetic parent,
or card statement changes the model identity and requires canonical
regeneration. A future trained release model must use its real dataset parent
and RFC-0008 evidence; it must never reuse this fixture's claims or identity.
