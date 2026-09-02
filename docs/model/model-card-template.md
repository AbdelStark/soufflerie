# Model bundle and card contract

Every deployable FNO is published as one immutable, content-addressed bundle.
The card is generated from strict `ModelCardMetadata`; it is not a free-form
file to edit after export. This keeps the human-readable claims bound to the
same model identity as the weights, preprocessing statistics, architecture,
runtime contract, and training lineage.

## Closed layout

```text
models/<model_id>/
  bundle.json
  model.safetensors
  preprocessing.json
  architecture.json
  model-card.md
  COMMITTED
```

`bundle.json` conforms to
[`model-bundle.json`](../../schemas/v1/model-bundle.json). It records the exact
28-tensor `fno2d-v1` allowlist, float32 dtypes, shapes of up to five dimensions,
per-tensor byte counts, the 151,123,216-byte decoded total, a 192 MiB file cap,
member digests, the full parent dataset digest, experiment identity, selected
epoch, clean source revision, lock digest, and bounded runtime compatibility.

The model ID is the first 20 hexadecimal characters of a SHA-256 over the
logical metadata. That identity includes the weights digest, preprocessing
digest, architecture digest, compatibility range, tensor contract, model-card
content, the authoritative full parent dataset digest, experiment identity,
seed, epoch, source revision, and lock digest. Storage paths and commit timing
are deliberately excluded.

## Card input template

Exporters supply the following bounded record. Each string is one line and
cannot contain a Markdown table separator. Gate names must be unique.

```python
ModelCardMetadata(
    display_name="<release-facing model name>",
    summary="<one factual sentence describing this trained model>",
    intended_uses=(
        "<supported workflow and operating domain>",
    ),
    limitations=(
        "<known scientific, domain, runtime, or deployment limitation>",
    ),
    gates=(
        ModelCardGate(
            name="<gate name>",
            status="not_evaluated",  # or green/red with evidence below
            threshold="<frozen acceptance threshold>",
            measured=None,
            evidence_sha256=None,
        ),
    ),
)
```

A `green` or `red` gate requires both a measured value and the SHA-256 of its
immutable evidence. A `not_evaluated` gate requires neither. Export rejects
partial or contradictory evidence. The deterministic renderer adds identity,
architecture, epoch, seed, weights digest, source revision, license, intended
uses, the complete gate table, and limitations. Hand-editing `model-card.md`
changes its digest and makes the bundle unreadable.

Model-card gates report captured validation state; they do not recompute it.
Scientific release acceptance still belongs to the validation report and
release gates in RFC-0008.

## Export and trust sequence

`snapshot_fno_weights` accepts only the fixed `FnoPredictor`, verifies the exact
state names, float32 shapes, contiguity, and finiteness, then makes owned,
read-only CPU NumPy copies. `build_model_bundle` validates those copies before
encoding `model.safetensors` and computing every identity.

`LocalModelBundleStore.publish` writes to a private staging directory, fsyncs
members, reopens them through the untrusted-reader path, writes `COMMITTED`, and
atomically renames the verified directory. A fault before rename never exposes
a partial model. Republishing identical content is idempotent.

Opening proceeds in this order:

1. Require the exact directory layout and regular files; reject symlinks.
2. Validate the ASCII commit marker and `bundle.json` digest.
3. Validate strict schema, compatibility, the full parent dataset digest,
   architecture, preprocessing lineage, card digest, and canonical card text.
4. Inspect the safetensor header, exact allowlist, shapes, dtypes, offsets, and
   allocation caps before decoding any tensor.
5. Verify all member digests, model identity, and committed byte count.
6. Construct the fixed predictor only on the explicitly requested `cpu` or
   `cuda[:index]` device and load the verified state strictly.

There is no architecture deserialization, dtype coercion, implicit device
fallback, pickle, or optimizer/RNG state in this boundary. Trusted resumable
training checkpoints are private artifacts and remain separate from public
deployable bundles.

## Validation

The framework-free suite covers identity binding, atomic publication, reader
limits, and tamper/missing/extra cases. With the locked ML profile installed,
it also exports a real PhysicsNeMo model, reloads it on CPU, and requires
bitwise-identical field and drag inference:

```bash
uv run pytest tests/surrogate/test_bundle.py tests/security/test_model_loading.py
uv run python scripts/export_schemas.py --check
```
