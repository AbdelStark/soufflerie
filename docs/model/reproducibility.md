# Checkpoint resume and validation selection

Training checkpoints are private, trusted epoch-boundary artifacts. They are
not model bundles and never ship in wheels or releases. Public bundles contain
only safe float32 weights plus reviewed JSON/Markdown sidecars.

## Complete identity and state

`capture_training_checkpoint` runs only after a completed epoch and records:

- experiment, full dataset, architecture, config, clean code, and lock digests;
- seed, completed epoch, global step, exact CUDA device/name/capability, and
  requested precision;
- model, AdamW, formula-scheduler, optional fp16 scaler, Python/NumPy RNG, and
  Torch CPU/all-CUDA RNG state.

Each member has a byte cap, byte count, and SHA-256 in
[`training-checkpoint.json`](../../schemas/v1/training-checkpoint.json). The
checkpoint ID hashes all identity fields and member digests. `model.pt`,
`optimizer.pt`, optional `scaler.pt`, and Torch RNG state use the pinned
framework codec and are decoded with `weights_only=True`; JSON members remain
strict and canonical. This trusted state is never accepted from a request or
copied into a deployable artifact.

`LocalTrainingCheckpointStore` writes members in a staging directory, fsyncs
them, writes the metadata-digest commit marker last, and atomically renames the
complete directory. Per-seed `latest.json` and `best.json` pointers are replaced
atomically and are never trusted without fully reopening the target.

## Resume boundary

`restore_training_checkpoint` compares every experiment/dataset/config/code/
lock/device/precision field before decoding state. It then restores model,
optimizer, scaler, Python, NumPy, Torch CPU, and all CUDA RNGs, verifies that
the restored optimizer learning rate agrees with scheduler state, sets the
global step, and returns `completed_epoch + 1`. A corrupt member, identity
mismatch, device change, missing scaler, mid-epoch request, or learning-rate
disagreement aborts resume. There is no partial restore or automatic fallback.

## Selection without test leakage

`ValidationCheckpointMetric` accepts only `split="validation"` and defines the
fixed score:

```text
median_velocity_relative_l2 + median_cd_head_relative_error
```

`freeze_validation_selection` chooses the minimum score per seed, breaking a
tie toward the earlier epoch. After all three seeds are frozen, it chooses the
deployable seed by minimum validation score and then lower seed. The resulting
[`training-selection.json`](../../schemas/v1/training-selection.json) binds all
three checkpoint IDs and carries the invariant `test_metrics_read=false`.
Unknown fields and test-split records are rejected structurally.

`export_selected_checkpoint_bundle` accepts only a checkpoint named by that
frozen selection. It decodes the selected model state, snapshots the exact FNO
allowlist, and delegates publication to the existing safetensors model-bundle
store. Optimizer, scaler, and RNG state cannot cross that boundary.

## Validation

```bash
uv run pytest \
  tests/training/test_checkpoint.py \
  tests/training/test_selection.py
uv run python scripts/validate_schemas.py
```

The local deterministic fixture compares the first post-checkpoint Python and
NumPy draws after restore with the uninterrupted path and verifies model,
optimizer, scheduler, framework RNG, and global-step restoration. Canonical
CUDA next-epoch equivalence is executed with the three-seed remote workflow in
issue #27.
