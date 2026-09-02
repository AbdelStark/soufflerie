# Deterministic mixed-precision training

The checked [`fno-v1.yaml`](../../configs/training/fno-v1.yaml) is the canonical
RFC-0007 optimization policy. It is bound to the published dataset
`4aefbbe88a18d233249b`, declares the three seeds `17/23/31`, and permits no
runtime-selected precision or hyperparameter fallback. `TrainingConfig`
rejects unknown fields, duplicate seeds, batch sizes outside `[1,64]`, and a
minimum learning rate above the maximum.

## Objective

`masked_training_loss` consumes raw normalized FNO output, its normalized
target, the exact fluid mask, and `TrainingConfig`. Each channel uses a
fluid-masked relative MSE with an energy denominator floored at `1e-6`.
Obstacle velocity is penalized independently, with normalized reference
velocity fixed to `1.0`; drag uses `max(abs(Cd), 0.1)` as its relative scale.
The fixed total is:

```text
1.0 * L_u + 1.0 * L_v + 0.25 * L_rho
            + 0.25 * L_obstacle + 0.5 * L_cd
```

All tensor reductions are explicitly float32, including under autocast.
`reference_training_loss` is an independent NumPy oracle used for tiny
hand-computed arrays, empty-energy channels, masks, and drag-floor boundaries.
Epoch means weight each batch by its retained sample count and accumulate with
Python's float64 `math.fsum`; a final partial batch is never dropped.

## Runtime policy

`prepare_training_session` performs checks in this order:

1. require the requested seed to be one of the three config seeds;
2. prove that the explicit CUDA device exists and supports the exact requested
   precision (native bf16 for the canonical config);
3. seed Python, NumPy, Torch CPU, and all CUDA generators;
4. enable deterministic algorithms and error-level deterministic debugging,
   disable cuDNN benchmarking, and disable TF32;
5. construct and move the FNO only after seeding;
6. build AdamW with `(beta1,beta2)=(0.9,0.999)`, epsilon `1e-8`, and separate
   zero-decay groups for biases and normalization parameters.

Unsupported CUDA, bf16, or deterministic controls fail before model
construction. A framework operation rejected by deterministic mode aborts the
epoch. There is no CPU precision fallback. Fp16 runs use a CUDA gradient
scaler; bf16 runs do not. Both unscale before the global gradient norm is
clipped to `1.0`, and a non-finite norm aborts the update.

The first five epoch learning rates are `10%`, `32.5%`, `55%`, `77.5%`, and
`100%` of the declared rate. Epochs 6 through 100 use cosine decay and reach
the declared minimum exactly at epoch 100. There is no early stopping.

## Authoritative evidence

`run_training_epoch` streams only the verified manifest's training split,
synchronizes CUDA around each measured optimization step, and appends one
[`training-epoch.json`](../../schemas/v1/training-epoch.json) record through
`EpochJsonlWriter`. Each record binds experiment/dataset/config/seed identity,
epoch and global-step continuity, every loss term, learning rate, requested
precision, device name and compute capability, synchronized compute/GPU time,
I/O and wall time, and peak allocated/reserved CUDA bytes.

The writer validates all existing lines before append. It rejects malformed or
partial JSON, identity drift, epoch gaps, global-step gaps, symlink targets,
and bounded-file violations, then flushes and fsyncs every complete record.
TensorBoard output may mirror this data later, but it is never authoritative.
Checkpoint publication and resume begin in issue #26 and do not weaken these
epoch-log invariants.

## Validation

Run the issue acceptance suite:

```bash
uv run pytest \
  tests/training/test_loss.py \
  tests/training/test_loop.py \
  tests/training/test_determinism.py
uv run python scripts/validate_schemas.py
```

Base CPU CI exercises the formulas, policy, sequencing, and failure paths with
contract runtimes. A locked CUDA/PhysicsNeMo environment supplies the real
mixed-precision acceptance evidence; a CPU pass is not reported as GPU
training proof.
