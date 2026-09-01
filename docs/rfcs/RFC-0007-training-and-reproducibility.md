# RFC-0007: Training, baselines, and reproducibility

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

Training is a single-GPU, three-seed, manifest-driven pipeline with deterministic initialization/data order, AdamW, cosine decay, mixed precision, epoch-boundary checkpoints, and validation-selected export. Mean-field and nearest-design baselines use the same frozen splits and metric implementation as the FNO.

## Motivation

The PRD requires the FNO to earn its place against two baselines, finish within one GPU-hour, use mixed precision/checkpointing, and record logs. Without a fixed loss, seed policy, selection rule, and resume semantics, reported improvements cannot be reproduced or distinguished from test-set selection.

## Goals

- Define complete training configuration, objective, scheduling, precision, and selection.
- Make repeated same-environment runs reproducible and resume-safe.
- Evaluate deterministic baselines through identical loaders/metrics.
- Keep test data sealed until the final validation stage.
- Produce deployable bundles and auditable training artifacts for three seeds.

## Non-Goals

- Hyperparameter sweeps over the test set.
- Distributed/multi-GPU training, online learning, or automatic mixed precision fallback.
- Claiming identical optimization across device/runtime versions.
- Shipping optimizer checkpoints as public model artifacts.

## Proposed Design

The checked-in `configs/training/fno-v1.yaml` parses strictly:

```python
class TrainingConfig(BaseModel):
    schema_version: Literal[1] = 1
    dataset_id: str
    seeds: tuple[int, int, int]
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    field_weights: tuple[float, float, float] = (1.0, 1.0, 0.25)
    cd_weight: float = 0.5
    obstacle_weight: float = 0.25
    precision: Literal["bf16", "fp16"] = "bf16"
    num_workers: int = 4
```

The remote device capability check uses bf16 when supported by the selected GPU; the checked-in canonical config declares bf16. The A10G fallback supports the functional profile; if a runtime reports unsupported precision, the run fails before training rather than changing precision silently. Architecture is fixed by RFC-0006.

For normalized prediction `y_hat`, target `y`, fluid mask `m_f`, obstacle mask `m_o`, and scalar drag:

```text
L_channel(c) = mean(m_f * (y_hat_c-y_c)^2) / max(mean(m_f*y_c^2), 1e-6)
L_obstacle = mean(m_o * (u_hat^2+v_hat^2)) / U_ref_norm^2
L_cd = mean(((Cd_hat-Cd)/max(abs(Cd),0.1))^2)
L_total = 1.0*L_u + 1.0*L_v + 0.25*L_rho + 0.25*L_obstacle + 0.5*L_cd
```

Loss reductions accumulate in fp32 and report fp64 epoch means. Density target is `rho-1` after training statistics. No field-derived Cd enters the optimization loss; it remains independent validation evidence.

AdamW uses betas `(0.9,0.999)`, epsilon `1e-8`, declared weight decay, no decay for biases/norm parameters, and global gradient-norm clipping. A five-epoch linear warmup rises from 10% to full learning rate; cosine decay reaches `min_learning_rate` at epoch 100. There is no early stopping. The selected checkpoint minimizes validation `score = median_velocity_relative_l2 + median_cd_head_relative_error`; ties select the earlier epoch. Test metrics are computed only after selection is frozen for all seeds.

Each seed controls Python, NumPy, framework CPU/CUDA RNGs, model initialization, and sampler order. Deterministic algorithms are enabled; framework benchmark/autotune that changes kernels is disabled. The loader uses a seeded generator and deterministic worker child seeds; batches do not drop. Dataset order derives from manifest rows, not filesystem listing. Training records any framework operation that cannot meet deterministic mode and fails unless explicitly accepted by a future RFC.

Epoch checkpoints contain trusted training-only state:

```python
class TrainingCheckpointMetadata(BaseModel):
    experiment_id: str
    dataset_id: str
    config_digest: str
    seed: int
    completed_epoch: int
    global_step: int
    model_digest: str
    optimizer_digest: str
    scheduler_digest: str
    scaler_digest: str | None
    rng_state_digest: str
```

Checkpoints publish atomically after every epoch and retain latest plus current best. Resume requires exact experiment/dataset/config/code/lock/device-precision identity and begins at the next epoch boundary. Mid-epoch resume is not supported. Uninterrupted and resumed runs must produce equivalent next-epoch metrics under the deterministic tolerance.

The mean-field baseline stores per-pixel/per-channel training mean fields and scalar mean Cd. The nearest-design baseline standardizes four design dimensions to `[0,1]`, finds Euclidean nearest training design with `design_id` tie-break, and returns that row's fields/Cd. Both implement `FlowPredictor`, use no validation/test fitting, and publish baseline metadata. FNO must beat both on test median velocity relative L2 and median Cd error; otherwise the validation report is red.

Artifacts per seed include canonical config, epoch JSONL metrics, trusted resume checkpoints, best safe model bundle, profiler/timing summary, and dependency/device provenance. TensorBoard event files are auxiliary; JSONL is authoritative. `experiment_id` hashes dataset, architecture, training config, seed set, code revision, and lock digest.

## Alternatives Considered

### Early stopping

It can save time but makes patience another tuning choice and complicates seed comparisons. Fixed epochs plus validation selection is deterministic and fits the declared budget.

### Test-selected seed or ensemble mean as primary model

Choosing by test performance leaks evaluation. Each seed selects by validation score; the deployable seed is the lowest validation score across seeds. The three models together serve OOD variance analysis, not test-driven cherry-picking.

### Field-derived Cd in the training loss

It could impose consistency but would weaken independence of the consistency check and add sensitivity to the control-volume implementation. Only solver Cd supervises the head in v0.1.

### Nearest neighbor over learned field embeddings

It would require additional fitting and obscure the simple baseline. Normalized design-space distance is transparent and deterministic.

## Drawbacks

- Fixed 100 epochs may waste time after convergence or undertrain difficult runs.
- Strict deterministic kernels can reduce throughput.
- One chosen configuration does not establish architecture optimality.
- Private framework optimizer checkpoints remain a trusted deserialization surface.

## Migration / Rollout

1. Implement dataset loader, preprocessing, deterministic sampler, and both baselines.
2. Add loss terms and one-batch overfit tests.
3. Implement training loop, JSONL metrics, checkpoints, resume checks, and validation selection.
4. Run a short smoke configuration remotely, then canonical three-seed training.
5. Export all best bundles safely; select deployable seed by validation score before test evaluation.

Any change to data, architecture, loss, optimizer, schedule, seed set, or precision creates a new experiment ID and cannot overwrite prior artifacts.

## Testing Strategy

- Hand-compute each loss term, mask denominator, channel weight, and zero-energy edge case.
- Assert no validation/test row contributes to statistics or baseline fitting.
- Verify batch order and augmentation-free samples repeat for each seed.
- Run one-batch overfit and gradient finite/clip tests.
- Compare uninterrupted with epoch-resumed training for a small deterministic fixture.
- Reject resume on every identity mismatch and corrupt state member.
- Verify best-epoch and best-seed tie-breaking without reading test metrics.
- Evaluate baselines and FNO through the exact same validation metric functions.
- Record remote wall time, GPU seconds, memory, and precision; enforce the one-hour budget.

## Open Questions

None for v0.1. Loss or schedule changes after canonical training begins are new named experiments owned by the ML maintainer; test results may not motivate retroactive selection.

## References

- [`prd.md#63-surrogate-physicsnemo-fno`](../../prd.md#63-surrogate-physicsnemo-fno)
- [`07-testing-strategy.md#ml-tests`](../spec/07-testing-strategy.md#ml-tests)
- [RFC-0005](RFC-0005-dataset-artifacts-and-sweep-lifecycle.md)
- [RFC-0006](RFC-0006-fno-surrogate-and-checkpoints.md)
- Loshchilov and Hutter, “Decoupled Weight Decay Regularization,” 2019.
