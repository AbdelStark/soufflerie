# RFC-0006: FNO surrogate and checkpoints

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

The v0.1 surrogate is a fixed two-dimensional Fourier neural operator that maps normalized ellipse SDF and Reynolds planes to mean `u`, `v`, and `rho-1` fields, with a pooled latent MLP for Cd. A safe, immutable model bundle binds weights, architecture, preprocessing statistics, training identity, and compatibility metadata.

## Motivation

The PRD selects an FNO and requires both learned-head Cd and a field-derived Cd for consistency. A model name alone does not define modes, widths, preprocessing, obstacle treatment, tensor shapes, or checkpoint safety. [`03-data-model.md#array-contracts`](../spec/03-data-model.md#array-contracts) and [`06-security.md#artifact-safety`](../spec/06-security.md#artifact-safety) require these decisions before training.

## Goals

- Lock model inputs, outputs, architecture, parameter initialization, and postprocessing.
- Keep framework-specific tensors behind a typed `FlowPredictor` adapter.
- Bind every deployable checkpoint to exact preprocessing and lineage.
- Support deterministic CPU inference from a small bundled checkpoint.
- Preserve raw model output for meaningful physics validation.

## Non-Goals

- Unsteady rollout, arbitrary resolution inference claims, MeshGraphNet, or distributed model parallelism.
- Predictive distributions from a single model.
- Hard-coding physical compliance by masking outputs after prediction.
- Loading arbitrary framework checkpoints from untrusted sources.

## Proposed Design

Preprocessing transforms each dataset row:

```text
x[:,0] = clip(sdf / D_lu, -1, 1)
x[:,1] = 2*(Re-40)/(300-40) - 1
y[:,0] = (u - mean_u_train) / std_u_train
y[:,1] = (v - mean_v_train) / std_v_train
y[:,2] = ((rho-1) - mean_rho_delta_train) / std_rho_delta_train
```

Statistics are scalar per output channel, accumulated in fp64 over training-split fluid and obstacle cells according to the stored arrays; standard deviations floor at `1e-6` with a recorded flag. No validation/test data contributes. SDF and Reynolds scaling are fixed by contract and need no fitted statistics.

The input/output contract is batch-first float32 `[B,C,320,256]`. The FNO has lifting projection `2 -> 64`, four spectral blocks, 24 retained modes in each spatial dimension, hidden width 64, GELU activation, and residual pointwise `1x1` convolution per block. The projection head maps `64 -> 128 -> 3` with GELU. No dropout or batch normalization is used. Spectral weights use the framework's documented default complex initialization under the experiment seed; all resolved module arguments are serialized.

The Cd head applies fluid-mask-aware global mean pooling of the final latent plus the normalized four design parameters `(aspect_ratio, rotation, scale, Re)` and uses `68 -> 64 -> 32 -> 1` with GELU. The binary fluid mask derives from input SDF but is not an FNO input channel because it is exactly recoverable by sign. The model returns unmasked raw fields. Neither service nor validator zeros obstacle values; obstacle compliance remains observable.

```python
@dataclass(frozen=True, slots=True)
class PredictionBatch:
    inputs: TorchFloat32Tensor      # [B,2,320,256]
    fluid_mask: TorchBoolTensor     # [B,1,320,256]
    design_params: TorchFloat32Tensor  # [B,4]

@dataclass(frozen=True, slots=True)
class PredictionBatchResult:
    fields_normalized: TorchFloat32Tensor  # [B,3,320,256]
    cd_head: TorchFloat32Tensor            # [B]

class FnoPredictor(FlowPredictor):
    def predict(self, batch: PredictionBatch) -> PredictionBatchResult: ...
```

Training forward uses autocast fp16/bfloat16 as selected in RFC-0007 but accumulation, loss scalars, optimizer state, validation, and exported inference weights have declared precision. Public NumPy output is de-normalized float32. Inputs with wrong dtype, shape, device, non-contiguity, or non-finite values fail before forward; adapters may perform an explicit documented contiguous copy but not implicit dtype/device changes.

The bundle layout is:

```text
models/<model_id>/
  bundle.json
  model.safetensors
  preprocessing.json
  architecture.json
  model-card.md
  COMMITTED
```

```python
class ModelBundleMetadata(BaseModel):
    schema_version: Literal[1] = 1
    model_id: str
    architecture: Literal["fno2d-v1"]
    dataset_id: str
    experiment_id: str
    seed: int
    selected_epoch: int
    weights_sha256: str
    preprocessing_sha256: str
    architecture_sha256: str
    code_revision: str
    lock_digest: str
    compatibility: CompatibilityRange
```

Weights use `safetensors`. `bundle.json` enumerates exact expected tensor names, shapes, dtypes, and total byte cap. Loading verifies commit marker, all digests, compatibility range, architecture allowlist, and tensor descriptors before allocation. Model ID hashes logical metadata and weights. Optimizer/RNG training checkpoints are separate private artifacts and never part of the deployable bundle.

CPU bundling MAY quantize neither weights nor fields in v0.1; the same float32 exported weights serve CPU and GPU. If bundle size makes source inclusion impractical, the wheel ships metadata and an immutable release-asset fetch command with checksum, while the fresh-clone demo setup documents the download. The quickstart smoke fixture remains small and committed.

## Alternatives Considered

### U-Net convolutional surrogate

It is simpler and a useful future baseline, but the PRD's learning objective specifically selects PhysicsNeMo FNO and spectral global interactions suit steady flow fields. Two non-neural baselines already challenge necessity.

### Geometry mask as a third input channel

It is exactly derivable from SDF sign and adds redundant representation. The mask is still provided to the Cd pooling and loss/metrics, where its discrete meaning matters.

### Hard zero velocity inside obstacles

It improves the compliance metric by construction and hides model behavior. Raw outputs must face the gate; visualization may overlay the obstacle without modifying returned arrays.

### Generic framework checkpoint serialization

It can include executable pickle and under-specifies architecture/preprocessing. Safe weights plus JSON metadata are accepted despite more export code.

## Drawbacks

- Fixed grid/modes limit generalization across resolution.
- Scalar global preprocessing may underrepresent spatially varying scale.
- The Cd head adds multi-task tuning and can disagree with field physics.
- Safe bundle export cannot resume training; separate trusted checkpoints are required.

## Migration / Rollout

1. Implement preprocessing/statistics with leakage tests and typed batch contracts.
2. Wrap the selected framework FNO with explicit arguments and shape tests.
3. Add Cd head, raw-output behavior, one-batch overfit, and CPU inference tests.
4. Implement safe bundle export/load and artifact integrity tests.
5. Train three seeded models under RFC-0007 and publish selected/all-seed bundles for validation.

Architecture changes create a new architecture identifier and model IDs. Existing bundles remain readable only within declared compatibility.

## Testing Strategy

- Golden-test preprocessing/deprocessing and training-only statistics with sentinel test data.
- Assert exact module arguments, parameter count range, tensor names, input/output shapes, and gradients.
- Test batch sizes 1 and >1 on CPU; test explicit dtype/device failures.
- Overfit one small batch by at least 90% loss reduction.
- Prove obstacle raw outputs are not post-masked.
- Round-trip safe bundle bytes and compare inference bitwise on the same CPU environment.
- Reject modified weights, missing tensors, extra tensors, oversized tensors, incompatible schema/code range, and mismatched preprocessing.
- Run installed-wheel bundled-model smoke without importing training/remote packages.

## Open Questions

None for v0.1. Architecture tuning may select only among experiments declared in RFC-0007; changing this structural contract requires a new RFC owned by the ML maintainer.

## References

- [`prd.md#63-surrogate-physicsnemo-fno`](../../prd.md#63-surrogate-physicsnemo-fno)
- [`03-data-model.md#array-contracts`](../spec/03-data-model.md#array-contracts)
- [RFC-0007](RFC-0007-training-and-reproducibility.md)
- [RFC-0008](RFC-0008-validation-and-release-gates.md)
- Li et al., “Fourier Neural Operator for Parametric Partial Differential Equations,” 2021.
