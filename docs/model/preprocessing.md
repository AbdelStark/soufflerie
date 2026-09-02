# Leakage-safe preprocessing

RFC-0006 fixes the conversion from verified dataset runs to FNO inputs and
targets. This module owns numerical preprocessing only. The manifest loader in
issue #24 remains responsible for opening `ManifestRow.run_uri`, verifying the
run digest and metadata, and then constructing a `PreprocessingSample` with the
matching dataset, case, split, design scalars, and curated arrays.

## Stored and model boundaries

A `PreprocessingSample` accepts the exact curated artifact contract:

```text
u_mean, v_mean, rho_mean, sdf  float16[320,256]
obstacle_mask                  uint8[320,256], values 0 or 1
```

Arrays must already be finite and C-contiguous. The adapter does not silently
repair dtype, shape, or layout. It explicitly restores stored fields to
float32 while creating model arrays:

```text
inputs             float32[B,2,320,256]
fields_normalized  float32[B,3,320,256]
fluid_mask         bool[B,1,320,256]
design_params      float32[B,4]
cd                 float32[B]
```

The two input channels are fixed and require no fitted state:

```text
inputs[:,0] = clip(sdf / 16, -1, 1)
inputs[:,1] = 2 * (Re - 40) / (300 - 40) - 1
```

The diameter `16` is `D_lu = ny/20` on the persisted `320 x 256` grid. It is
the isotropically reduced counterpart of the solver grid's diameter. The four
Cd-head design parameters `(aspect_ratio, rotation_deg, scale, Re)` use the
same affine `[-1,1]` mapping over their frozen ranges
`[0.5,1]`, `[0,30]`, `[0.75,1.25]`, and `[40,300]`.

The model fluid mask is exactly `sdf > 0`, as required by RFC-0006. The stored
`obstacle_mask` is still validated at the artifact boundary, but it is not used
to construct the model mask: the persisted mask is downsampled from the solver
grid while SDF is recomputed on the model grid, so their boundary pixels need
not be identical. Raw output fields are not masked during preprocessing,
training, inference, or de-normalization.

## Training-only statistics

`fit_preprocessing_statistics` accepts samples from one dataset but updates
moments only for rows whose split is exactly `train`. Validation and test
arrays never enter the accumulation loop. Training samples sort by `case_id`
before fitting so caller order cannot change the durable record.

Each output channel uses scalar population moments across every stored cell,
including fluid and obstacle cells:

```text
u_target         = (u_mean - mean_u_train) / std_u_train
v_target         = (v_mean - mean_v_train) / std_v_train
rho_delta_target = ((rho_mean - 1) - mean_rho_delta_train)
                   / std_rho_delta_train
```

Moments combine deterministic chunks in float64. The raw population standard
deviation is `sqrt(M2/N)`. Values below `1e-6` use an exact `1e-6` denominator,
and every channel records both the raw deviation and a `floored` flag. The
checked [`preprocessing.json`](../../schemas/v1/preprocessing.json) schema also
binds the dataset ID, training case/cell counts, channel order, grid, ranges,
stored/fit/model dtypes, and floor policy.

`denormalize_fields` accepts only finite, C-contiguous
`float32[B,3,320,256]` arrays and returns unmasked public float32 `(u,v,rho)`
channels. Golden tests cover the normalization/de-normalization round trip
against hand-computed population moments.

## Optional Torch boundary

`prediction_batch_to_torch` imports Torch only when called, makes an explicit
owned C-order copy of the NumPy inputs, and moves tensors only to the named
`cpu` or `cuda[:index]` device. A base install therefore imports
`soufflerie.surrogate` without Torch.

`PredictionBatch` validates the RFC-0006 inference inputs before model forward:

- inputs and design parameters are exactly `torch.float32`;
- the fluid mask is exactly `torch.bool`;
- all tensors have the fixed batch-first shapes and one shared device;
- every tensor is contiguous, and numeric tensors are finite;
- an explicitly expected device must match; there is no fallback or cast.

Missing Torch raises `DEPENDENCY_UNAVAILABLE`. An invalid or unavailable device
raises `DEVICE_UNAVAILABLE`. Malformed persisted arrays, statistics, or tensors
fail with `ARTIFACT_INTEGRITY` before model execution.

## Validation

Run the issue-specific contract suite and schema check:

```bash
uv run pytest tests/surrogate/test_preprocessing.py
uv run python scripts/export_schemas.py --check
```
