# Fixed FNO architecture

RFC-0006 defines one deployable neural architecture. There is no architecture
search or environment-dependent default resolution in v0.1. The checked
[`architecture.json`](../../schemas/v1/architecture.json) schema and
`FnoArchitecture` record bind every module argument used by training, bundle
export, loading, and inference.

## Field path

The model receives the strict `PredictionBatch` produced by preprocessing:

```text
inputs         float32[B,2,320,256]
fluid_mask     bool[B,1,320,256]
design_params  float32[B,4]
```

The PhysicsNeMo 2.2.1 FNO core is constructed with these resolved arguments:

```text
dimension             2
input channels        2
latent channels       64
spectral blocks       4
retained modes        [24,24]
padding               [8,8], constant
spectral activation   GELU
coordinate features   disabled
projection            64 -> 128 -> 3, GELU
dropout                none
batch normalization   none
```

PhysicsNeMo's stock encoder normally adds coordinate channels and uses a
two-stage `2 -> 32 -> 64` lift. RFC-0006 instead requires exactly two input
channels and a `2 -> 64` lifting projection. `FnoPredictor` therefore disables
coordinate features and replaces only the stock lift with one pointwise
`Conv2d(2, 64, kernel_size=1)`. The four PhysicsNeMo spectral convolutions and
their residual pointwise `1x1` convolutions remain unchanged. The stock decoder
is resolved to one hidden layer, giving the required pointwise
`64 -> 128 -> 3` projection.

Under the pinned framework version this model has 37,780,804 trainable
parameters and 28 state tensors, including the drag head. Tests assert both the
resolved constructor arguments and module topology so a framework-default
change cannot silently alter the architecture.

## Drag path

The final 64-channel latent is pooled over fluid cells only. For each sample,
with boolean fluid mask `m`, the pooled feature is:

```text
pooled = sum(latent * m, spatial) / sum(m, spatial)  # [B,64]
```

A sample with no fluid cell fails before division. The pooled latent is
concatenated with normalized `(aspect_ratio, rotation_deg, scale, Re)` and sent
through the fixed head:

```text
68 -> 64 -> 32 -> 1
     GELU  GELU
```

The result is `cd_head float32[B]`. The mask affects only this pooling path.
The three normalized field channels are never multiplied by either mask, so
raw obstacle predictions remain available to compliance validation.

## Runtime boundary

Importing `soufflerie` or `soufflerie.surrogate.fno` does not import Torch or
PhysicsNeMo. `FnoPredictor()` loads both only at construction and requires the
exact PhysicsNeMo version recorded by `FnoArchitecture`; a missing or mismatched
runtime raises `DEPENDENCY_UNAVAILABLE`.

`forward(batch)` is differentiable and is used by the trainer. `predict(batch)`
temporarily selects evaluation mode, runs under inference mode, and restores
the caller's prior train/eval state. Both paths reject non-Torch tensors,
non-float32 parameters, batch/model device mismatches, malformed latent shapes,
and empty fluid masks. Public results are contiguous float32 tensors with exact
shapes:

```text
fields_normalized  float32[B,3,320,256]  # raw, unmasked
cd_head            float32[B]
```

The root state dictionary namespaces members under `core.` and `cd_head.` so
issue #22 can publish a closed tensor allowlist in safe model bundles.

## Validation

Run the architecture, gradient, and deterministic fixture-overfit tests with
the locked ML runtime available:

```bash
uv run pytest tests/surrogate/test_fno.py tests/surrogate/test_overfit.py
```

The full-resolution one-batch fixture must reduce its initial joint field/Cd
loss by at least 90%. Framework-free contract doubles exercise construction and
failure behavior in base CPU CI; the real-runtime shape, parameter, gradient,
and overfit cases run whenever the optional ML packages are installed.
