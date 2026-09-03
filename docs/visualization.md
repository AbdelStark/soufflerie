# Deterministic field visualization

Soufflerie renders solver and surrogate arrays through one versioned PNG
contract. Rendering is a view over validated `FlowFields`: it copies and
revalidates every array, and the neutral obstacle overlay never replaces or
zeros the raw values used by NPZ artifacts or metrics.

Install the locked visualization profile before rendering:

```bash
uv sync --frozen --extra viz
```

The public API lives in `soufflerie.demo`:

```python
from soufflerie.demo import RenderSpec, render_comparison, render_fields

spec = RenderSpec(
    annotation="aspect=0.75 | rotation=12 deg | scale=1.0 | Re=100",
    provenance="Solver reference | case 0123456789abcdefabcd",
)
standalone_png = render_fields(fields, spec)
comparison_png = render_comparison(prediction, reference, spec)
```

`PngArtifact.data` contains the encoded image. `sha256` binds the exact PNG
bytes, while `contract_sha256` binds its dimensions, accessible description,
panel roles, labels, units, colormaps, and numerical ranges. The artifact also
exposes those panel records directly for API and regression checks. PNG metadata
stores the same description as `alt_text`.

## Numerical and visual contract

The schema-version-1 rules are fixed:

| Field | Definition | Display range | Colormap |
| --- | --- | --- | --- |
| Velocity magnitude | `sqrt(u**2 + v**2)` | `0` to finite fluid-cell p99.5 | `viridis` |
| Pressure proxy | `(rho - 1) / 3` | symmetric fluid-cell maximum absolute value | `coolwarm` |
| Vorticity | second-order `dv/dx - du/dy` | symmetric fluid-cell p99 absolute value | `RdBu_r` |

Constant fields receive the smallest positive float32 scale instead of a
degenerate color range. Non-finite, non-contiguous, incorrectly typed, or
geometry-inconsistent arrays fail before plotting. Vorticity uses the declared
`spacing_lu`, which defaults to two lattice units for the public downsampled
field grid.

Comparison output has reference, prediction, and error rows. Reference and
prediction use the exact same per-variable limits computed only from the
reference. Velocity error is vector magnitude on `magma`; pressure-proxy and
vorticity errors remain signed on symmetric diverging scales. This prevents an
independently rescaled prediction from concealing amplitude error.

Every panel uses equal physical aspect, lattice-unit axes, a visible flow arrow,
a labeled colorbar, and nearest-neighbor raster display. The last rule is
important: rendering never interpolates pixels in a way that implies more
numerical resolution than the source field contains. Obstacles use one opaque
neutral overlay while fluid values retain the selected scientific colormap.

## Reproducibility and accessibility

`RenderSpec` bounds the output to 600–2400 pixels wide, 300–1600 pixels high,
72–300 DPI, and four million pixels total. Annotation and provenance strings
are normalized and bounded before entering the figure or PNG metadata. The
Agg backend, DejaVu Sans font, explicit layout, nearest interpolation, fixed PNG
compression, and locked plotting dependencies make repeated renders
byte-identical in the supported environment.

Color is not the only carrier of meaning: panel titles, row roles, units,
numeric colorbars, direction, annotation, provenance, and a full textual
description accompany the raster. Consumers should publish `alt_text` beside
the image and verify `sha256` when moving it across an artifact boundary.

Run the rendering contract directly with:

```bash
uv run pytest tests/demo/test_rendering.py
```

The golden suite checks dimensions, labels, ranges, semantic and byte digests,
constant/non-finite behavior, shared comparison scales, lazy optional imports,
and byte-for-byte preservation of every input array.
