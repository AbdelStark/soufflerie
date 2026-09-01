# Geometry and preflight

Soufflerie has one geometry contract for solver masks, persisted datasets, model inputs,
service validation, and rendering. All consumers call `soufflerie.geometry`; they must not
reimplement obstacle placement, axes, or masking.

## Coordinates and signed distance

Arrays use `[y, x]` order and lattice-cell-center coordinates. For a grid `(ny, nx)`, the
ellipse is centered at `(0.30 * (nx - 1), 0.50 * (ny - 1))`. Its unscaled reference major-axis
diameter is `D_lu = ny / 20`. Scale changes the major semi-axis, aspect ratio changes the
minor semi-axis, and rotation is counter-clockwise in degrees.

`ellipse_sdf(shape, grid)` returns a finite, C-contiguous, read-only `float32[ny, nx]` array.
Values are negative inside the ellipse, zero on its analytic contour, and positive outside.
The implementation is an algebraic ellipse distance: its sign and zero contour are exact, but
its magnitude away from the boundary is only an approximation to Euclidean distance.

`obstacle_mask(sdf)` returns a read-only Boolean array with `sdf <= 0` treated as solid.
`normalized_sdf_input(shape, grid)` produces the model channel
`clip(sdf / D_lu, -1, 1)` as read-only `float32`. These functions preserve the RFC-0003
convention; changing any of them invalidates downstream dataset and model identities.

## Fail-closed preflight

Call `validate_geometry(shape, grid)` before allocating solver state. It checks inexpensive
analytic constraints first, then raster connectivity:

1. the scaled minor diameter is at least 12 lattice cells;
2. inlet, outlet, and wall clearances are at least `2 * D_lu`, `4 * D_lu`, and `D_lu`;
3. the obstacle remains outside the last `max(16, round(2 * D_lu))` sponge columns; and
4. an eight-output-cell control surface fits inside the fluid, sponge-free domain; and
5. a one-cell-wide fluid path connects inlet to outlet between the no-slip walls.

Invalid cases raise `DomainError` with stable `GEO-1` or `GEO-2` diagnostic text. Resolution,
clearance, and sponge rejection occur before a dense geometry raster is allocated. Successful
preflight returns `GeometryDiagnostics`, including the resolved axes, four clearances, sponge
extent, exact output-grid control surface, solid/fluid cell counts, and connectivity result. The
surface is selected from an SDF recomputed on the fixed `(ny=320, nx=256)` output grid, so the algebraic
distance magnitude is checked directly rather than approximated from the ellipse's Euclidean
extent. These values are the case-level evidence to log when a requested geometry is accepted.

## Public domain decision

The v0.1 public aspect-ratio range is `0.5` to `1.0`, inclusive. On the canonical
`(ny=640, nx=512)` grid, the smallest public ellipse (`aspect_ratio=0.5`, `scale=0.75`) has exactly
12 cells across its minor diameter. The earlier draft floor of `0.3` would have produced only
7.2 cells. Soufflerie keeps the 12-cell boundary-resolution gate, the reference diameter, and
the reference diameter unchanged rather than weakening numerical acceptance. The fixed
`(ny=320, nx=256)` tensor is an isotropic two-cell area reduction, so masks, SDF, and fields retain
one shared coordinate system.

Reduced grids used by isolated solver unit tests are internal numerical fixtures. They are not
accepted public geometry cases. Geometry preflight itself is CPU-only and is tested on the
canonical grid.

## Example

```python
from soufflerie import GridSpec, ShapeParams, ellipse_sdf, obstacle_mask, validate_geometry

grid = GridSpec(nx=512, ny=640)
shape = ShapeParams(aspect_ratio=0.75, rotation_deg=15.0, scale=1.0)
diagnostics = validate_geometry(shape, grid)
sdf = ellipse_sdf(shape, grid)
solid = obstacle_mask(sdf)
```

The solver, dataset builder, service, and renderer must share these exact `sdf` and `solid`
semantics. `reference_diameter_lu(grid)` is likewise the only source for Reynolds and force
normalization.
