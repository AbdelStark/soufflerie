# RFC-0003: Geometry, boundaries, and force diagnostics

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

Elliptical obstacles are represented by one analytic signed-distance contract shared by solver and surrogate. The solver uses half-way link bounce-back, no-slip channel walls, regularized velocity inlet, zero-gradient outlet with a fixed sponge zone, momentum-exchange forces, and lift-spectrum Strouhal estimation.

## Motivation

Boundary conditions and force normalization dominate whether the compact solver can pass the cylinder and conservation gates. The same geometry signal must feed numerical masks, datasets, the model, API validation, and visualization without drift. This RFC locks those choices for [`01-architecture.md#primary-data-flow`](../spec/01-architecture.md#primary-data-flow).

## Goals

- Define one deterministic ellipse SDF, placement, mask, and reference length.
- Specify inlet, outlet, wall, obstacle, and sponge behavior including corner ownership.
- Define Cd, Cl, and Strouhal calculations with sampling windows.
- Guarantee geometry stays resolved and separated from external boundaries.
- Provide failure diagnostics specific enough to reproduce rejected cases.

## Non-Goals

- Arbitrary user SDFs, multiple obstacles, moving geometry, or shape topology changes.
- Sub-grid interpolated bounce-back or curved-boundary order claims.
- Physical-unit conversion or engineering force estimates.
- General spectral analysis tooling.

## Proposed Design

`ShapeParams` follows [`03-data-model.md#canonical-types`](../spec/03-data-model.md#canonical-types). The obstacle center is `(x_c=0.30*(nx-1), y_c=0.50*(ny-1))`. The unscaled reference major-axis diameter is `D_lu=0.125*ny`; `scale` produces major semi-axis `a=0.5*D_lu*scale` and minor semi-axis `b=a*aspect_ratio`. At cell-center coordinates, rotate into obstacle axes:

```text
x' =  cos(theta)*(x-xc) + sin(theta)*(y-yc)
y' = -sin(theta)*(x-xc) + cos(theta)*(y-yc)
q = sqrt((x'/a)^2 + (y'/b)^2)
sdf = (q - 1) * min(a, b)
mask = sdf <= 0
```

This algebraic ellipse distance has exact sign and zero contour but approximate Euclidean magnitude away from the boundary. The surrogate contract depends only on signed, clipped magnitude; the approximation is accepted and named honestly. `sdf_input = clip(sdf / D_lu, -1, 1)`.

```python
def ellipse_sdf(shape: ShapeParams, grid: GridSpec) -> Float32Array: ...
def obstacle_mask(sdf: Float32Array) -> BoolArray: ...
def validate_geometry(shape: ShapeParams, grid: GridSpec) -> GeometryDiagnostics: ...
```

Preflight requires at least 12 cells across the scaled minor diameter; at least `2*D_lu` clearance to inlet, `4*D_lu` to outlet, and `1*D_lu` to either channel wall; and one-cell fluid connectivity from inlet to outlet. The public aspect-ratio range is `[0.5,1.0]`, so its minimum with `scale=0.75` is exactly 12 cells on the canonical `(ny=256,nx=512)` grid. Reduced solver smoke grids are internal numerical fixtures, not accepted public geometry cases; canonical-grid geometry preflight remains a CPU test. Any requested grid that cannot resolve the obstacle is rejected before solver-state allocation rather than rasterized inconsistently.

This `[0.5,1.0]` floor is the v0.1 resolution decision. Retaining the earlier `0.3` draft bound would produce a 7.2-cell minor diameter while relaxing the 12-cell gate would weaken the accepted boundary model. Increasing the full grid would multiply solver and sweep cost, and increasing `D_lu/ny` would change blockage and invalidate the cylinder reference. The bounded domain cut preserves the numerical, reference, and performance contracts together.

Boundary ownership order is: obstacle links, channel walls, inlet, outlet, then interior. The outer rows are inactive wall nodes and physical channel fluid spans rows `1..ny-2`. At each of the four corner-adjacent fluid cells, wall-crossing diagonal populations remain wall-owned; inlet or outlet reconstruction owns only the remaining streamwise incoming populations. Every destination population has one writer and the four mappings are fixture-backed. Top/bottom channel walls and obstacles use half-way link bounce-back: when pull streaming crosses a solid link, the destination population receives the opposite post-collision population from the fluid node. Force contribution on each obstacle link is accumulated in fixed direction/tile order:

```text
delta_p = 2 * f_post_i(x_fluid) * c_i
Fx,Fy = deterministic_sum(delta_p over obstacle links)
Cd = Fx / (0.5 * rho_ref * U_ref^2 * D_lu)
Cl = Fy / (0.5 * rho_ref * U_ref^2 * D_lu)
```

Channel walls are excluded from obstacle force.

The inlet uses a regularized Zou/He velocity boundary with prescribed `u=U_ref*ramp(t)`, `v=0`, and density reconstructed from known populations. Away from wall corners, the inlet owns all nine populations and projects the provisional Zou/He state onto the second-order non-equilibrium Hermite tensor while preserving reconstructed density and velocity. The outlet copies first-order non-equilibrium populations from the adjacent interior column with zero streamwise gradient and reconstructs equilibrium at the local density/velocity. The last `max(16, round_half_up(2*D_lu))` fluid columns form a sponge applied after BGK collision and before pull streaming: collision relaxes toward the current ramped inlet equilibrium with strength increasing quadratically from `0` to `0.15`. The obstacle cannot enter the sponge.

Force histories are sampled every 10 steps after `warmup_steps`. Mean Cd/Cl use the last 4,000 steps. Strouhal requires at least eight resolved lift cycles and a non-degenerate lift signal. It subtracts the mean, applies a Hann window, uses an `rfft` in fp64, ignores zero frequency, and selects the strongest frequency in dimensionless `St in [0.05, 0.4]`. Parabolic peak interpolation refines the bin. If prerequisites fail, `strouhal=None` with a typed diagnostic; the canonical cylinder gate requires a value.

Predicted-field Cd uses a rectangular control surface whose nearest side is at least eight output cells from the SDF zero contour and inside the sponge-free domain. With `p=cs2*rho`, it deterministically integrates pressure and convective streamwise momentum flux per unit span. Cases without a valid control surface are rejected at geometry preflight. This field estimate is a consistency signal, not the solver's momentum-exchange label.

<a id="acceptance-invariants"></a>
### Acceptance invariants

- `GEO-1 SINGLE SOURCE`: solver mask, dataset SDF, service validation, and rendering use the same geometry function/version.
- `GEO-2 RESOLUTION`: every accepted case passes minor-axis and clearance bounds.
- `BC-1 OWNERSHIP`: every incoming population at every boundary is assigned exactly once.
- `BC-2 NO SLIP`: obstacle and wall behavior passes channel/obstacle regression tests.
- `FORCE-1 NORMALIZATION`: Cd/Cl always use the declared `D_lu`, `rho_ref`, and `U_ref`.
- `CYL-1 REFERENCE`: the circular `Re=100` reference yields `St in [0.15,0.19]` and mean `Cd in [1.1475,1.5525]`.

## Alternatives Considered

### Exact Euclidean ellipse distance

Iterative closest-point evaluation would improve distance magnitude but costs more and creates CPU/GPU algorithm parity concerns. The algebraic signed distance is deterministic, sufficient for masking/input, and explicitly not claimed exact away from the surface.

### Stair-step on-node bounce-back

It is simpler but places the effective boundary inconsistently and worsens force/grid convergence. Half-way link bounce-back gives a clearer geometric convention at small additional complexity.

### Convective outlet boundary

It can reduce reflection but adds a tunable wave speed and previous-step state. A zero-gradient non-equilibrium reconstruction plus bounded sponge is easier to test for the fixed domain.

### Force from pressure integration on the obstacle

The stair-step boundary lacks a clean continuous normal/stress estimate. Momentum exchange is native to bounce-back and becomes the solver label; control-volume integration is retained only as an independent surrogate consistency estimate.

## Drawbacks

- The rasterized ellipse and half-way bounce-back retain grid-orientation error.
- Fixed placement and clearance reduce geometry flexibility.
- The sponge changes downstream dynamics and needs regression coverage.
- Lift FFT may be undefined for short/non-periodic histories.

## Migration / Rollout

1. Implement geometry/SDF with property and snapshot tests.
2. Land walls/inlet/outlet and Poiseuille evidence.
3. Add obstacle link bounce-back and deterministic force reduction.
4. Add sponge, force history, and Strouhal estimator.
5. Run grid-sensitivity and full cylinder acceptance remotely; freeze the canonical config.

Changing geometry or force semantics invalidates downstream datasets/models and requires new schema/config identities.

## Testing Strategy

- Property-test SDF sign, circle rotation invariance, scale monotonicity, and grid determinism.
- Test exact preflight thresholds and connected-channel constraint.
- Compare each boundary population against hand-computed fixtures, including four corners.
- Verify a no-obstacle channel has zero obstacle force.
- Verify force sign under mirrored lift and drag symmetry under positive/negative rotation where applicable.
- Test FFT on synthetic sinusoidal lift with off-bin frequency, noise, insufficient cycles, and zero signal.
- Validate control-volume Cd on manufactured constant and symmetric fields.
- Run cylinder `Re=100`, grid sensitivity, and mass gates on the accepted remote config.

## Open Questions

None for v0.1. A failure of the fixed boundary scheme to pass reference gates is owned by the solver maintainer and requires a replacement RFC before dataset generation.

## References

- [`prd.md#61-solver-warp-mini-lbm`](../../prd.md#61-solver-warp-mini-lbm)
- [`03-data-model.md#units-and-coordinates`](../spec/03-data-model.md#units-and-coordinates)
- [RFC-0002](RFC-0002-d2q9-lbm-core.md)
- Latt et al., “Straight velocity boundaries in the lattice Boltzmann method,” 2008.
- Ladd, “Numerical simulations of particulate suspensions via a discretized Boltzmann equation,” 1994.
