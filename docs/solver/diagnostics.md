# Strouhal and field-drag diagnostics

Soufflerie exposes two independent force diagnostics. The solver label remains the
momentum-exchange Cd described in [forces.md](forces.md). A lift spectrum derives Strouhal from
that solver history, while a rectangular control-volume integral derives `cd_field` from mean
flow fields. The latter is a consistency signal for predicted fields, never a replacement label
or training target.

## Force-history summary

`mean_force_coefficients(history)` accumulates `float32` Cd/Cl samples as fp64 scalars in
chronological order. It selects the right-closed interval
`(last_sample_step - 4000, last_sample_step]`; with the canonical ten-step cadence this is
exactly 400 samples. An empty history is invalid rather than silently producing NaN.

`estimate_strouhal(history, config)` requires post-warmup samples at the configuration's exact,
uniform cadence. It converts lift to fp64, subtracts its fp64 mean, applies a Hann window, and
runs `numpy.fft.rfft`. Zero frequency is excluded. The strongest bin whose dimensionless
frequency satisfies `0.05 <= St <= 0.4` is selected; a three-bin parabola over log magnitude
refines an interior peak and is bounded to half a bin and the accepted band.

The observed duration is `last_step - first_step`. A value is published only when the refined
frequency spans at least eight cycles over that duration. Short histories, irregular cadence,
float32-degenerate lift, an unresolvable band, zero windowed energy, and insufficient cycles all
return `StrouhalEstimate(strouhal=None, reason=...)`. `StrouhalUnavailableReason` is a stable
string enum, so callers do not need to parse prose or fabricate a scalar. The canonical cylinder
acceptance in issue #14 requires an available value.

## Control-surface selection

`select_control_surface(fields, sponge_start_x=...)` consumes validated fp32 `FlowFields` whose
SDF is expressed in output-cell lattice units. It forms the strict clearance band `sdf < 8` and
selects the tightest axis-aligned rectangle one cell outside that band's bounding box. Therefore
every inclusive side sample has `sdf >= 8`. The rectangle must remain off the inactive wall rows,
away from the inlet edge, and strictly before the sponge. A missing zero contour or a rectangle
that cannot fit raises `DomainError` with `GEO-2 CONTROL_SURFACE`; it cannot yield a misleading
Cd.

The canonical solver grid is `512 x 256` and its fixed output grid is `256 x 128`, so eight output
cells correspond to 16 full-grid lattice units. Because the RFC uses an algebraic rather than
Euclidean SDF magnitude, `validate_geometry` recomputes the exact output-grid SDF and runs the
same selector used by field drag. Successful `GeometryDiagnostics` retain that output control
surface; an invalid case is rejected during preflight.

## Field-drag integral

`field_drag_coefficient` uses `p = rho / 3` and a fixed increasing-index fp64 sum on all four
sides. For an outward control-volume normal, the force on the obstacle is the negative of the
fluid momentum balance. In inclusive output-cell coordinates this is:

```text
Fx_pressure   = sum_left(p) - sum_right(p)
Fx_convective = sum_left(rho*u^2) - sum_right(rho*u^2)
              + sum_bottom(rho*u*v) - sum_top(rho*u*v)
Cd_field      = (Fx_pressure + Fx_convective)
              / (0.5 * rho_ref * U_ref^2 * D_lu)
```

`rho_ref` is fixed at one. `U_ref` is the declared unramped inlet velocity and `D_lu` is expressed
in the same output-cell coordinate system as the surface. `FieldDragEstimate` retains the
surface, both force components, total force, normalization, and Cd so downstream validation can
show how the consistency value was obtained.

Run the acceptance slice with:

```bash
uv run pytest tests/solver/test_strouhal.py tests/validation/test_field_drag.py
```
