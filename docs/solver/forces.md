# Obstacle bounce-back and forces

Soufflerie derives obstacle links only from the shared geometry mask. Channel walls are handled
by the boundary operator and never enter obstacle-force coefficients.

## Link convention

`enumerate_obstacle_links(mask)` scans fluid cells in row-major `[y, x]` order and directions
`1` through `8` in ascending D2Q9 order. A link `(y, x, i)` means that `(y, x)` is fluid and
its neighbor `(y + c_iy, x + c_ix)` is solid. Each link is unique and the returned coordinate
and direction arrays are C-contiguous and read-only.

For an outgoing fluid-to-solid direction `i`, half-way bounce-back assigns the opposite incoming
population at the same fluid node:

```text
f_streamed_opp_i(x_fluid) = f_post_i(x_fluid)
```

Solid nodes are inactive and retain their post-collision state. Obstacle links have priority
over channel boundary pulls. Geometry preflight keeps the obstacle away from walls, inlet,
outlet, and sponge, so valid public cases have no ambiguous mixed-owner link.

## Momentum exchange and signs

The force on the obstacle from one link is the momentum transferred by reflection:

```text
delta_p = 2 * f_post_i(x_fluid) * c_i
```

The sign is the force on the obstacle: positive `Fx` is drag in the positive streamwise
direction, and positive `Fy` follows the D2Q9 positive-y direction. `momentum_exchange_force`
uses the post-collision state after sponge application and before pull streaming. Geometry
exclusion means sponge relaxation cannot alter an obstacle-link population.

Links are reduced in their declared order as fp64 scalars, in fixed 256-link tiles followed by
fixed tile order. At declared force-sample steps, the Warp path copies the post-collision buffer
to the host for this reduction; unsampled steps perform no host copy and the reduction uses no
nondeterministic atomics. This is the correctness-first v0.1 path. A later optimized
reduction must remain bitwise equal at persisted boundaries or replace the contract through an
RFC change.

Coefficients use the declared, unramped reference velocity:

```text
normalization = 0.5 * rho_ref * U_ref^2 * D_lu
Cd = Fx / normalization
Cl = Fy / normalization
rho_ref = 1
U_ref = config.inlet_velocity_lu
D_lu = config.reference_diameter_lu
```

Using the instantaneous ramp velocity would make early coefficients singular or incomparable,
so it is forbidden.

## Force histories

`NumpyObstacleStepper` and `WarpObstacleStepper` compose collision, sponge, force reduction,
obstacle-priority channel streaming, and macroscopic reduction. They sample force after warmup
at `warmup_steps + n * sample_interval`. Frozen histories store steps as `int64`, raw forces as
`float64`, and Cd/Cl as the canonical persisted `float32` vectors.

## Validation

```bash
uv run pytest tests/solver/test_obstacle.py tests/solver/test_forces.py
```

The suite covers exact single-cell link enumeration, unique bounce destinations, solid-node
inactivity, link/mask mismatch rejection, no-obstacle zero force, wall exclusion, drag sign,
mirrored lift, declared normalization, history cadence, repeatability, and exact CPU Warp/NumPy
state and force parity.
