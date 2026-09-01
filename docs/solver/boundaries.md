# Channel boundaries

The v0.1 solver uses one explicit, deterministic ownership rule for D2Q9 pull streaming. The
NumPy oracle in `soufflerie.solver.boundaries` and the unfused Warp stages implement the same
fp32 calculations. Neither path uses atomics or relies on array overwrite order.

## Stage order

One channel step runs these stages in order:

1. BGK collision produces `f_post`.
2. The outlet sponge relaxes fluid populations toward the current ramped inlet equilibrium.
3. Pull streaming assigns every destination population using wall, inlet, outlet, or interior
   ownership.
4. Density and velocity are reduced from the completed streamed state.

The two outer rows (`y=0` and `y=ny-1`) are inactive wall nodes. Physical channel fluid spans
`y=1` through `y=ny-2`. When a pull link crosses either wall, half-way bounce-back reads the
opposite post-collision population from the same fluid node. This is the same wall operator used
by the Poiseuille acceptance fixture.

## Inlet and outlet

At `x=0`, the three west-originating populations use Zou/He density reconstruction with the
current half-cosine-ramped target `u` and `v=0`. Away from the corners, the inlet then owns all
nine populations and projects the provisional state onto the second-order non-equilibrium
Hermite tensor. This regularization removes unsupported higher-order content while preserving
the reconstructed density and velocity.

At `x=nx-1`, incoming populations use non-equilibrium extrapolation from `x=nx-2`:

```text
f_out_i = feq_i(rho_adj, u_adj) + (f_adj_i - feq_i(rho_adj, u_adj))
```

The adjacent density and velocity are evaluated after collision, sponge relaxation, and pull.
With the zero-gradient target this expression is algebraically the adjacent population, but the
equilibrium/non-equilibrium split states the contract used by later boundary variants.

## Corner ownership

Wall ownership has priority over inlet and outlet ownership. Only the remaining streamwise
incoming links are reconstructed at corner-adjacent fluid cells:

| Fluid cell | Wall-owned directions | Streamwise-owned directions |
|---|---|---|
| lower inlet `(1, 0)` | `2, 5, 6` | inlet `1, 8` |
| upper inlet `(ny-2, 0)` | `4, 7, 8` | inlet `1, 5` |
| lower outlet `(1, nx-1)` | `2, 5, 6` | outlet `3, 7` |
| upper outlet `(ny-2, nx-1)` | `4, 7, 8` | outlet `3, 6` |

All other directions at those cells are ordinary interior pulls. Use
`channel_boundary_ownership(grid)` to inspect the read-only owner array; its values are the
`BoundaryOwner` enum. Obstacle-link priority is added by the obstacle bounce-back stage and
remains outside this boundary-only contract.

## Sponge

The final `max(16, round_half_up(2 * D_lu))` columns form the sponge. Strength is zero at its
first column and rises quadratically to `0.15` at the outlet. Relaxation is

```text
f_post = f_post + strength * (feq(rho=1, u=inlet_ramp, v=0) - f_post)
```

It applies only to physical fluid rows. Geometry preflight guarantees the obstacle ends before
the sponge, and both execution paths redundantly reject an obstacle mask that violates that
invariant. A sponge that would occupy the entire streamwise grid is rejected before a kernel
launch.

## Validation

Run the focused boundary and shared analytic-channel suite with:

```bash
uv run pytest tests/solver/test_boundaries.py tests/solver/test_poiseuille.py
```

The tests pin all four corner mappings, inlet macroscopic reconstruction, outlet extrapolation,
sponge endpoints and exclusions, one-writer Warp source structure, exact CPU Warp/NumPy parity,
and both Poiseuille grid gates.
