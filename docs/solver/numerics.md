# D2Q9 numerical foundation

Soufflerie's first solver layer is a deterministic, isothermal D2Q9 BGK
contract. It is intentionally split into framework-independent lattice logic
and a small periodic NumPy oracle. The oracle is a correctness reference for
CPU/GPU kernels; it is not the physical channel solver and does not implement
walls, inlet/outlet treatment, obstacle links, forces, or convergence claims.

## Layout and constants

Population state is C-contiguous `float32[ny, nx, 9]`; velocity is
`float32[ny, nx, 2]`. Directions use the fixed RFC-0002 order:

```text
c = [(0,0), (1,0), (0,1), (-1,0), (0,-1),
     (1,1), (-1,1), (-1,-1), (1,-1)]
w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]
```

The exported NumPy constants are read-only. `equilibrium` and
`macroscopic_moments` reject wrong shapes, non-`float32` state, non-contiguous
buffers, non-finite values, and non-positive recovered density rather than
silently casting or repairing them.

## Lattice derivation and preflight

`derive_lattice(case)` maps `CaseConfig` to `DerivedLatticeConfig`. The
unscaled reference diameter is `D_lu = 0.125 * ny`, the diagnostic sample
interval is 10 steps, and the inlet ramp is capped at 2,000 warm-up steps.
The exact derived quantities are:

```text
nu_lu = inlet_velocity_lu * D_lu / Reynolds
tau = 3 * nu_lu + 0.5
omega = 1 / tau
Mach = inlet_velocity_lu / sqrt(1/3)
```

Preflight requires grids of at least 3 by 3, a coherent positive run schedule,
positive finite Reynolds number and diameter, inlet velocity in `(0, 0.1]`,
`tau in [0.5005, 1.95]`, and nominal Mach at most `0.1733`. Boundaries are
inclusive where shown. Invalid requests raise `DomainError`; tau, velocity, or
Reynolds values are never clipped. Geometry clearance and obstacle resolution
remain the responsibility of the shared geometry preflight.

## NumPy oracle

`initialize_numpy` creates density one, the declared streamwise inlet velocity
in fluid cells, zero velocity in an optional boolean obstacle mask, and
equilibrium populations. It performs no random operation.

`collide_numpy` applies fp32 BGK collision. `pull_stream_periodic_numpy` then
pulls direction `i` at destination `(y, x)` from upstream
`((y-c_y) mod ny, (x-c_x) mod nx)`. `numpy_periodic_step` composes those stages
and recovers next-step moments. Inputs are never mutated and no global RNG
state is read or written. Issue #7's Warp kernels must compare both stage
outputs with this oracle at the declared fp32 tolerance.

Run the issue acceptance slice with:

```bash
uv run pytest tests/solver/test_lattice.py tests/solver/test_numpy_oracle.py
```
