# RFC-0002: D2Q9 lattice Boltzmann core

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

The v0.1 solver is a deterministic, weakly compressible, isothermal D2Q9 lattice Boltzmann method with single-relaxation-time BGK collision, pull streaming, fp32 state, and explicit preflight stability bounds. It exposes validated mean fields and histories rather than framework buffers.

## Motivation

The PRD requires a compact GPU-native solver that teaches explicit kernels and supplies data for the surrogate. The numerical method must be precise enough to implement and test without treating the framework or visual output as a correctness oracle. [`00-overview.md#hardest-problems`](../spec/00-overview.md#hardest-problems) identifies numerical fidelity as a release risk.

## Goals

- Specify the lattice, equilibrium, collision, streaming, macroscopic reduction, initialization, and averaging algorithms.
- Enforce stable mappings from physical design parameters to lattice units.
- Produce deterministic fp32 results on the same hardware/runtime lock.
- Pass analytic channel flow, mass conservation, and cylinder reference gates.
- Measure kernel and end-to-end performance without weakening correctness.

## Non-Goals

- Compressible, thermal, multiphase, entropic, MRT, or turbulence models.
- Adaptive grids, body-fitted meshes, or three dimensions.
- Bitwise equality across different device classes or runtime versions.
- Solver autodifferentiation in v0.1.

## Proposed Design

The lattice velocities and weights are fixed:

```text
c = [(0,0), (1,0), (0,1), (-1,0), (0,-1),
     (1,1), (-1,1), (-1,-1), (1,-1)]
w = [4/9, 1/9, 1/9, 1/9, 1/9,
     1/36, 1/36, 1/36, 1/36]
opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]
cs2 = 1/3
```

For each fluid cell:

```text
rho = sum_i(f_i)
u = sum_i(f_i * c_ix) / rho
v = sum_i(f_i * c_iy) / rho
cu_i = c_ix*u + c_iy*v
feq_i = w_i*rho*(1 + 3*cu_i + 4.5*cu_i^2 - 1.5*(u^2+v^2))
f_post_i = f_i - omega*(f_i-feq_i)
omega = 1/tau
tau = 3*nu_lu + 0.5
nu_lu = inlet_velocity_lu * D_lu / reynolds
```

`inlet_velocity_lu` defaults to `0.05` and MUST be in `(0, 0.1]`. Preflight requires `tau in [0.5005, 1.95]`, maximum nominal Mach `inlet_velocity_lu / sqrt(1/3) <= 0.1733`, `rho_ref=1.0`, and the geometry-clearance invariants in RFC-0003. If a requested Reynolds number violates tau at the selected `D_lu`, the caller must increase obstacle/grid resolution; clipping tau, velocity, or Reynolds number is forbidden.

Core interfaces are:

```python
@dataclass(frozen=True, slots=True)
class LatticeConfig:
    nx: int
    ny: int
    steps: int
    warmup_steps: int
    sample_interval: int
    inlet_velocity_lu: float
    reynolds: float
    reference_diameter_lu: float

@dataclass(slots=True)
class LatticeState:
    f: wp.array3d              # [ny, nx, 9], float32
    f_next: wp.array3d         # same
    rho: wp.array2d            # [ny, nx], float32
    velocity: wp.array3d       # [ny, nx, 2], float32

def derive_lattice(case: CaseConfig) -> DerivedLatticeConfig: ...
def initialize(config: DerivedLatticeConfig, mask: BoolArray, device: str) -> LatticeState: ...
def advance(state: LatticeState, config: DerivedLatticeConfig, mask: BoolArray) -> StepDiagnostics: ...
def solve(case: CaseConfig, *, device: str = "cpu") -> SolverResult: ...
```

The memory layout is row-major `[ny, nx, q]`; velocity is `[ny, nx, 2]`. Kernels use one thread per cell and fixed-order unrolled nine-direction operations. Pull streaming reads populations from upstream neighbors, applies boundary ownership, and writes each destination exactly once, avoiding atomics. Collision and streaming may fuse only if regression and profiler evidence show unchanged outputs/contracts.

Initialization sets `rho=1`, velocity to the inlet profile in fluid, zero in obstacles, and populations to equilibrium. A deterministic smooth ramp multiplies inlet velocity for the first `min(2_000, warmup_steps)` steps:

```text
ramp(t) = 0.5 * (1 - cos(pi * min(t/ramp_steps, 1)))
```

The solver samples diagnostics every `sample_interval` and checks finite populations, `rho in [0.5, 1.5]`, and maximum speed `<=0.2`. Violation raises `NumericalStabilityError` immediately. Time averaging begins at `warmup_steps`, accumulates fp64 sums on the host or deterministic reduction path, and returns fp32 means. The final averaging window contains at least 4,000 steps and 200 samples in canonical cases.

The solver seed is reserved for any deterministic initialization perturbation and recorded even when zero perturbation is used. No random operation may read global RNG state. Kernel launches synchronize at diagnostic, timing, and output boundaries. Same case, code, lock, device class, and determinism mode must persist bitwise-identical arrays and histories.

<a id="acceptance-invariants"></a>
### Acceptance invariants

- `LBM-1 CONSERVATION`: canonical 20,000-step cases have total density drift `<0.001`.
- `LBM-2 FINITE`: all persisted values are finite and density remains positive.
- `LBM-3 STABILITY`: preflight and runtime tau/Mach/range bounds are never bypassed.
- `LBM-4 POISEUILLE`: analytic profile relative L2 and max error are each `<=0.01` after declared exclusions.
- `LBM-5 DETERMINISM`: same-environment repeated runs are bitwise equal at persisted boundaries.
- `LBM-6 PRECISION`: state and kernels compute in fp32; metric reductions use fp64.
- `LBM-7 PERFORMANCE`: the reference full run meets [`08-performance-budget.md#budgets`](../spec/08-performance-budget.md#budgets) after correctness gates.

Errors propagate as typed errors; numerical invalidity is never retried unchanged. A failed diagnostic artifact may retain scalar histories and configuration, but it cannot masquerade as `SolverResult` or enter a dataset.

## Alternatives Considered

### Finite-volume Navier-Stokes solver

It would resemble conventional CFD more closely but requires pressure coupling, meshing or immersed-boundary complexity, and a larger validation burden. The Cartesian LBM directly serves the educational and GPU-kernel scope.

### Multiple-relaxation-time collision

MRT can improve stability, especially at higher Reynolds numbers, but adds basis transforms and parameters that obscure the compact v0.1 method. BGK is accepted with strict stability/domain gates; MRT requires a superseding RFC if BGK cannot meet them.

### Push streaming

Push streaming naturally emits outgoing populations but causes competing writes or requires careful scatter buffers. Pull streaming owns each destination and gives deterministic write structure.

### fp64 solver state

It can reduce round-off but conflicts with the explicit fp32 learning/performance goal and is not required by declared validation gates. fp64 remains limited to diagnostic reductions.

## Drawbacks

- BGK and simple grids limit stable resolution/Re combinations.
- Weak compressibility and boundary placement produce discretization error.
- Same-device bitwise determinism constrains kernel optimization and reductions.
- Host fp64 averaging may add transfer cost if not carefully batched.

## Migration / Rollout

1. Implement lattice constants, config derivation, allocation, initialization, and macroscopic reduction with unit tests.
2. Add collision/streaming with periodic/simple channel boundaries and mass tests.
3. Add Poiseuille fixture and determinism evidence before obstacle work.
4. Integrate RFC-0003 boundaries/forces and execute full remote acceptance.
5. Record the accepted numerical configuration as a versioned config and golden summary.

Schema-breaking numerical changes require new dataset/model identities and regeneration; old outputs are never re-labelled.

## Testing Strategy

- Hand-compute equilibrium at rest and low velocity; verify symmetry and density/momentum moments.
- Property-test equilibrium finite values and moment recovery within fp32 tolerance.
- Test tau/Re derivation at boundaries and rejection just outside each bound.
- Compare one-step kernel output with a pure NumPy reference on small periodic grids.
- Verify pull-stream index mapping and opposite-direction table exhaustively.
- Run Poiseuille at two grid sizes and require convergence plus the `1%` gate.
- Run 20,000-step mass and same-device bitwise determinism tests.
- Inject NaN, density, and speed violations and assert immediate typed failure.
- Benchmark synchronized kernel and end-to-end timing only after all gates pass.

## Open Questions

None for v0.1. Failure to cover the declared Reynolds domain under these bounds is a release blocker and triggers a replacement RFC owned by the solver maintainer, not an implementation-local relaxation.

## References

- [`prd.md#61-solver-warp-mini-lbm`](../../prd.md#61-solver-warp-mini-lbm)
- [`03-data-model.md#units-and-coordinates`](../spec/03-data-model.md#units-and-coordinates)
- [`07-testing-strategy.md#numerical-oracles`](../spec/07-testing-strategy.md#numerical-oracles)
- [RFC-0003](RFC-0003-geometry-boundaries-and-forces.md)
- Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*, 2017.
