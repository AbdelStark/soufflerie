# Solver lifecycle and operations

The lifecycle layer owns schedule enforcement, synchronization boundaries,
runtime numerical checks, deterministic host accumulation, and the distinction
between completed and failed runs. Boundary physics remains a stepper concern:
the lifecycle supplies each step's half-cosine inlet target so issue #11's
channel stepper can apply it without duplicating schedule logic.

## Run preflight

A run is rejected before allocation unless its post-warmup window contains at
least 4,000 steps and at least 200 samples at the declared interval. Averaging
samples occur at `warmup_steps + n * sample_interval`; the first sample is one
full interval after warmup and the last cannot exceed `steps`. These gates are
fixed defaults, not test-only knobs.

The obstacle mask must be C-contiguous boolean data with shape `(ny, nx)`. The
framework-independent `NumpyPeriodicStepper` provides local contract evidence.
`WarpPeriodicStepper` constructs the optional synchronized Warp adapter. Both
are periodic drivers; they do not claim physical inlet, outlet, wall, or
obstacle behavior.

## Loop and synchronization

For completed step `t`, the lifecycle computes:

```text
ramp(t) = 0.5 * (1 - cos(pi * min(t / ramp_steps, 1)))
U(t) = inlet_velocity_lu * ramp(t)
```

The target `U(t)` is passed to the stepper. Diagnostic snapshots occur every
`sample_interval`, at averaging samples when the warmup alignment differs, and
at the final step. Warp snapshots synchronize first. Each snapshot is a
detached fp32 host copy; asynchronous work cannot race diagnostics or
accumulation.

Runtime checks reject non-finite populations, density, or velocity; density
outside `[0.5, 1.5]`; non-positive mass; and maximum fp32 speed above `0.2`.
The inclusive `0.2` endpoint uses its nearest fp32 representation. A violation
raises `SolverStabilityFailure`, stops at that inspection boundary, and carries
only a scalar `FailedSolverRun`. It is not retryable unchanged and has no mean
fields or conversion path to `SolverResult`.

## Averaging and completion

At each declared averaging sample, `u`, `v`, and `rho` are converted explicitly
to fp64 and added in chronological order. Only after all samples are consumed
are means divided and cast once to validated C-contiguous fp32 arrays. Scalar
mass, density, and speed summaries are derived with fp64 host reductions.

Final mass drift must be strictly below `0.001`. A finite run at or above that
threshold raises `SolverConvergenceFailure` with `SolverDiagnostics(valid=false,
converged=false)`; it cannot return `CompletedLatticeRun`. Successful runs bind
the fp32 means, exact sample steps, diagnostic history, device class, and valid
`SolverDiagnostics`. Geometry, SDF fields, force histories, provenance, and
final `SolverResult` assembly remain owned by their dedicated downstream
issues.

Run the acceptance slice with:

```bash
uv run pytest tests/solver/test_lifecycle.py tests/solver/test_failures.py
```
