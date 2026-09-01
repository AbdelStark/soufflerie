# Warp collision and streaming kernels

Soufflerie's Warp adapter implements the unfused fp32 D2Q9 stages defined by
RFC-0002. It is an optional runtime boundary: importing `soufflerie` or
`soufflerie.solver` does not import Warp. Constructing `WarpKernelAdapter`
loads the backend and raises `DependencyUnavailableError` with installation
guidance when the `solver` extra is absent. Unknown devices raise
`DeviceUnavailableError` before allocation.

## State ownership

`LatticeState` owns exactly four mutable Warp arrays for one run:

```text
f         float32[ny, nx, 9]  current/streamed populations
f_next    float32[ny, nx, 9]  post-collision populations
rho       float32[ny, nx]     recovered density
velocity  float32[ny, nx, 2]  recovered (u, v)
```

`allocate` zero-initializes all buffers. `initialize` uses the deterministic
NumPy equilibrium contract and uploads it to the selected device. The adapter
rejects grid mismatches, buffer-shape drift, and mixed-device state rather than
launching against an incoherent state.

## Unfused stage sequence

`collide` launches one thread per cell. Each thread reads the nine populations
in fixed direction order, computes rho and momentum in fp32, and writes every
post-collision direction explicitly to `f_next`.

`pull_stream_periodic` launches one thread for each `(y, x, direction)`
destination. That thread reads exactly one periodic upstream population from
`f_next` and assigns exactly one element in `f`; there are no atomics or
scatter writes. Physical channel, inlet/outlet, wall, and obstacle ownership
remain separate boundary stages in issues #11 and #12.

`reduce_macroscopic` launches one thread per cell and writes rho plus both
velocity components. `step` composes collision, streaming, and reduction
without fusion. Each public stage synchronizes at its output boundary so host
comparisons, diagnostics, and timing never observe pending writes.

## Oracle and determinism contract

Small CPU states compare with the periodic NumPy oracle using `rtol=2e-6` and
`atol=2e-7` for fp32 arithmetic; the pull index mapping itself is exact. A
snapshot synchronizes and returns detached C-contiguous fp32 host arrays.
Repeated runs with identical state, lock, device class, and stage order must be
bitwise equal at those snapshot boundaries. This is same-environment evidence,
not a cross-device bitwise claim.

The JIT executes Warp DSL bodies outside Python's line tracer, so the backend
module is explicitly omitted from Python line coverage. Its collision,
streaming, reduction, ownership, and determinism behavior is exercised through
the CPU adapter tests and compared with the independently implemented NumPy
oracle.

Install and run the issue acceptance slice with:

```bash
uv sync --extra solver
uv run pytest tests/solver/test_kernels.py tests/solver/test_determinism.py
```
