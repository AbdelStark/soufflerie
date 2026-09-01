# CPU solver gate evidence

`cpu-gates.json` is the first schema-v1 numerical regression summary generated
for issue #9. It records two independent NumPy D2Q9 analytic-channel fixtures
and two same-environment, 20,000-step Warp CPU periodic regressions.

The Poiseuille fixture deliberately uses periodic streamwise flow, half-way
no-slip transverse walls, and a constant Guo body force. It is the RFC-0002
simple-channel analytic fixture, not evidence for RFC-0003's production
regularized inlet, outlet reconstruction, sponge, corners, or obstacles. Issue
#11 must rerun the 1% gate with those full boundaries.

The checked-in record was generated under the platform and locked versions
named in the JSON. Cross-device results use the same thresholds but are not
expected to reproduce its bitwise digests. On the recorded platform, verify it
with:

```bash
uv run python scripts/generate_cpu_solver_gates.py \
  --check reports/solver/cpu-gates.json
```

Updating the golden requires a review note naming the numerical reason, a new
`generation_revision` when the algorithm or fixture changes, rerunning both
Poiseuille grids, and rerunning the two 20,000-step mass/determinism repetitions.
Formatting-only or tolerance-relaxation updates are not valid rationales.
