# Performance budget

<a id="measurement-policy"></a>
## Measurement policy

Performance claims are valid only with a recorded source revision, dependency lock digest, configuration, warm/cold state, device class, sample count, and percentile definition. Wall time uses a monotonic clock and synchronizes asynchronous device work at measurement boundaries. Correctness gates run before benchmarks. Failed or retried runs remain in operational accounting and are not silently excluded.

<a id="reference-workloads"></a>
## Reference workloads

- `solver-full`: fp32, `(ny=256, nx=512)`, 20,000 steps, one valid ellipse case, L40S-class GPU, end-to-end remote invocation including artifact I/O.
- `predict-gpu`: batch 1, `(128, 256)`, loaded/warm FNO bundle, GPU service worker; includes preprocessing, forward, consistency, and PNG/NPZ encoding.
- `predict-cpu`: same request and bundled model on the documented reference CPU; hardware model and thread count recorded.
- `sweep-1000`: canonical 1,000 cases with bounded `50-100` task fan-out; measures total wall time and total GPU seconds separately.
- `train-mvp`: three seeds and selected configuration on one remote GPU per seed, including validation checkpoint production.

<a id="budgets"></a>
## Budgets

| Operation | Budget | Gate |
|---|---:|---|
| Full solver | `< 60 s` end-to-end on L40S-class GPU | p95 over 10 representative cases |
| Solver throughput | reported LUPS, no fixed claim until first accepted baseline | mandatory measurement |
| Warm GPU prediction | `< 100 ms` | p95 over 200 requests after 20 warmups |
| Local CPU prediction | `< 1,000 ms` | p95 over 50 requests after 5 warmups |
| Predict service queue | `< 100 ms` at supported concurrency | p95 in load test |
| Full model training | `< 60 min` per seed | wall time on reference GPU |
| Dataset payload | `< 2 GiB` for 1,000 curated runs | manifest sum of archive sizes |
| Service resident GPU memory | `< 12 GiB` | peak allocated after warmup |
| API response | `< 2 MiB` compressed/body | hard rejection in encoder |
| Remote image cold build | recorded, not included in warm prediction | informational |

The A10G fallback must pass functional and memory gates. L40S latency targets do not transfer to it; results must name the device class.

<a id="resource-controls"></a>
## Resource controls

Solver buffers are preallocated and reused within a run. Snapshot retention is configured and bounded; v0.1 persists time-averaged fields plus force history and only declared visualization frames. Dataset readers stream per-run archives and avoid holding the full dataset in host memory. Training loaders use bounded prefetch and pinned memory only on CUDA. Response encoding limits decoded array dimensions before allocation.

<a id="profiling-plan"></a>
## Profiling plan

1. Record kernel-stage time for collision, streaming, boundaries, macroscopic reduction, force reduction, and I/O.
2. Report LUPS as `nx * ny * executed_steps / synchronized_kernel_seconds` and separately report end-to-end time.
3. Profile FNO preprocessing, forward, Cd heads, consistency calculation, and encoders independently.
4. Capture CPU and GPU memory peaks.
5. Store raw benchmark JSON in `bench/results/<machine>/`; commit only reviewed baselines in `bench/results/baseline/`.

Optimization is accepted only when the same correctness inputs and outputs pass. Precision may not drop below the declared fp32 compute/fp16 curated-storage contract without a superseding RFC.

<a id="regression-policy"></a>
## Regression policy

A change fails the performance gate when median or p95 regresses more than `10%` against the matching baseline across at least three benchmark repetitions and measurement noise is below `5%`. Intentional regressions require a documented trade-off in the changelog and an updated accepted budget; a single faster sample is not evidence.
