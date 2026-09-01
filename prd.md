# Soufflerie

**A miniature GPU-native virtual wind tunnel.** A lattice Boltzmann solver written in NVIDIA Warp generates 2D flow data across shapes and Reynolds numbers. A PhysicsNeMo surrogate learns the mapping from geometry to flow field and drag. A validation harness decides, with numbers, whether the surrogate deserves trust. A small web demo turns the whole thing into a millisecond design loop: drag a slider, watch the flow change.

*Soufflerie* is French for wind tunnel. Repo: `soufflerie`. License: Apache 2.0. Public from day one.

---

## 1. Purpose

Learn the NVIDIA simulation-AI stack by building one coherent artifact instead of reading about it. Each component maps to a layer of the stack:

| Component | Stack layer it teaches |
|---|---|
| `solver/` (Warp LBM) | GPU kernels, CFL and stability, why explicit methods parallelize, Warp autodiff |
| `datagen/` | Design-of-experiments thinking, dataset curation for physics ML |
| `surrogate/` (PhysicsNeMo) | FNO training, the PhysicsNeMo module and checkpoint conventions, data pipelines |
| `validation/` | Physics-consistency checks, the reason low MSE is not trust (mirrors PhysicsNeMo-CFD) |
| `api/` + `ui/` | Packaging a model as a service, the interactive-design-loop story |
| `infra/` (Modal) | GPU as code: images, functions, volumes; driving remote CUDA from a local loop |
| Stretch goals | Distributed training, sensitivities through autodiff, agentic orchestration |

Secondary purpose: a public repo with visuals good enough to explain the solver-to-surrogate pattern to other developers.

## 2. What it does

A visitor (or I, locally) opens the demo and:

1. Picks an obstacle shape: ellipse with aspect ratio 0.3-1.0, rotation 0-30 degrees, size scale.
2. Picks a Reynolds number in 40-300.
3. Gets, in under 100 ms on GPU: predicted velocity and pressure fields rendered as images, a drag coefficient, and a consistency panel (divergence residual, out-of-distribution flag).
4. Can press **Solve for real**: the Warp solver runs the same case in seconds and displays solver vs surrogate side by side with the error field.

The repo README shows three GIFs: the solver shedding vortices, the surrogate matching it, the slider loop running live.

## 3. Users

Primary: me, as the builder. Secondary: developers who read the repo and the accompanying posts. No production users, no uptime promises.

## 4. Non-goals

- No 3D. 2D captures every concept this project exists to teach at 1/1000 the cost.
- No meshing. The LBM Cartesian grid sidesteps it by construction (state this openly in the README; meshing is the industry bottleneck and this project deliberately routes around it).
- No turbulence modeling. Re is capped at 300, laminar-to-transitional vortex shedding.
- No engineering-grade accuracy claims. Validation reports what the surrogate gets wrong as prominently as what it gets right.
- No Omniverse in the MVP. Optional USD export is a stretch.
- No local GPU dependency. The dev machine stays CPU-only; every CUDA task runs on Modal. No distributed training in the MVP.

## 5. Architecture

```
params (shape, Re)
      |
      v
[ solver/  Warp D2Q9 LBM ]  --->  fields u,v,rho + Cd, Cl, St
      |                                   |
      v                                   v
[ datagen/ sweep runner ]  --->  dataset/ (npz shards + manifest.parquet)
                                          |
                                          v
                          [ surrogate/  PhysicsNeMo FNO ]
                                          |
                                          v
                          [ validation/  consistency harness ]
                                          |
                              gate: ship only if green
                                          |
                                          v
                          [ api/ FastAPI ]  <-->  [ ui/ Gradio ]
```

## 6. Component specs

### 6.0 infra/ : Modal execution layer

All GPU execution goes through Modal, via the Python SDK, on my Modal account. Rationale: infra as code (the environment lives in the repo, not in a console), agent-friendly (an agent can define, run, and iterate on GPU jobs from the same codebase it edits), zero local CUDA setup.

- One `modal.App("soufflerie")` and one shared `modal.Image`: Debian slim, uv-installed pinned deps (`warp-lang`, `nvidia-physicsnemo[cu12]`, `torch`). Every function reuses this image; build it once at session start so later runs start warm.
- GPU selection in one constant: default `gpu="L40S"`, fallback list `["L40S", "A10G"]`. This workload is small; anything Ada or Ampere class is fine.
- One `modal.Volume("soufflerie-data")` mounted by all functions: dataset shards, manifests, checkpoints, validation reports persist across runs.
- Entrypoints under `infra/`: `solve.py` (single run), `sweep.py` (datagen fan-out), `train.py`, `validate.py`, `serve.py`. Each is `modal run`-able and importable.
- Local rule: pure-Python logic, tests, and plotting run locally on CPU; anything importing CUDA runs inside a Modal function. CI never touches Modal.
- Cost posture: the whole build is minutes of GPU time per milestone; single-digit dollars total. Log GPU seconds per milestone in the README for honesty.

### 6.1 solver/ : Warp mini-LBM

- D2Q9 lattice, BGK collision, half-way bounce-back on obstacles, velocity inlet, zero-gradient outlet with a thin sponge zone to kill reflections.
- Obstacle defined as a signed distance function from shape params, rasterized to a boolean mask. Same SDF later feeds the surrogate as an input channel.
- Outputs per run: instantaneous snapshots, time-averaged u, v, rho fields, and force history via momentum exchange, reduced to Cd, Cl, and Strouhal number (FFT of Cl).
- Grid: 256 x 512 default, configurable. Everything in fp32.
- Written as Warp kernels: stream, collide, boundary, macroscopic reduction. Target under 400 lines including comments.

**Correctness gates (pytest, run in CI on CPU at small grid, on GPU locally):**

| Test | Pass condition |
|---|---|
| Poiseuille channel | velocity profile within 1% of analytic parabola |
| Cylinder Re=100 | Strouhal in [0.15, 0.19]; Cd within 15% of the 2D reference value ~1.35 |
| Mass conservation | total density drift under 0.1% over 20k steps |
| Determinism | fixed seed reproduces fields bitwise on same hardware |

**Performance target:** a 256 x 512 run of 20k steps completes in under 60 s on an L40S-class Modal GPU including I/O and container overhead. Log lattice updates per second; it is the solver's honest speed metric.

### 6.2 datagen/ : sweep runner

- Config-driven (YAML): shape family, param ranges, Re range, sample count, grid, step count, seed.
- Latin hypercube sampling over (aspect ratio, rotation, scale, Re). MVP dataset: 1,000 runs. Split 600/200/200 train/val/test by design points, never by snapshots.
- Stored per run: time-averaged fields downsampled to 128 x 256 fp16, SDF channel, scalars (Cd, Cl, St, params), run metadata. Whole MVP dataset under 2 GB.
- `manifest.parquet` indexes everything. Runner is resumable and idempotent; a killed sweep continues where it stopped.
- Wall-clock budget: fan the sweep out with Modal `.starmap()` across 50-100 containers; 1,000 runs land in the shared volume in minutes, not hours. The runner stays resumable anyway (containers get preempted).

### 6.3 surrogate/ : PhysicsNeMo FNO

- Model: `physicsnemo` FNO. Input channels: SDF, normalized Re broadcast as a constant plane. Output channels: mean u, v, rho. Plus a small MLP head from the FNO latent for Cd.
- The pair (Cd from head, Cd integrated from the predicted field by momentum balance) becomes a free consistency check: disagreement flags an untrustworthy prediction.
- Training: AdamW, cosine schedule, mixed precision, checkpointing with PhysicsNeMo conventions, TensorBoard logging. Budget: under 1 hour on one Modal GPU for the MVP model; checkpoints write to the volume.
- Baselines to beat, in order: mean-field predictor, nearest-neighbor-in-param-space, then FNO. Keep the baseline table in the README; it is the honest way to show the model earns its place.
- Stretch: MeshGraphNet variant on boundary points plus grid subsample (`nvidia-physicsnemo[gnns]`), to learn the geometry-native modeling style.

### 6.4 validation/ : the consistency harness

This is the differentiating component. It produces `reports/validation.md` automatically after every training run, with plots, and it gates the demo.

| Metric | Definition | Ship gate |
|---|---|---|
| Field error | relative L2 on u, v over test set | median under 8% |
| Cd error | percent error vs solver | median under 5% |
| Head-vs-field Cd gap | disagreement between the two Cd estimates | flag samples over 10% |
| Divergence residual | mean abs div(u) on predictions vs solver baseline | under 3x solver baseline |
| Obstacle compliance | velocity magnitude inside the mask | under 1% of inlet velocity |
| OOD behavior | 3-seed ensemble variance at Re 20 and Re 400 (outside training) | must visibly increase; report it |
| Sensitivity sanity | sign of dCd/d(rotation) via autograd vs central finite differences on 10 test shapes | at least 8/10 agree |

If a gate fails, the demo still runs but shows a visible "surrogate unvalidated" banner. Never hide a red gate.

### 6.5 api/ + ui/

- FastAPI service: `POST /predict` takes shape params and Re, returns fields (PNG plus raw npz), Cd, latency, and the consistency flags. `POST /solve` launches a real solver run and streams status. `GET /health` returns model hash and validation status.
- Gradio front end for the MVP: three sliders, live prediction on release, compare button. Latency target: under 100 ms per prediction on GPU, under 1 s on CPU.
- Deployment is `modal deploy infra/serve.py`: the FastAPI app (Gradio mounted) runs as a Modal web endpoint with a GPU attached and prints a shareable URL. A CPU-only `uv run soufflerie demo` mode stays for local runs against a bundled checkpoint.

## 7. Milestones

Built with agentic coding (Claude Code or equivalent), all of it shipping today in one continuous session. Working agreement: write the failing test in the prompt before the implementation, land milestones in order, `main` stays green after each one, every milestone ends with something visual.

| Milestone | Scope | Done when |
|---|---|---|
| M0 | Repo scaffold: uv project, `warp-lang` + `nvidia-physicsnemo[cu12]` pinned, pytest, ruff, CI smoke test; Modal app + image + volume defined, `modal run infra/solve.py --smoke` executes a Warp kernel on GPU | `uv run pytest` green locally and in CI; smoke run returns from Modal |
| M1 | LBM core: stream, collide, channel BCs | Poiseuille test passes; velocity field renders |
| M2 | Obstacles, forces, diagnostics | Cylinder gates pass on a Modal run; vortex-street GIF in README |
| M3 | Sweep runner + dataset | 1,000-run dataset lands in the Modal volume via fan-out; manifest stats notebook |
| M4 | FNO training + validation harness | validation.md generated; beats both baselines; gates evaluated |
| M5 | API, Gradio, Modal deploy | `modal deploy` prints a live demo URL; slider loop works end to end; README complete with 3 GIFs |
| M6 (pick one) | Stretch (section 9) | the chosen stretch has its own test and README section |

If a milestone drags, cut scope inside it (section 10); the gates and the milestone order do not move.

## 8. Tech stack

| Layer | Choice | Note |
|---|---|---|
| Language | Python 3.11, uv | lockfile committed |
| Kernels | warp-lang | fp32; CPU fallback for CI |
| ML | nvidia-physicsnemo[cu12], torch | pin exact versions; PhysicsNeMo requires Python >= 3.10 |
| Data | npz shards + parquet manifest | no database |
| Service | FastAPI, Gradio | Gradio only in MVP UI; deployed as a Modal web endpoint |
| GPU infra | modal (Python SDK) | app, image, volume, entrypoints in `infra/`; auth via `modal setup` on my account |
| Quality | pytest, ruff, pre-commit | CI on CPU with a 64 x 128 grid; CI never calls Modal |
| Viz | matplotlib, imageio for GIFs | consistent colormap everywhere |

**Hardware:** none locally. The dev machine runs CPU-only Python (tests, plotting, orchestration); all CUDA work executes on Modal GPUs (L40S default, A10G fallback, both PhysicsNeMo-supported classes). CPU inference mode exists for the bundled-checkpoint local demo.

## 9. Stretch goals (pick by appetite, each maps to a stack layer)

1. **Sensitivity explorer.** Autograd dCd with respect to shape params through the surrogate; render the gradient as arrows on the shape. Teaches the adjoint-style story end to end.
2. **Unsteady rollout.** Train a second FNO to advance snapshots in time; report where autoregressive drift breaks physics. Teaches why temporal surrogates are hard.
3. **MeshGraphNet track.** Same task, geometry-native model, honest comparison table vs FNO.
4. **Distributed training.** PhysicsNeMo DistributedManager on a multi-GPU Modal function; document the delta from single-GPU code.
5. **Agent runner.** A small agent (Nemotron via API, or any local model) that takes "sweep aspect ratio at Re 150 and report the drag minimum" and emits config, launches runs, writes a summary table. Teaches the orchestration layer honestly at toy scale.
6. **TensorRT export.** ONNX then TensorRT engine for the FNO; measure the latency delta in the API. Teaches the deployment path.
7. **USD export.** Write obstacle geometry and a field slice to OpenUSD for viewing in an Omniverse-compatible viewer.

## 10. Risks and planned cuts

| Risk | Mitigation / cut |
|---|---|
| LBM unstable at Re near 300 on coarse grids | cap lattice velocity at 0.1, add sponge zone; if still unstable, cap Re at 200 and say so |
| Surrogate Cd misses the 5% gate | ship fields-only demo with the red banner; add Cd-weighted loss in a follow-up; never quietly relax the gate |
| Modal cold starts or image rebuilds eat the session | one lean pinned image shared by all functions, built once up front; GPU fallback list `["L40S", "A10G"]` |
| Sweep fan-out hits account concurrency limits | batch the map; 300 runs is enough for the demo, keep the data-scaling curve as a finding |
| PhysicsNeMo or Warp API drift | exact version pins in uv.lock; record versions in validation.md |
| Time pressure | cut order: stretch goals, then MeshGraphNet, then custom UI polish; the solver, the FNO, and the validation harness are never cut |

## 11. Definition of done

- Public repo, Apache 2.0, tagged v0.1.
- Fresh-clone quickstart works: `uv sync`, `uv run pytest` (CPU), `modal run infra/sweep.py --n 8` as a smoke sweep, `modal deploy infra/serve.py` prints the demo URL, `uv run soufflerie demo` for the local CPU path (small bundled checkpoint).
- README: three GIFs, baseline table, validation summary, honest limitations section.
- `reports/validation.md` checked in from the shipped checkpoint.
- Optional but intended: a three-post build log (writing a GPU solver in Warp; training a surrogate with PhysicsNeMo; deciding when to trust it). The third post is the one worth writing.