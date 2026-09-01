# Overview

<a id="thesis"></a>
## Thesis

Soufflerie makes the complete simulation-to-surrogate trust loop inspectable at small scale: generate flow with a numerical solver, curate design-point-separated data, train a surrogate, challenge it with physical and statistical checks, and expose both predictions and failed checks in an interactive loop. Its distinct requirement is not low prediction error alone; the shipped interface must make disagreement, physical residuals, out-of-distribution behavior, and provenance visible.

The primary user is the maintainer learning and demonstrating the stack. Secondary users are contributors and technical readers who must reproduce claims from a fresh clone. No public availability or engineering certification is promised.

<a id="problem-statement"></a>
## Problem statement

Most compact surrogate demonstrations omit at least one load-bearing step: solver correctness, leakage-resistant dataset construction, baseline comparison, physical validation, reproducible artifact lineage, or honest serving behavior when validation fails. Soufflerie v0.1 must put those steps in one repository without requiring a local GPU.

<a id="v01-goals"></a>
## v0.1 goals

1. Implement a deterministic fp32 D2Q9 BGK solver for two-dimensional channel flow around parameterized ellipses.
2. Demonstrate numerical correctness with analytic channel flow and bounded cylinder-reference checks.
3. Generate exactly 1,000 valid design points over the declared parameter domain and split by design point.
4. Train a specified FNO and two deterministic baselines against the same immutable split.
5. Emit a schema-versioned validation report that applies every gate and cannot conceal a failure.
6. Serve typed prediction and asynchronous solver workflows, with CPU local inference and remote GPU execution.
7. Ship as a typed, tested, reproducible Python 3.11 open-source package with documented security and limitations.

<a id="non-goals"></a>
## Non-goals

- Three-dimensional, compressible, thermal, multiphase, or turbulence-modelled flow.
- Arbitrary geometry, body-fitted meshing, moving bodies, fluid-structure coupling, or topology optimization.
- Reynolds numbers outside `[40, 300]` as supported predictions. Values `20` and `400` are validation probes only.
- Engineering-grade accuracy, certification, design sign-off, or comparison to a validated industrial solver.
- Distributed training, MeshGraphNet, unsteady surrogate rollout, TensorRT, USD export, or agent orchestration.
- Multi-tenant accounts, user-owned data, uptime commitments, or a durable public job queue.
- Backpropagation through the numerical solver. Autograd is used only for surrogate sensitivity checks.

These exclusions keep v0.1 buildable while preserving the PRD's solver, FNO, validation, and interaction thesis. Stretch ideas require new RFCs and milestones.

<a id="success-criteria"></a>
## Success criteria

The release is complete only when all conditions hold:

| ID | Criterion | Authoritative evidence |
|---|---|---|
| `S-SOLVER` | All numerical gates in [RFC-0002](../rfcs/RFC-0002-d2q9-lbm-core.md#acceptance-invariants) and [RFC-0003](../rfcs/RFC-0003-geometry-boundaries-and-forces.md#acceptance-invariants) pass. | Checked CPU reports plus one recorded remote GPU acceptance run |
| `S-DATA` | The canonical manifest contains 1,000 unique successful design points and immutable `600/200/200` splits. | Manifest validator output and content digest |
| `S-MODEL` | The selected FNO beats both baselines on field and drag primary metrics. | Checked-in validation report tied to checkpoint and dataset digests |
| `S-GATES` | Every validation gate is evaluated; the overall status equals the conjunction of required gates. | Machine-readable report plus Markdown rendering |
| `S-SERVICE` | Prediction, solve-status streaming, health, local demo, and visible invalid-state behavior pass contract tests. | HTTP, CLI, and browser smoke artifacts |
| `S-REPRO` | A fresh clone can lock, test, build, install, and reproduce smoke artifacts without network/GPU during tests. | CI release gate and installed-wheel smoke test |
| `S-OSS` | License, contribution, conduct, security, citation, changelog, and limitations documents ship in source distributions. | Package-content test |

Performance thresholds are release requirements only on the reference classes and measurement protocols in [`08-performance-budget.md`](08-performance-budget.md#budgets).

<a id="hardest-problems"></a>
## Hardest technical problems

1. **Numerical fidelity under a compact solver.** Boundary treatment, lattice stability, force normalization, and transient removal interact. Analytic and reference-case gates constrain these choices.
2. **Leakage-free, reproducible ML evidence.** Design points, not snapshots, define splits; data, config, code, and model digests must survive remote fan-out and checkpoint selection.
3. **Trust-preserving product behavior.** Validation state and uncertainty must travel from report to API to UI without being reduced to a green-looking scalar.

<a id="load-bearing-abstractions"></a>
## Load-bearing abstractions

- `CaseConfig` defines one physical/numerical case and rejects unstable or unsupported inputs before allocation.
- `SolverResult` binds fields, force history, diagnostics, and provenance to one case.
- `DatasetManifest` is the only training-data index and enforces unique design points and split immutability.
- `ModelBundle` binds architecture, preprocessing statistics, checkpoint weights, and their digests.
- `ValidationReport` binds test-set metrics and release gates to exact dataset/model/solver identities.
- `PredictionResponse` transports fields, scalar results, latency, validation state, OOD state, and artifact identities together.

<a id="contributor-contract"></a>
## Contributor contract

A contributor arriving six months after v0.1 starts from `SPEC.md`, selects one implementation issue, follows its RFC interfaces and invariants, adds the named evidence, and changes normative behavior only through a spec/RFC update in the same pull request. Maintainers must not require private design context to complete a scoped issue. Checked-in configs, schemas, fixtures, acceptance commands, and generated-artifact rules are part of that handoff contract.

<a id="risk-register"></a>
## Risk register

- `RISK: high-Re lattice instability.` Owner: solver maintainer. Resolution: enforce the lattice-velocity/tau contract, run the full domain stability sweep before dataset production, and block v0.1 if valid cases cannot cover `Re <= 300`. Changing the supported domain requires an RFC amendment; it is never a silent cut.
- `RISK: drag target misses its gate.` Owner: ML maintainer. Resolution: retain the red validation state, publish baseline-relative evidence, and iterate loss weighting or data only through versioned experiments. The gate stays fixed.
- `RISK: remote concurrency or preemption prevents a 1,000-case sweep.` Owner: infrastructure maintainer. Resolution: resume from committed per-run state, lower fan-out without lowering the sample count, and record remote timing/cost.
- `RISK: dependency/API drift.` Owner: release maintainer. Resolution: exact lockfile, compatibility smoke jobs, recorded package versions, and automated dependency updates through reviewed pull requests.
- `RISK: public solve endpoint can exhaust budget.` Owner: service maintainer. Resolution: strict input validation, bounded concurrency, request timeout, rate limit, and kill-switch; disable solve while keeping prediction available if the budget guard trips.

<a id="assumptions"></a>
## Assumptions

- Python 3.11 is available locally and in the remote image.
- Remote credentials and billing authority are configured outside the repository.
- An L40S-class GPU is the performance reference; an A10G-class GPU is a functional fallback, not a performance-equivalent target.
- Cylinder reference intervals are regression gates for this educational discretization, not external validation of engineering accuracy.
- The bundled checkpoint is small enough for source-release or release-asset distribution; its license and checksum are recorded before inclusion.
