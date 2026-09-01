# Soufflerie v0.1 Specification

<a id="status"></a>
## Status

- Specification version: `0.1.0`
- Target release: `v0.1`
- Status: Accepted for implementation
- Canonical source of product intent: [`prd.md`](prd.md)
- Normative language: **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have their usual requirements meaning.

This corpus defines Soufflerie v0.1. Implementation, tests, examples, generated artifacts, and public claims must conform to it. When the PRD and this corpus differ in implementation detail, this corpus controls; scope may change only through an accepted RFC amendment.

<a id="executive-summary"></a>
## Executive summary

Soufflerie is a two-dimensional, GPU-executed virtual wind-tunnel reference project. A deterministic D2Q9 lattice Boltzmann solver generates a versioned dataset over elliptical obstacle geometry and Reynolds number. A Fourier neural operator predicts mean flow fields and drag. A validation harness compares the surrogate with solver-held-out cases and exposes every failed gate. A typed HTTP service and a small interactive client provide prediction and asynchronous reference-solve workflows. Local development and continuous integration remain CPU-only; authenticated remote jobs own CUDA execution and persistent large artifacts.

The v0.1 claim is deliberately bounded: it demonstrates a reproducible solver-to-surrogate workflow and reports measured limitations. It does not provide engineering-grade CFD accuracy, three-dimensional flow, turbulence models, arbitrary geometry, or service-level guarantees.

<a id="corpus-index"></a>
## Corpus index

| Document | Normative subject |
|---|---|
| [`00-overview.md`](docs/spec/00-overview.md) | Thesis, users, scope, success, and risks |
| [`01-architecture.md`](docs/spec/01-architecture.md) | Module boundaries, dependency direction, and data flow |
| [`02-public-api.md`](docs/spec/02-public-api.md) | Python, CLI, HTTP, and artifact surfaces |
| [`03-data-model.md`](docs/spec/03-data-model.md) | Typed records, schemas, units, and invariants |
| [`04-error-model.md`](docs/spec/04-error-model.md) | Stable error taxonomy and recovery |
| [`05-observability.md`](docs/spec/05-observability.md) | Events, metrics, provenance, and redaction |
| [`06-security.md`](docs/spec/06-security.md) | Threat model, trust boundaries, and secrets |
| [`07-testing-strategy.md`](docs/spec/07-testing-strategy.md) | Test layers, numerical oracles, and CI gates |
| [`08-performance-budget.md`](docs/spec/08-performance-budget.md) | Latency, throughput, memory, and measurement |
| [`09-release-and-versioning.md`](docs/spec/09-release-and-versioning.md) | Compatibility, releases, deprecation, and governance |
| [`10-glossary.md`](docs/spec/10-glossary.md) | Canonical terminology and symbols |

<a id="rfc-index"></a>
## RFC index

| RFC | Decision | Status |
|---|---|---|
| [RFC-0001](docs/rfcs/RFC-0001-package-and-runtime-boundaries.md) | Package and runtime boundaries | Accepted |
| [RFC-0002](docs/rfcs/RFC-0002-d2q9-lbm-core.md) | D2Q9 lattice Boltzmann core | Accepted |
| [RFC-0003](docs/rfcs/RFC-0003-geometry-boundaries-and-forces.md) | Geometry, boundaries, and force diagnostics | Accepted |
| [RFC-0004](docs/rfcs/RFC-0004-experiment-config-and-design.md) | Experiment configuration and design of experiments | Accepted |
| [RFC-0005](docs/rfcs/RFC-0005-dataset-artifacts-and-sweep-lifecycle.md) | Dataset artifacts and resumable sweep lifecycle | Accepted |
| [RFC-0006](docs/rfcs/RFC-0006-fno-surrogate-and-checkpoints.md) | FNO surrogate and checkpoint contract | Accepted |
| [RFC-0007](docs/rfcs/RFC-0007-training-and-reproducibility.md) | Training, baselines, and reproducibility | Accepted |
| [RFC-0008](docs/rfcs/RFC-0008-validation-and-release-gates.md) | Validation metrics and release gates | Accepted |
| [RFC-0009](docs/rfcs/RFC-0009-inference-and-solve-api.md) | Prediction and asynchronous solve API | Accepted |
| [RFC-0010](docs/rfcs/RFC-0010-interactive-demo-and-visualization.md) | Interactive demo and visualization | Accepted |
| [RFC-0011](docs/rfcs/RFC-0011-remote-execution-and-persistence.md) | Remote execution and persistence | Accepted |
| [RFC-0012](docs/rfcs/RFC-0012-observability-provenance-and-security.md) | Observability, provenance, and security | Accepted |
| [RFC-0013](docs/rfcs/RFC-0013-testing-packaging-and-release.md) | Testing, packaging, and release | Accepted |

<a id="change-control"></a>
## Change control

Normative behavior changes require a pull request that updates affected specs, RFCs, schemas, tests, and the implementation tracker together. A superseding RFC must name the replaced RFC and migration. Generated validation reports cannot amend gates. No implementation convenience may silently weaken a numerical, validation, security, or reproducibility invariant.

<a id="implementation-plan"></a>
## Implementation plan

[`docs/roadmap/IMPLEMENTATION.md`](docs/roadmap/IMPLEMENTATION.md) maps the accepted corpus to the complete `v0.1` GitHub issue graph. GitHub is authoritative for live state; the tracker is updated whenever issue scope, dependencies, or status change.
