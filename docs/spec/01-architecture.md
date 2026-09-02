# Architecture

<a id="system-context"></a>
## System context

Soufflerie has two execution planes connected by immutable artifacts:

```text
local CPU / CI                             authenticated remote GPU
-----------------------------              -----------------------------
typed config + orchestration  -----------> solver fan-out / training
unit, contract, smoke tests                 GPU acceptance benchmarks
local bundled-model demo      <----------- dataset/checkpoint/report store
```

The local plane owns configuration, validation of inputs, plotting, package tests, and CPU inference. The remote plane owns CUDA solver runs, full sweep execution, training, validation, and deployed GPU inference. Artifact digests, not process memory or mutable paths, connect stages.

<a id="module-map"></a>
## Module map

The implementation uses a `src` layout:

```text
src/soufflerie/
  config.py          # strict configuration parsing and canonicalization
  geometry.py        # ellipse SDF and masks; no solver dependency
  numerics.py        # allocation-free lattice configuration preflight
  schemas.py         # shared Pydantic/dataclass contracts and schema versions
  solver/            # lattice state, kernels, boundaries, forces, diagnostics
  datagen/           # design sampling, split assignment, manifests, sweep state
  surrogate/         # preprocessing, baselines, FNO adapter, bundles, inference
  validation/        # metrics, gates, report model and renderer
  service/           # HTTP application, solve job lifecycle, response encoding
  demo/              # UI composition and plotting
  observability.py   # structured events, metric names, redaction
  cli.py             # local command surface
infra/
  app.py             # remote app, image, volume and shared policy
  solve.py           # remote single-case entrypoint
  sweep.py           # remote fan-out entrypoint
  train.py           # remote training entrypoint
  validate.py        # remote validation entrypoint
  serve.py           # remote ASGI deployment entrypoint
configs/             # versioned YAML experiment and runtime configs
tests/               # CPU-default tests and marked remote acceptance tests
```

The public package name is `soufflerie`. Importing `soufflerie` MUST NOT import CUDA-only, remote-execution, serving, plotting, or training packages.

### Runtime profiles

`pyproject.toml` is the package and direct-dependency source of truth; `uv.lock`
records the resolved Python 3.11 environment. Base dependencies contain the
schema, array, table, configuration, safe-artifact, and CLI primitives used by
public CPU contracts. Optional runtimes have explicit ownership:

| Extra | Owner | Import boundary |
|---|---|---|
| `solver` | Warp numerical adapter | Loaded only when a solver adapter is constructed |
| `ml` | PhysicsNeMo, PyTorch, and training support | Loaded only by surrogate/training adapters |
| `remote` | Remote execution adapter | Loaded only from `infra` operational entrypoints |
| `serve` | HTTP and interactive service adapters | Loaded only when a service is constructed |
| `viz` | Plot and image renderers | Loaded only by rendering call sites |

The `dev` dependency group owns CPU test, lint, formatting, typing, and local
hook tools. The `release` group owns distribution, SBOM, dependency-audit, and
lock tooling. Package import tests run in an isolated interpreter, and a static
AST contract rejects upward package dependencies, peer-domain coupling, and any
domain import of root-level `infra`.

<a id="dependency-direction"></a>
## Dependency direction

Allowed dependencies are one-way:

```text
config, schemas, geometry, numerics
    -> solver, datagen, surrogate
        -> validation
            -> service, demo, cli

infra -> public package interfaces
```

- Domain modules MUST NOT import `infra`.
- Training MUST NOT import CLI or UI modules.
- Models MUST return values; they MUST NOT write reports or remote artifacts directly.
- External framework calls live behind thin adapters in `solver`, `surrogate`, `service`, or `infra`.
- Optional imports occur inside the owning adapter and raise typed `DependencyUnavailableError` errors.

<a id="primary-data-flow"></a>
## Primary data flow

1. A versioned YAML config is parsed into `SweepConfig`; canonical JSON yields `config_digest`.
2. Latin hypercube sampling emits 1,000 physical design points, binds them to
   `CaseConfig` records, preflights all cases, and freezes deterministic split
   assignments before execution.
3. The remote sweep claims each `case_id`, runs the solver, validates the result, and atomically publishes a run archive and terminal state.
4. A manifest builder admits only valid, checksum-matching runs, freezes splits, and publishes `DatasetManifest`.
5. Training loads only the manifest, fits preprocessing statistics on the training split, trains both baselines and three seeded FNO runs, and publishes immutable model bundles.
6. Validation evaluates the selected bundle and all seeds against the frozen test split plus OOD probes, then writes JSON, Markdown, plots, and a gate summary atomically.
7. The service loads one bundle and its matching report at startup. It refuses mismatched identities and exposes failed validation state without suppressing predictions.
8. The UI renders the service response, uses one field rendering contract, and launches reference solve jobs only through the service boundary.

<a id="state-ownership"></a>
## State ownership

| State | Owner | Mutability |
|---|---|---|
| User/config input | `config` / `service` | Immutable after validation |
| Lattice buffers | `solver` | Mutable only during one run |
| Run archive | `datagen` | Write once, content addressed |
| Sweep state | `datagen` | State machine with compare-before-write claims |
| Dataset manifest | `datagen` | Immutable after publication |
| Checkpoint bundle | `surrogate` | Immutable after publication |
| Validation report | `validation` | Immutable per model/dataset pair |
| Solve job status | `service` | Bounded ephemeral state with terminal retention |
| Remote volume | `infra` | Physical storage; domain modules own layout contracts |

<a id="concurrency"></a>
## Concurrency model

Sweep cases are independent. A case transitions `pending -> running -> succeeded|failed`; expired `running` leases may be reclaimed. Publishing uses a temporary key followed by atomic rename/commit. Duplicate successful attempts are accepted only when their content digests match; divergent duplicates fail the sweep integrity check.

Training is single-process/single-GPU in v0.1. API prediction concurrency is bounded by one worker per GPU and a configurable in-process queue. Reference solve jobs use a separate concurrency limit so they cannot starve prediction. No distributed consensus or durable general-purpose queue is in scope.

<a id="extension-points"></a>
## Extension points

Only imminent multiplicity receives an abstraction:

- `FlowPredictor` protocol has mean-field, nearest-neighbor, and FNO implementations.
- `ArtifactStore` protocol has local-filesystem and remote-volume adapters.
- `ExecutionBackend` has local CPU and remote GPU orchestration adapters.

Solver collision models, arbitrary shape families, and multiple UI frameworks do not receive plugin systems in v0.1.
