# RFC-0001: Package and runtime boundaries

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

Soufflerie uses one typed `src/soufflerie` package with deep domain modules, thin adapters for optional frameworks, and separate local CPU and remote GPU execution planes. Runtime extras prevent imports of training, remote, serving, or visualization dependencies from contaminating core and CI paths.

## Motivation

The PRD requires CPU-only local development and CI while solver-scale, training, and deployed inference run remotely on CUDA. It also requires installed CLI, service, UI, numerical solver, ML, and release surfaces in one repository. [`01-architecture.md#module-map`](../spec/01-architecture.md#module-map) needs a dependency structure that keeps those concerns testable and prevents optional import failures.

## Goals

- Make `import soufflerie` deterministic and lightweight on Python 3.11 CPU hosts.
- Define module ownership and one-way dependency rules.
- Provide explicit extras for solver, ML, remote operations, service, visualization, and development.
- Keep operational entrypoints importable and command-line runnable without duplicating domain logic.
- Make installed wheel and source distribution contents verifiable.

## Non-Goals

- Supporting multiple numerical or training frameworks in v0.1.
- A general plugin registry.
- Local CUDA environment management.
- Distributed training or a durable workflow engine.

## Proposed Design

The repository follows [`01-architecture.md#module-map`](../spec/01-architecture.md#module-map). `src/soufflerie/__init__.py` exports only version data, canonical schemas, and lazy callable façades. `solver`, `datagen`, `surrogate`, and `validation` each own a coherent contract. `service`, `demo`, `cli`, and `infra` depend inward and never become dependencies of domain code.

`pyproject.toml` declares Python `>=3.11,<3.12`, base metadata, the `soufflerie = "soufflerie.cli:app"` script, `py.typed`, and profiles:

```toml
[project.optional-dependencies]
solver = ["warp-lang==1.17.0"]
ml = ["nvidia-physicsnemo[cu12]==2.2.1", "torch==2.10.0"]
remote = ["modal==1.5.5"]
serve = ["fastapi==0.141.1", "gradio==6.26.0"]
viz = ["matplotlib==3.11.1", "imageio==2.37.4"]

[dependency-groups]
dev = [
  "pytest==9.1.1", "pytest-cov==7.1.0", "hypothesis==6.167.1",
  "ruff==0.16.5", "mypy==2.3.1", "pre-commit==4.6.2",
]
```

Shared schema/data dependencies may be base requirements when required by public schemas and smoke workflows. Exact compatible transitive versions live in `uv.lock`; the remote image installs `solver,ml,remote,serve,viz` from the lock. CPU CI installs base, solver, viz, and dev only. ML contract tests use adapters/fakes unless a small CPU-compatible framework install is explicitly assigned to a separate job.

Optional adapters use this pattern:

```python
def require_physicsnemo() -> ModuleType:
    try:
        import physicsnemo
    except ImportError as exc:
        raise DependencyUnavailableError(
            "ML runtime requires the 'ml' extra"
        ) from exc
    return physicsnemo
```

Imports occur at adapter construction, not package import. Domain protocols are introduced only for current multiplicity:

```python
class FlowPredictor(Protocol):
    @property
    def metadata(self) -> PredictorMetadata: ...
    def predict(self, batch: PredictionBatch) -> PredictionBatchResult: ...

class ArtifactStore(Protocol):
    def read_bytes(self, key: ArtifactKey, *, max_bytes: int) -> bytes: ...
    def publish(self, artifact: StagedArtifact) -> ArtifactRef: ...

class ExecutionBackend(Protocol):
    def submit_solve(self, case: CaseConfig) -> JobHandle: ...
```

Configuration is parsed once into strict, frozen Pydantic settings. Domain functions accept typed values and do not read environment variables. `infra/app.py` owns the single remote application, image, volume, GPU selection, timeouts, concurrency, and secrets wiring. Each operational file imports shared policy from `infra.app` and domain callables from the installed package.

All public functions declare input/output types. Arrays follow [`03-data-model.md#array-contracts`](../spec/03-data-model.md#array-contracts). Framework tensors cannot cross a public domain boundary; adapters convert them to validated NumPy records. `mypy --strict` applies to `src/soufflerie`; targeted exceptions require inline rationale.

Failure propagation follows [`04-error-model.md#taxonomy`](../spec/04-error-model.md#taxonomy). Missing extras and unavailable devices are distinguishable. No adapter performs implicit remote calls, downloads, device fallback, or artifact writes on import.

## Alternatives Considered

### Flat top-level directories as import packages

This matches the PRD sketch but makes distribution contents and dependency rules ambiguous. A `src` package catches editable-install assumptions and gives one public namespace, so the flat layout is rejected for importable code; `infra/` remains operational at the root.

### One environment containing the entire stack

It simplifies a single remote image but forces large CUDA/ML/service packages into local tests and package imports. Named profiles retain one lock while keeping runtime ownership explicit.

### Separate packages for solver, ML, and service

They could enforce boundaries mechanically, but v0.1 has one maintainer, one release, and shared schemas. Multi-package version coordination is unjustified now.

### Generic plugin entry points

Only a few implementations exist and they are known at build time. Protocols at real seams are sufficient; plugin discovery adds compatibility and security surface without a v0.1 user.

## Drawbacks

- Extras and dependency groups increase release-matrix complexity.
- The remote image still contains a large combined dependency set.
- Runtime optional imports shift some failures from installation to adapter construction.
- A single package relies on lint/type/import tests rather than package boundaries alone.

## Migration / Rollout

1. Add project metadata, lockfile, `src` skeleton, schemas, `py.typed`, and CLI version command.
2. Add import-isolation and installed-wheel tests before optional adapters.
3. Land domain modules in dependency order: geometry/config, solver, datagen, surrogate, validation, service/demo.
4. Add remote entrypoints only after their domain functions have CPU contract tests.
5. Any dependency set incompatible with Python 3.11 updates the spec and lock in the same reviewed change.

No legacy package exists, so no user data migration is required.

## Testing Strategy

- Build wheel/sdist and inspect required/forbidden members.
- Install the wheel into clean base-only and each-extra environments.
- Assert `import soufflerie` does not add `torch`, `physicsnemo`, `modal`, `fastapi`, `gradio`, or plotting modules to `sys.modules`.
- Run every CLI `--help` without optional runtime extras and assert typed missing-extra errors only when executing those commands.
- Enforce dependency direction with an import graph test.
- Type-check public façades and adapters under strict mode.
- Run the bundled CPU smoke prediction from the installed wheel.

## Open Questions

None for v0.1. Dependency compatibility changes follow the lock-and-spec update path owned by the release maintainer before the first implementation merge.

## References

- [`SPEC.md#executive-summary`](../../SPEC.md#executive-summary)
- [`01-architecture.md`](../spec/01-architecture.md)
- [`09-release-and-versioning.md#dependency-policy`](../spec/09-release-and-versioning.md#dependency-policy)
- [RFC-0011](RFC-0011-remote-execution-and-persistence.md)
- [RFC-0013](RFC-0013-testing-packaging-and-release.md)
