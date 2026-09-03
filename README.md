# Soufflerie

Soufflerie is a reproducible wind-tunnel workflow that connects a D2Q9 lattice
Boltzmann reference solver, immutable simulation datasets, learned flow models,
validation gates, and an interactive comparison service. The project targets
Python 3.11 and keeps CUDA, ML, serving, remote, and visualization runtimes
behind explicit optional dependencies.

The repository is under active v0.1 implementation. Public claims are limited
to checked-in tests and artifacts; the roadmap in
[`docs/roadmap/IMPLEMENTATION.md`](docs/roadmap/IMPLEMENTATION.md) records which
subsystems are complete.

## Install

Install the base package and command-line interface:

```bash
python -m pip install .
soufflerie version
```

Install only the runtime needed for an operation:

```bash
python -m pip install ".[solver]"      # Warp solver adapter
python -m pip install ".[ml]"          # model and training runtime
python -m pip install ".[serve,viz]"   # service, demo, and rendering
python -m pip install ".[remote]"      # authenticated remote adapter
```

Optional frameworks are loaded when their command or adapter executes. Importing
`soufflerie`, displaying CLI help, and running `soufflerie version` do not import
CUDA, ML, remote, service, or plotting frameworks.

## Command line

```text
soufflerie solve --config PATH --output PATH [--device cpu]
soufflerie dataset validate --manifest PATH [--json]
soufflerie model inspect --bundle PATH [--json]
soufflerie validate --config PATH --output-dir PATH [--device cpu]
soufflerie demo --bundle PATH [--host 127.0.0.1] [--port 7860]
soufflerie version [--json]
```

Every command supports `--help`. Human results use stdout. Command failures use
a schema-v1 JSON record on stderr and stable exit codes; `version`, dataset
validation, and model inspection also expose successful JSON output. Domain
commands load their owning implementation lazily, so a base-only installation
reports the required extra without a traceback.

See [`docs/cli.md`](docs/cli.md) for JSON fields, failure codes, dependency
profiles, and automation guidance.

## Bundled CPU smoke model

The wheel and source distribution include a checksum-bound synthetic FNO for a
real installed-package CPU smoke. It is an untrained plumbing fixture, not a
scientific model and not evidence of surrogate accuracy. With the ML profile
installed, materialize its losslessly compressed float32 weights and run the
fixed PhysicsNeMo predictor with:

```bash
uv sync --frozen --extra ml
uv run python - <<'PY'
from pathlib import Path
from soufflerie.surrogate import run_bundled_cpu_smoke

result = run_bundled_cpu_smoke(Path("/tmp/soufflerie-model-smoke"))
print(result.model_dump_json(indent=2))
PY
```

The result is schema-v1 JSON with finite `[1,3,320,256]` fields, one drag value,
and exact output digests. Materialization verifies the packaged descriptor,
compressed and decoded weight checksums, bundle members, synthetic parent,
compatibility, and closed 28-tensor allowlist before atomically publishing the
local bundle. It imports no training, remote, service, or visualization code.
See the [bundled CPU smoke contract](docs/model/bundled-cpu-smoke.md) for size,
identity, regeneration, and installed-wheel acceptance details.

## Remote solver smoke

Authenticated CUDA execution uses one locked Modal app, image, and persistent
volume. From a clean commit, run the resumable non-release eight-case smoke with:

```bash
uv run --extra remote modal run infra/sweep.py \
  --config configs/sweeps/mvp-v1.yaml \
  --n 8
```

The command injects one retryable first-attempt failure on a fresh smoke
identity, resumes only missing work, verifies committed digests, and returns a
small typed summary. `--n 8` is intentionally distinct from and cannot satisfy
the 1,000-case release dataset. See
[`docs/operations/remote-runtime.md`](docs/operations/remote-runtime.md) for
single-solve commands, identity rules, persistence, fallback, and cost controls.

The release-eligible command deliberately has no `--n` override:

```bash
uv run --extra remote modal run infra/sweep.py \
  --config configs/sweeps/mvp-v1.yaml \
  --output /tmp/soufflerie-sweep-summary.json
```

It binds the frozen 1,000-point maximin LHS design to a clean source and lock,
uses at most 100 idempotent workers, retains failed-attempt codes across
retries, and publishes a dataset manifest only after all 1,000 run archives
verify. Checked release statistics and execution evidence live in
[`reports/dataset/README.md`](reports/dataset/README.md) once that authenticated
run completes.

## Canonical three-seed training

The canonical L40S/bf16 experiment trained 100 epochs for each configured seed
against dataset `4aefbbe88a18d233249b` from source revision `e1feea4d`. The
validation-only selection is immutable and chose seed 17; scientific validation
and release approval remain separate downstream gates.

| Seed | Model ID | Validation score | Wall time |
| ---: | --- | ---: | ---: |
| 17 | `7b6fd39c0ec78f452163` | 0.0229791 | 21.3 min |
| 23 | `8847109ea4eaf87aa5ae` | 0.0238489 | 20.9 min |
| 31 | `bec20fbbb9dfd64f0639` | 0.0230755 | 22.6 min |

The full request, model references, parent digests, accounting, and frozen
selection are in [`reports/training/index.json`](reports/training/index.json).
Verify the checked evidence with:

```bash
uv run python scripts/check_training_run.py reports/training/index.json
```

To submit a new clean-source experiment, see the
[remote training operations guide](docs/operations/training.md).

## Cylinder acceptance

The checked-in Re=100 report binds the canonical and sensitivity runs to exact
source, lock, configuration, device, field archive, and run artifact digests.
Regenerate and verify it from a clean authenticated checkout with:

```bash
uv run --extra remote modal run infra/solve.py \
  --config configs/cases/cylinder-re100.yaml
uv run python scripts/check_solver_report.py \
  reports/solver/cylinder-re100.json
```

The immutable reference gates are `Cd in [1.1475, 1.5525]`,
`St in [0.15, 0.19]`, at least eight resolved lift cycles, and mass drift below
`0.1%`. The command writes the rendered companion to
`reports/solver/cylinder-re100.md`.

## Development checks

The default development environment is CPU-first:

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run python scripts/validate_schemas.py
```

Authenticated GPU operations and release evidence use the separately documented
remote workflow in
[`docs/operations/remote-runtime.md`](docs/operations/remote-runtime.md).
