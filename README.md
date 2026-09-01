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
