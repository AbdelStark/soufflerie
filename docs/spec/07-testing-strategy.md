# Testing strategy

<a id="test-principles"></a>
## Test principles

Tests mirror the architecture, run deterministically, and distinguish CPU contract evidence from remote GPU acceptance evidence. Default tests require no network, credential, GPU, large dataset, or downloaded model. Every fixed bug adds a regression test at the lowest layer that reproduces it.

<a id="test-layers"></a>
## Test layers

| Layer | Scope | Default environment |
|---|---|---|
| Unit | formulas, validators, identities, sampling, gates, redaction | CPU, in-process |
| Property | SDF geometry, conservation primitives, schema round trips, split invariants | CPU, bounded generated cases |
| Numerical regression | Poiseuille, mass drift, deterministic small-grid golden summaries | CPU Warp backend |
| Contract | Python exports, CLI, schemas, HTTP/OpenAPI, artifact readers | CPU, temporary filesystem |
| Integration | miniature solve -> manifest -> baseline -> report -> API path | CPU, synthetic/tiny real artifacts |
| UI | field rendering, controls, red banner, solve state transitions | Headless browser, mocked service fixtures |
| Remote acceptance | full cylinder, 1,000-case sweep, training, validation, GPU benchmarks, deployment smoke | Authenticated remote GPU, manually dispatched |
| Release | lock, lint, typing, docs links, build/install/package contents, audits | CI CPU |

<a id="numerical-oracles"></a>
## Numerical oracles

- Poiseuille: after declared convergence, compare the centerline velocity profile with the analytic parabola using relative L2 and max error; both must be at most `1%`, excluding one cell adjacent to each wall.
- Cylinder `Re=100`: on the reference grid/run, `St in [0.15, 0.19]`, mean `Cd in [1.1475, 1.5525]`, periodic lift established, and mass drift below `0.1%`.
- Mass: `abs(sum(rho_t)-sum(rho_0))/sum(rho_0) < 0.001` at 20,000 steps.
- Determinism: identical inputs and environment yield bitwise-equal persisted fields and histories on the same device class/runtime lock; cross-device results use numerical tolerances and are never called bitwise deterministic.
- Geometry properties: SDF sign, rotation symmetry at circular aspect ratio, scale monotonicity, mask containment, and minimum clearance hold across generated valid parameters.

Reference fixtures record schema version, configuration, platform class, and generation revision. Updating a numerical golden requires a review note explaining the physical/numerical reason and re-running analytic/reference gates.

<a id="ml-tests"></a>
## ML-specific tests

- Split tests prove exact `600/200/200` counts, unique case membership, fixed seed reproduction, and absence of snapshot leakage.
- Preprocessing fits only on training rows; a sentinel in validation/test must not influence statistics.
- One-batch overfit decreases the declared loss by at least `90%` in a fixed small fixture.
- Resume equivalence compares uninterrupted and checkpoint-resumed training at an epoch boundary within deterministic tolerance.
- Each baseline is deterministic and evaluated through the same metric code as FNO.
- Metric unit tests use hand-computable arrays, including zero-denominator behavior.
- Three independent model seeds are loaded for ensemble/OOD evidence; tests catch accidental weight reuse.
- Sensitivity compares autograd with double-sided finite differences away from parameter boundaries.
- The report evaluator tests every gate exactly at, below, and above its threshold and proves `overall_green == all(required_gates)`.

<a id="service-and-security-tests"></a>
## Service and security tests

HTTP tests cover valid requests, each numeric boundary, unknown fields, booleans, NaN/infinity encodings, oversize bodies, capacity exhaustion, job reconnect, monotonic events, missing jobs, timeout, and sanitized internal errors. Artifact fuzz tests cover bad digests, unsupported schemas, oversized arrays, pickle payload attempts, traversal names, and truncated files. Redaction tests plant sentinel values in settings, headers, URLs, exception text, and nested fields.

UI tests assert that red validation state appears above results, cannot be dismissed, survives compare mode, and has equal prominence in narrow and wide layouts. The UI must not derive green/red state from metric formatting.

<a id="ci-gates"></a>
## CI gates

Pull requests must pass:

```text
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -m "not remote" --cov=src/soufflerie --cov-report=term-missing --cov-fail-under=90
uv run python scripts/validate_schemas.py
uv run python scripts/check_docs.py
uv build
uv run python scripts/check_distribution.py dist/*
```

The installed-wheel job creates a clean environment, installs the built wheel, verifies `soufflerie version`, `soufflerie --help`, import isolation, and the bundled model smoke prediction. Coverage is a floor, not proof of numerical correctness.

Remote acceptance is required before release but excluded from pull-request CI. Its signed-off report contains the source revision and exact artifact identities.

<a id="test-evidence"></a>
## Evidence retention

CI logs retain normal test output. Release evidence checked into `reports/` includes validation Markdown/JSON, plots, GPU performance summary, manifest statistics, and an index of full remote artifact digests. Large datasets, raw checkpoints, and transient logs remain in external storage or release assets.
