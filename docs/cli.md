# Command-line interface

The installed `soufflerie` command is a thin adapter over typed domain
interfaces. Importing it does not initialize a solver, load a model, inspect the
environment, contact a remote service, or write an artifact. Optional frameworks
are imported only after command arguments have passed adapter-level validation.

## Commands

| Command | Required options | Optional options | Successful output |
|---|---|---|---|
| `solve` | `--config PATH`, `--output PATH` | `--device cpu` | human result |
| `dataset validate` | `--manifest PATH` | `--json` | human or `DatasetValidationResult` |
| `model inspect` | `--bundle PATH` | `--json` | human or `ModelInspectionResult` |
| `validate` | `--config PATH`, `--output-dir PATH` | `--device cpu` | human result |
| `demo` | `--bundle PATH` | `--host 127.0.0.1`, `--port 7860` | listening address |
| `version` | none | `--json` | installed package and Python versions |

Run `soufflerie COMMAND --help` before automation rather than relying on shell
completion. `device` accepts `cpu`, `cuda`, or `cuda:<nonnegative index>` and is
normalized to lowercase. There is no implicit device fallback.

Remote operational entrypoints are intentionally absent. Maintainers use
`modal run infra/{solve,sweep,train,validate}.py` and
`modal deploy infra/serve.py` under the locked remote-runtime policy.

## Machine-readable success

`--json` writes one compact, newline-terminated schema-v1 object to stdout and
nothing to stderr.

The checked-in contracts are [`cli-version.json`](../schemas/v1/cli-version.json),
[`dataset-validation-result.json`](../schemas/v1/dataset-validation-result.json),
[`model-inspection-result.json`](../schemas/v1/model-inspection-result.json), and
[`cli-error.json`](../schemas/v1/cli-error.json).

```json
{"package":"soufflerie","python":"3.11.14","schema_version":1,"version":"0.1.0"}
```

Dataset validation succeeds only after the domain reader verifies the bounded
Parquet schema, rows, referenced identities, size/split gates, and recomputed
logical dataset digest. Parent run archives are fully opened by the manifest
builder; standalone validation cannot attest to parents that are not present.
Its JSON contains `schema_version`, `valid=true`, `manifest`, `dataset_id`, and
the exact `case_count=1000`. Model inspection succeeds
only after safe bundle verification and contains `schema_version`, `valid=true`,
`bundle`, `model_id`, `dataset_id`, and `architecture`. A failed gate or invalid
artifact is an error, never a JSON success with `valid=false`.

## Errors and exit codes

Command-execution failures write one compact schema-v1 JSON object to stderr and
leave stdout empty:

```json
{
  "schema_version": 1,
  "error": {
    "code": "DEPENDENCY_UNAVAILABLE",
    "message": "solve requires the 'solver' extra; install soufflerie[solver]",
    "retryable": false
  }
}
```

Unknown exceptions become `INTERNAL_ERROR` without their message or traceback.
Typed messages pass through the same URL/path/exception sanitization used by
structured observability. Argument-parser diagnostics such as a missing option
also use stderr and exit `2` before a command handler runs.

| Exit | Meaning | Error families |
|---:|---|---|
| `0` | completed; requested gates passed | none |
| `2` | usage or configuration invalid | `CONFIG_INVALID` |
| `3` | domain or numerical case invalid | `CASE_OUT_OF_DOMAIN`, `SOLVER_UNSTABLE`, `SOLVER_NOT_CONVERGED` |
| `4` | artifact or schema invalid | `ARTIFACT_INTEGRITY`, `SCHEMA_UNSUPPORTED` |
| `5` | dependency or device unavailable | `DEPENDENCY_UNAVAILABLE`, `DEVICE_UNAVAILABLE` |
| `6` | validation completed with a red required gate | `VALIDATION_RED` |
| `7` | remote or capacity failure | `REMOTE_EXECUTION`, `CAPACITY_EXHAUSTED` |
| `70` | internal invariant or unexpected failure | `INTERNAL_INVARIANT`, `INTERNAL_ERROR` |

Automation should branch on the integer exit code and stable `error.code`, not
on human message text.

## Optional runtime boundaries

| Operation | Extra | First lazy framework check |
|---|---|---|
| solve | `solver` | `warp` |
| validation | `ml` | `torch` |
| local demo/service | `serve`, then `viz` | `gradio`, then `matplotlib` |

Dataset validation and model inspection use base safe-artifact dependencies and
do not import training frameworks. The CLI backend remains injectable so domain
modules can own algorithms and tests can exercise every output/error contract
without downloading models, allocating a device, or calling remote services.
