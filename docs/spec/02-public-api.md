# Public API

<a id="compatibility-surface"></a>
## Compatibility surface

The supported public surface comprises:

- names exported from `soufflerie.__all__`;
- the `soufflerie` console command and documented flags;
- versioned YAML, JSON, NPZ, Parquet, and checkpoint schemas;
- documented HTTP paths, JSON fields, and error codes;
- checked-in configuration examples.

Internal modules may change within `0.x`; documented surfaces follow [`09-release-and-versioning.md`](09-release-and-versioning.md#compatibility-policy).

<a id="python-api"></a>
## Python API

The package exports these typed contracts; complete schemas are in [`03-data-model.md`](03-data-model.md#canonical-types) and their owning RFCs.

```python
from collections.abc import Iterator, Sequence
from pathlib import Path

def solve(case: CaseConfig, *, device: str = "cpu") -> SolverResult: ...
def sample_design(config: SweepConfig) -> tuple[DesignPoint, ...]: ...
def build_manifest(run_root: Path, *, config: SweepConfig) -> DatasetManifest: ...
def load_bundle(path: Path, *, device: str = "cpu") -> ModelBundle: ...
def predict(bundle: ModelBundle, case: PredictionCase) -> Prediction: ...
def validate(
    bundles: Sequence[ModelBundle],
    dataset: DatasetManifest,
    *,
    output_dir: Path,
) -> ValidationReport: ...
```

`device` accepts normalized strings `cpu` and `cuda[:index]`. A requested unavailable device raises `DeviceUnavailableError`; implicit fallback is forbidden because it invalidates latency and determinism evidence.

<a id="cli"></a>
## CLI

```text
soufflerie solve --config PATH --output PATH [--device cpu]
soufflerie dataset validate --manifest PATH
soufflerie model inspect --bundle PATH
soufflerie validate --config PATH --output-dir PATH [--device cpu]
soufflerie demo --bundle PATH [--host 127.0.0.1] [--port 7860]
soufflerie version [--json]
```

All commands support `--help`. Commands emit human text to stdout and structured errors to stderr. `--json` is available for `version`, `model inspect`, and dataset validation. Exit codes are specified in [`04-error-model.md`](04-error-model.md#cli-exit-codes). Remote operational entrypoints remain `modal run infra/{solve,sweep,train,validate}.py` and `modal deploy infra/serve.py`; these are maintainer surfaces, not installed console commands.

<a id="http-api"></a>
## HTTP API

The HTTP schema version is `1`, carried in every response. Request bodies reject unknown fields.

```text
POST /predict
POST /solve
GET  /solve/{job_id}
GET  /solve/{job_id}/events
GET  /health
GET  /openapi.json
```

`POST /predict` accepts `PredictionRequest` and returns `PredictionResponse`. Field payloads are base64-encoded PNG and compressed NPZ bytes with declared SHA-256 digests. The measured `inference_ms` excludes network transfer and includes preprocessing, model forward, consistency calculations, and encoding; `request_ms` records total server handling time.

`POST /solve` validates the same case, reserves bounded capacity, and returns `202` with `SolveAccepted`. `GET /solve/{job_id}/events` uses server-sent events named `queued`, `running`, `progress`, `completed`, and `failed`. Each event carries monotonically increasing `sequence`, `job_id`, `timestamp`, and typed data. Clients reconnect with `Last-Event-ID`; terminal events remain readable for 60 minutes. `GET /solve/{job_id}` is the polling fallback.

`GET /health` returns process liveness, model and validation identities, validation status, device class, and readiness. It never returns secret, host, account, or filesystem values.

<a id="request-limits"></a>
## Request limits

| Field | Constraint |
|---|---|
| `aspect_ratio` | finite float in `[0.3, 1.0]` |
| `rotation_deg` | finite float in `[0.0, 30.0]` |
| `scale` | finite float in `[0.75, 1.25]` |
| `reynolds` | finite float in `[40.0, 300.0]` for public predictions |
| request body | at most 16 KiB |
| concurrent predicts | configured, default `8` queued plus active worker |
| concurrent solves | configured, default `2` active and `8` queued |

NaN, infinity, booleans supplied as numbers, unknown fields, unsupported media types, and out-of-domain values are rejected before any GPU work.

<a id="artifact-api"></a>
## Artifact API

Artifact roots contain a `metadata.json` with `schema_version`, `artifact_type`, SHA-256 digest, producing code revision, dependency-lock digest, configuration digest, creation time, and parent artifact digests. Readers verify type, version, size bounds, and digest before deserialization. NPZ readers set `allow_pickle=False`.

<a id="public-api-ownership"></a>
## Ownership

The solver maintainer owns numerical Python interfaces; the ML maintainer owns bundle and predictor interfaces; the service maintainer owns HTTP schemas; the release maintainer owns CLI and compatibility policy. Cross-owner changes require updates to contract tests and all linked RFCs.
