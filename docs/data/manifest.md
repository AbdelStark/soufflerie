# Immutable dataset manifest

The Parquet manifest is the only authoritative training index. Directory
contents, successful-looking run roots, and sweep state alone never grant
dataset membership.

## Build trust boundary

`build_manifest` requires the artifact root, the frozen `SweepConfig`, and
exactly 1,000 explicit run references:

```python
manifest = build_manifest(
    artifact_root,
    config=sweep_config,
    run_references=successful_run_references,
)
```

The builder does not list directories. It fully opens every supplied parent
through `LocalRunArtifactStore`, reproduces the canonical design, and requires:

- exactly 1,000 unique case, design, and run identities;
- exact `600/200/200` design-level splits;
- each case and split to match the frozen design and numerical controls;
- valid, converged run metadata and checksum-verified field archives;
- one clean source revision and one dependency-lock digest;
- aggregate referenced run bytes strictly below 2 GiB.

Any mismatch aborts the whole build. There is no partial manifest and no
replacement sample.

## Parquet contract

Rows sort by `design_id`. The fixed columns and Arrow types are:

| Column | Arrow type | Nullable |
|---|---|---|
| `schema_version` | `int16` | no |
| `dataset_id`, `case_id`, `design_id`, `split` | `string` | no |
| `aspect_ratio`, `rotation_deg`, `scale`, `reynolds` | `double` | no |
| `run_uri`, `run_digest` | `string` | no |
| `bytes` | `int64` | no |
| `cd`, `cl_mean` | `double` | no |
| `strouhal` | `double` | yes |
| `solver_valid` | `bool` | no |

`run_uri` must equal `runs/<case_id>/<run_digest>` and is always relative to
the artifact root. Parquet uses four fixed row groups with sizes
`256/256/256/232`, no inferred object columns, and schema metadata recording
the schema fingerprint, writer/version, config/design/split identities,
source revision, and lock digest.

The JSON contracts are
[`dataset-manifest.json`](../../schemas/v1/dataset-manifest.json),
[`manifest-row.json`](../../schemas/v1/manifest-row.json), and
[`dataset-statistics.json`](../../schemas/v1/dataset-statistics.json).

## Identity and statistics

The full logical dataset SHA-256 covers the exact Arrow schema descriptor,
config/design/split digests, and sorted logical rows. Row `dataset_id` and
physical `run_uri` are excluded to avoid self-reference and storage-root
coupling; the run digest remains included. The 20-character `dataset_id` is the
full digest prefix.

`statistics.json` is derived from the rows and records exact split/count/byte
totals plus finite fp64 summaries for geometry, Reynolds, run bytes, Cd, Cl,
and nullable Strouhal. `metadata.json` binds the logical identity, Parquet and
statistics byte digests, all 1,000 direct run digests, source/lock identity,
writer version, schema fingerprint, and size gate.

`LocalDatasetArtifactStore` publishes:

```text
datasets/<dataset_id>/
  manifest.parquet
  metadata.json
  statistics.json
  COMMITTED
```

It stages, fsyncs, reopens every member through bounded readers, writes the
metadata-digest marker last, and atomically renames the complete root. Matching
publication in another physical store yields the same dataset identity.

## Standalone validation

Validate a received manifest without training or remote dependencies:

```bash
uv run soufflerie dataset validate --manifest PATH
```

This command safely checks Parquet byte/row/column/group limits, exact schema
and metadata, row validation and order, uniqueness, splits, payload size, and
recomputed logical identity. Parent field archives are verified at build time;
a standalone manifest cannot prove that unavailable external parents still
exist.

The checked
[`tests/fixtures/dataset/manifest.parquet`](../../tests/fixtures/dataset/manifest.parquet)
is synthetic contract data for this command. It is not evidence that the
canonical 1,000 remote solves have completed. That release evidence belongs to
issue #19.

Run the focused acceptance command:

```bash
uv run pytest tests/datagen/test_manifest.py
uv run soufflerie dataset validate \
  --manifest tests/fixtures/dataset/manifest.parquet
```
