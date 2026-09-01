# Artifact trust boundary

Soufflerie treats every persisted artifact as untrusted until its location, byte
limits, digest, schema, and lineage have all been checked. The public helpers in
`soufflerie.artifacts` implement that boundary for JSON records, NPZ arrays,
Parquet tables, and safetensor weights.

## Identity and provenance

An `ArtifactEnvelope` separates immutable logical identity from mutable storage
metadata. Its SHA-256 covers the artifact type, logical metadata, member
digests, and reproducibility provenance. It deliberately excludes the envelope
URI, creation time, and provenance start/completion wall-clock times. Logical
metadata rejects location, timestamp, and digest-shaped field names so those
mutable values cannot silently enter the identity boundary.

`capture_provenance` records a full Git revision, dirty state, lock digest,
explicitly allowlisted package versions, platform and numerical policy, config
digest, parent digests, seeds, determinism, timing, and GPU seconds. It does not
capture arbitrary environment variables. Before release,
`validate_release_provenance` rejects dirty source; source, lock, config, or
package drift; missing or extra parents; and non-deterministic execution.
The canonical standalone provenance contract remains
[`schemas/v1/provenance.json`](../../schemas/v1/provenance.json); artifact
envelopes embed that same model through
[`schemas/v1/artifact-envelope.json`](../../schemas/v1/artifact-envelope.json).

## Lineage

Each `LineageNode` binds an artifact type and content ID to its full digest and
typed direct parents. `verify_lineage` rejects duplicate identities, prefix
collisions, missing parents, incorrect parent types, disallowed or missing
required parent classes, unknown artifact classes, oversized graphs, and
cycles. The default policy is:

| Child | Required direct parents | Additional allowed parents |
|---|---|---|
| run | none | none |
| dataset | run | none |
| model | dataset | none |
| baseline | dataset | none |
| report | dataset, model | baseline |

Consumers must call `verify_consumer_identities` with the exact dataset, model,
and report content IDs they intend to use. Acceptance requires one valid DAG in
which the selected model references the selected dataset and the selected
report references both.

## Safe reader contract

All reader inputs are root-relative POSIX keys, never arbitrary paths or URLs.
Absolute paths, traversal, alternate separators, colon-bearing schemes, and
symbolic-link components fail closed. `ReaderLimits` applies byte, row, column,
member, compression, JSON-depth, and tensor-allocation ceilings before a
format-specific decoder allocates payload data.

- JSON is UTF-8, duplicate-key-free, finite, depth/key bounded, schema-v1
  checked, and validated into an explicitly supplied strict Pydantic model.
- NPZ central-directory entries and NPY headers are checked before
  `numpy.load(..., allow_pickle=False)`. Members must exactly match supplied
  dtype, shape, C-order, finite, and no-pickle descriptors.
- Parquet metadata is checked before reading and only an exact declared set of
  primitive columns is projected. Row groups, rows, columns, file bytes, and
  decoded bytes are bounded.
- Safetensor headers, names, dtypes, shapes, non-overlapping contiguous offsets,
  and total bytes are checked before the safe decoder runs. Returned arrays are
  read-only.

Callers should pass a reviewed member SHA-256 to every reader and retain the
verified `ArtifactEnvelope` and lineage graph beside the decoded value. A safe
reader establishes structural and integrity validity; it does not establish
scientific correctness or model quality.
