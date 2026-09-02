# Canonical experiment design

The `mvp-v1` design is a frozen set of 1,000 ellipse/Reynolds points. It is
generated locally before any remote solver work, and split membership belongs to
the physical design point rather than to a run or snapshot.

## Reproduce the design

The checked-in config is [`configs/sweeps/mvp-v1.yaml`](../../configs/sweeps/mvp-v1.yaml).
Generate the evidence from its strict typed content:

```bash
uv run python scripts/generate_design_summary.py \
  --output reports/data/design-mvp-v1.json
```

Verify the checked-in evidence without updating it:

```bash
uv run python scripts/generate_design_summary.py \
  --check reports/data/design-mvp-v1.json
```

Generation performs the RFC-0002 scalar lattice preflight and the RFC-0003
dense geometry/connectivity preflight for every point. A failure aborts the
whole design; it never triggers resampling.

## Sampling contract

The seed `20260901` initializes a local NumPy `Generator(PCG64(seed))`; global
NumPy RNG state is untouched. The first 32 raw uint64 values seed independent
candidate generators. Each candidate contains one jittered value in every one
of the 1,000 strata on each normalized dimension. Candidate selection maximizes
the minimum pairwise Euclidean distance in normalized four-dimensional space,
with the lowest index winning an exact tie.

The locked design selects candidate `10` with minimum normalized distance
`0.03666928448214767`. Values are mapped linearly and retained as binary64 for
identity and preflight:

| Dimension | Range |
|---|---:|
| aspect ratio | `[0.5, 1.0]` |
| rotation | `[0, 30]` degrees |
| scale | `[0.75, 1.25]` |
| Reynolds number | `[40, 300]` |

`design_id` is the first 20 lowercase hexadecimal characters of the SHA-256 of
schema version, shape family, physical ellipse parameters, and Reynolds number.
Grid, run schedule, seed, path, and execution order are excluded. Binding a
point to those numerical controls produces a distinct `case_id`.

## Frozen splits and digests

For each point, split ranking hashes the ASCII salt `split-v1`, the unsigned
seed as exactly eight big-endian bytes, and canonical physical design JSON.
Full digests are sorted ascending with `design_id` as the deterministic tie
breaker. Ranks `0:600` are train, `600:800` validation, and `800:1000` test.

| Identity | SHA-256 |
|---|---|
| Config | `04ffaf7dcb027482d82c921fa617a429872a06e5d0a92545e3cfaf724a011333` |
| Design | `352a060bdb7ef2ff3e9432d7eff4333d6d3c9bd9aca33c6f29f6f56a307250c1` |
| Split membership | `406a35bdef46cd10e77efc7fc6b301ffbb089783c2454dc86c8308a093e9d027` |
| Candidate seeds | `3a742d5398c7134219807743428650bd1a4a6b5750c10f6839a893a2e0ae8b01` |

The machine-readable summary is
[`reports/data/design-mvp-v1.json`](../../reports/data/design-mvp-v1.json),
validated by [`schemas/v1/design-summary.json`](../../schemas/v1/design-summary.json).
