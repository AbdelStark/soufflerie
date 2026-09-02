# Manifest loader and deterministic baselines

The training data boundary starts from one committed dataset `ArtifactRef`.
`open_manifest_dataset(root, reference)` fully verifies the dataset marker,
metadata, statistics, Parquet schema, logical identity, and fixed `600/200/200`
membership before it returns a `ManifestDataset`. It does not discover samples
by listing directories.

## Membership and batches

Every sample open reconstructs the exact run reference from its manifest row:
artifact type, digest, byte count, and canonical URI. The run store verifies its
commit marker, metadata, safe NPZ members, and checksums. The loader then binds
case, design, split, drag, and physical parameters back to the manifest row
before producing a `PreprocessingSample`. A stray file or run directory can
therefore never become training membership.

Training order is a stable SHA-256 rank over the full dataset digest, seed,
epoch, and design ID. The same tuple repeats exactly; changing seed or epoch
changes the train permutation. Validation and test rows always retain canonical
design-ID order. `batch_rows` and `iter_batches` accept sizes from 1 through 64,
retain the final partial batch, and open only one bounded slice at a time.

```python
from pathlib import Path

from soufflerie.training import open_manifest_dataset

dataset = open_manifest_dataset(Path("/artifacts"), dataset_reference)
for batch in dataset.iter_batches(
    preprocessing_statistics,
    "train",
    batch_size=8,
    seed=17,
    epoch=0,
):
    train_step(batch.data)
```

The reference and statistics in this example are already verified typed
records. Training orchestration supplies `train_step`; the loader performs no
implicit framework import, device transfer, or artifact publication.

## Baselines

`fit_baselines(dataset, statistics)` fits exactly two RFC-0007 predictors:

- `MeanFieldBaseline` streams the 600 training rows in design-ID order. It
  accumulates normalized per-pixel/per-channel fields and scalar drag in
  float64, then freezes float32 prediction state.
- `NearestDesignBaseline` maps aspect ratio, rotation, scale, and Reynolds
  number to `[0,1]`. It computes Euclidean distance in float64 and resolves an
  exact tie by the lexicographically smaller training `design_id`. It stores
  only the design index and lazily opens the selected checksum-verified run, so
  the complete training field corpus is not duplicated in memory.

Both implement the same `FlowPredictor.predict(PredictionBatch)` interface as
the FNO and return `PredictionBatchResult`. Output tensors are created through
the input tensor backend on its existing device. Validation can consequently
run learned and baseline outputs through one metric implementation without a
baseline-specific branch.

Neither fitter accepts caller-supplied sample collections. Dataset identity,
preprocessing identity, the exact 600-row membership digest, baseline state,
distance contract, and dtypes are bound into `BaselineMetadata`. Its full
SHA-256 produces the public baseline ID, and the checked
[`baseline-metadata.json`](../../schemas/v1/baseline-metadata.json) schema is
the durable publication contract. Validation and test rows never contribute to
fitting or identity.

These baselines are deterministic comparison models, not evidence that the FNO
is accurate. Release status remains owned by the RFC-0008 validation gates,
which must show that the selected FNO beats both baselines on the frozen test
split.
