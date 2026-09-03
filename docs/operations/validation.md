# Remote validation

Validation consumes a frozen `ValidationConfig` plus the complete training
index. It opens every dataset and model artifact again, refits the two fixed
train-only baselines, verifies their full identities, and refuses any parent
drift before evaluating test data.

```bash
uv run --extra remote modal run infra/validate.py \
  --config configs/validation/release-v1.yaml \
  --training-index reports/training/index.json \
  --output reports/validation-receipt.json
```

The command requires a clean validation source revision plus the training
index's lock, dataset, model ensemble, selected model, baseline IDs, and
precision. Model bundles remain bound to the training source revision recorded
in that index; the report is independently bound to the validation source
revision. The local adapter constructs one canonical bounded request with all
full parent digests and the requested validation device class. The remote
worker has a 30-minute timeout, one-container concurrency, one explicit GPU,
and no provider retry.

The provider function only verifies/stages the request and calls the
provider-neutral validation pipeline. Metric formulas, gate definitions,
selection rules, OOD probes, and sensitivity rules remain in the domain
modules. The pipeline:

1. verifies the 1,000-case manifest and all three safe model bundles;
2. proves model dataset/source/lock/preprocessing identity and three distinct
   training seeds;
3. fits mean-field and nearest-design baselines from the train split only;
4. evaluates the selected FNO and both baselines on the same frozen 200-case
   test membership using physical fp32 fields and fp64 reductions;
5. runs the fixed 10-geometry OOD/ID ensemble probes and 10 rotation
   autograd/central-difference probes; and
6. calls the immutable RFC-0008 gate evaluator, renders JSON/Markdown/eight SVG
   plots, verifies the complete file set, commits it, and returns a small
   receipt.

The solver parent is a canonical digest of the dataset manifest's ordered
1,000 full run digests. The report parent map must contain exactly `dataset`,
`solver`, `ensemble_model_0..2`, and `baseline_0..1`. A mismatch has a distinct
`REMOTE_VALIDATE_*_MISMATCH` error for dataset, model, selected model, baseline,
solver, config, source, lock, or parent map. There is no fallback to mutable
paths and no default-green behavior.

Reports publish below
`/data/soufflerie/v1/validation/<report-id>/`. The directory contains canonical
`report.json`, `report.md`, a plot manifest, and the fixed SVG set. A report
with red gates is still an honest successful validation artifact and its
receipt reports `overall_status: red`; it does not satisfy release readiness.
Malformed, incomplete, non-finite, or lineage-incoherent evidence fails
publication instead of being omitted.

CI imports these entrypoints through a stubbed Modal SDK and exercises request,
receipt, tamper, and parent-mismatch contracts without authenticating or
spending GPU time. Authenticated acceptance is manual and must use a clean
commit with a real training index.

After a smoke training index exists and its generated model/baseline IDs replace
the explicit sentinels in `configs/validation/smoke.yaml`, run the smaller
bootstrap acceptance with:

```bash
uv run --extra remote modal run infra/validate.py \
  --config configs/validation/smoke.yaml \
  --training-index reports/training/index.json
```

This smoke reduces only bootstrap resampling from 2,000 to 100. It still opens
real artifacts and evaluates all fixed test, OOD, sensitivity, and gate
memberships; it cannot manufacture release evidence from sentinel IDs.
