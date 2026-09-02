# OOD ensemble and rotation sensitivity probes

RFC-0008 defines two report-only diagnostics over immutable test evidence. They
do not expand the supported Reynolds range, certify extrapolation, or provide
calibrated uncertainty. The checked-in implementation is a pure evaluator: the
canonical model artifacts and their measured results are produced by the later
training and release-evaluation runs.

## Deterministic probe selection

`select_probe_geometries` accepts manifest rows from one dataset and never
changes split membership. It admits only `split="test"` rows, binds each probe
to the dataset, source case and design, geometry, and source Reynolds value,
then ranks the complete identity by SHA-256. The first ten ranks are frozen in
canonical order. Sensitivity selection first excludes rotations outside
`[1, 29]` degrees so both finite-difference points remain within the public
geometry bounds.

Selection fails closed for mixed datasets, duplicate design identities, fewer
than ten eligible rows, or a non-test source. A stored `ProbeGeometry`
recomputes its rank digest during validation, so changing a bound value without
changing the identity is rejected.

## OOD ensemble variance

The same ten OOD-selected geometries are evaluated with three independently
trained models at:

- OOD controls: `Re=20` and `Re=400`.
- In-domain boundary controls: `Re=40` and `Re=300`.

For each geometry and Reynolds value, `evaluate_ensemble_variance` requires
three distinct model IDs and three distinct full model digests. Every model
must return the same finite, C-contiguous `float32[3,H,W]` field and nonempty
boolean fluid mask. Population variance is computed across models in `float64`,
divided by the corresponding training-set variance for each output channel,
then averaged over channels and fluid cells.

`summarize_ood_evaluation` requires all 40 probe results. It records the median
OOD value, median in-domain boundary value, and their ratio. A zero in-domain
median is explicit invalid evidence, not infinity. The immutable gate is green
when the ratio is at least `1.5`. The durable machine contract is
[`ood-evaluation.json`](../../schemas/v1/ood-evaluation.json).

## Rotation sensitivity

`RotationSensitivityPredictor` is the adapter boundary for one selected model.
Its autograd method must return the Cd head and
`dCd_head/drotation_deg` from the same graph, expressed per physical degree.
The direct method evaluates identical model and preprocessing inputs at the
center and at `rotation_deg +/- 0.25`.

`evaluate_rotation_sensitivity` records all three Cd values, the autograd
derivative, the central difference, the fixed step, the `1e-5 Cd/degree`
magnitude tolerance, and the agreement decision. Agreement means either:

- both magnitudes exceed the tolerance and their signs match; or
- both magnitudes are at or below the tolerance.

One small and one material derivative is a disagreement. Center prediction
mismatch, non-finite evidence, or adapter failure aborts the probe.
`evaluate_sensitivity_probes` requires ten distinct probes in canonical order;
the gate is green for at least eight agreements. The durable contract is
[`sensitivity-evaluation.json`](../../schemas/v1/sensitivity-evaluation.json).

## Report integration and validation

`ValidationReport` can bind the complete OOD and sensitivity records. When
present, their three ensemble model IDs must match the report ensemble, and the
sensitivity model must be the selected model. Gate evidence references the
model identities and exact probe counts; renderers consume these records and
must not recompute them.

Run the focused contract suite and schema check with:

```bash
uv run pytest tests/validation/test_ood.py tests/validation/test_sensitivity.py
uv run python scripts/export_schemas.py --check
```
