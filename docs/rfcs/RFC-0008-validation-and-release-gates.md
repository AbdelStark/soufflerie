# RFC-0008: Validation metrics and release gates

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

Validation is a pure, schema-versioned evaluation pipeline over the frozen test split, two baselines, three FNO seeds, and fixed out-of-domain probes. It computes precisely defined accuracy, physics, consistency, uncertainty, and sensitivity metrics; emits JSON/Markdown/plots; and sets overall green only when every required gate passes.

## Motivation

The validation harness is the project’s differentiating component. The PRD explicitly rejects MSE-only trust and requires red failures to remain visible. The gate implementation must be independent of report rendering and serving so no layer can reinterpret or hide a result.

## Goals

- Define every metric, aggregation, threshold, and edge case mechanically.
- Compare the selected FNO with both baselines without test leakage.
- Quantify OOD ensemble response and gradient sanity.
- Bind validation evidence to exact solver, dataset, model, code, and dependency identities.
- Generate machine-readable and human-readable evidence from one report model.

## Non-Goals

- Engineering certification, uncertainty calibration guarantees, or extrapolation support.
- Adjusting thresholds after seeing test results.
- Treating the numerical solver as error-free physical truth.
- Hiding prediction service when a model is red.

## Proposed Design

Metric calculations use de-normalized float32 fields, boolean masks, and fp64 reductions. For sample `j`, with fluid cells `F`:

```text
velocity_rel_l2_j = sqrt(sum_F((u_hat-u)^2+(v_hat-v)^2))
                    / max(sqrt(sum_F(u^2+v^2)), 1e-8)
cd_head_pct_j = 100*abs(cd_head-cd_solver)/max(abs(cd_solver), 0.1)
cd_field_pct_j = 100*abs(cd_field-cd_solver)/max(abs(cd_solver), 0.1)
head_field_gap_pct_j = 100*abs(cd_head-cd_field)
                       / max(abs(cd_solver), 0.1)
div_mean_abs_j = mean(interior_F(abs(du/dx + dv/dy)))
obstacle_ratio_j = mean_obstacle(sqrt(u_hat^2+v_hat^2)) / U_ref
```

Spatial derivatives use second-order central differences on interior fluid cells whose required neighbors are fluid; one-sided stencils are excluded. Solver baseline divergence uses the same downsampled mask/stencil and test rows. `cd_field` follows RFC-0003. A metric with no valid cells or non-finite input is an explicit failed gate, not a skipped sample.

```python
class GateResult(BaseModel):
    name: str
    required: bool
    status: Literal["green", "red"]
    value: float | int | bool
    operator: Literal["lt", "le", "gt", "ge", "eq"]
    threshold: float | int | bool
    units: str
    evidence: list[str]

class ValidationReport(BaseModel):
    schema_version: Literal[1] = 1
    report_id: str
    dataset_id: str
    selected_model_id: str
    ensemble_model_ids: tuple[str, str, str]
    baseline_ids: tuple[str, str]
    metrics: dict[str, MetricSummary]
    gates: tuple[GateResult, ...]
    overall_status: Literal["green", "red"]
    provenance: Provenance
```

Required gates are fixed:

| Gate | Calculation | Green condition |
|---|---|---|
| Field error | median test `velocity_rel_l2` | `< 0.08` |
| Cd head error | median test `cd_head_pct` | `< 5.0%` |
| Head/field consistency | fraction test samples with gap `<=10%` | `>= 0.95` |
| Divergence | median prediction `div_mean_abs` / median solver value | `< 3.0` |
| Obstacle compliance | p95 test `obstacle_ratio` | `< 0.01` |
| Mean baseline field | selected FNO median field error | `<` mean-field baseline |
| Nearest baseline field | selected FNO median field error | `<` nearest baseline |
| Mean baseline Cd | selected FNO median Cd error | `<` mean-field baseline |
| Nearest baseline Cd | selected FNO median Cd error | `<` nearest baseline |
| OOD variance increase | OOD normalized ensemble variance / ID boundary variance | `>= 1.5` |
| Sensitivity sign | agreed central-difference signs | `>= 8 of 10` |
| Evidence integrity | all IDs/digests/split/counts valid | `true` |

Strict inequalities remain strict at the boundary. In addition to required aggregates, the report lists distribution count, median, p90, p95, maximum, bootstrap 95% interval for medians with a fixed report seed, and the worst 20 design IDs for each primary metric.

OOD cases use the ten fixed test geometries whose hash rank is lowest, evaluated at both `Re=20` and `Re=400`. ID boundary controls use the same geometries at `Re=40` and `Re=300`. For each case, ensemble variance is the mean across fluid cells/channels of variance across the three independently trained predictions, normalized by training output variance. The gate compares median OOD to median ID. OOD outputs are report-only and never returned as supported public predictions.

Sensitivity uses the ten fixed in-domain test cases whose hash rank is lowest and whose rotation lies in `[1,29]` degrees. Autograd computes `dCd_head/drotation_deg`. Central difference uses `h=0.25 degrees` with identical model and preprocessing. A sign agrees when both magnitudes exceed `1e-5 Cd/degree` and signs match; if both are at or below the tolerance, it also agrees. One-small/one-zero is disagreement. The report lists each pair.

`overall_status = green` if and only if every required `GateResult` is green. Renderers accept only `ValidationReport`; they do not recompute status. Markdown starts with status, model/dataset IDs, and a complete gate table. Red status uses text plus color and appears before favorable plots. Plots include representative/worst solver-surrogate-error fields, baseline comparison, error by design parameters, head-vs-field scatter, divergence/compliance distributions, OOD variance, and sensitivity pairs. Generated Markdown values come from JSON and carry a generator version; hand editing fails regeneration checks.

The service loads report JSON, verifies its digest and model/dataset identities, and propagates `overall_status` plus per-prediction consistency/OOD flags. It serves a red model with the explicit banner contract. Missing/mismatched report makes readiness false; it does not default to green.

## Alternatives Considered

### One weighted trust score

It is easy to display but lets strong metrics hide categorical failures and makes weights arbitrary. Independent required gates preserve failure meaning.

### Mean aggregation

Means overemphasize extreme cases at this dataset size and can obscure typical behavior in either direction. Medians with tails/worst cases provide both robust gate and failure visibility.

### Monte Carlo dropout for OOD

It changes the fixed architecture/training and often underestimates epistemic uncertainty. Three independently seeded models already exist and produce a clearer ensemble signal.

### Field masking before metrics

It would guarantee obstacle compliance by construction. Raw predictions must be evaluated so the gate remains meaningful.

## Drawbacks

- Fixed thresholds can block release even when visuals appear acceptable.
- Solver-relative metrics inherit solver discretization error.
- A three-seed variance ratio is a limited OOD heuristic, not calibrated uncertainty.
- Control-volume Cd may be noisy at the downsampled resolution.

## Migration / Rollout

1. Implement metric functions with manufactured-array tests.
2. Implement immutable gate definitions and boundary tests.
3. Add baseline/ensemble/OOD/sensitivity evaluators.
4. Implement JSON report, deterministic Markdown/plots, and regeneration checks.
5. Run validation once after all model selection is frozen; check evidence into `reports/`.

Changing a metric, threshold, probe selection, or aggregation creates a new validation schema/RFC and report ID; prior reports remain intact.

## Testing Strategy

- Hand-calculate every metric on tiny arrays, including masks, stencils, zero denominators, and non-finite inputs.
- Test each gate just below, exactly at, and just above threshold.
- Assert overall status equals conjunction for every generated Boolean gate vector.
- Prove test rows do not affect model/baseline/preprocessing selection.
- Verify OOD/ID and sensitivity case selection by stable hash and three distinct model IDs.
- Compare autograd and central difference on analytic toy predictors before FNO cases.
- Golden-test JSON schema, deterministic Markdown, plot manifest, and regeneration cleanliness.
- Contract-test service startup mismatch and UI red-banner behavior.

## Open Questions

None for v0.1. The OOD ratio is explicitly a heuristic; calibration or support claims require a future RFC owned by the validation maintainer.

## References

- [`prd.md#64-validation-the-consistency-harness`](../../prd.md#64-validation-the-consistency-harness)
- [`00-overview.md#success-criteria`](../spec/00-overview.md#success-criteria)
- [`07-testing-strategy.md#ml-tests`](../spec/07-testing-strategy.md#ml-tests)
- [RFC-0003](RFC-0003-geometry-boundaries-and-forces.md)
- [RFC-0006](RFC-0006-fno-surrogate-and-checkpoints.md)
- [RFC-0007](RFC-0007-training-and-reproducibility.md)
