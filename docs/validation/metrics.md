# Validation metrics and immutable gates

RFC-0008 validation consumes de-normalized, C-contiguous `float32` fields and
boolean masks. Array shapes and dtypes are checked without casting. Numerical
reductions use `float64`; raw predictions are never masked before obstacle
compliance is measured.

## Per-case protocol

`evaluate_case_metrics` computes seven named observations for one frozen test
case:

```text
velocity_rel_l2 = sqrt(sum_F((u_hat-u)^2 + (v_hat-v)^2))
                  / max(sqrt(sum_F(u^2+v^2)), 1e-8)
cd_head_pct = 100 * abs(cd_head-cd_solver) / max(abs(cd_solver), 0.1)
cd_field_pct = 100 * abs(cd_field-cd_solver) / max(abs(cd_solver), 0.1)
head_field_gap_pct = 100 * abs(cd_head-cd_field) / max(abs(cd_solver), 0.1)
obstacle_ratio = mean_obstacle(sqrt(u_hat^2+v_hat^2)) / U_ref
```

Prediction and solver divergence use the same mask. A valid center and its
left, right, top, and bottom neighbors must all be fluid. The derivative is a
second-order central difference; domain edges and one-sided stencils are
excluded. The canonical curated grid is a two-to-one reduction of the solver
grid, so its default spacing is `2.0` solver lattice units.

A non-finite input, empty fluid/obstacle mask, or absent divergence stencil
produces a typed `status="invalid"` observation with a stable `VAL-1` or
`VAL-2` reason. It is not dropped and is not represented by NaN.

## Distribution summaries

`summarize_metric` sorts observations by case ID before reduction. A complete
valid distribution records count, median, linear-interpolated p90/p95,
maximum, the worst 20 case IDs with case-ID tie breaks, and a 95% percentile
interval for the median. Bootstrap sampling uses NumPy PCG64, the frozen
report seed, replacement at the original sample count, and the declared
`bootstrap_resamples` count.

If even one case is invalid, the summary is invalid: it names every invalid
case and carries no aggregate or worst-case ranking. This prevents a report
from silently improving by omitting failed samples. Machine contracts are
checked in as [`case-metrics.json`](../../schemas/v1/case-metrics.json) and
[`metric-summary.json`](../../schemas/v1/metric-summary.json).

## Required gates

`REQUIRED_GATE_DEFINITIONS` is the immutable ordered set. Dynamic baseline
thresholds are supplied as evidence; all other thresholds are fixed.

| Gate | Operator | Threshold |
|---|---:|---:|
| Field error | `<` | `0.08` median velocity relative L2 |
| Cd head error | `<` | `5.0%` median |
| Head/field consistency | `>=` | `0.95` fraction with gap `<=10%` |
| Divergence | `<` | `3.0` prediction/solver median ratio |
| Obstacle compliance | `<` | `0.01` p95 ratio |
| Mean baseline field | `<` | mean-field baseline median |
| Nearest baseline field | `<` | nearest-design baseline median |
| Mean baseline Cd | `<` | mean-field baseline median |
| Nearest baseline Cd | `<` | nearest-design baseline median |
| OOD variance increase | `>=` | `1.5` |
| Sensitivity sign | `>=` | `8` of `10` |
| Evidence integrity | `==` | `true` |

Strict inequalities remain red at equality. Invalid numeric gate input is
encoded as `value=false` with an explicit failure evidence string; mixed
boolean/numeric comparison is always red, so it cannot accidentally satisfy a
numeric threshold. [`gate-result.json`](../../schemas/v1/gate-result.json)
binds the value, operator, threshold, units, status, and evidence.

`ValidationReport` requires all 12 gates exactly once and accepts green only
when every required result is green. It binds the dataset, selected model,
three distinct ensemble models, two distinct baselines, metric summaries,
gates, provenance, and overall status into both a full SHA-256 and its
20-character report ID. The durable contract is
[`validation-report.json`](../../schemas/v1/validation-report.json).

## Validation

```bash
uv run pytest tests/validation/test_metrics.py tests/validation/test_gates.py
uv run python scripts/export_schemas.py --check
```
