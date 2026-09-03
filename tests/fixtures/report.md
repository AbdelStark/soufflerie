# Validation status: RED

- Overall status: **RED**
- Report ID: `f5d6f4ea4b2aea70590a`
- Dataset ID: `11111111111111111111`
- Selected model ID: `33333333333333333333`
- Ensemble model IDs: `33333333333333333333`, `44444444444444444444`, `55555555555555555555`
- Baseline IDs: `66666666666666666666`, `77777777777777777777`
- Generator: `validation-report-v1`

> **Release blocked:** one or more required gates are red. The plots below
> are diagnostic evidence and do not override this status.

## Required gates

| Gate | Status | Value | Operator | Threshold | Units | Evidence |
|---|---|---:|:---:|---:|---|---|
| field_error | RED | 0.09 | lt | 0.08 | ratio | fixture:median_velocity_rel_l2 |
| cd_head_error | GREEN | 3 | lt | 5 | percent | fixture:median_cd_head_pct |
| head_field_consistency | GREEN | 1 | ge | 0.95 | fraction | fixture:fraction_head_field_gap_le_10_pct |
| divergence | GREEN | 2 | lt | 3 | ratio | fixture:median_prediction_divergence_over_solver |
| obstacle_compliance | GREEN | 0.0084 | lt | 0.01 | ratio | fixture:p95_obstacle_ratio |
| mean_baseline_field | GREEN | 0.09 | lt | 0.15 | ratio | fixture:selected_fno_median_velocity_rel_l2 |
| nearest_baseline_field | GREEN | 0.09 | lt | 0.12 | ratio | fixture:selected_fno_median_velocity_rel_l2 |
| mean_baseline_cd | GREEN | 3 | lt | 7 | percent | fixture:selected_fno_median_cd_head_pct |
| nearest_baseline_cd | GREEN | 3 | lt | 5.5 | percent | fixture:selected_fno_median_cd_head_pct |
| ood_variance_increase | GREEN | 3 | ge | 1.5 | ratio | fixture:median_ood_variance_over_id_boundary |
| sensitivity_sign | GREEN | 9 | ge | 8 | count_of_10 | fixture:agreed_sensitivity_signs |
| evidence_integrity | GREEN | true | eq | true | boolean | fixture:all_lineage_split_and_count_checks |

## Metric distributions

| Metric | Status | Count | Median | P90 | P95 | Maximum | Bootstrap median 95% |
|---|---|---:|---:|---:|---:|---:|---|
| cd_field_pct | valid | 11 | 2.4 | 4 | 4.2 | 4.4 | 1.2 - 3.6 |
| cd_head_pct | valid | 11 | 3 | 5 | 5.25 | 5.5 | 1.5 - 4.5 |
| head_field_gap_pct | valid | 11 | 0.49505 | 0.883002 | 0.9018 | 0.920598 | 0.19685 - 0.763359 |
| obstacle_ratio | valid | 11 | 0.0048 | 0.008 | 0.0084 | 0.0088 | 0.0024 - 0.0072 |
| prediction_div_mean_abs | valid | 11 | 0.6 | 1 | 1.05 | 1.1 | 0.3 - 0.9 |
| solver_div_mean_abs | valid | 11 | 0.3 | 0.5 | 0.525 | 0.55 | 0.15 - 0.45 |
| velocity_rel_l2 | valid | 11 | 0.09 | 0.13 | 0.135 | 0.14 | 0.06 - 0.12 |

Worst `cd_field_pct` cases: `0000000000000000000b`, `0000000000000000000a`, `00000000000000000009`, `00000000000000000008`, `00000000000000000007`, `00000000000000000006`, `00000000000000000005`, `00000000000000000004`, `00000000000000000003`, `00000000000000000002`, `00000000000000000001`.

Worst `cd_head_pct` cases: `0000000000000000000b`, `0000000000000000000a`, `00000000000000000009`, `00000000000000000008`, `00000000000000000007`, `00000000000000000006`, `00000000000000000005`, `00000000000000000004`, `00000000000000000003`, `00000000000000000002`, `00000000000000000001`.

Worst `head_field_gap_pct` cases: `00000000000000000001`, `0000000000000000000b`, `0000000000000000000a`, `00000000000000000002`, `00000000000000000009`, `00000000000000000008`, `00000000000000000003`, `00000000000000000007`, `00000000000000000004`, `00000000000000000006`, `00000000000000000005`.

Worst `obstacle_ratio` cases: `0000000000000000000b`, `0000000000000000000a`, `00000000000000000009`, `00000000000000000008`, `00000000000000000007`, `00000000000000000006`, `00000000000000000005`, `00000000000000000004`, `00000000000000000003`, `00000000000000000002`, `00000000000000000001`.

Worst `prediction_div_mean_abs` cases: `0000000000000000000b`, `0000000000000000000a`, `00000000000000000009`, `00000000000000000008`, `00000000000000000007`, `00000000000000000006`, `00000000000000000005`, `00000000000000000004`, `00000000000000000003`, `00000000000000000002`, `00000000000000000001`.

Worst `solver_div_mean_abs` cases: `0000000000000000000b`, `0000000000000000000a`, `00000000000000000009`, `00000000000000000008`, `00000000000000000007`, `00000000000000000006`, `00000000000000000005`, `00000000000000000004`, `00000000000000000003`, `00000000000000000002`, `00000000000000000001`.

Worst `velocity_rel_l2` cases: `0000000000000000000b`, `0000000000000000000a`, `00000000000000000009`, `00000000000000000008`, `00000000000000000007`, `00000000000000000006`, `00000000000000000005`, `00000000000000000004`, `00000000000000000003`, `00000000000000000002`, `00000000000000000001`.

## OOD heuristic

- Status: `valid`
- Median OOD normalized variance: `3`
- Median ID-boundary normalized variance: `1`
- OOD / ID-boundary ratio: `3`

## Rotation sensitivity

Agreed signs: **9 of 10**.

| Case | Autograd Cd/degree | Central difference Cd/degree | Agrees |
|---|---:|---:|:---:|
| `00000000000000000067` | 1 | 1 | yes |
| `0000000000000000006b` | 1 | 1 | yes |
| `00000000000000000064` | 1 | 1 | yes |
| `00000000000000000065` | 1 | 1 | yes |
| `00000000000000000066` | 1 | 1 | yes |
| `0000000000000000006d` | 1 | 1 | yes |
| `0000000000000000006a` | 1 | 1 | yes |
| `0000000000000000006c` | 1 | 1 | yes |
| `00000000000000000069` | 1 | 1 | yes |
| `00000000000000000068` | -1 | 1 | no |

## Diagnostic plots

- [Representative flow fields](report.plots/representative-fields.svg)
- [Worst-case flow fields](report.plots/worst-fields.svg)
- [Model and baseline comparison](report.plots/baseline-comparison.svg)
- [Error by design parameter](report.plots/error-by-design.svg)
- [Cd head and field consistency](report.plots/head-vs-field.svg)
- [Divergence and obstacle compliance](report.plots/divergence-compliance.svg)
- [OOD ensemble variance](report.plots/ood-variance.svg)
- [Rotation sensitivity agreement](report.plots/sensitivity.svg)

## Provenance

- Source revision: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
- Source dirty: `false`
- Lock SHA-256: `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`
- Config SHA-256: `cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc`
- Device class: `L40S-fixture`
- Report SHA-256: `f5d6f4ea4b2aea70590a6a84644db82f254f9c5d9494c2edfe15ca72e2fce35b`
