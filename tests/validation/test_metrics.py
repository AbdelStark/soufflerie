from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from soufflerie.errors import ArtifactIntegrityError
from soufflerie.validation import (
    MetricObservation,
    evaluate_case_metrics,
    summarize_metric,
)


def _case_id(index: int) -> str:
    return f"{index:020x}"


def _observation(value: float) -> MetricObservation:
    return MetricObservation(
        name="velocity_rel_l2",
        status="valid",
        value=value,
        units="ratio",
    )


def test_every_case_metric_matches_manufactured_arrays() -> None:
    shape = (5, 5)
    fluid = np.zeros(shape, dtype=np.bool_)
    fluid[1:4, 1:4] = True
    obstacle = np.logical_not(fluid)
    solver_u = np.zeros(shape, dtype=np.float32)
    prediction_u = np.full(shape, np.float32(0.05), dtype=np.float32)
    for column in range(1, 4):
        solver_u[1:4, column] = np.float32(2 * column)
        prediction_u[1:4, column] = np.float32(4 * column)
    solver_v = np.zeros(shape, dtype=np.float32)
    prediction_v = np.zeros(shape, dtype=np.float32)

    metrics = evaluate_case_metrics(
        case_id="a" * 20,
        prediction_u=prediction_u,
        prediction_v=prediction_v,
        solver_u=solver_u,
        solver_v=solver_v,
        fluid_mask=fluid,
        obstacle_mask=obstacle,
        cd_head=2.1,
        cd_field=1.8,
        cd_solver=2.0,
        inlet_velocity_lu=0.1,
        spacing_lu=2.0,
    )

    assert metrics.velocity_rel_l2.value == pytest.approx(1.0)
    assert metrics.cd_head_pct.value == pytest.approx(5.0)
    assert metrics.cd_field_pct.value == pytest.approx(10.0)
    assert metrics.head_field_gap_pct.value == pytest.approx(15.0)
    assert metrics.prediction_div_mean_abs.value == pytest.approx(2.0)
    assert metrics.solver_div_mean_abs.value == pytest.approx(1.0)
    assert metrics.obstacle_ratio.value == pytest.approx(0.5)
    assert all(
        getattr(metrics, name).status == "valid"
        for name in (
            "velocity_rel_l2",
            "cd_head_pct",
            "cd_field_pct",
            "head_field_gap_pct",
            "prediction_div_mean_abs",
            "solver_div_mean_abs",
            "obstacle_ratio",
        )
    )


def test_zero_denominator_uses_exact_floor() -> None:
    zeros = np.zeros((3, 3), dtype=np.float32)
    prediction = zeros.copy()
    prediction[1, 1] = np.float32(1e-8)
    fluid = np.ones((3, 3), dtype=np.bool_)
    obstacle = np.zeros((3, 3), dtype=np.bool_)
    metrics = evaluate_case_metrics(
        case_id="b" * 20,
        prediction_u=prediction,
        prediction_v=zeros,
        solver_u=zeros,
        solver_v=zeros,
        fluid_mask=fluid,
        obstacle_mask=obstacle,
        cd_head=0.0,
        cd_field=0.0,
        cd_solver=0.0,
        inlet_velocity_lu=0.1,
    )
    assert metrics.velocity_rel_l2.value == pytest.approx(1.0)
    assert metrics.cd_head_pct.value == 0.0
    assert metrics.obstacle_ratio.status == "invalid"
    assert "NO_VALID_CELLS" in str(metrics.obstacle_ratio.failure)


def test_nonfinite_and_no_valid_cell_inputs_are_explicit_invalid_evidence() -> None:
    zeros = np.zeros((3, 3), dtype=np.float32)
    prediction = zeros.copy()
    prediction[0, 0] = np.nan
    empty = np.zeros((3, 3), dtype=np.bool_)
    metrics = evaluate_case_metrics(
        case_id="c" * 20,
        prediction_u=prediction,
        prediction_v=zeros,
        solver_u=zeros,
        solver_v=zeros,
        fluid_mask=empty,
        obstacle_mask=empty,
        cd_head=math.inf,
        cd_field=0.0,
        cd_solver=0.0,
        inlet_velocity_lu=math.nan,
    )
    assert metrics.velocity_rel_l2.status == "invalid"
    assert metrics.cd_head_pct.status == "invalid"
    assert metrics.head_field_gap_pct.status == "invalid"
    assert metrics.prediction_div_mean_abs.status == "invalid"
    assert metrics.solver_div_mean_abs.status == "invalid"
    assert metrics.obstacle_ratio.status == "invalid"
    assert metrics.cd_field_pct.value == 0.0


def test_metric_array_contract_rejects_cast_layout_shape_and_overlapping_masks() -> None:
    values = np.zeros((3, 3), dtype=np.float32)
    mask = np.ones((3, 3), dtype=np.bool_)
    arguments: Any = {
        "case_id": "d" * 20,
        "prediction_u": values,
        "prediction_v": values,
        "solver_u": values,
        "solver_v": values,
        "fluid_mask": mask,
        "obstacle_mask": np.zeros((3, 3), dtype=np.bool_),
        "cd_head": 1.0,
        "cd_field": 1.0,
        "cd_solver": 1.0,
        "inlet_velocity_lu": 0.1,
    }
    with pytest.raises(ArtifactIntegrityError, match="DTYPE"):
        evaluate_case_metrics(**{**arguments, "prediction_u": values.astype(np.float64)})
    with pytest.raises(ArtifactIntegrityError, match="LAYOUT"):
        evaluate_case_metrics(**{**arguments, "prediction_u": values[:, ::-1]})
    with pytest.raises(ArtifactIntegrityError, match="SHAPE"):
        evaluate_case_metrics(**{**arguments, "prediction_u": values[:2]})
    with pytest.raises(ArtifactIntegrityError, match="must not overlap"):
        evaluate_case_metrics(**{**arguments, "obstacle_mask": mask})


def test_summary_is_deterministic_order_independent_and_records_tails() -> None:
    observations = {_case_id(index): _observation(float(index)) for index in range(1, 26)}
    summary = summarize_metric(
        "velocity_rel_l2",
        observations,
        report_seed=123,
        bootstrap_resamples=200,
    )
    reversed_summary = summarize_metric(
        "velocity_rel_l2",
        dict(reversed(tuple(observations.items()))),
        report_seed=123,
        bootstrap_resamples=200,
    )

    assert summary == reversed_summary
    assert summary.status == "valid"
    assert summary.count == summary.valid_count == 25
    assert summary.median == 13.0
    assert summary.p90 == pytest.approx(22.6)
    assert summary.p95 == pytest.approx(23.8)
    assert summary.maximum == 25.0
    assert summary.worst_case_ids[0] == _case_id(25)
    assert summary.worst_case_ids[-1] == _case_id(6)
    assert len(summary.worst_case_ids) == 20


def test_summary_never_aggregates_around_an_invalid_case() -> None:
    invalid = MetricObservation(
        name="velocity_rel_l2",
        status="invalid",
        value=None,
        units="ratio",
        failure="VAL-2 NO_VALID_CELLS: no fluid cells",
    )
    summary = summarize_metric(
        "velocity_rel_l2",
        {_case_id(1): _observation(0.1), _case_id(2): invalid},
        report_seed=1,
        bootstrap_resamples=100,
    )
    assert summary.status == "invalid"
    assert summary.valid_count == 1
    assert summary.invalid_case_ids == (_case_id(2),)
    assert summary.median is None
    assert summary.worst_case_ids == ()
