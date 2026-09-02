from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import product
from typing import Literal

import pytest
from pydantic import ValidationError

from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import Provenance
from soufflerie.validation import (
    REQUIRED_GATE_DEFINITIONS,
    GateDefinition,
    GateEvidence,
    GateResult,
    MetricObservation,
    MetricSummary,
    ValidationReport,
    divergence_gate_evidence,
    evaluate_gate,
    evaluate_required_gates,
    head_field_consistency_gate_evidence,
    overall_gate_status,
    summarize_metric,
)


def _threshold(definition: GateDefinition) -> float:
    if definition.threshold is None:
        return 10.0
    assert not isinstance(definition.threshold, bool)
    return float(definition.threshold)


def _evidence(
    definition: GateDefinition,
    value: float | int | bool,
) -> GateEvidence:
    return GateEvidence(
        name=definition.name,
        value=value,
        comparison_threshold=(10.0 if definition.threshold is None else None),
        evidence=(f"metric:{definition.metric}",),
    )


@pytest.mark.parametrize(
    "definition",
    tuple(item for item in REQUIRED_GATE_DEFINITIONS if item.name != "evidence_integrity"),
    ids=lambda item: item.name,
)
def test_every_numeric_gate_is_exact_below_at_and_above_threshold(
    definition: GateDefinition,
) -> None:
    threshold = _threshold(definition)
    delta = max(abs(threshold) * 1e-6, 1e-9)
    observed = {
        "below": evaluate_gate(definition, _evidence(definition, threshold - delta)).status,
        "at": evaluate_gate(definition, _evidence(definition, threshold)).status,
        "above": evaluate_gate(definition, _evidence(definition, threshold + delta)).status,
    }
    if definition.operator == "lt":
        assert observed == {"below": "green", "at": "red", "above": "red"}
    elif definition.operator == "ge":
        assert observed == {"below": "red", "at": "green", "above": "green"}
    else:  # pragma: no cover - frozen definitions make this unreachable
        raise AssertionError(f"unexpected numeric operator {definition.operator}")


def test_boolean_integrity_gate_and_invalid_numeric_evidence_are_red() -> None:
    integrity = REQUIRED_GATE_DEFINITIONS[-1]
    assert evaluate_gate(integrity, _evidence(integrity, True)).status == "green"
    assert evaluate_gate(integrity, _evidence(integrity, False)).status == "red"

    field = REQUIRED_GATE_DEFINITIONS[0]
    result = evaluate_gate(
        field,
        GateEvidence(
            name=field.name,
            value=None,
            evidence=("case:a",),
            failure="VAL-2 NO_VALID_CELLS: metric distribution is invalid",
        ),
    )
    assert result.status == "red"
    assert result.value is False
    assert "NO_VALID_CELLS" in result.evidence[-1]


def test_required_gate_evaluation_demands_exactly_all_definitions() -> None:
    evidence = tuple(
        _evidence(
            definition,
            (
                True
                if definition.operator == "eq"
                else _threshold(definition) - 1e-6
                if definition.operator == "lt"
                else _threshold(definition)
            ),
        )
        for definition in REQUIRED_GATE_DEFINITIONS
    )
    results = evaluate_required_gates(evidence)
    assert len(results) == 12
    assert all(result.status == "green" for result in results)
    with pytest.raises(ArtifactIntegrityError, match="every required gate once"):
        evaluate_required_gates(evidence[:-1])
    with pytest.raises(ArtifactIntegrityError, match="every required gate once"):
        evaluate_required_gates((*evidence, evidence[0]))


def test_consistency_fraction_uses_inclusive_ten_percent_and_rejects_invalid_cases() -> None:
    def gap(value: float) -> MetricObservation:
        return MetricObservation(
            name="head_field_gap_pct",
            status="valid",
            value=value,
            units="percent",
        )

    evidence = head_field_consistency_gate_evidence(
        {"1" * 20: gap(9.0), "2" * 20: gap(10.0), "3" * 20: gap(10.001)}
    )
    assert evidence.value == pytest.approx(2 / 3)
    assert evaluate_gate(REQUIRED_GATE_DEFINITIONS[2], evidence).status == "red"

    invalid = MetricObservation(
        name="head_field_gap_pct",
        status="invalid",
        value=None,
        units="percent",
        failure="VAL-1 NONFINITE: drag input is not finite",
    )
    failed = head_field_consistency_gate_evidence({"1" * 20: gap(1.0), "2" * 20: invalid})
    assert failed.value is None
    assert evaluate_gate(REQUIRED_GATE_DEFINITIONS[2], failed).status == "red"


def test_divergence_gate_uses_ratio_of_medians_and_zero_solver_is_red() -> None:
    def summary(
        name: Literal["prediction_div_mean_abs", "solver_div_mean_abs"],
        values: tuple[float, ...],
    ) -> MetricSummary:
        observations = {
            f"{index:020x}": MetricObservation(
                name=name,
                status="valid",
                value=value,
                units="inverse_lattice_unit",
            )
            for index, value in enumerate(values, start=1)
        }
        return summarize_metric(
            name,
            observations,
            report_seed=1,
            bootstrap_resamples=100,
        )

    prediction = summary("prediction_div_mean_abs", (1.0, 2.0, 3.0))
    solver = summary("solver_div_mean_abs", (0.5, 1.0, 1.5))
    evidence = divergence_gate_evidence(prediction, solver)
    assert evidence.value == 2.0
    assert evaluate_gate(REQUIRED_GATE_DEFINITIONS[3], evidence).status == "green"

    zero_solver = summary("solver_div_mean_abs", (0.0,))
    failed = divergence_gate_evidence(prediction, zero_solver)
    assert failed.value is None
    assert evaluate_gate(REQUIRED_GATE_DEFINITIONS[3], failed).status == "red"


def test_overall_status_is_the_conjunction_for_every_gate_vector() -> None:
    names = tuple(definition.name for definition in REQUIRED_GATE_DEFINITIONS)
    for vector in product((False, True), repeat=len(names)):
        gates = tuple(
            GateResult(
                name=name,
                required=True,
                status="green" if value else "red",
                value=value,
                operator="eq",
                threshold=True,
                units="boolean",
                evidence=(f"vector:{index}",),
            )
            for index, (name, value) in enumerate(zip(names, vector, strict=True))
        )
        assert (overall_gate_status(gates) == "green") is all(vector)


def _provenance() -> Provenance:
    started = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)
    return Provenance(
        source_revision="a" * 40,
        source_dirty=False,
        python_version="3.11.14",
        lock_sha256="b" * 64,
        packages={"soufflerie": "0.1.0"},
        os="linux",
        architecture="x86_64",
        device_class="L40S",
        dtype_policy="fields-fp32-metrics-fp64",
        config_sha256="c" * 64,
        parent_sha256={"dataset": "d" * 64, "selected_model": "e" * 64},
        seeds=(17, 23, 31),
        deterministic=True,
        started_at=started,
        completed_at=started + timedelta(minutes=1),
        gpu_seconds=60.0,
    )


def test_validation_report_binds_identity_and_cannot_hide_a_red_gate() -> None:
    observation = MetricObservation(
        name="velocity_rel_l2",
        status="valid",
        value=0.01,
        units="ratio",
    )
    summary = summarize_metric(
        "velocity_rel_l2",
        {"9" * 20: observation},
        report_seed=42,
        bootstrap_resamples=100,
    )
    evidence = tuple(
        _evidence(
            definition,
            (
                True
                if definition.operator == "eq"
                else _threshold(definition) - 1e-6
                if definition.operator == "lt"
                else _threshold(definition)
            ),
        )
        for definition in REQUIRED_GATE_DEFINITIONS
    )
    gates = evaluate_required_gates(evidence)
    report = ValidationReport.create(
        dataset_id="1" * 20,
        selected_model_id="2" * 20,
        ensemble_model_ids=("2" * 20, "3" * 20, "4" * 20),
        baseline_ids=("5" * 20, "6" * 20),
        metrics={"velocity_rel_l2": summary},
        gates=gates,
        overall_status="green",
        provenance=_provenance(),
    )
    assert ValidationReport.model_validate_json(report.model_dump_json()) == report

    payload = report.model_dump(mode="python")
    with pytest.raises(ValidationError, match="conjunction"):
        ValidationReport.model_validate({**payload, "overall_status": "red"})
    with pytest.raises(ValidationError, match="every required gate once"):
        ValidationReport.model_validate({**payload, "gates": gates[:-1]})
