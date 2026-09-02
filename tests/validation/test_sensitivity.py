from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import pytest

from soufflerie.datagen.manifest import ManifestRow
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.validation import (
    AutogradCdResult,
    ProbeGeometry,
    ProbeModelIdentity,
    SensitivityEvaluation,
    SensitivityProbeResult,
    evaluate_gate,
    evaluate_rotation_sensitivity,
    evaluate_sensitivity_probes,
    select_probe_geometries,
    sensitivity_gate_evidence,
)
from soufflerie.validation.gates import GATE_DEFINITIONS_BY_NAME


def _row(
    index: int,
    *,
    split: Literal["train", "validation", "test"] = "test",
    rotation: float = 15.0,
) -> ManifestRow:
    case_id = f"{index:020x}"
    design_id = f"{index + 1000:020x}"
    digest = f"{index + 1:064x}"
    return ManifestRow(
        dataset_id="d" * 20,
        case_id=case_id,
        design_id=design_id,
        split=split,
        aspect_ratio=0.75,
        rotation_deg=rotation,
        scale=1.0,
        reynolds=100.0,
        run_uri=f"runs/{case_id}/{digest}",
        run_digest=digest,
        bytes=100,
        cd=1.0,
        cl_mean=0.0,
        strouhal=0.17,
    )


@dataclass(frozen=True)
class AnalyticPredictor:
    model: ProbeModelIdentity
    reported_gradient_scale: float = 1.0
    center_offset: float = 0.0

    @staticmethod
    def _cd(rotation_deg: float) -> float:
        return rotation_deg * rotation_deg + 0.5 * rotation_deg + 2.0

    def cd_and_rotation_gradient(self, probe: ProbeGeometry) -> AutogradCdResult:
        return AutogradCdResult(
            cd_head=self._cd(probe.rotation_deg) + self.center_offset,
            rotation_gradient_cd_per_degree=(2.0 * probe.rotation_deg + 0.5)
            * self.reported_gradient_scale,
        )

    def cd_at_rotation(self, probe: ProbeGeometry, rotation_deg: float) -> float:
        return self._cd(rotation_deg)


def _model() -> ProbeModelIdentity:
    return ProbeModelIdentity(model_id="a" * 20, model_sha256="a" * 64)


def _probes() -> tuple[ProbeGeometry, ...]:
    return select_probe_geometries(
        tuple(_row(index, rotation=float(index + 2)) for index in range(12)),
        selection="sensitivity",
    )


def test_analytic_quadratic_matches_autograd_and_fixed_central_difference() -> None:
    probe = _probes()[0]
    result = evaluate_rotation_sensitivity(probe, AnalyticPredictor(_model()))
    expected = 2.0 * probe.rotation_deg + 0.5
    assert result.h_degrees == 0.25
    assert result.magnitude_tolerance_cd_per_degree == 1e-5
    assert result.autograd_cd_per_degree == pytest.approx(expected)
    assert result.central_difference_cd_per_degree == pytest.approx(expected)
    assert result.agrees is True


def test_sign_policy_rejects_one_small_gradient_and_center_mismatch() -> None:
    probe = _probes()[0]
    mismatch = evaluate_rotation_sensitivity(
        probe,
        AnalyticPredictor(_model(), reported_gradient_scale=0.0),
    )
    assert mismatch.agrees is False

    with pytest.raises(ArtifactIntegrityError, match="center predictions differ"):
        evaluate_rotation_sensitivity(
            probe,
            AnalyticPredictor(_model(), center_offset=0.1),
        )


@dataclass(frozen=True)
class FlatPredictor:
    model: ProbeModelIdentity

    def cd_and_rotation_gradient(self, probe: ProbeGeometry) -> AutogradCdResult:
        return AutogradCdResult(cd_head=2.0, rotation_gradient_cd_per_degree=0.0)

    def cd_at_rotation(self, probe: ProbeGeometry, rotation_deg: float) -> float:
        return 2.0


def test_both_below_magnitude_tolerance_agree() -> None:
    result = evaluate_rotation_sensitivity(_probes()[0], FlatPredictor(_model()))
    assert result.autograd_cd_per_degree == 0.0
    assert result.central_difference_cd_per_degree == 0.0
    assert result.agrees is True


def test_recorded_sensitivity_evidence_rejects_contract_tampering() -> None:
    result = evaluate_rotation_sensitivity(_probes()[0], AnalyticPredictor(_model()))
    payload = result.model_dump(mode="json")
    with pytest.raises(ValueError, match="step must remain"):
        SensitivityProbeResult.model_validate({**payload, "h_degrees": 0.5})
    with pytest.raises(ValueError, match="central difference"):
        SensitivityProbeResult.model_validate(
            {**payload, "central_difference_cd_per_degree": 123.0}
        )
    with pytest.raises(ValueError, match="fixed sign policy"):
        SensitivityProbeResult.model_validate({**payload, "agrees": not result.agrees})


def test_sensitivity_adapter_failures_are_fail_closed() -> None:
    ood_probe = select_probe_geometries(tuple(_row(index) for index in range(10)), selection="ood")[
        0
    ]
    with pytest.raises(ArtifactIntegrityError, match="selected sensitivity probe"):
        evaluate_rotation_sensitivity(ood_probe, AnalyticPredictor(_model()))

    invalid_model = AnalyticPredictor(cast(Any, object()))
    with pytest.raises(ArtifactIntegrityError, match="model identity is invalid"):
        evaluate_rotation_sensitivity(_probes()[0], invalid_model)

    @dataclass(frozen=True)
    class FailingPredictor:
        model: ProbeModelIdentity

        def cd_and_rotation_gradient(self, probe: ProbeGeometry) -> AutogradCdResult:
            raise RuntimeError("analytic test failure")

        def cd_at_rotation(self, probe: ProbeGeometry, rotation_deg: float) -> float:
            return 0.0

    with pytest.raises(ArtifactIntegrityError, match="predictor evaluation failed"):
        evaluate_rotation_sensitivity(_probes()[0], FailingPredictor(_model()))


def test_complete_sensitivity_evaluation_records_ten_values_and_gate_count() -> None:
    evaluation = evaluate_sensitivity_probes(_probes(), AnalyticPredictor(_model()))
    assert evaluation.agreement_count == 10
    assert len(evaluation.results) == 10
    gate = evaluate_gate(
        GATE_DEFINITIONS_BY_NAME["sensitivity_sign"],
        sensitivity_gate_evidence(evaluation),
    )
    assert gate.status == "green"
    assert SensitivityEvaluation.model_validate_json(evaluation.model_dump_json()) == evaluation

    with pytest.raises(ArtifactIntegrityError, match="exactly ten"):
        evaluate_sensitivity_probes(_probes()[:-1], AnalyticPredictor(_model()))


def test_sensitivity_evaluation_requires_one_dataset_identity() -> None:
    model = _model()
    predictor = AnalyticPredictor(model)
    results = tuple(evaluate_rotation_sensitivity(probe, predictor) for probe in _probes())
    original = results[0]
    source_index = int(original.geometry.source_design_id, 16) - 1000
    mixed_probe = ProbeGeometry.from_manifest(
        _row(
            source_index,
            rotation=original.geometry.rotation_deg,
        ).model_copy(update={"dataset_id": "e" * 20}),
        selection="sensitivity",
    )
    mixed_result = evaluate_rotation_sensitivity(mixed_probe, predictor)
    mixed_results = tuple(
        sorted(
            (mixed_result, *results[1:]),
            key=lambda item: (item.geometry.rank_sha256, item.geometry.source_design_id),
        )
    )
    with pytest.raises(ValueError, match="one dataset identity"):
        SensitivityEvaluation(
            model=model,
            results=mixed_results,
            agreement_count=sum(item.agrees for item in mixed_results),
        )
