"""Rotation sensitivity probes with autograd/central-difference agreement."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Protocol, Self

from pydantic import Field, model_validator

from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import StrictFrozenModel
from soufflerie.validation.ood import PROBE_COUNT, ProbeGeometry, ProbeModelIdentity

CENTRAL_DIFFERENCE_H_DEG = 0.25
SENSITIVITY_MAGNITUDE_TOLERANCE = 1e-5
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class AutogradCdResult(StrictFrozenModel):
    """Cd and its physical-degree rotation derivative from one autograd graph."""

    cd_head: FiniteFloat
    rotation_gradient_cd_per_degree: FiniteFloat


class RotationSensitivityPredictor(Protocol):
    """Adapter contract; autograd is evaluated at the physical degree boundary."""

    @property
    def model(self) -> ProbeModelIdentity: ...

    def cd_and_rotation_gradient(self, probe: ProbeGeometry) -> AutogradCdResult: ...

    def cd_at_rotation(self, probe: ProbeGeometry, rotation_deg: float) -> float: ...


class SensitivityProbeResult(StrictFrozenModel):
    """One fixed-h central difference and autograd sign comparison."""

    geometry: ProbeGeometry
    model: ProbeModelIdentity
    h_degrees: float = Field(
        default=0.25,
        allow_inf_nan=False,
        json_schema_extra={"const": 0.25},
    )
    magnitude_tolerance_cd_per_degree: float = Field(
        default=1e-5,
        allow_inf_nan=False,
        json_schema_extra={"const": 1e-5},
    )
    cd_minus: FiniteFloat
    cd_center: FiniteFloat
    cd_plus: FiniteFloat
    autograd_cd_per_degree: FiniteFloat
    central_difference_cd_per_degree: FiniteFloat
    agrees: bool

    @model_validator(mode="after")
    def _comparison_is_exact(self) -> Self:
        if self.geometry.selection != "sensitivity":
            raise ValueError("sensitivity result requires a sensitivity-selected geometry")
        if self.h_degrees != CENTRAL_DIFFERENCE_H_DEG:
            raise ValueError("sensitivity central-difference step must remain 0.25 degrees")
        if self.magnitude_tolerance_cd_per_degree != SENSITIVITY_MAGNITUDE_TOLERANCE:
            raise ValueError("sensitivity magnitude tolerance must remain 1e-5")
        expected_difference = (self.cd_plus - self.cd_minus) / (2.0 * self.h_degrees)
        if not math.isclose(
            self.central_difference_cd_per_degree,
            expected_difference,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("central difference does not match the recorded Cd values")
        autograd_magnitude = abs(self.autograd_cd_per_degree)
        central_magnitude = abs(self.central_difference_cd_per_degree)
        tolerance = self.magnitude_tolerance_cd_per_degree
        expected_agreement = (
            autograd_magnitude <= tolerance and central_magnitude <= tolerance
        ) or (
            autograd_magnitude > tolerance
            and central_magnitude > tolerance
            and math.copysign(1.0, self.autograd_cd_per_degree)
            == math.copysign(1.0, self.central_difference_cd_per_degree)
        )
        if self.agrees != expected_agreement:
            raise ValueError("sensitivity agreement does not match the fixed sign policy")
        return self


def evaluate_rotation_sensitivity(
    probe: ProbeGeometry,
    predictor: RotationSensitivityPredictor,
) -> SensitivityProbeResult:
    """Compare one autograd derivative with the fixed double-sided difference."""

    if not isinstance(probe, ProbeGeometry) or probe.selection != "sensitivity":
        raise ArtifactIntegrityError("VAL-7 SENSITIVITY: selected sensitivity probe is required")
    model = predictor.model
    if not isinstance(model, ProbeModelIdentity):
        raise ArtifactIntegrityError("VAL-7 SENSITIVITY: predictor model identity is invalid")
    try:
        autograd = predictor.cd_and_rotation_gradient(probe)
        center = float(predictor.cd_at_rotation(probe, probe.rotation_deg))
        minus = float(
            predictor.cd_at_rotation(probe, probe.rotation_deg - CENTRAL_DIFFERENCE_H_DEG)
        )
        plus = float(predictor.cd_at_rotation(probe, probe.rotation_deg + CENTRAL_DIFFERENCE_H_DEG))
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("VAL-7 SENSITIVITY: predictor evaluation failed") from error
    if not isinstance(autograd, AutogradCdResult) or not all(
        math.isfinite(value) for value in (center, minus, plus)
    ):
        raise ArtifactIntegrityError("VAL-7 SENSITIVITY: predictor returned non-finite evidence")
    if not math.isclose(autograd.cd_head, center, rel_tol=1e-8, abs_tol=1e-10):
        raise ArtifactIntegrityError(
            "VAL-7 SENSITIVITY: autograd and direct center predictions differ"
        )
    central = (plus - minus) / (2.0 * CENTRAL_DIFFERENCE_H_DEG)
    tolerance = SENSITIVITY_MAGNITUDE_TOLERANCE
    autograd_magnitude = abs(autograd.rotation_gradient_cd_per_degree)
    central_magnitude = abs(central)
    agrees = (autograd_magnitude <= tolerance and central_magnitude <= tolerance) or (
        autograd_magnitude > tolerance
        and central_magnitude > tolerance
        and math.copysign(1.0, autograd.rotation_gradient_cd_per_degree)
        == math.copysign(1.0, central)
    )
    return SensitivityProbeResult(
        geometry=probe,
        model=model,
        cd_minus=minus,
        cd_center=center,
        cd_plus=plus,
        autograd_cd_per_degree=autograd.rotation_gradient_cd_per_degree,
        central_difference_cd_per_degree=central,
        agrees=agrees,
    )


class SensitivityEvaluation(StrictFrozenModel):
    """Complete ten-case sensitivity evidence for one selected model."""

    model: ProbeModelIdentity
    results: tuple[SensitivityProbeResult, ...]
    agreement_count: int = Field(ge=0, le=PROBE_COUNT)

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_results(cls, value: object) -> object:
        if isinstance(value, Mapping) and isinstance(value.get("results"), list):
            return {**value, "results": tuple(value["results"])}
        return value

    @model_validator(mode="after")
    def _evaluation_is_complete(self) -> Self:
        if len(self.results) != PROBE_COUNT:
            raise ValueError("sensitivity evaluation requires exactly ten probes")
        if (
            tuple(
                sorted(
                    self.results,
                    key=lambda item: (
                        item.geometry.rank_sha256,
                        item.geometry.source_design_id,
                    ),
                )
            )
            != self.results
        ):
            raise ValueError("sensitivity results must use canonical probe order")
        if len({item.geometry.source_design_id for item in self.results}) != PROBE_COUNT:
            raise ValueError("sensitivity probe identities must be distinct")
        if len({item.geometry.dataset_id for item in self.results}) != 1:
            raise ValueError("sensitivity probes must share one dataset identity")
        if any(item.model != self.model for item in self.results):
            raise ValueError("sensitivity result model identities differ")
        if self.agreement_count != sum(item.agrees for item in self.results):
            raise ValueError("sensitivity agreement count does not match per-case evidence")
        return self


def evaluate_sensitivity_probes(
    probes: Sequence[ProbeGeometry],
    predictor: RotationSensitivityPredictor,
) -> SensitivityEvaluation:
    """Evaluate exactly ten canonical sensitivity probes with one model."""

    materialized = tuple(probes)
    if len(materialized) != PROBE_COUNT or any(
        not isinstance(probe, ProbeGeometry) or probe.selection != "sensitivity"
        for probe in materialized
    ):
        raise ArtifactIntegrityError("VAL-7 SENSITIVITY: exactly ten selected probes are required")
    ordered = tuple(
        sorted(materialized, key=lambda item: (item.rank_sha256, item.source_design_id))
    )
    results = tuple(evaluate_rotation_sensitivity(probe, predictor) for probe in ordered)
    return SensitivityEvaluation(
        model=predictor.model,
        results=results,
        agreement_count=sum(item.agrees for item in results),
    )


__all__ = [
    "CENTRAL_DIFFERENCE_H_DEG",
    "SENSITIVITY_MAGNITUDE_TOLERANCE",
    "AutogradCdResult",
    "RotationSensitivityPredictor",
    "SensitivityEvaluation",
    "SensitivityProbeResult",
    "evaluate_rotation_sensitivity",
    "evaluate_sensitivity_probes",
]
