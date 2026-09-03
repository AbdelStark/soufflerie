"""Immutable RFC-0008 release gates and validation report identity."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, TypeAlias, cast

from pydantic import Field, StringConstraints, model_validator

from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import (
    ContentId,
    Provenance,
    Sha256,
    StrictFrozenModel,
    VersionedModel,
    canonical_sha256,
)
from soufflerie.validation.metrics import MetricObservation, MetricSummary
from soufflerie.validation.ood import OodEvaluation
from soufflerie.validation.plot_data import ValidationPlotData
from soufflerie.validation.sensitivity import SensitivityEvaluation

GateName: TypeAlias = Literal[
    "field_error",
    "cd_head_error",
    "head_field_consistency",
    "divergence",
    "obstacle_compliance",
    "mean_baseline_field",
    "nearest_baseline_field",
    "mean_baseline_cd",
    "nearest_baseline_cd",
    "ood_variance_increase",
    "sensitivity_sign",
    "evidence_integrity",
]
GateOperator: TypeAlias = Literal["lt", "le", "gt", "ge", "eq"]
GateStatus: TypeAlias = Literal["green", "red"]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
GateScalar: TypeAlias = FiniteFloat | int | bool
NonEmptyEvidence = Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class GateDefinition(StrictFrozenModel):
    """One immutable required gate and its fixed or comparison threshold source."""

    name: GateName
    metric: NonEmptyEvidence
    required: Literal[True] = True
    operator: GateOperator
    threshold: GateScalar | None = None
    threshold_metric: NonEmptyEvidence | None = None
    units: NonEmptyEvidence

    @model_validator(mode="after")
    def _threshold_source_is_unambiguous(self) -> Self:
        if (self.threshold is None) == (self.threshold_metric is None):
            raise ValueError("gate definition requires exactly one threshold source")
        return self


REQUIRED_GATE_DEFINITIONS: tuple[GateDefinition, ...] = (
    GateDefinition(
        name="field_error",
        metric="median_velocity_rel_l2",
        operator="lt",
        threshold=0.08,
        units="ratio",
    ),
    GateDefinition(
        name="cd_head_error",
        metric="median_cd_head_pct",
        operator="lt",
        threshold=5.0,
        units="percent",
    ),
    GateDefinition(
        name="head_field_consistency",
        metric="fraction_head_field_gap_le_10_pct",
        operator="ge",
        threshold=0.95,
        units="fraction",
    ),
    GateDefinition(
        name="divergence",
        metric="median_prediction_divergence_over_solver",
        operator="lt",
        threshold=3.0,
        units="ratio",
    ),
    GateDefinition(
        name="obstacle_compliance",
        metric="p95_obstacle_ratio",
        operator="lt",
        threshold=0.01,
        units="ratio",
    ),
    GateDefinition(
        name="mean_baseline_field",
        metric="selected_fno_median_velocity_rel_l2",
        operator="lt",
        threshold_metric="mean_field_baseline_median_velocity_rel_l2",
        units="ratio",
    ),
    GateDefinition(
        name="nearest_baseline_field",
        metric="selected_fno_median_velocity_rel_l2",
        operator="lt",
        threshold_metric="nearest_baseline_median_velocity_rel_l2",
        units="ratio",
    ),
    GateDefinition(
        name="mean_baseline_cd",
        metric="selected_fno_median_cd_head_pct",
        operator="lt",
        threshold_metric="mean_field_baseline_median_cd_pct",
        units="percent",
    ),
    GateDefinition(
        name="nearest_baseline_cd",
        metric="selected_fno_median_cd_head_pct",
        operator="lt",
        threshold_metric="nearest_baseline_median_cd_pct",
        units="percent",
    ),
    GateDefinition(
        name="ood_variance_increase",
        metric="median_ood_variance_over_id_boundary",
        operator="ge",
        threshold=1.5,
        units="ratio",
    ),
    GateDefinition(
        name="sensitivity_sign",
        metric="agreed_sensitivity_signs",
        operator="ge",
        threshold=8,
        units="count_of_10",
    ),
    GateDefinition(
        name="evidence_integrity",
        metric="all_lineage_split_and_count_checks",
        operator="eq",
        threshold=True,
        units="boolean",
    ),
)
GATE_DEFINITIONS_BY_NAME: Mapping[GateName, GateDefinition] = MappingProxyType(
    {definition.name: definition for definition in REQUIRED_GATE_DEFINITIONS}
)


def _comparison(operator: GateOperator, value: GateScalar, threshold: GateScalar) -> bool:
    value_is_bool = isinstance(value, bool)
    threshold_is_bool = isinstance(threshold, bool)
    if value_is_bool or threshold_is_bool:
        return operator == "eq" and value_is_bool and threshold_is_bool and value is threshold
    left = float(value)
    right = float(threshold)
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return {
        "lt": left < right,
        "le": left <= right,
        "gt": left > right,
        "ge": left >= right,
        "eq": left == right,
    }[operator]


class GateResult(StrictFrozenModel):
    """One self-verifying evaluated gate; invalid numeric evidence uses false/red."""

    name: GateName
    required: bool
    status: GateStatus
    value: GateScalar
    operator: GateOperator
    threshold: GateScalar
    units: NonEmptyEvidence
    evidence: tuple[NonEmptyEvidence, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_evidence(cls, value: object) -> object:
        if isinstance(value, Mapping) and isinstance(value.get("evidence"), list):
            return {**value, "evidence": tuple(value["evidence"])}
        return value

    @model_validator(mode="after")
    def _status_matches_comparison(self) -> Self:
        expected: GateStatus = (
            "green" if _comparison(self.operator, self.value, self.threshold) else "red"
        )
        if self.status != expected:
            raise ValueError("gate status does not match its exact comparison")
        return self


class GateEvidence(StrictFrozenModel):
    """Typed inputs to one gate, including explicit invalid-metric evidence."""

    name: GateName
    value: GateScalar | None
    comparison_threshold: GateScalar | None = None
    evidence: tuple[NonEmptyEvidence, ...] = Field(min_length=1, max_length=64)
    failure: NonEmptyEvidence | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_evidence(cls, value: object) -> object:
        if isinstance(value, Mapping) and isinstance(value.get("evidence"), list):
            return {**value, "evidence": tuple(value["evidence"])}
        return value

    @model_validator(mode="after")
    def _failure_is_explicit(self) -> Self:
        if self.failure is None and self.value is None:
            raise ValueError("valid gate evidence requires a value")
        if self.failure is not None and self.value is not None:
            raise ValueError("failed gate evidence must not carry a numeric value")
        return self


def evaluate_gate(definition: GateDefinition, evidence: GateEvidence) -> GateResult:
    """Apply one exact comparator; invalid inputs always become explicit red evidence."""

    if not isinstance(definition, GateDefinition) or not isinstance(evidence, GateEvidence):
        raise TypeError("definition and evidence must use validation gate models")
    if definition.name != evidence.name:
        raise ArtifactIntegrityError("VAL-4 GATE: definition and evidence names differ")
    threshold: GateScalar
    if definition.threshold is not None:
        if evidence.comparison_threshold is not None:
            raise ArtifactIntegrityError("VAL-4 GATE: fixed gate received a comparison threshold")
        threshold = definition.threshold
    else:
        if evidence.failure is None and evidence.comparison_threshold is None:
            raise ArtifactIntegrityError("VAL-4 GATE: comparison threshold is missing")
        threshold = (
            evidence.comparison_threshold if evidence.comparison_threshold is not None else False
        )
    value: GateScalar = evidence.value if evidence.value is not None else False
    status: GateStatus = "green" if _comparison(definition.operator, value, threshold) else "red"
    messages = evidence.evidence
    if evidence.failure is not None:
        messages = (*messages, evidence.failure)
    return GateResult(
        name=definition.name,
        required=definition.required,
        status=status,
        value=value,
        operator=definition.operator,
        threshold=threshold,
        units=definition.units,
        evidence=messages,
    )


def evaluate_required_gates(evidence: Sequence[GateEvidence]) -> tuple[GateResult, ...]:
    """Evaluate exactly the 12 required gates in frozen RFC order."""

    materialized = tuple(evidence)
    if any(not isinstance(item, GateEvidence) for item in materialized):
        raise TypeError("evidence must contain GateEvidence instances")
    by_name = {item.name: item for item in materialized}
    if len(by_name) != len(materialized) or set(by_name) != set(GATE_DEFINITIONS_BY_NAME):
        raise ArtifactIntegrityError("VAL-4 GATE: evidence must cover every required gate once")
    return tuple(
        evaluate_gate(definition, by_name[definition.name])
        for definition in REQUIRED_GATE_DEFINITIONS
    )


def head_field_consistency_gate_evidence(
    observations: Mapping[str, MetricObservation],
) -> GateEvidence:
    """Compute the exact fraction of complete cases whose gap is at most 10%."""

    if not observations:
        raise ArtifactIntegrityError("VAL-4 GATE: consistency observations are empty")
    ordered = tuple(sorted(observations.items()))
    if any(item.name != "head_field_gap_pct" for _, item in ordered):
        raise ArtifactIntegrityError("VAL-4 GATE: consistency observation metric differs")
    invalid = tuple(case_id for case_id, item in ordered if item.status == "invalid")
    base_evidence = (f"cases:{len(ordered)}", "condition:head_field_gap_pct<=10.0")
    if invalid:
        return GateEvidence(
            name="head_field_consistency",
            value=None,
            evidence=(*base_evidence, f"invalid_cases:{','.join(invalid)}"),
            failure="VAL-2 NO_VALID_CELLS: consistency distribution contains invalid cases",
        )
    passing = sum(cast(float, item.value) <= 10.0 for _, item in ordered)
    return GateEvidence(
        name="head_field_consistency",
        value=passing / len(ordered),
        evidence=base_evidence,
    )


def divergence_gate_evidence(
    prediction: MetricSummary,
    solver: MetricSummary,
) -> GateEvidence:
    """Compute prediction median divergence divided by the solver median."""

    if (
        not isinstance(prediction, MetricSummary)
        or not isinstance(solver, MetricSummary)
        or prediction.name != "prediction_div_mean_abs"
        or solver.name != "solver_div_mean_abs"
    ):
        raise ArtifactIntegrityError("VAL-4 GATE: divergence summary metric differs")
    evidence = (
        f"prediction_count:{prediction.count}",
        f"solver_count:{solver.count}",
    )
    if (
        prediction.status == "invalid"
        or solver.status == "invalid"
        or prediction.median is None
        or solver.median is None
        or solver.median <= 0.0
    ):
        return GateEvidence(
            name="divergence",
            value=None,
            evidence=evidence,
            failure="VAL-2 NO_VALID_CELLS: divergence median ratio is undefined",
        )
    return GateEvidence(
        name="divergence",
        value=prediction.median / solver.median,
        evidence=evidence,
    )


def ood_gate_evidence(evaluation: OodEvaluation) -> GateEvidence:
    """Translate complete OOD evidence into the fixed variance-increase gate."""

    if not isinstance(evaluation, OodEvaluation):
        raise TypeError("evaluation must be an OodEvaluation")
    evidence = (
        f"model_ids:{','.join(evaluation.model_ids)}",
        f"probe_count:{len(evaluation.results)}",
    )
    if evaluation.status == "invalid" or evaluation.variance_ratio is None:
        return GateEvidence(
            name="ood_variance_increase",
            value=None,
            evidence=evidence,
            failure=evaluation.failure or "VAL-6 ENSEMBLE: OOD evidence is invalid",
        )
    return GateEvidence(
        name="ood_variance_increase",
        value=evaluation.variance_ratio,
        evidence=evidence,
    )


def sensitivity_gate_evidence(evaluation: SensitivityEvaluation) -> GateEvidence:
    """Translate ten recorded sign comparisons into the fixed count gate."""

    if not isinstance(evaluation, SensitivityEvaluation):
        raise TypeError("evaluation must be a SensitivityEvaluation")
    return GateEvidence(
        name="sensitivity_sign",
        value=evaluation.agreement_count,
        evidence=(
            f"model_id:{evaluation.model.model_id}",
            f"probe_count:{len(evaluation.results)}",
        ),
    )


def overall_gate_status(gates: Sequence[GateResult]) -> GateStatus:
    """Return green iff every required gate is present and green."""

    materialized = tuple(gates)
    if any(not isinstance(gate, GateResult) for gate in materialized):
        raise TypeError("gates must contain GateResult instances")
    required_names = {definition.name for definition in REQUIRED_GATE_DEFINITIONS}
    observed_names = {gate.name for gate in materialized if gate.required}
    if len(observed_names) != len(materialized) or observed_names != required_names:
        raise ArtifactIntegrityError("VAL-4 GATE: results must contain every required gate once")
    return "green" if all(gate.status == "green" for gate in materialized) else "red"


class ValidationReport(VersionedModel):
    """Identity-bound machine report whose overall status cannot hide a red gate."""

    report_id: ContentId
    report_sha256: Sha256
    dataset_id: ContentId
    selected_model_id: ContentId
    ensemble_model_ids: tuple[ContentId, ContentId, ContentId]
    baseline_ids: tuple[ContentId, ContentId]
    metrics: dict[str, MetricSummary] = Field(min_length=1, max_length=128)
    gates: tuple[GateResult, ...]
    overall_status: GateStatus
    provenance: Provenance
    generator_version: Literal["validation-report-v1"] = "validation-report-v1"
    ood: OodEvaluation | None = None
    sensitivity: SensitivityEvaluation | None = None
    plot_data: ValidationPlotData | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_members(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("ensemble_model_ids", "baseline_ids", "gates"):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        provenance = normalized.get("provenance")
        if isinstance(provenance, Mapping):
            normalized_provenance = dict(provenance)
            seeds = normalized_provenance.get("seeds")
            if isinstance(seeds, list):
                normalized_provenance["seeds"] = tuple(seeds)
            for name in ("started_at", "completed_at"):
                timestamp = normalized_provenance.get(name)
                if isinstance(timestamp, str):
                    with suppress(ValueError):
                        normalized_provenance[name] = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
            normalized["provenance"] = normalized_provenance
        return normalized

    def logical_identity(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"report_id", "report_sha256"})

    @model_validator(mode="after")
    def _report_is_coherent(self) -> Self:
        if len(set(self.ensemble_model_ids)) != 3:
            raise ValueError("validation report requires three distinct ensemble models")
        if self.selected_model_id not in self.ensemble_model_ids:
            raise ValueError("selected model must be one of the ensemble models")
        if len(set(self.baseline_ids)) != 2 or set(self.baseline_ids) & set(
            self.ensemble_model_ids
        ):
            raise ValueError("baseline identities must be distinct from every ensemble model")
        expected_gate_order = tuple(definition.name for definition in REQUIRED_GATE_DEFINITIONS)
        observed_gate_order = tuple(gate.name for gate in self.gates)
        if (
            len(observed_gate_order) == len(expected_gate_order)
            and set(observed_gate_order) == set(expected_gate_order)
            and observed_gate_order != expected_gate_order
        ):
            raise ValueError("validation report gates must use the immutable RFC order")
        if self.ood is not None and self.ood.model_ids != tuple(sorted(self.ensemble_model_ids)):
            raise ValueError("OOD and report ensemble model identities differ")
        if self.ood is not None and any(
            item.geometry.dataset_id != self.dataset_id for item in self.ood.results
        ):
            raise ValueError("OOD and report dataset identities differ")
        if (
            self.sensitivity is not None
            and self.sensitivity.model.model_id != self.selected_model_id
        ):
            raise ValueError("sensitivity and selected report model identities differ")
        if self.sensitivity is not None and any(
            item.geometry.dataset_id != self.dataset_id for item in self.sensitivity.results
        ):
            raise ValueError("sensitivity and report dataset identities differ")
        if self.plot_data is not None:
            plot_ids = tuple(item.artifact_id for item in self.plot_data.baselines)
            expected_plot_ids = (self.selected_model_id, *self.baseline_ids)
            if plot_ids != expected_plot_ids:
                raise ValueError("plot model and baseline identities differ from the report")
            case_count = len(self.plot_data.cases)
            if any(summary.count != case_count for summary in self.metrics.values()):
                raise ValueError("plot cases must cover every summarized metric case")
        for name, summary in self.metrics.items():
            if name != summary.name and not name.endswith(f".{summary.name}"):
                raise ValueError("metric mapping keys must name the summarized metric")
        try:
            expected_status = overall_gate_status(self.gates)
        except ArtifactIntegrityError as error:
            raise ValueError(str(error)) from error
        if self.overall_status != expected_status:
            raise ValueError("overall status must equal the conjunction of required gates")
        digest = canonical_sha256(self.logical_identity())
        if self.report_sha256 != digest or self.report_id != digest[:20]:
            raise ValueError("validation report identity does not bind its complete evidence")
        return self

    @classmethod
    def create(cls, **values: object) -> ValidationReport:
        draft = cls.model_construct(
            **cast(
                Any,
                {"report_id": "0" * 20, "report_sha256": "0" * 64, **values},
            )
        )
        logical = draft.model_dump(mode="json", exclude={"report_id", "report_sha256"})
        digest = canonical_sha256(logical)
        return cls.model_validate({"report_id": digest[:20], "report_sha256": digest, **values})


__all__ = [
    "GATE_DEFINITIONS_BY_NAME",
    "REQUIRED_GATE_DEFINITIONS",
    "GateDefinition",
    "GateEvidence",
    "GateName",
    "GateResult",
    "ValidationReport",
    "divergence_gate_evidence",
    "evaluate_gate",
    "evaluate_required_gates",
    "head_field_consistency_gate_evidence",
    "ood_gate_evidence",
    "overall_gate_status",
    "sensitivity_gate_evidence",
]
