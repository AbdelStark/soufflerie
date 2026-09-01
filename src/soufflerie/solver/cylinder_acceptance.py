"""Typed, digest-bound evidence for the remote Re=100 cylinder acceptance."""

from __future__ import annotations

import json
import math
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from soufflerie.schemas import (
    ArtifactRef,
    CaseConfig,
    Sha256,
    VersionedModel,
    canonical_sha256,
)
from soufflerie.solver.diagnostics import MIN_RESOLVED_LIFT_CYCLES

CYLINDER_CD_MIN = 1.1475
CYLINDER_CD_MAX = 1.5525
CYLINDER_ST_MIN = 0.15
CYLINDER_ST_MAX = 0.19
CYLINDER_MASS_DRIFT_MAX = 0.001
CYLINDER_GRID_SCALES = (0.75, 1.0, 1.25)
CYLINDER_REPORT_REVISION = "cylinder-re100-v1"
CYLINDER_CONFIG_PATH = "configs/cases/cylinder-re100.yaml"


class CylinderThresholds(VersionedModel):
    """Immutable numerical-oracle thresholds copied from the testing specification."""

    cd_min: float = CYLINDER_CD_MIN
    cd_max: float = CYLINDER_CD_MAX
    strouhal_min: float = CYLINDER_ST_MIN
    strouhal_max: float = CYLINDER_ST_MAX
    mass_drift_max_exclusive: float = CYLINDER_MASS_DRIFT_MAX
    resolved_lift_cycles_min: float = MIN_RESOLVED_LIFT_CYCLES

    @model_validator(mode="after")
    def _thresholds_are_frozen(self) -> Self:
        expected = (
            CYLINDER_CD_MIN,
            CYLINDER_CD_MAX,
            CYLINDER_ST_MIN,
            CYLINDER_ST_MAX,
            CYLINDER_MASS_DRIFT_MAX,
            MIN_RESOLVED_LIFT_CYCLES,
        )
        actual = (
            self.cd_min,
            self.cd_max,
            self.strouhal_min,
            self.strouhal_max,
            self.mass_drift_max_exclusive,
            self.resolved_lift_cycles_min,
        )
        if actual != expected:
            raise ValueError("cylinder thresholds are immutable specification values")
        return self


class CylinderRunEvidence(VersionedModel):
    """Small verified projection of one committed remote run artifact."""

    artifact: ArtifactRef
    case: CaseConfig
    config_sha256: Sha256
    fields_sha256: Sha256
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: Sha256
    device_class: str = Field(min_length=1)
    dtype_policy: str = Field(min_length=1)
    cd: float = Field(allow_inf_nan=False)
    cl_mean: float = Field(allow_inf_nan=False)
    strouhal: float = Field(gt=0.0, allow_inf_nan=False)
    mass_drift_ratio: float = Field(ge=0.0, allow_inf_nan=False)
    sample_count: int = Field(ge=4)
    sample_interval_steps: int = Field(ge=1)
    observed_duration_steps: int = Field(ge=1)
    resolved_lift_cycles: float = Field(ge=0.0, allow_inf_nan=False)
    lift_rms: float = Field(ge=0.0, allow_inf_nan=False)
    lift_peak_to_peak: float = Field(ge=0.0, allow_inf_nan=False)
    diagnostics_valid: Literal[True] = True
    diagnostics_converged: Literal[True] = True
    cd_reference_passed: bool
    strouhal_reference_passed: bool
    periodic_lift_passed: bool
    mass_passed: bool
    wall_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    evidence_sha256: Sha256

    @model_validator(mode="after")
    def _evidence_is_coherent(self) -> Self:
        if self.artifact.artifact_type != "run":
            raise ValueError("cylinder evidence requires a run artifact")
        expected_uri_prefix = f"runs/{self.case.case_id}/"
        if not self.artifact.uri.startswith(expected_uri_prefix):
            raise ValueError("run artifact path does not match the cylinder case")
        if self.config_sha256 != self.case.sha256:
            raise ValueError("config digest does not match the cylinder case")
        if self.observed_duration_steps != (self.sample_count - 1) * self.sample_interval_steps:
            raise ValueError("observed duration does not match the sampled history")
        diameter = self.case.ny / 20.0
        expected_cycles = (
            self.strouhal * self.case.inlet_velocity_lu * self.observed_duration_steps / diameter
        )
        if not math.isclose(self.resolved_lift_cycles, expected_cycles, rel_tol=1e-12):
            raise ValueError("resolved lift cycles do not match Strouhal and the run schedule")
        expected_gates = (
            CYLINDER_CD_MIN <= self.cd <= CYLINDER_CD_MAX,
            CYLINDER_ST_MIN <= self.strouhal <= CYLINDER_ST_MAX,
            self.resolved_lift_cycles >= MIN_RESOLVED_LIFT_CYCLES
            and self.lift_rms > 0.0
            and self.lift_peak_to_peak > float(np.finfo(np.float32).eps),
            self.mass_drift_ratio < CYLINDER_MASS_DRIFT_MAX,
        )
        actual_gates = (
            self.cd_reference_passed,
            self.strouhal_reference_passed,
            self.periodic_lift_passed,
            self.mass_passed,
        )
        if actual_gates != expected_gates:
            raise ValueError("stored cylinder gate results do not match the recorded metrics")
        payload = self.model_dump(mode="json", exclude={"evidence_sha256"})
        if self.evidence_sha256 != canonical_sha256(payload):
            raise ValueError("evidence_sha256 does not match the cylinder run evidence")
        return self

    @property
    def reference_passed(self) -> bool:
        return (
            self.cd_reference_passed
            and self.strouhal_reference_passed
            and self.periodic_lift_passed
            and self.mass_passed
        )

    @property
    def sensitivity_health_passed(self) -> bool:
        return self.periodic_lift_passed and self.mass_passed

    @classmethod
    def create(cls, **values: object) -> Self:
        payload: dict[str, object] = {"schema_version": 1, **values}
        return cls.model_validate({**payload, "evidence_sha256": canonical_sha256(payload)})


class CylinderGridSensitivity(VersionedModel):
    """Three-grid observations without inventing a new reference tolerance."""

    scales: tuple[float, float, float] = CYLINDER_GRID_SCALES
    coarse_to_canonical_cd_abs_change: float = Field(ge=0.0, allow_inf_nan=False)
    canonical_to_fine_cd_abs_change: float = Field(ge=0.0, allow_inf_nan=False)
    coarse_to_canonical_strouhal_abs_change: float = Field(ge=0.0, allow_inf_nan=False)
    canonical_to_fine_strouhal_abs_change: float = Field(ge=0.0, allow_inf_nan=False)
    cd_change_contracts: bool
    strouhal_change_contracts: bool
    normalized_domain_preserved: Literal[True] = True
    all_grids_periodic_and_mass_stable: bool
    passed: bool

    @model_validator(mode="before")
    @classmethod
    def _json_scale_list_to_tuple(cls, value: object) -> object:
        if isinstance(value, dict) and isinstance(value.get("scales"), list):
            return {**value, "scales": tuple(value["scales"])}
        return value

    @model_validator(mode="after")
    def _result_is_coherent(self) -> Self:
        if self.scales != CYLINDER_GRID_SCALES:
            raise ValueError("cylinder sensitivity grid scales are immutable")
        expected = self.normalized_domain_preserved and self.all_grids_periodic_and_mass_stable
        if self.passed != expected:
            raise ValueError("grid sensitivity result does not match its required checks")
        return self


class CylinderDeterminism(VersionedModel):
    """Independent-operation equality evidence for one canonical numerical case."""

    primary_operation_sha256: Sha256
    repeat_operation_sha256: Sha256
    artifact_digests_equal: bool
    field_archives_equal: bool
    passed: bool

    @model_validator(mode="after")
    def _result_is_coherent(self) -> Self:
        independent = self.primary_operation_sha256 != self.repeat_operation_sha256
        expected = independent and self.artifact_digests_equal and self.field_archives_equal
        if self.passed != expected:
            raise ValueError("determinism result does not match independent-operation evidence")
        return self


class CylinderAcceptanceReport(VersionedModel):
    """Self-verifying checked-in report for issue #14."""

    generation_revision: Literal["cylinder-re100-v1"] = "cylinder-re100-v1"
    config_path: Literal["configs/cases/cylinder-re100.yaml"] = "configs/cases/cylinder-re100.yaml"
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: Sha256
    config_sha256: Sha256
    device_class: str = Field(min_length=1)
    thresholds: CylinderThresholds
    coarse: CylinderRunEvidence
    canonical: CylinderRunEvidence
    fine: CylinderRunEvidence
    canonical_repeat: CylinderRunEvidence
    grid_sensitivity: CylinderGridSensitivity
    determinism: CylinderDeterminism
    total_gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    overall_passed: bool
    report_sha256: Sha256

    @model_validator(mode="after")
    def _report_is_coherent(self) -> Self:
        runs = (self.coarse, self.canonical, self.fine, self.canonical_repeat)
        identities = {(run.source_revision, run.lock_sha256, run.device_class) for run in runs}
        if identities != {(self.source_revision, self.lock_sha256, self.device_class)}:
            raise ValueError("all cylinder runs must share source, lock, and device identities")
        if self.config_sha256 != self.canonical.config_sha256:
            raise ValueError("report config digest does not match the canonical case")
        if self.canonical.case != self.canonical_repeat.case:
            raise ValueError("deterministic rerun must use the exact canonical case")
        if self.determinism.passed and self.canonical != self.canonical_repeat:
            raise ValueError("equal deterministic artifacts must yield identical run evidence")
        baseline = self.canonical.case
        for run in (self.coarse, self.fine):
            comparable = run.case.model_dump(mode="python", exclude={"nx", "ny"})
            expected = baseline.model_dump(mode="python", exclude={"nx", "ny"})
            if comparable != expected:
                raise ValueError("sensitivity grids may change only nx and ny")
        normalized = (
            self.coarse.case.nx * 4 == baseline.nx * 3
            and self.coarse.case.ny * 4 == baseline.ny * 3
            and self.fine.case.nx * 4 == baseline.nx * 5
            and self.fine.case.ny * 4 == baseline.ny * 5
        )
        if normalized != self.grid_sensitivity.normalized_domain_preserved:
            raise ValueError("sensitivity grid scales do not preserve the normalized domain")
        expected_changes = (
            abs(self.coarse.cd - self.canonical.cd),
            abs(self.canonical.cd - self.fine.cd),
            abs(self.coarse.strouhal - self.canonical.strouhal),
            abs(self.canonical.strouhal - self.fine.strouhal),
        )
        actual_changes = (
            self.grid_sensitivity.coarse_to_canonical_cd_abs_change,
            self.grid_sensitivity.canonical_to_fine_cd_abs_change,
            self.grid_sensitivity.coarse_to_canonical_strouhal_abs_change,
            self.grid_sensitivity.canonical_to_fine_strouhal_abs_change,
        )
        if any(
            not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)
            for actual, expected in zip(actual_changes, expected_changes, strict=True)
        ):
            raise ValueError("grid-sensitivity deltas do not match run metrics")
        expected_health = all(
            run.sensitivity_health_passed for run in (self.coarse, self.canonical, self.fine)
        )
        if self.grid_sensitivity.all_grids_periodic_and_mass_stable != expected_health:
            raise ValueError("grid-sensitivity health result does not match run metrics")
        expected_determinism = (
            self.canonical.artifact.sha256 == self.canonical_repeat.artifact.sha256,
            self.canonical.fields_sha256 == self.canonical_repeat.fields_sha256,
        )
        if (
            self.determinism.artifact_digests_equal,
            self.determinism.field_archives_equal,
        ) != expected_determinism:
            raise ValueError("determinism comparisons do not match the canonical runs")
        expected_gpu_seconds = sum(run.gpu_seconds for run in runs)
        if not math.isclose(self.total_gpu_seconds, expected_gpu_seconds, rel_tol=1e-12):
            raise ValueError("total GPU seconds do not match the four run records")
        expected_pass = (
            self.canonical.reference_passed
            and self.grid_sensitivity.passed
            and self.determinism.passed
        )
        if self.overall_passed != expected_pass:
            raise ValueError("overall_passed does not match the required cylinder gates")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != canonical_sha256(payload):
            raise ValueError("report_sha256 does not match the cylinder report")
        return self

    @classmethod
    def create(
        cls,
        *,
        coarse: CylinderRunEvidence,
        canonical: CylinderRunEvidence,
        fine: CylinderRunEvidence,
        canonical_repeat: CylinderRunEvidence,
        primary_operation_sha256: str,
        repeat_operation_sha256: str,
    ) -> Self:
        cd_first = abs(coarse.cd - canonical.cd)
        cd_second = abs(canonical.cd - fine.cd)
        st_first = abs(coarse.strouhal - canonical.strouhal)
        st_second = abs(canonical.strouhal - fine.strouhal)
        sensitivity = CylinderGridSensitivity(
            coarse_to_canonical_cd_abs_change=cd_first,
            canonical_to_fine_cd_abs_change=cd_second,
            coarse_to_canonical_strouhal_abs_change=st_first,
            canonical_to_fine_strouhal_abs_change=st_second,
            cd_change_contracts=cd_second <= cd_first,
            strouhal_change_contracts=st_second <= st_first,
            all_grids_periodic_and_mass_stable=all(
                run.sensitivity_health_passed for run in (coarse, canonical, fine)
            ),
            passed=all(run.sensitivity_health_passed for run in (coarse, canonical, fine)),
        )
        determinism = CylinderDeterminism(
            primary_operation_sha256=primary_operation_sha256,
            repeat_operation_sha256=repeat_operation_sha256,
            artifact_digests_equal=canonical.artifact.sha256 == canonical_repeat.artifact.sha256,
            field_archives_equal=canonical.fields_sha256 == canonical_repeat.fields_sha256,
            passed=(
                primary_operation_sha256 != repeat_operation_sha256
                and canonical.artifact.sha256 == canonical_repeat.artifact.sha256
                and canonical.fields_sha256 == canonical_repeat.fields_sha256
            ),
        )
        values: dict[str, object] = {
            "schema_version": 1,
            "generation_revision": CYLINDER_REPORT_REVISION,
            "config_path": CYLINDER_CONFIG_PATH,
            "source_revision": canonical.source_revision,
            "lock_sha256": canonical.lock_sha256,
            "config_sha256": canonical.config_sha256,
            "device_class": canonical.device_class,
            "thresholds": CylinderThresholds(),
            "coarse": coarse,
            "canonical": canonical,
            "fine": fine,
            "canonical_repeat": canonical_repeat,
            "grid_sensitivity": sensitivity,
            "determinism": determinism,
            "total_gpu_seconds": sum(
                run.gpu_seconds for run in (coarse, canonical, fine, canonical_repeat)
            ),
            "overall_passed": (
                canonical.reference_passed and sensitivity.passed and determinism.passed
            ),
        }
        return cls.model_validate({**values, "report_sha256": canonical_sha256(values)})


def render_cylinder_report(report: CylinderAcceptanceReport) -> str:
    """Render one concise human-readable companion from the typed report."""

    rows = []
    for name, run in (
        ("coarse", report.coarse),
        ("canonical", report.canonical),
        ("fine", report.fine),
        ("canonical repeat", report.canonical_repeat),
    ):
        rows.append(
            f"| {name} | {run.case.nx} x {run.case.ny} | {run.cd:.6f} | "
            f"{run.strouhal:.6f} | {run.mass_drift_ratio:.3e} | "
            f"{run.resolved_lift_cycles:.2f} | {run.gpu_seconds:.2f} | "
            f"`{run.artifact.sha256}` |"
        )
    status = "PASS" if report.overall_passed else "FAIL"
    return "\n".join(
        [
            "# Cylinder Re=100 acceptance",
            "",
            f"Overall: **{status}**",
            "",
            "| Run | Grid (nx x ny) | Cd | St | Mass drift | Lift cycles | GPU s | "
            "Artifact SHA-256 |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            f"- Source revision: `{report.source_revision}`",
            f"- Lock SHA-256: `{report.lock_sha256}`",
            f"- Canonical config SHA-256: `{report.config_sha256}`",
            f"- Device class: `{report.device_class}`",
            f"- Deterministic rerun: `{report.determinism.passed}`",
            f"- Normalized three-grid study: `{report.grid_sensitivity.passed}`",
            f"- Cd adjacent changes contract: `{report.grid_sensitivity.cd_change_contracts}`",
            "- Strouhal adjacent changes contract: "
            f"`{report.grid_sensitivity.strouhal_change_contracts}`",
            f"- Report SHA-256: `{report.report_sha256}`",
            "",
            "The immutable reference intervals were not relaxed. Grid contraction booleans are",
            "observations, not extra pass criteria; the sensitivity gate requires preserved "
            "normalized",
            "geometry plus periodic, mass-stable runs on all three grids.",
            "",
        ]
    )


def rendered_cylinder_schema_documents() -> dict[str, str]:
    """Render the durable cylinder report JSON Schema."""

    document = CylinderAcceptanceReport.model_json_schema()
    document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    document["$id"] = (
        "https://github.com/AbdelStark/soufflerie/schemas/v1/cylinder-acceptance-report.json"
    )
    return {
        "cylinder-acceptance-report.json": json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    }


__all__ = [
    "CYLINDER_CD_MAX",
    "CYLINDER_CD_MIN",
    "CYLINDER_CONFIG_PATH",
    "CYLINDER_GRID_SCALES",
    "CYLINDER_MASS_DRIFT_MAX",
    "CYLINDER_REPORT_REVISION",
    "CYLINDER_ST_MAX",
    "CYLINDER_ST_MIN",
    "CylinderAcceptanceReport",
    "CylinderDeterminism",
    "CylinderGridSensitivity",
    "CylinderRunEvidence",
    "CylinderThresholds",
    "render_cylinder_report",
    "rendered_cylinder_schema_documents",
]
