from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scripts.check_solver_report import main as check_report
from soufflerie.schemas import ArtifactRef, CaseConfig, ShapeParams
from soufflerie.solver.cylinder_acceptance import (
    CYLINDER_CD_MAX,
    CYLINDER_CD_MIN,
    CYLINDER_MASS_DRIFT_MAX,
    CYLINDER_ST_MAX,
    CYLINDER_ST_MIN,
    CylinderAcceptanceReport,
    CylinderRunEvidence,
    render_cylinder_report,
)

SOURCE = "1" * 40
LOCK = "2" * 64
PRIMARY_OPERATION = "3" * 64
REPEAT_OPERATION = "4" * 64


def _case(*, nx: int = 512, ny: int = 640) -> CaseConfig:
    return CaseConfig(
        shape=ShapeParams(aspect_ratio=1.0, rotation_deg=0.0, scale=1.0),
        reynolds=100.0,
        nx=nx,
        ny=ny,
        steps=60_000,
        warmup_steps=20_000,
        inlet_velocity_lu=0.05,
        seed=20260901,
    )


def _evidence(
    case: CaseConfig,
    *,
    digest: str,
    fields_digest: str | None = None,
    cd: float = 1.4,
    strouhal: float = 0.17,
    mass_drift: float = 0.0002,
) -> CylinderRunEvidence:
    observed_duration = 39_990
    cycles = strouhal * case.inlet_velocity_lu * observed_duration / (case.ny / 20.0)
    return CylinderRunEvidence.create(
        artifact=ArtifactRef(
            artifact_type="run",
            artifact_id=digest[:20],
            sha256=digest,
            size_bytes=100,
            uri=f"runs/{case.case_id}/{digest}",
        ),
        case=case,
        config_sha256=case.sha256,
        fields_sha256=fields_digest or digest,
        source_revision=SOURCE,
        lock_sha256=LOCK,
        device_class="L40S",
        dtype_policy="fp32-lbm-fp64-reduction",
        cd=cd,
        cl_mean=-0.01,
        strouhal=strouhal,
        mass_drift_ratio=mass_drift,
        sample_count=4_000,
        sample_interval_steps=10,
        observed_duration_steps=observed_duration,
        resolved_lift_cycles=cycles,
        lift_rms=0.2,
        lift_peak_to_peak=0.6,
        diagnostics_valid=True,
        diagnostics_converged=True,
        cd_reference_passed=CYLINDER_CD_MIN <= cd <= CYLINDER_CD_MAX,
        strouhal_reference_passed=CYLINDER_ST_MIN <= strouhal <= CYLINDER_ST_MAX,
        periodic_lift_passed=cycles >= 8.0,
        mass_passed=mass_drift < CYLINDER_MASS_DRIFT_MAX,
        wall_seconds=10.0,
        gpu_seconds=9.0,
    )


def _report(**canonical_updates: Any) -> CylinderAcceptanceReport:
    canonical_case = _case()
    digest = "b" * 64
    canonical = _evidence(canonical_case, digest=digest, **canonical_updates)
    return CylinderAcceptanceReport.create(
        coarse=_evidence(_case(nx=384, ny=480), digest="a" * 64, cd=1.35, strouhal=0.165),
        canonical=canonical,
        fine=_evidence(_case(nx=640, ny=800), digest="c" * 64, cd=1.42, strouhal=0.172),
        canonical_repeat=canonical,
        primary_operation_sha256=PRIMARY_OPERATION,
        repeat_operation_sha256=REPEAT_OPERATION,
    )


def test_green_report_binds_grids_determinism_and_rendered_evidence() -> None:
    report = _report()

    assert report.overall_passed
    assert report.grid_sensitivity.passed
    assert report.grid_sensitivity.cd_change_contracts
    assert report.grid_sensitivity.strouhal_change_contracts
    assert report.determinism.passed
    assert report.total_gpu_seconds == 36.0
    rendered = render_cylinder_report(report)
    assert "Overall: **PASS**" in rendered
    assert report.report_sha256 in rendered


@pytest.mark.parametrize("cd", [CYLINDER_CD_MIN, CYLINDER_CD_MAX])
def test_cd_reference_interval_is_closed_at_both_edges(cd: float) -> None:
    assert _report(cd=cd).canonical.cd_reference_passed


@pytest.mark.parametrize("strouhal", [CYLINDER_ST_MIN, CYLINDER_ST_MAX])
def test_strouhal_reference_interval_is_closed_at_both_edges(strouhal: float) -> None:
    assert _report(strouhal=strouhal).canonical.strouhal_reference_passed


@pytest.mark.parametrize(
    ("updates", "gate"),
    [
        ({"cd": CYLINDER_CD_MIN - 1e-6}, "cd_reference_passed"),
        ({"cd": CYLINDER_CD_MAX + 1e-6}, "cd_reference_passed"),
        ({"strouhal": CYLINDER_ST_MIN - 1e-6}, "strouhal_reference_passed"),
        ({"strouhal": CYLINDER_ST_MAX + 1e-6}, "strouhal_reference_passed"),
        ({"mass_drift": CYLINDER_MASS_DRIFT_MAX}, "mass_passed"),
    ],
)
def test_reference_values_outside_or_at_exclusive_limit_are_red(
    updates: dict[str, float],
    gate: str,
) -> None:
    report = _report(**updates)

    assert not getattr(report.canonical, gate)
    assert not report.overall_passed


def test_report_rejects_digest_tampering_and_nonindependent_repeat() -> None:
    report = _report()
    tampered = report.model_dump(mode="python")
    tampered["canonical"] = {
        **report.canonical.model_dump(mode="python"),
        "cd": 1.5,
    }
    with pytest.raises(ValidationError, match=r"evidence_sha256|gate results"):
        CylinderAcceptanceReport.model_validate(tampered)

    repeated_operation = CylinderAcceptanceReport.create(
        coarse=report.coarse,
        canonical=report.canonical,
        fine=report.fine,
        canonical_repeat=report.canonical_repeat,
        primary_operation_sha256=PRIMARY_OPERATION,
        repeat_operation_sha256=PRIMARY_OPERATION,
    )
    assert not repeated_operation.determinism.passed
    assert not repeated_operation.overall_passed


def test_checked_report_and_rendered_companion_pass_cli_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report()
    report_path = tmp_path / "cylinder-re100.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.with_suffix(".md").write_text(
        render_cylinder_report(report),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["check_solver_report.py", str(report_path)])

    assert check_report() == 0
    assert "cylinder_acceptance=PASS" in capsys.readouterr().out
