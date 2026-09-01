from __future__ import annotations

import math
import platform
from pathlib import Path

import pytest
from pydantic import ValidationError

from soufflerie.errors import SchemaVersionError
from soufflerie.solver.numerical_gates import (
    CpuSolverGateSummary,
    load_cpu_gate_summary,
    mass_gate_passes,
    run_periodic_regression,
)

REPORT = Path(__file__).parents[2] / "reports" / "solver" / "cpu-gates.json"


def test_twenty_thousand_step_mass_and_determinism_gates() -> None:
    result = run_periodic_regression()

    assert result.mass_drift_ratio < 0.001
    assert mass_gate_passes(result.mass_drift_ratio)
    assert result.final_arrays_bitwise_equal
    assert result.histories_bitwise_equal
    assert len(result.final_state_sha256) == 64
    assert len(result.history_sha256) == 64

    recorded = load_cpu_gate_summary(REPORT)
    current_platform = f"{platform.system().lower()}-{platform.machine().lower()}-cpu"
    if recorded.platform_class == current_platform:
        assert result.mass_drift_ratio == recorded.mass.mass_drift_ratio
        assert result.final_state_sha256 == recorded.determinism.final_state_sha256
        assert result.history_sha256 == recorded.determinism.history_sha256


def test_mass_gate_is_strict_at_the_declared_threshold() -> None:
    assert mass_gate_passes(math.nextafter(0.001, 0.0))
    assert not mass_gate_passes(0.001)
    assert not mass_gate_passes(math.nextafter(0.001, math.inf))
    assert not mass_gate_passes(math.nan)
    assert not mass_gate_passes(math.inf)


def test_checked_in_cpu_report_is_strict_coherent_and_green() -> None:
    report = load_cpu_gate_summary(REPORT)
    assert report.schema_version == 1
    assert report.generation_revision == "cpu-gates-v1"
    assert report.python_version.startswith("3.11.")
    assert report.numpy_version == "2.2.6"
    assert report.warp_version == "1.17.0"
    assert len(report.poiseuille) == 2
    assert all(item.passed for item in report.poiseuille)
    assert report.mass.steps == 20_000
    assert report.mass.passed
    assert report.determinism.repetitions == 2
    assert report.determinism.steps_per_repetition == 20_000
    assert report.determinism.passed
    assert report.overall_passed

    contradictory = report.model_dump(mode="python")
    contradictory["overall_passed"] = False
    with pytest.raises(ValidationError, match="overall_passed"):
        CpuSolverGateSummary.model_validate(contradictory)


def test_report_rejects_unknown_schema_version() -> None:
    payload = load_cpu_gate_summary(REPORT).model_dump(mode="python")
    payload["schema_version"] = 2
    with pytest.raises(SchemaVersionError, match="unsupported schema version"):
        CpuSolverGateSummary.model_validate(payload)
