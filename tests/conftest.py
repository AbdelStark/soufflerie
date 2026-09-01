from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from soufflerie.geometry import ellipse_sdf, obstacle_mask
from soufflerie.schemas import (
    CaseConfig,
    FlowFields,
    Provenance,
    ShapeParams,
    SolverDiagnostics,
    SolverResult,
)


@pytest.fixture
def run_case() -> CaseConfig:
    return CaseConfig(
        shape=ShapeParams(aspect_ratio=0.75, rotation_deg=12.0, scale=1.0),
        reynolds=100.0,
        nx=512,
        ny=256,
        steps=20_000,
        warmup_steps=10_000,
        inlet_velocity_lu=0.05,
        seed=20260901,
    )


@pytest.fixture
def solver_result(run_case: CaseConfig) -> SolverResult:
    y, x = np.mgrid[: run_case.ny, : run_case.nx]
    u = np.ascontiguousarray(0.02 + 1e-5 * x + 2e-5 * y, dtype=np.float32)
    v = np.ascontiguousarray(-0.01 + 1e-5 * y, dtype=np.float32)
    rho = np.ascontiguousarray(1.0 + 1e-6 * x - 5e-7 * y, dtype=np.float32)
    sdf = ellipse_sdf(run_case.shape, run_case.grid)
    fields = FlowFields(
        u=u,
        v=v,
        rho=rho,
        sdf=sdf,
        obstacle_mask=obstacle_mask(sdf),
    )
    diagnostics = SolverDiagnostics(
        steps_completed=20_000,
        sample_count=4,
        initial_mass=100.0,
        final_mass=100.01,
        mass_drift_ratio=0.0001,
        min_rho=float(np.min(rho)),
        max_rho=float(np.max(rho)),
        max_speed_lu=0.03,
        converged=True,
        valid=True,
        messages=(),
    )
    started = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    provenance = Provenance(
        source_revision="a" * 40,
        source_dirty=False,
        python_version="3.11.14",
        lock_sha256="b" * 64,
        packages={"numpy": "2.2.6", "soufflerie": "0.1.0"},
        os="darwin",
        architecture="arm64",
        device_class="cpu-test",
        dtype_policy="solver-fp32-storage-fp16",
        config_sha256=run_case.sha256,
        parent_sha256={},
        seeds=(run_case.seed,),
        deterministic=True,
        started_at=started,
        completed_at=started + timedelta(seconds=2),
        gpu_seconds=0.0,
    )
    return SolverResult(
        case_id=run_case.case_id,
        fields=fields,
        cd=1.3,
        cl_mean=0.01,
        strouhal=0.17,
        force_steps=np.array([10_010, 10_020, 10_030, 10_040], dtype=np.int64),
        cd_history=np.array([1.2, 1.3, 1.4, 1.3], dtype=np.float32),
        cl_history=np.array([-0.2, 0.2, -0.2, 0.2], dtype=np.float32),
        diagnostics=diagnostics,
        provenance=provenance,
    )
