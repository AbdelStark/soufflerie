from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from soufflerie.cli import rendered_cli_schema_documents
from soufflerie.config import rendered_config_schema_documents
from soufflerie.errors import ArtifactIntegrityError, SchemaVersionError
from soufflerie.observability import rendered_observability_schema_documents
from soufflerie.schemas import (
    ArrayDescriptor,
    CaseConfig,
    FlowFields,
    GridSpec,
    Provenance,
    ShapeParams,
    SolverDiagnostics,
    SolverResult,
    rendered_schema_documents,
    validate_array,
    validate_field_units,
    validate_parent_digests,
    validate_schema_version,
    validate_split_membership,
)


def _case() -> CaseConfig:
    return CaseConfig(
        shape=ShapeParams(aspect_ratio=0.75, rotation_deg=12.0, scale=1.0),
        reynolds=100.0,
        nx=64,
        ny=32,
        steps=500,
        warmup_steps=100,
        inlet_velocity_lu=0.05,
        seed=7,
    )


def _provenance(*, parents: dict[str, str] | None = None) -> Provenance:
    started = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    return Provenance(
        source_revision="a" * 40,
        source_dirty=False,
        python_version="3.11.14",
        lock_sha256="b" * 64,
        packages={"numpy": "2.2.6", "soufflerie": "0.1.0"},
        os="linux",
        architecture="x86_64",
        device_class="cpu",
        dtype_policy="solver-fp32-metrics-fp64",
        config_sha256="c" * 64,
        parent_sha256=parents or {},
        seeds=(7,),
        deterministic=True,
        started_at=started,
        completed_at=started + timedelta(seconds=2),
        gpu_seconds=0.0,
    )


def _fields() -> FlowFields:
    sdf = np.array([[-1.0, 1.0], [2.0, -0.5]], dtype=np.float32)
    return FlowFields(
        u=np.zeros((2, 2), dtype=np.float32),
        v=np.zeros((2, 2), dtype=np.float32),
        rho=np.ones((2, 2), dtype=np.float32),
        sdf=sdf,
        obstacle_mask=sdf <= 0,
    )


def test_shared_records_are_strict_frozen_and_versioned() -> None:
    case = _case()
    assert case.grid == GridSpec(nx=64, ny=32)
    assert case.grid.shape == (32, 64)

    with pytest.raises(ValidationError):
        ShapeParams.model_validate({"aspect_ratio": "0.75", "rotation_deg": 12.0, "scale": 1.0})
    with pytest.raises(ValidationError):
        ShapeParams(aspect_ratio=0.75, rotation_deg=12.0, scale=1.0, unknown=True)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        CaseConfig(**{**case.model_dump(), "warmup_steps": 500})
    with pytest.raises(SchemaVersionError):
        GridSpec.model_validate({"schema_version": 2, "nx": 64, "ny": 32})
    with pytest.raises(SchemaVersionError):
        validate_schema_version(2)


def test_flow_fields_enforce_dm1_dm2_dm3_and_dm8() -> None:
    fields = _fields()
    assert fields.shape == (2, 2)
    validate_field_units(fields.descriptors())

    wrong_shape = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ArtifactIntegrityError, match="DM-1 SHAPE"):
        FlowFields(
            u=fields.u,
            v=wrong_shape,
            rho=fields.rho,
            sdf=fields.sdf,
            obstacle_mask=fields.obstacle_mask,
        )

    with pytest.raises(ArtifactIntegrityError, match="DM-3 DTYPE"):
        FlowFields(
            u=fields.u.astype(np.float64),
            v=fields.v,
            rho=fields.rho,
            sdf=fields.sdf,
            obstacle_mask=fields.obstacle_mask,
        )

    non_contiguous = np.zeros((2, 4), dtype=np.float32)[:, ::2]
    with pytest.raises(ArtifactIntegrityError, match="C-contiguous"):
        validate_array(
            non_contiguous,
            name="field",
            dtype=np.dtype(np.float32),
            shape=(2, 2),
        )

    with pytest.raises(ArtifactIntegrityError, match="dimensions must be positive"):
        validate_array(
            np.empty((0, 2), dtype=np.float32),
            name="empty",
            dtype=np.dtype(np.float32),
        )

    rho = fields.rho.copy()
    rho[0, 0] = np.nan
    with pytest.raises(ArtifactIntegrityError, match="DM-2 FINITE"):
        FlowFields(
            u=fields.u,
            v=fields.v,
            rho=rho,
            sdf=fields.sdf,
            obstacle_mask=fields.obstacle_mask,
        )

    wrong_mask = fields.obstacle_mask.copy()
    wrong_mask[0, 0] = False
    with pytest.raises(ArtifactIntegrityError, match="obstacle_mask"):
        FlowFields(
            u=fields.u,
            v=fields.v,
            rho=fields.rho,
            sdf=fields.sdf,
            obstacle_mask=wrong_mask,
        )


def test_array_descriptor_enforces_dtype_shape_and_no_pickle() -> None:
    descriptor = ArrayDescriptor(dtype="float16", shape=(2, 2), unit="lattice_velocity")
    descriptor.validate_array(np.ones((2, 2), dtype=np.float16), name="u_mean")

    with pytest.raises(ArtifactIntegrityError, match="DM-5 NO_PICKLE"):
        validate_array(
            np.array([[object()]], dtype=object),
            name="payload",
            dtype=np.dtype(object),
        )


def test_solver_result_binds_histories_diagnostics_and_provenance() -> None:
    initial_mass = 100.0
    final_mass = 100.01
    diagnostics = SolverDiagnostics(
        steps_completed=500,
        sample_count=2,
        initial_mass=initial_mass,
        final_mass=final_mass,
        mass_drift_ratio=abs(final_mass - initial_mass) / initial_mass,
        min_rho=0.99,
        max_rho=1.01,
        max_speed_lu=0.08,
        converged=True,
        valid=True,
    )
    result = SolverResult(
        case_id=_case().case_id,
        fields=_fields(),
        cd=1.3,
        cl_mean=0.0,
        strouhal=0.17,
        force_steps=np.array([100, 110], dtype=np.int64),
        cd_history=np.array([1.2, 1.3], dtype=np.float32),
        cl_history=np.array([-0.1, 0.1], dtype=np.float32),
        diagnostics=diagnostics,
        provenance=_provenance(),
    )
    assert result.diagnostics.valid

    with pytest.raises(ArtifactIntegrityError, match="history length"):
        SolverResult(
            case_id=result.case_id,
            fields=result.fields,
            cd=result.cd,
            cl_mean=result.cl_mean,
            strouhal=result.strouhal,
            force_steps=np.array([100], dtype=np.int64),
            cd_history=np.array([1.2], dtype=np.float32),
            cl_history=np.array([0.1], dtype=np.float32),
            diagnostics=diagnostics,
            provenance=result.provenance,
        )


def test_dm6_split_and_dm7_parent_validators() -> None:
    first = "1" * 20
    second = "2" * 20
    assert validate_split_membership([(first, "train"), (second, "test")]) == {
        first: "train",
        second: "test",
    }
    with pytest.raises(ArtifactIntegrityError, match="DM-6 SPLIT"):
        validate_split_membership([(first, "train"), (first, "validation")])

    provenance = _provenance(parents={"dataset": "d" * 64})
    validate_parent_digests(provenance, required_parents=["dataset"])
    with pytest.raises(ArtifactIntegrityError, match="DM-7 PROVENANCE"):
        validate_parent_digests(provenance, required_parents=["dataset", "model"])

    wrong_units = _fields().descriptors()
    wrong_units["u"] = wrong_units["u"].model_copy(update={"unit": "dimensionless"})
    with pytest.raises(ArtifactIntegrityError, match="DM-8 UNITS"):
        validate_field_units(wrong_units)


def test_checked_in_schema_v1_documents_are_current() -> None:
    root = Path(__file__).parents[1] / "schemas" / "v1"
    expected = {
        **rendered_schema_documents(),
        **rendered_config_schema_documents(),
        **rendered_observability_schema_documents(),
        **rendered_cli_schema_documents(),
    }
    assert {path.name for path in root.glob("*.json")} == set(expected)
    for name, content in expected.items():
        assert (root / name).read_text(encoding="utf-8") == content
