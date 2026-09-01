from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from soufflerie.artifacts import ReaderLimits, safe_read_npz
from soufflerie.datagen.run_artifact import (
    OUTPUT_SHAPE,
    RUN_MEMBER_ORDER,
    CuratedRunFields,
    RunMetadata,
    curate_solver_result,
    encode_run_fields,
    run_member_descriptors,
)
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.geometry import ellipse_sdf
from soufflerie.schemas import CaseConfig, SolverResult, sha256_bytes


def test_curation_area_averages_recomputes_sdf_and_records_quantization(
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    fields, statistics = curate_solver_result(run_case, solver_result)

    assert fields.u_mean.shape == OUTPUT_SHAPE
    assert fields.u_mean.dtype == np.float16
    assert fields.obstacle_mask.dtype == np.uint8
    assert all(not array.flags.writeable for array in fields.members().values())
    expected_first_u = float(np.mean(solver_result.fields.u[:2, :2], dtype=np.float64))
    assert float(fields.u_mean[0, 0]) == pytest.approx(float(np.float16(expected_first_u)), abs=0.0)
    output_grid = run_case.grid.model_copy(update={"nx": 256, "ny": 128})
    expected_sdf = ellipse_sdf(run_case.shape, output_grid).astype(np.float16)
    np.testing.assert_array_equal(fields.sdf, expected_sdf)
    y_indices = np.rint(np.linspace(0, run_case.ny - 1, OUTPUT_SHAPE[0])).astype(np.int64)
    x_indices = np.rint(np.linspace(0, run_case.nx - 1, OUTPUT_SHAPE[1])).astype(np.int64)
    expected_mask = solver_result.fields.obstacle_mask[np.ix_(y_indices, x_indices)].astype(
        np.uint8
    )
    np.testing.assert_array_equal(fields.obstacle_mask, expected_mask)
    assert set(statistics) == {"u_mean", "v_mean", "rho_mean", "sdf"}
    assert all(
        statistic.max_abs_error >= statistic.mean_abs_error for statistic in statistics.values()
    )


def test_npz_encoding_is_byte_reproducible_fixed_member_and_no_pickle(
    run_case: CaseConfig,
    solver_result: SolverResult,
    tmp_path: Path,
) -> None:
    fields, _ = curate_solver_result(run_case, solver_result)
    first = encode_run_fields(fields)
    second = encode_run_fields(fields)

    assert first == second
    path = tmp_path / "fields.npz"
    path.write_bytes(first)
    with zipfile.ZipFile(io.BytesIO(first), "r") as archive:
        assert archive.namelist() == [f"{name}.npy" for name in RUN_MEMBER_ORDER]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    loaded = safe_read_npz(
        tmp_path,
        "fields.npz",
        expected=run_member_descriptors(fields.sample_count),
        expected_sha256=sha256_bytes(first),
    )
    np.testing.assert_array_equal(loaded["cd_history"], fields.cd_history)


def test_run_reader_contract_rejects_pickle_traversal_wrong_members_and_oversize(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    expected = run_member_descriptors(1)
    np.savez(
        tmp_path / "pickle.npz",
        **{  # type: ignore[arg-type]  # NumPy stubs reject a dynamically keyed fixture.
            name: np.array([object()]) for name in expected
        },
    )
    with pytest.raises(ArtifactIntegrityError, match=r"NO_PICKLE|dtype/shape"):
        safe_read_npz(tmp_path, "pickle.npz", expected=expected)

    member = io.BytesIO()
    np.save(member, np.zeros(OUTPUT_SHAPE, dtype=np.float16), allow_pickle=False)
    info = zipfile.ZipInfo("../u_mean.npy")
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with zipfile.ZipFile(tmp_path / "traversal.npz", "w") as archive:
        archive.writestr(info, member.getvalue())
    with pytest.raises(ArtifactIntegrityError, match=r"artifact key|exactly match"):
        safe_read_npz(tmp_path, "traversal.npz", expected=expected)

    np.savez(tmp_path / "wrong-members.npz", unexpected=np.ones((1,), dtype=np.float32))
    with pytest.raises(ArtifactIntegrityError, match="exactly match"):
        safe_read_npz(tmp_path, "wrong-members.npz", expected=expected)

    fields, _ = curate_solver_result(run_case, solver_result)
    expected_fields = run_member_descriptors(fields.sample_count)
    wrong_dtype = fields.members()
    wrong_dtype["u_mean"] = fields.u_mean.astype(np.float32)
    np.savez(
        tmp_path / "wrong-dtype.npz",
        **wrong_dtype,  # type: ignore[arg-type]  # Dynamic exact-member corruption fixture.
    )
    with pytest.raises(ArtifactIntegrityError, match="dtype/shape"):
        safe_read_npz(tmp_path, "wrong-dtype.npz", expected=expected_fields)

    wrong_shape = fields.members()
    wrong_shape["u_mean"] = fields.u_mean[:-1, :]
    np.savez(
        tmp_path / "wrong-shape.npz",
        **wrong_shape,  # type: ignore[arg-type]  # Dynamic exact-member corruption fixture.
    )
    with pytest.raises(ArtifactIntegrityError, match="dtype/shape"):
        safe_read_npz(tmp_path, "wrong-shape.npz", expected=expected_fields)

    valid_path = tmp_path / "valid.npz"
    valid_path.write_bytes(encode_run_fields(fields))
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        safe_read_npz(
            tmp_path,
            "valid.npz",
            expected=expected_fields,
            expected_sha256="0" * 64,
        )

    (tmp_path / "truncated.npz").write_bytes(b"PK\x03\x04")
    with pytest.raises(ArtifactIntegrityError, match=r"invalid|unsupported"):
        safe_read_npz(tmp_path, "truncated.npz", expected=expected)

    (tmp_path / "oversize.npz").write_bytes(b"x" * 1024)
    with pytest.raises(ArtifactIntegrityError, match="byte limit"):
        safe_read_npz(
            tmp_path,
            "oversize.npz",
            expected=expected,
            limits=ReaderLimits(max_file_bytes=128),
        )


def test_metadata_digest_detects_logical_tampering(
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    fields, statistics = curate_solver_result(run_case, solver_result)
    encoded = encode_run_fields(fields)
    metadata = RunMetadata.create(
        design_id="d" * 20,
        split="train",
        case=run_case,
        result=solver_result,
        fields=fields,
        quantization=statistics,
        fields_sha256=sha256_bytes(encoded),
    )

    with pytest.raises(ValidationError, match="artifact_digest"):
        RunMetadata.model_validate(metadata.model_dump(mode="python") | {"cd": 9.0})


def test_curated_fields_reject_wrong_shape_dtype_and_mask_values(
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    fields, _ = curate_solver_result(run_case, solver_result)
    invalid_mask = fields.obstacle_mask.copy()
    invalid_mask[0, 0] = np.uint8(2)
    invalid_mask.flags.writeable = False
    with pytest.raises(ArtifactIntegrityError, match="zero or one"):
        CuratedRunFields(
            u_mean=fields.u_mean,
            v_mean=fields.v_mean,
            rho_mean=fields.rho_mean,
            sdf=fields.sdf,
            obstacle_mask=invalid_mask,
            force_steps=fields.force_steps,
            cd_history=fields.cd_history,
            cl_history=fields.cl_history,
        )

    wrong_case = run_case.model_copy(update={"nx": 256})
    with pytest.raises(ArtifactIntegrityError, match=r"solver result|requires solver shape"):
        curate_solver_result(wrong_case, solver_result)
