from __future__ import annotations

import io
import json
import os
import stat
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError
from safetensors.numpy import save as save_tensors

from soufflerie.artifacts import (
    ReaderLimits,
    resolve_artifact_key,
    safe_read_bytes,
    safe_read_json,
    safe_read_npz,
    safe_read_parquet,
    safe_read_tensors,
)
from soufflerie.errors import ArtifactIntegrityError, SchemaVersionError
from soufflerie.schemas import ArrayDescriptor, VersionedModel, sha256_bytes


class Manifest(VersionedModel):
    name: str
    count: int


def _array_descriptor(
    *, dtype: str = "float32", shape: tuple[int, ...] = (2, 2)
) -> ArrayDescriptor:
    return ArrayDescriptor(dtype=dtype, shape=shape, unit="dimensionless")  # type: ignore[arg-type]


def test_root_keys_reject_traversal_urls_alternate_separators_and_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "nested" / "value.bin").write_bytes(b"safe")
    assert resolve_artifact_key(root, "nested/value.bin") == root / "nested" / "value.bin"
    assert safe_read_bytes(root, "nested/value.bin", max_bytes=4) == b"safe"

    for key in (
        "../outside.bin",
        "/etc/passwd",
        "https://example.test/a",
        "nested\\value.bin",
        "nested/../value.bin",
        "nested//value.bin",
        "./nested/value.bin",
    ):
        with pytest.raises(ArtifactIntegrityError, match="artifact key"):
            resolve_artifact_key(root, key)

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"unsafe")
    os.symlink(outside, root / "link.bin")
    with pytest.raises(ArtifactIntegrityError, match="symbolic link"):
        safe_read_bytes(root, "link.bin", max_bytes=10)


def test_bounded_byte_reader_checks_size_digest_and_regular_file(tmp_path: Path) -> None:
    content = b"verified"
    (tmp_path / "artifact.bin").write_bytes(content)
    assert (
        safe_read_bytes(
            tmp_path,
            "artifact.bin",
            max_bytes=len(content),
            expected_sha256=sha256_bytes(content),
        )
        == content
    )
    with pytest.raises(ArtifactIntegrityError, match="byte limit"):
        safe_read_bytes(tmp_path, "artifact.bin", max_bytes=len(content) - 1)
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        safe_read_bytes(
            tmp_path,
            "artifact.bin",
            max_bytes=len(content),
            expected_sha256="0" * 64,
        )
    (tmp_path / "directory").mkdir()
    with pytest.raises(ArtifactIntegrityError, match="regular file"):
        safe_read_bytes(tmp_path, "directory", max_bytes=10)


def test_json_reader_requires_bounded_finite_unique_schema_valid_records(tmp_path: Path) -> None:
    valid = b'{"schema_version":1,"name":"run","count":2}'
    (tmp_path / "manifest.json").write_bytes(valid)
    manifest = safe_read_json(
        tmp_path,
        "manifest.json",
        model=Manifest,
        expected_sha256=sha256_bytes(valid),
    )
    assert manifest == Manifest(name="run", count=2)

    hostile = {
        "duplicate.json": b'{"schema_version":1,"name":"first","name":"second","count":2}',
        "nan.json": b'{"schema_version":1,"name":"run","count":NaN}',
        "missing-schema.json": b'{"name":"run","count":2}',
        "bad-type.json": b'{"schema_version":1,"name":"run","count":"2"}',
    }
    for name, content in hostile.items():
        (tmp_path / name).write_bytes(content)
    with pytest.raises(ArtifactIntegrityError, match=r"duplicate|malformed"):
        safe_read_json(tmp_path, "duplicate.json", model=Manifest)
    with pytest.raises(ArtifactIntegrityError, match=r"malformed|non-finite"):
        safe_read_json(tmp_path, "nan.json", model=Manifest)
    with pytest.raises(ArtifactIntegrityError, match="schema_version"):
        safe_read_json(tmp_path, "missing-schema.json", model=Manifest)
    with pytest.raises(ArtifactIntegrityError, match="does not match"):
        safe_read_json(tmp_path, "bad-type.json", model=Manifest)


def test_json_reader_rejects_wrong_version_depth_keys_and_size(tmp_path: Path) -> None:
    documents = {
        "wrong-version.json": {"schema_version": 2, "name": "run", "count": 2},
        "deep.json": {"schema_version": 1, "name": "run", "count": 2, "x": [[[1]]]},
        "keys.json": {"schema_version": 1, "name": "run", "count": 2},
    }
    for name, value in documents.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SchemaVersionError):
        safe_read_json(tmp_path, "wrong-version.json", model=Manifest)
    with pytest.raises(ArtifactIntegrityError, match="maximum depth"):
        safe_read_json(
            tmp_path,
            "deep.json",
            model=Manifest,
            limits=ReaderLimits(max_json_depth=3),
        )
    with pytest.raises(ArtifactIntegrityError, match="key count"):
        safe_read_json(
            tmp_path,
            "keys.json",
            model=Manifest,
            limits=ReaderLimits(max_json_keys=2),
        )
    with pytest.raises(ArtifactIntegrityError, match="byte limit"):
        safe_read_json(
            tmp_path,
            "keys.json",
            model=Manifest,
            limits=ReaderLimits(max_json_bytes=8),
        )


def test_npz_reader_preflights_exact_descriptors_and_returns_read_only_arrays(
    tmp_path: Path,
) -> None:
    u = np.arange(4, dtype=np.float32).reshape(2, 2)
    mask = np.array([[True, False], [False, True]], dtype=np.bool_)
    np.savez_compressed(tmp_path / "fields.npz", u=u, mask=mask)
    content = (tmp_path / "fields.npz").read_bytes()
    arrays = safe_read_npz(
        tmp_path,
        "fields.npz",
        expected={
            "u": _array_descriptor(),
            "mask": _array_descriptor(dtype="bool"),
        },
        expected_sha256=sha256_bytes(content),
    )
    np.testing.assert_array_equal(arrays["u"], u)
    assert arrays["u"].flags.writeable is False
    with pytest.raises(ValueError):
        arrays["u"][0, 0] = 9.0

    with pytest.raises(ArtifactIntegrityError, match="dtype/shape descriptor"):
        safe_read_npz(
            tmp_path,
            "fields.npz",
            expected={
                "u": _array_descriptor(shape=(4,)),
                "mask": _array_descriptor(dtype="bool"),
            },
        )
    with pytest.raises(ArtifactIntegrityError, match="exactly match"):
        safe_read_npz(tmp_path, "fields.npz", expected={"u": _array_descriptor()})


def test_npz_reader_rejects_pickle_traversal_and_oversize_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np.savez(tmp_path / "pickle.npz", payload=np.array([object()], dtype=object))
    with pytest.raises(ArtifactIntegrityError, match="NO_PICKLE"):
        safe_read_npz(
            tmp_path,
            "pickle.npz",
            expected={"payload": _array_descriptor(dtype="int64", shape=(1,))},
        )

    member = io.BytesIO()
    np.save(member, np.ones((1,), dtype=np.float32), allow_pickle=False)
    with zipfile.ZipFile(tmp_path / "traversal.npz", "w") as archive:
        archive.writestr("../array.npy", member.getvalue())
    with pytest.raises(ArtifactIntegrityError, match="artifact key"):
        safe_read_npz(
            tmp_path,
            "traversal.npz",
            expected={"../array": _array_descriptor(shape=(1,))},
        )

    executable = zipfile.ZipInfo("array.npy")
    executable.external_attr = (stat.S_IFREG | 0o755) << 16
    with zipfile.ZipFile(tmp_path / "executable.npz", "w") as archive:
        archive.writestr(executable, member.getvalue())
    with pytest.raises(ArtifactIntegrityError, match="executable"):
        safe_read_npz(
            tmp_path,
            "executable.npz",
            expected={"array": _array_descriptor(shape=(1,))},
        )

    np.savez(tmp_path / "large.npz", u=np.ones((4,), dtype=np.float32))

    def forbidden_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("NumPy decoder ran before the member-size preflight")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(ArtifactIntegrityError, match="byte cap"):
        safe_read_npz(
            tmp_path,
            "large.npz",
            expected={"u": _array_descriptor(shape=(4,))},
            limits=ReaderLimits(max_member_bytes=8),
        )


def test_parquet_reader_projects_only_an_exact_bounded_primitive_schema(tmp_path: Path) -> None:
    table = pa.table(
        {
            "case_id": pa.array(["a", "b"], type=pa.string()),
            "score": pa.array([0.1, 0.2], type=pa.float64()),
        }
    )
    pq.write_table(table, tmp_path / "metrics.parquet")
    content = (tmp_path / "metrics.parquet").read_bytes()
    loaded = safe_read_parquet(
        tmp_path,
        "metrics.parquet",
        expected_columns={"case_id": "string", "score": "double"},
        expected_sha256=sha256_bytes(content),
    )
    assert loaded.equals(table)

    with pytest.raises(ArtifactIntegrityError, match="exactly match"):
        safe_read_parquet(
            tmp_path,
            "metrics.parquet",
            expected_columns={"score": "double"},
        )
    with pytest.raises(ArtifactIntegrityError, match="has type"):
        safe_read_parquet(
            tmp_path,
            "metrics.parquet",
            expected_columns={"case_id": "binary", "score": "double"},
        )
    with pytest.raises(ArtifactIntegrityError, match="row count"):
        safe_read_parquet(
            tmp_path,
            "metrics.parquet",
            expected_columns={"case_id": "string", "score": "double"},
            limits=ReaderLimits(max_parquet_rows=1),
        )


def test_safetensor_reader_preflights_descriptors_offsets_and_allocation_caps(
    tmp_path: Path,
) -> None:
    weights = np.arange(4, dtype=np.float32).reshape(2, 2)
    content = save_tensors({"weights": weights})
    (tmp_path / "model.safetensors").write_bytes(content)
    loaded = safe_read_tensors(
        tmp_path,
        "model.safetensors",
        expected={"weights": _array_descriptor()},
        expected_sha256=sha256_bytes(content),
    )
    np.testing.assert_array_equal(loaded["weights"], weights)
    assert loaded["weights"].flags.writeable is False

    with pytest.raises(ArtifactIntegrityError, match="dtype/shape descriptor"):
        safe_read_tensors(
            tmp_path,
            "model.safetensors",
            expected={"weights": _array_descriptor(shape=(4,))},
        )
    with pytest.raises(ArtifactIntegrityError, match="member byte cap"):
        safe_read_tensors(
            tmp_path,
            "model.safetensors",
            expected={"weights": _array_descriptor()},
            limits=ReaderLimits(max_member_bytes=8),
        )

    tampered = bytearray(content)
    tampered[-1] ^= 0x01
    (tmp_path / "tampered.safetensors").write_bytes(tampered)
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        safe_read_tensors(
            tmp_path,
            "tampered.safetensors",
            expected={"weights": _array_descriptor()},
            expected_sha256=sha256_bytes(content),
        )

    (tmp_path / "truncated.safetensors").write_bytes(b"\xff" * 8)
    with pytest.raises(ArtifactIntegrityError, match="header"):
        safe_read_tensors(
            tmp_path,
            "truncated.safetensors",
            expected={"weights": _array_descriptor()},
        )


def test_reader_contracts_are_strict_models() -> None:
    with pytest.raises(ValidationError):
        ReaderLimits(max_json_bytes="4096")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported expected Parquet types"):
        safe_read_parquet(Path("."), "metrics.parquet", expected_columns={"x": "list<int64>"})
