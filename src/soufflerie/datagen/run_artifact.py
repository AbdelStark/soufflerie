"""Deterministic run curation, safe codecs, and atomic local publication."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from errno import EEXIST, ENOTEMPTY
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from soufflerie.artifacts import (
    DEFAULT_READER_LIMITS,
    ReaderLimits,
    safe_read_bytes,
    safe_read_json,
    safe_read_npz,
)
from soufflerie.datagen._local_files import ensure_real_directory, fsync_directory, fsync_file
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.geometry import OUTPUT_GRID_NX, OUTPUT_GRID_NY, ellipse_sdf
from soufflerie.schemas import (
    ArrayDescriptor,
    ArtifactRef,
    CaseConfig,
    ContentId,
    JsonValue,
    Provenance,
    Sha256,
    SolverDiagnostics,
    SolverResult,
    Split,
    VersionedModel,
    canonical_sha256,
    sha256_bytes,
    validate_array,
)

Float16Array = npt.NDArray[np.float16]
Float32Array = npt.NDArray[np.float32]
Int64Array = npt.NDArray[np.int64]
UInt8Array = npt.NDArray[np.uint8]

OUTPUT_SHAPE = (OUTPUT_GRID_NY, OUTPUT_GRID_NX)
SOLVER_SHAPE = (2 * OUTPUT_GRID_NY, 2 * OUTPUT_GRID_NX)
RUN_MEMBER_ORDER = (
    "u_mean",
    "v_mean",
    "rho_mean",
    "sdf",
    "obstacle_mask",
    "force_steps",
    "cd_history",
    "cl_history",
)
RUN_ROOT_PREFIX = "runs"
RUN_FIELDS_NAME = "fields.npz"
RUN_METADATA_NAME = "metadata.json"
RUN_COMMIT_NAME = "COMMITTED"
RUN_MAX_FILE_BYTES = 16 * 1024 * 1024
_CONTENT_ID_PATTERN = re.compile(r"^[0-9a-f]{20}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class QuantizationStatistic(VersionedModel):
    """Error introduced by one explicit fp32-to-fp16 storage cast."""

    source_dtype: Literal["float32"] = "float32"
    stored_dtype: Literal["float16"] = "float16"
    max_abs_error: float = Field(ge=0.0, allow_inf_nan=False)
    mean_abs_error: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _maximum_bounds_mean(self) -> Self:
        if self.mean_abs_error > self.max_abs_error:
            raise ValueError("mean_abs_error must not exceed max_abs_error")
        return self


def run_member_descriptors(sample_count: int) -> dict[str, ArrayDescriptor]:
    """Return the exact RFC-0005 member contract for one history length."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ArtifactIntegrityError("RUN-1 MEMBERS: sample_count must be a positive integer")
    return {
        "u_mean": ArrayDescriptor(dtype="float16", shape=OUTPUT_SHAPE, unit="lattice_velocity"),
        "v_mean": ArrayDescriptor(dtype="float16", shape=OUTPUT_SHAPE, unit="lattice_velocity"),
        "rho_mean": ArrayDescriptor(dtype="float16", shape=OUTPUT_SHAPE, unit="lattice_density"),
        "sdf": ArrayDescriptor(dtype="float16", shape=OUTPUT_SHAPE, unit="lattice_distance"),
        "obstacle_mask": ArrayDescriptor(dtype="uint8", shape=OUTPUT_SHAPE, unit="dimensionless"),
        "force_steps": ArrayDescriptor(dtype="int64", shape=(sample_count,), unit="step"),
        "cd_history": ArrayDescriptor(dtype="float32", shape=(sample_count,), unit="dimensionless"),
        "cl_history": ArrayDescriptor(dtype="float32", shape=(sample_count,), unit="dimensionless"),
    }


@dataclass(frozen=True, slots=True)
class CuratedRunFields:
    """Validated read-only arrays stored in one run NPZ."""

    u_mean: Float16Array
    v_mean: Float16Array
    rho_mean: Float16Array
    sdf: Float16Array
    obstacle_mask: UInt8Array
    force_steps: Int64Array
    cd_history: Float32Array
    cl_history: Float32Array

    def __post_init__(self) -> None:
        descriptors = run_member_descriptors(int(self.force_steps.size))
        for name in RUN_MEMBER_ORDER:
            array = getattr(self, name)
            descriptors[name].validate_array(array, name=name)
            if array.flags.writeable:
                raise ArtifactIntegrityError(f"RUN-1 MEMBERS: {name} must be read-only")
        if np.any(self.obstacle_mask > np.uint8(1)):
            raise ArtifactIntegrityError("RUN-1 MEMBERS: obstacle_mask values must be zero or one")
        if np.any(self.force_steps < 0) or np.any(np.diff(self.force_steps) <= 0):
            raise ArtifactIntegrityError(
                "RUN-1 MEMBERS: force_steps must be nonnegative and strictly increasing"
            )

    @property
    def sample_count(self) -> int:
        return int(self.force_steps.size)

    def members(self) -> dict[str, npt.NDArray[np.generic]]:
        return {name: getattr(self, name) for name in RUN_MEMBER_ORDER}


def _readonly(array: npt.NDArray[Any], *, dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.flags.writeable = False
    return result


def _area_average(array: Float32Array) -> Float32Array:
    validate_array(array, name="mean_field", dtype=np.dtype(np.float32), shape=SOLVER_SHAPE)
    averaged = (
        array.astype(np.float64)
        .reshape(
            OUTPUT_GRID_NY,
            2,
            OUTPUT_GRID_NX,
            2,
        )
        .mean(axis=(1, 3), dtype=np.float64)
    )
    return cast(Float32Array, np.ascontiguousarray(averaged, dtype=np.float32))


def _nearest_center_mask(mask: npt.NDArray[np.bool_]) -> UInt8Array:
    validate_array(mask, name="obstacle_mask", dtype=np.dtype(np.bool_), shape=SOLVER_SHAPE)
    y_indices = np.rint(np.linspace(0, SOLVER_SHAPE[0] - 1, OUTPUT_GRID_NY)).astype(np.int64)
    x_indices = np.rint(np.linspace(0, SOLVER_SHAPE[1] - 1, OUTPUT_GRID_NX)).astype(np.int64)
    sampled = mask[np.ix_(y_indices, x_indices)]
    return cast(UInt8Array, np.ascontiguousarray(sampled, dtype=np.uint8))


def _quantize(array: Float32Array) -> tuple[Float16Array, QuantizationStatistic]:
    stored = np.ascontiguousarray(array, dtype=np.float16)
    if not np.isfinite(stored).all():
        raise ArtifactIntegrityError("RUN-2 QUANTIZATION: fp16 cast produced NaN or infinity")
    restored = stored.astype(np.float32)
    error = np.abs(restored.astype(np.float64) - array.astype(np.float64))
    result = stored
    result.flags.writeable = False
    return result, QuantizationStatistic(
        max_abs_error=float(np.max(error)),
        mean_abs_error=float(np.mean(error, dtype=np.float64)),
    )


def curate_solver_result(
    case: CaseConfig,
    result: SolverResult,
) -> tuple[CuratedRunFields, dict[str, QuantizationStatistic]]:
    """Downsample validated solver fields and cast only at the storage boundary."""

    if not isinstance(case, CaseConfig):
        raise TypeError("case must be a CaseConfig instance")
    if not isinstance(result, SolverResult):
        raise TypeError("result must be a SolverResult instance")
    if result.case_id != case.case_id:
        raise ArtifactIntegrityError("RUN-1 IDENTITY: solver result does not match the case")
    if result.fields.shape != SOLVER_SHAPE or (case.ny, case.nx) != SOLVER_SHAPE:
        raise ArtifactIntegrityError(
            f"RUN-1 SHAPE: v0.1 run curation requires solver shape {SOLVER_SHAPE}"
        )

    output_grid = case.grid.model_copy(update={"nx": OUTPUT_GRID_NX, "ny": OUTPUT_GRID_NY})
    continuous = {
        "u_mean": _area_average(result.fields.u),
        "v_mean": _area_average(result.fields.v),
        "rho_mean": _area_average(result.fields.rho),
        "sdf": ellipse_sdf(case.shape, output_grid).copy(),
    }
    stored: dict[str, Float16Array] = {}
    statistics: dict[str, QuantizationStatistic] = {}
    for name, array in continuous.items():
        stored[name], statistics[name] = _quantize(array)

    mask = _nearest_center_mask(result.fields.obstacle_mask)
    mask.flags.writeable = False
    return (
        CuratedRunFields(
            u_mean=stored["u_mean"],
            v_mean=stored["v_mean"],
            rho_mean=stored["rho_mean"],
            sdf=stored["sdf"],
            obstacle_mask=mask,
            force_steps=cast(Int64Array, _readonly(result.force_steps, dtype=np.dtype(np.int64))),
            cd_history=cast(Float32Array, _readonly(result.cd_history, dtype=np.dtype(np.float32))),
            cl_history=cast(Float32Array, _readonly(result.cl_history, dtype=np.dtype(np.float32))),
        ),
        statistics,
    )


def _identity_provenance(provenance: Provenance) -> dict[str, JsonValue]:
    value = provenance.model_dump(
        mode="json",
        exclude={"started_at", "completed_at", "gpu_seconds"},
    )
    return cast(dict[str, JsonValue], value)


def _run_logical_identity(
    *,
    case_id: str,
    design_id: str,
    split: Split,
    case: CaseConfig,
    cd: float,
    cl_mean: float,
    strouhal: float | None,
    diagnostics: SolverDiagnostics,
    field_members: Mapping[str, ArrayDescriptor],
    quantization: Mapping[str, QuantizationStatistic],
    provenance: Provenance,
    fields_sha256: str,
) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            "schema_version": 1,
            "case_id": case_id,
            "design_id": design_id,
            "split": split,
            "case": case.model_dump(mode="json"),
            "cd": cd,
            "cl_mean": cl_mean,
            "strouhal": strouhal,
            "diagnostics": diagnostics.model_dump(mode="json"),
            "field_members": {
                name: descriptor.model_dump(mode="json")
                for name, descriptor in sorted(field_members.items())
            },
            "quantization": {
                name: statistic.model_dump(mode="json")
                for name, statistic in sorted(quantization.items())
            },
            "provenance": _identity_provenance(provenance),
            "fields_sha256": fields_sha256,
        },
    )


class RunMetadata(VersionedModel):
    """Immutable logical metadata for one checksum-verified run artifact."""

    expected_member_names: ClassVar[frozenset[str]] = frozenset(RUN_MEMBER_ORDER)
    expected_quantized_names: ClassVar[frozenset[str]] = frozenset(
        {"u_mean", "v_mean", "rho_mean", "sdf"}
    )

    case_id: ContentId
    design_id: ContentId
    split: Split
    case: CaseConfig
    cd: float = Field(allow_inf_nan=False)
    cl_mean: float = Field(allow_inf_nan=False)
    strouhal: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    diagnostics: SolverDiagnostics
    field_members: dict[str, ArrayDescriptor]
    quantization: dict[str, QuantizationStatistic]
    provenance: Provenance
    fields_sha256: Sha256
    artifact_digest: Sha256

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_containers(cls, value: object) -> object:
        """Convert only JSON's lossless representations for strict nested records."""

        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        diagnostics = normalized.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            normalized_diagnostics = dict(diagnostics)
            messages = normalized_diagnostics.get("messages")
            if isinstance(messages, list):
                normalized_diagnostics["messages"] = tuple(messages)
            normalized["diagnostics"] = normalized_diagnostics
        field_members = normalized.get("field_members")
        if isinstance(field_members, Mapping):
            normalized_members: dict[object, object] = {}
            for name, descriptor in field_members.items():
                if isinstance(descriptor, Mapping):
                    normalized_descriptor = dict(descriptor)
                    shape = normalized_descriptor.get("shape")
                    if isinstance(shape, list):
                        normalized_descriptor["shape"] = tuple(shape)
                    normalized_members[name] = normalized_descriptor
                else:
                    normalized_members[name] = descriptor
            normalized["field_members"] = normalized_members
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

    def logical_identity(self) -> dict[str, JsonValue]:
        return _run_logical_identity(
            case_id=self.case_id,
            design_id=self.design_id,
            split=self.split,
            case=self.case,
            cd=self.cd,
            cl_mean=self.cl_mean,
            strouhal=self.strouhal,
            diagnostics=self.diagnostics,
            field_members=self.field_members,
            quantization=self.quantization,
            provenance=self.provenance,
            fields_sha256=self.fields_sha256,
        )

    @model_validator(mode="after")
    def _metadata_is_coherent(self) -> Self:
        if self.case_id != self.case.case_id:
            raise ValueError("case_id must match the canonical case identity")
        if not self.diagnostics.valid or not self.diagnostics.converged:
            raise ValueError("run metadata requires valid converged diagnostics")
        expected = run_member_descriptors(self.diagnostics.sample_count)
        if self.field_members != expected or set(self.field_members) != self.expected_member_names:
            raise ValueError("field_members do not exactly match the run artifact contract")
        if set(self.quantization) != self.expected_quantized_names:
            raise ValueError("quantization statistics do not exactly match continuous members")
        if self.provenance.config_sha256 != self.case.sha256:
            raise ValueError("provenance config digest must match the case")
        if self.artifact_digest != canonical_sha256(self.logical_identity()):
            raise ValueError("artifact_digest does not match the logical run content")
        return self

    @classmethod
    def create(
        cls,
        *,
        design_id: str,
        split: Split,
        case: CaseConfig,
        result: SolverResult,
        fields: CuratedRunFields,
        quantization: Mapping[str, QuantizationStatistic],
        fields_sha256: str,
    ) -> Self:
        field_members = run_member_descriptors(fields.sample_count)
        quantization_values = dict(quantization)
        artifact_digest = canonical_sha256(
            _run_logical_identity(
                case_id=result.case_id,
                design_id=design_id,
                split=split,
                case=case,
                cd=result.cd,
                cl_mean=result.cl_mean,
                strouhal=result.strouhal,
                diagnostics=result.diagnostics,
                field_members=field_members,
                quantization=quantization_values,
                provenance=result.provenance,
                fields_sha256=fields_sha256,
            )
        )
        values: dict[str, object] = {
            "schema_version": 1,
            "case_id": result.case_id,
            "design_id": design_id,
            "split": split,
            "case": case,
            "cd": result.cd,
            "cl_mean": result.cl_mean,
            "strouhal": result.strouhal,
            "diagnostics": result.diagnostics,
            "field_members": field_members,
            "quantization": quantization_values,
            "provenance": result.provenance,
            "fields_sha256": fields_sha256,
            "artifact_digest": artifact_digest,
        }
        return cls.model_validate(values)


def _npy_bytes(array: npt.NDArray[np.generic]) -> bytes:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def encode_run_fields(fields: CuratedRunFields) -> bytes:
    """Encode fixed members in a byte-reproducible uncompressed NPZ archive."""

    if not isinstance(fields, CuratedRunFields):
        raise TypeError("fields must be a CuratedRunFields instance")
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in fields.members().items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, _npy_bytes(array))
    return output.getvalue()


def _metadata_bytes(metadata: RunMetadata) -> bytes:
    payload = metadata.model_dump(mode="json")
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class RunArtifact:
    """One fully verified committed run and its small portable reference."""

    reference: ArtifactRef
    metadata: RunMetadata
    fields: CuratedRunFields
    metadata_sha256: str


class RunArtifactStore(Protocol):
    """Current local/remote multiplicity boundary for immutable run artifacts."""

    def publish_run(
        self,
        *,
        attempt_id: str,
        design_id: str,
        split: Split,
        case: CaseConfig,
        result: SolverResult,
    ) -> ArtifactRef: ...

    def open_run(self, reference: ArtifactRef) -> RunArtifact: ...

    def verify_run(self, *, case_id: str, run_digest: str) -> ArtifactRef: ...


class LocalRunArtifactStore:
    """Filesystem adapter with stage-verify-commit-rename publication."""

    def __init__(
        self,
        root: Path,
        *,
        limits: ReaderLimits = DEFAULT_READER_LIMITS,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.limits = limits
        self._field_limits = limits.model_copy(
            update={
                "max_file_bytes": min(limits.max_file_bytes, RUN_MAX_FILE_BYTES),
                "max_member_bytes": min(limits.max_member_bytes, RUN_MAX_FILE_BYTES),
                "max_total_uncompressed_bytes": min(
                    limits.max_total_uncompressed_bytes, RUN_MAX_FILE_BYTES
                ),
            }
        )
        self._fault_injector = fault_injector
        self.root.mkdir(parents=True, exist_ok=True)

    def _inject(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)

    def _committed_size(self, uri: str) -> int:
        fields = safe_read_bytes(
            self.root,
            f"{uri}/{RUN_FIELDS_NAME}",
            max_bytes=RUN_MAX_FILE_BYTES,
        )
        metadata = safe_read_bytes(
            self.root,
            f"{uri}/{RUN_METADATA_NAME}",
            max_bytes=min(self.limits.max_file_bytes, self.limits.max_json_bytes),
        )
        marker = safe_read_bytes(
            self.root,
            f"{uri}/{RUN_COMMIT_NAME}",
            max_bytes=65,
        )
        return len(fields) + len(metadata) + len(marker)

    @staticmethod
    def _validate_identity(name: str, value: str, pattern: re.Pattern[str]) -> None:
        if pattern.fullmatch(value) is None:
            raise ArtifactIntegrityError(f"RUN-1 IDENTITY: invalid {name}")

    def publish_run(
        self,
        *,
        attempt_id: str,
        design_id: str,
        split: Split,
        case: CaseConfig,
        result: SolverResult,
    ) -> ArtifactRef:
        """Publish one run atomically; a matching committed target is a no-op."""

        self._validate_identity("attempt_id", attempt_id, _ATTEMPT_ID_PATTERN)
        self._validate_identity("design_id", design_id, _CONTENT_ID_PATTERN)
        fields, quantization = curate_solver_result(case, result)
        fields_bytes = encode_run_fields(fields)
        if len(fields_bytes) > RUN_MAX_FILE_BYTES:
            raise ArtifactIntegrityError("RUN-1 SIZE: fields archive exceeds the run byte cap")
        fields_sha256 = sha256_bytes(fields_bytes)
        metadata = RunMetadata.create(
            design_id=design_id,
            split=split,
            case=case,
            result=result,
            fields=fields,
            quantization=quantization,
            fields_sha256=fields_sha256,
        )
        metadata_bytes = _metadata_bytes(metadata)
        metadata_sha256 = sha256_bytes(metadata_bytes)
        relative_root = f"{RUN_ROOT_PREFIX}/{case.case_id}/{metadata.artifact_digest}"
        reference = ArtifactRef(
            artifact_type="run",
            artifact_id=metadata.artifact_digest[:20],
            sha256=metadata.artifact_digest,
            size_bytes=len(fields_bytes) + len(metadata_bytes) + 65,
            uri=relative_root,
        )
        target = self.root / relative_root
        if target.exists() or target.is_symlink():
            existing_size = self._committed_size(relative_root)
            loaded = self.open_run(reference.model_copy(update={"size_bytes": existing_size}))
            return loaded.reference

        staging_parent = ensure_real_directory(self.root, ".staging", case.case_id)
        staging = Path(tempfile.mkdtemp(prefix=f"{attempt_id}-", dir=staging_parent))
        try:
            fields_path = staging / RUN_FIELDS_NAME
            fields_path.write_bytes(fields_bytes)
            fsync_file(fields_path)
            self._inject("fields_written")

            metadata_path = staging / RUN_METADATA_NAME
            metadata_path.write_bytes(metadata_bytes)
            fsync_file(metadata_path)
            self._inject("metadata_written")

            safe_read_npz(
                staging,
                RUN_FIELDS_NAME,
                expected=metadata.field_members,
                limits=self._field_limits,
                expected_sha256=metadata.fields_sha256,
            )
            safe_read_json(
                staging,
                RUN_METADATA_NAME,
                model=RunMetadata,
                limits=self.limits,
                expected_sha256=metadata_sha256,
            )
            self._inject("verified")

            commit_path = staging / RUN_COMMIT_NAME
            commit_path.write_text(metadata_sha256 + "\n", encoding="ascii")
            fsync_file(commit_path)
            fsync_directory(staging)
            self._inject("committed")

            target_parent = ensure_real_directory(self.root, RUN_ROOT_PREFIX, case.case_id)
            try:
                os.rename(staging, target)
            except OSError as error:
                if error.errno not in {EEXIST, ENOTEMPTY}:
                    raise
                existing_size = self._committed_size(relative_root)
                loaded = self.open_run(reference.model_copy(update={"size_bytes": existing_size}))
                reference = loaded.reference
            fsync_directory(target_parent)
            self._inject("published")
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return reference

    def open_run(self, reference: ArtifactRef) -> RunArtifact:
        """Open only a complete marker-bound run and revalidate every member."""

        if not isinstance(reference, ArtifactRef) or reference.artifact_type != "run":
            raise ArtifactIntegrityError("RUN-1 IDENTITY: expected a run ArtifactRef")
        parts = reference.uri.split("/")
        if len(parts) != 3 or parts[0] != RUN_ROOT_PREFIX:
            raise ArtifactIntegrityError("RUN-1 IDENTITY: run URI does not match the store layout")
        case_id, run_digest = parts[1], parts[2]
        self._validate_identity("case_id", case_id, _CONTENT_ID_PATTERN)
        self._validate_identity("run_digest", run_digest, _SHA256_PATTERN)
        if reference.sha256 != run_digest or reference.artifact_id != run_digest[:20]:
            raise ArtifactIntegrityError("RUN-1 IDENTITY: reference and run path disagree")

        marker = safe_read_bytes(
            self.root,
            f"{reference.uri}/{RUN_COMMIT_NAME}",
            max_bytes=65,
        )
        try:
            marker_digest = marker.decode("ascii").removesuffix("\n")
        except UnicodeDecodeError as error:
            raise ArtifactIntegrityError("RUN-3 COMMIT: marker is not ASCII") from error
        if len(marker) != 65 or _SHA256_PATTERN.fullmatch(marker_digest) is None:
            raise ArtifactIntegrityError("RUN-3 COMMIT: marker must contain one metadata digest")

        metadata = safe_read_json(
            self.root,
            f"{reference.uri}/{RUN_METADATA_NAME}",
            model=RunMetadata,
            limits=self.limits,
            expected_sha256=marker_digest,
        )
        if metadata.case_id != case_id or metadata.artifact_digest != run_digest:
            raise ArtifactIntegrityError("RUN-1 IDENTITY: metadata does not match its run path")
        arrays = safe_read_npz(
            self.root,
            f"{reference.uri}/{RUN_FIELDS_NAME}",
            expected=metadata.field_members,
            limits=self._field_limits,
            expected_sha256=metadata.fields_sha256,
        )
        fields = CuratedRunFields(
            u_mean=cast(Float16Array, arrays["u_mean"]),
            v_mean=cast(Float16Array, arrays["v_mean"]),
            rho_mean=cast(Float16Array, arrays["rho_mean"]),
            sdf=cast(Float16Array, arrays["sdf"]),
            obstacle_mask=cast(UInt8Array, arrays["obstacle_mask"]),
            force_steps=cast(Int64Array, arrays["force_steps"]),
            cd_history=cast(Float32Array, arrays["cd_history"]),
            cl_history=cast(Float32Array, arrays["cl_history"]),
        )
        actual_size = self._committed_size(reference.uri)
        if actual_size != reference.size_bytes:
            raise ArtifactIntegrityError("RUN-1 SIZE: reference byte count does not match content")
        return RunArtifact(
            reference=reference,
            metadata=metadata,
            fields=fields,
            metadata_sha256=marker_digest,
        )

    def verify_run(self, *, case_id: str, run_digest: str) -> ArtifactRef:
        """Resolve a deterministic run key and fully verify its committed content."""

        self._validate_identity("case_id", case_id, _CONTENT_ID_PATTERN)
        self._validate_identity("run_digest", run_digest, _SHA256_PATTERN)
        uri = f"{RUN_ROOT_PREFIX}/{case_id}/{run_digest}"
        reference = ArtifactRef(
            artifact_type="run",
            artifact_id=run_digest[:20],
            sha256=run_digest,
            size_bytes=self._committed_size(uri),
            uri=uri,
        )
        return self.open_run(reference).reference


__all__ = [
    "OUTPUT_SHAPE",
    "RUN_MEMBER_ORDER",
    "SOLVER_SHAPE",
    "CuratedRunFields",
    "LocalRunArtifactStore",
    "QuantizationStatistic",
    "RunArtifact",
    "RunArtifactStore",
    "RunMetadata",
    "curate_solver_result",
    "encode_run_fields",
    "run_member_descriptors",
]
