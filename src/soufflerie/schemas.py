"""Canonical shared records, identities, and array invariants."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, TypeAlias
from urllib.parse import urlsplit

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from soufflerie.errors import ArtifactIntegrityError, SchemaVersionError

SchemaVersion: TypeAlias = Literal[1]
Split: TypeAlias = Literal["train", "validation", "test"]
RunState: TypeAlias = Literal["pending", "running", "succeeded", "failed"]
ArrayDType: TypeAlias = Literal["float16", "float32", "float64", "int64", "uint8", "bool"]
ArrayUnit: TypeAlias = Literal[
    "lattice_velocity",
    "lattice_density",
    "lattice_distance",
    "dimensionless",
    "step",
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ContentId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{20}$")]
ArtifactType = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
NonEmptyString = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CONTENT_ID_LENGTH = 20
UINT64_MAX = 2**64 - 1
SUPPORTED_SCHEMA_VERSIONS = (1,)


class StrictFrozenModel(BaseModel):
    """Base for immutable records that reject coercion and unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class VersionedModel(StrictFrozenModel):
    """Base for durable schema-v1 records with an explicit version failure."""

    schema_version: SchemaVersion = 1

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_schema_version(cls, value: object) -> object:
        if isinstance(value, Mapping):
            version = value.get("schema_version", 1)
            if isinstance(version, int) and not isinstance(version, bool) and version != 1:
                raise SchemaVersionError(version)
        return value


class ShapeParams(StrictFrozenModel):
    """Ellipse parameters at the public degree/lattice-unit boundary."""

    aspect_ratio: float = Field(ge=0.5, le=1.0, allow_inf_nan=False)
    rotation_deg: float = Field(ge=0.0, le=30.0, allow_inf_nan=False)
    scale: float = Field(ge=0.75, le=1.25, allow_inf_nan=False)


class GridSpec(VersionedModel):
    """A row-major two-dimensional lattice grid."""

    nx: int = Field(ge=3)
    ny: int = Field(ge=3)
    axis_order: Literal["yx"] = "yx"
    spacing_unit: Literal["lattice_unit"] = "lattice_unit"
    timestep_unit: Literal["lattice_step"] = "lattice_step"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)


class CaseConfig(VersionedModel):
    """One validated physical and numerical solver case."""

    shape: ShapeParams
    reynolds: float = Field(ge=40.0, le=300.0, allow_inf_nan=False)
    nx: int = Field(ge=3)
    ny: int = Field(ge=3)
    steps: int = Field(ge=1)
    warmup_steps: int = Field(ge=0)
    inlet_velocity_lu: float = Field(gt=0.0, le=0.1, allow_inf_nan=False)
    seed: int = Field(ge=0, le=UINT64_MAX)

    @model_validator(mode="after")
    def _warmup_precedes_completion(self) -> Self:
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be less than steps")
        return self

    @property
    def grid(self) -> GridSpec:
        return GridSpec(nx=self.nx, ny=self.ny)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self)

    @property
    def case_id(self) -> str:
        return self.sha256[:CONTENT_ID_LENGTH]


class ArrayDescriptor(VersionedModel):
    """JSON-native contract for one persisted array member."""

    dtype: ArrayDType
    shape: tuple[int, ...] = Field(min_length=1, max_length=4)
    unit: ArrayUnit
    order: Literal["C"] = "C"
    finite: Literal[True] = True
    allow_pickle: Literal[False] = False

    @model_validator(mode="after")
    def _shape_is_positive(self) -> Self:
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("array dimensions must be positive")
        return self

    def validate_array(self, array: npt.NDArray[Any], *, name: str) -> None:
        validate_array(
            array,
            name=name,
            dtype=np.dtype(self.dtype),
            shape=self.shape,
            finite=self.finite,
        )


class SolverDiagnostics(VersionedModel):
    """Finite, internally coherent scalar solver diagnostics."""

    steps_completed: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    initial_mass: float = Field(gt=0.0, allow_inf_nan=False)
    final_mass: float = Field(gt=0.0, allow_inf_nan=False)
    mass_drift_ratio: float = Field(ge=0.0, allow_inf_nan=False)
    min_rho: float = Field(gt=0.0, allow_inf_nan=False)
    max_rho: float = Field(gt=0.0, allow_inf_nan=False)
    max_speed_lu: float = Field(ge=0.0, allow_inf_nan=False)
    converged: bool
    valid: bool
    messages: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def _diagnostics_are_coherent(self) -> Self:
        if self.min_rho > self.max_rho:
            raise ValueError("min_rho must not exceed max_rho")
        expected_drift = abs(self.final_mass - self.initial_mass) / self.initial_mass
        if not math.isclose(self.mass_drift_ratio, expected_drift, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("mass_drift_ratio does not match initial_mass and final_mass")
        return self


class Provenance(VersionedModel):
    """Reproducibility and direct-lineage record for a durable artifact."""

    model_config = ConfigDict(
        json_schema_extra={
            "$comment": (
                "Release consumers must bind source_revision, lock_sha256, config_sha256, "
                "packages, and every direct parent digest to reviewed expected values."
            )
        }
    )

    source_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    source_dirty: bool
    python_version: NonEmptyString
    lock_sha256: Sha256
    packages: dict[NonEmptyString, NonEmptyString] = Field(min_length=1)
    os: NonEmptyString
    architecture: NonEmptyString
    device_class: NonEmptyString
    dtype_policy: NonEmptyString
    config_sha256: Sha256
    parent_sha256: dict[NonEmptyString, Sha256] = Field(default_factory=dict)
    seeds: tuple[int, ...]
    deterministic: bool
    started_at: datetime
    completed_at: datetime
    gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _provenance_is_coherent(self) -> Self:
        for label, timestamp in (
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if timestamp.utcoffset() is None:
                raise ValueError(f"{label} must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if any(seed < 0 or seed > UINT64_MAX for seed in self.seeds):
            raise ValueError("seeds must be unsigned 64-bit integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        return self


class ArtifactRef(VersionedModel):
    """Portable reference to content whose identity is independent of location."""

    artifact_type: ArtifactType
    artifact_id: ContentId
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    uri: NonEmptyString

    @model_validator(mode="after")
    def _reference_is_safe_and_coherent(self) -> Self:
        if self.artifact_id != self.sha256[:CONTENT_ID_LENGTH]:
            raise ValueError("artifact_id must be the 20-character prefix of sha256")
        parsed = urlsplit(self.uri)
        path = PurePosixPath(self.uri)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or self.uri.startswith("/")
            or "\\" in self.uri
            or "\x00" in self.uri
            or str(path) != self.uri
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("uri must be a normalized artifact-root-relative POSIX path")
        return self

    @classmethod
    def from_bytes(
        cls,
        *,
        artifact_type: str,
        uri: str,
        content: bytes,
    ) -> ArtifactRef:
        digest = sha256_bytes(content)
        return cls(
            artifact_type=artifact_type,
            artifact_id=digest[:CONTENT_ID_LENGTH],
            sha256=digest,
            size_bytes=len(content),
            uri=uri,
        )


def _canonical_value(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not allow NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise ValueError("canonical JSON requires timezone-aware datetimes")
        normalized = value.astimezone(UTC)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            result[key] = _canonical_value(item)
        return result
    if isinstance(value, np.ndarray):
        raise TypeError("array bytes and descriptors must be hashed separately")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return stable, finite, whitespace-free JSON for a typed logical value."""

    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: object) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def verify_sha256(content: bytes, expected_sha256: str) -> None:
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise ArtifactIntegrityError("DM-4 IDENTITY: expected digest is not lowercase SHA-256")
    actual = sha256_bytes(content)
    if not hmac.compare_digest(actual, expected_sha256):
        raise ArtifactIntegrityError(
            f"DM-4 IDENTITY: SHA-256 mismatch (expected {expected_sha256}, got {actual})"
        )


def validate_schema_version(version: object) -> None:
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in SUPPORTED_SCHEMA_VERSIONS
    ):
        raise SchemaVersionError(version)


def validate_array(
    array: npt.NDArray[Any],
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    finite: bool = True,
) -> None:
    """Enforce DM-1/2/3/5 for an in-memory public array."""

    if not isinstance(array, np.ndarray):
        raise ArtifactIntegrityError(f"DM-1 SHAPE: {name} must be a NumPy array")
    if array.dtype.hasobject:
        raise ArtifactIntegrityError(f"DM-5 NO_PICKLE: {name} must not contain Python objects")
    if any(dimension <= 0 for dimension in array.shape):
        raise ArtifactIntegrityError(f"DM-1 SHAPE: {name} dimensions must be positive")
    if array.dtype != dtype:
        raise ArtifactIntegrityError(
            f"DM-3 DTYPE: {name} must have dtype {dtype.name}, got {array.dtype.name}"
        )
    if shape is not None and array.shape != shape:
        raise ArtifactIntegrityError(
            f"DM-1 SHAPE: {name} must have shape {shape}, got {array.shape}"
        )
    if ndim is not None and array.ndim != ndim:
        raise ArtifactIntegrityError(
            f"DM-1 SHAPE: {name} must have {ndim} dimensions, got {array.ndim}"
        )
    if not array.flags.c_contiguous:
        raise ArtifactIntegrityError(f"DM-1 SHAPE: {name} must be C-contiguous")
    if finite and np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ArtifactIntegrityError(f"DM-2 FINITE: {name} contains NaN or infinity")


FLOW_FIELD_UNITS = MappingProxyType(
    {
        "u": "lattice_velocity",
        "v": "lattice_velocity",
        "rho": "lattice_density",
        "sdf": "lattice_distance",
        "obstacle_mask": "dimensionless",
    }
)


@dataclass(frozen=True, slots=True)
class FlowFields:
    """Validated row-major fp32 solver fields."""

    u: npt.NDArray[np.float32]
    v: npt.NDArray[np.float32]
    rho: npt.NDArray[np.float32]
    sdf: npt.NDArray[np.float32]
    obstacle_mask: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        validate_array(self.u, name="u", dtype=np.dtype(np.float32), ndim=2)
        shape = self.u.shape
        validate_array(self.v, name="v", dtype=np.dtype(np.float32), shape=shape)
        validate_array(self.rho, name="rho", dtype=np.dtype(np.float32), shape=shape)
        validate_array(self.sdf, name="sdf", dtype=np.dtype(np.float32), shape=shape)
        validate_array(
            self.obstacle_mask,
            name="obstacle_mask",
            dtype=np.dtype(np.bool_),
            shape=shape,
            finite=False,
        )
        if not np.all(self.rho > 0):
            raise ArtifactIntegrityError("DM-2 FINITE: rho must be strictly positive")
        if not np.array_equal(self.obstacle_mask, self.sdf <= 0):
            raise ArtifactIntegrityError("DM-1 SHAPE: obstacle_mask must equal sdf <= 0")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.u.shape[0], self.u.shape[1])

    def descriptors(self) -> dict[str, ArrayDescriptor]:
        shape = self.shape
        return {
            "u": ArrayDescriptor(dtype="float32", shape=shape, unit="lattice_velocity"),
            "v": ArrayDescriptor(dtype="float32", shape=shape, unit="lattice_velocity"),
            "rho": ArrayDescriptor(dtype="float32", shape=shape, unit="lattice_density"),
            "sdf": ArrayDescriptor(dtype="float32", shape=shape, unit="lattice_distance"),
            "obstacle_mask": ArrayDescriptor(
                dtype="bool", shape=shape, unit="dimensionless", finite=True
            ),
        }


@dataclass(frozen=True, slots=True)
class SolverResult:
    """One complete solver output bound to diagnostics and provenance."""

    case_id: str
    fields: FlowFields
    cd: float
    cl_mean: float
    strouhal: float | None
    force_steps: npt.NDArray[np.int64]
    cd_history: npt.NDArray[np.float32]
    cl_history: npt.NDArray[np.float32]
    diagnostics: SolverDiagnostics
    provenance: Provenance

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{20}", self.case_id):
            raise ArtifactIntegrityError("DM-4 IDENTITY: case_id must be 20 lowercase hex digits")
        for name, value in (("cd", self.cd), ("cl_mean", self.cl_mean)):
            if not math.isfinite(value):
                raise ArtifactIntegrityError(f"DM-2 FINITE: {name} must be finite")
        if self.strouhal is not None and (not math.isfinite(self.strouhal) or self.strouhal < 0):
            raise ArtifactIntegrityError("DM-2 FINITE: strouhal must be finite and nonnegative")
        validate_array(self.force_steps, name="force_steps", dtype=np.dtype(np.int64), ndim=1)
        history_shape = self.force_steps.shape
        validate_array(
            self.cd_history,
            name="cd_history",
            dtype=np.dtype(np.float32),
            shape=history_shape,
        )
        validate_array(
            self.cl_history,
            name="cl_history",
            dtype=np.dtype(np.float32),
            shape=history_shape,
        )
        if history_shape[0] != self.diagnostics.sample_count:
            raise ArtifactIntegrityError(
                "DM-1 SHAPE: history length must match diagnostics.sample_count"
            )
        if np.any(self.force_steps < 0) or np.any(np.diff(self.force_steps) <= 0):
            raise ArtifactIntegrityError(
                "DM-1 SHAPE: force_steps must be nonnegative and strictly increasing"
            )
        if not self.diagnostics.valid:
            raise ArtifactIntegrityError("invalid diagnostics cannot be published as SolverResult")


def validate_split_membership(assignments: Iterable[tuple[str, Split]]) -> dict[str, Split]:
    """Enforce DM-6 by rejecting duplicate or cross-split case identities."""

    result: dict[str, Split] = {}
    for case_id, split in assignments:
        if not re.fullmatch(r"[0-9a-f]{20}", case_id):
            raise ArtifactIntegrityError("DM-6 SPLIT: case_id must be 20 lowercase hex digits")
        if split not in {"train", "validation", "test"}:
            raise ArtifactIntegrityError(f"DM-6 SPLIT: unsupported split {split!r}")
        if case_id in result:
            raise ArtifactIntegrityError(f"DM-6 SPLIT: duplicate case_id {case_id}")
        result[case_id] = split
    return result


def validate_parent_digests(
    provenance: Provenance,
    *,
    required_parents: Iterable[str],
) -> None:
    """Enforce DM-7 for the direct parents required by an artifact type."""

    required = set(required_parents)
    missing = sorted(required - provenance.parent_sha256.keys())
    if missing:
        raise ArtifactIntegrityError(
            f"DM-7 PROVENANCE: missing direct parent digests: {', '.join(missing)}"
        )


def validate_field_units(descriptors: Mapping[str, ArrayDescriptor]) -> None:
    """Enforce the fixed DM-8 units for canonical solver fields."""

    if descriptors.keys() != FLOW_FIELD_UNITS.keys():
        raise ArtifactIntegrityError("DM-8 UNITS: field descriptor names do not match the contract")
    for name, expected_unit in FLOW_FIELD_UNITS.items():
        if descriptors[name].unit != expected_unit:
            raise ArtifactIntegrityError(
                f"DM-8 UNITS: {name} must use {expected_unit}, got {descriptors[name].unit}"
            )


SCHEMA_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "array-descriptor": ArrayDescriptor,
        "artifact-ref": ArtifactRef,
        "case-config": CaseConfig,
        "grid-spec": GridSpec,
        "provenance": Provenance,
        "shape-params": ShapeParams,
        "solver-diagnostics": SolverDiagnostics,
    }
)


def schema_documents() -> dict[str, dict[str, Any]]:
    """Generate deterministic JSON Schema draft 2020-12 documents."""

    result: dict[str, dict[str, Any]] = {}
    for name, model in SCHEMA_MODELS.items():
        document = model.model_json_schema()
        document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        document["$id"] = f"https://github.com/AbdelStark/soufflerie/schemas/v1/{name}.json"
        result[name] = document
    return result


def rendered_schema_documents() -> dict[str, str]:
    return {
        f"{name}.json": json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        for name, document in schema_documents().items()
    }


__all__ = [
    "FLOW_FIELD_UNITS",
    "SCHEMA_MODELS",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ArrayDType",
    "ArrayDescriptor",
    "ArrayUnit",
    "ArtifactRef",
    "CaseConfig",
    "ContentId",
    "FlowFields",
    "GridSpec",
    "JsonScalar",
    "JsonValue",
    "Provenance",
    "RunState",
    "SchemaVersion",
    "ShapeParams",
    "SolverDiagnostics",
    "SolverResult",
    "Split",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "rendered_schema_documents",
    "schema_documents",
    "sha256_bytes",
    "validate_array",
    "validate_field_units",
    "validate_parent_digests",
    "validate_schema_version",
    "validate_split_membership",
    "verify_sha256",
]
