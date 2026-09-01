"""Artifact identity, lineage, provenance, and bounded safe readers."""

from __future__ import annotations

import heapq
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import stat
import struct
import subprocess
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)
from pydantic import (
    JsonValue as PydanticJsonValue,
)

from soufflerie.errors import ArtifactIntegrityError, SchemaVersionError
from soufflerie.schemas import (
    ArrayDescriptor,
    ArtifactRef,
    ArtifactType,
    ContentId,
    JsonValue,
    Provenance,
    Sha256,
    StrictFrozenModel,
    VersionedModel,
    canonical_sha256,
    sha256_bytes,
    validate_array,
    validate_schema_version,
)

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore[import-untyped]

TModel = TypeVar("TModel", bound=BaseModel)
ParentRole = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
]
ArtifactKeyText = Annotated[str, StringConstraints(min_length=1, max_length=1024)]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_LOGICAL_METADATA_FORBIDDEN_KEYS = frozenset(
    {
        "artifact_id",
        "committed_at",
        "created_at",
        "location",
        "path",
        "published_at",
        "sha256",
        "timestamp",
        "updated_at",
        "uri",
    }
)
_NPY_SUFFIX = ".npy"
_SAFE_ZIP_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_SAFE_PARQUET_TYPES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float",
        "double",
        "string",
        "binary",
    }
)
_SAFETENSOR_DTYPES: Mapping[str, np.dtype[Any]] = MappingProxyType(
    {
        "BOOL": np.dtype("bool"),
        "F16": np.dtype("float16"),
        "F32": np.dtype("float32"),
        "F64": np.dtype("float64"),
        "I64": np.dtype("int64"),
        "U8": np.dtype("uint8"),
    }
)


class ReaderLimits(StrictFrozenModel):
    """Allocation and parser ceilings applied before format deserialization."""

    max_file_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    max_json_bytes: int = Field(default=4 * 1024 * 1024, ge=1)
    max_json_depth: int = Field(default=32, ge=1, le=128)
    max_json_keys: int = Field(default=10_000, ge=1)
    max_archive_members: int = Field(default=128, ge=1)
    max_member_bytes: int = Field(default=256 * 1024 * 1024, ge=1)
    max_total_uncompressed_bytes: int = Field(default=1024 * 1024 * 1024, ge=1)
    max_compression_ratio: float = Field(default=1000.0, ge=1.0, allow_inf_nan=False)
    max_npy_header_bytes: int = Field(default=64 * 1024, ge=128)
    max_parquet_rows: int = Field(default=1_000_000, ge=1)
    max_parquet_columns: int = Field(default=128, ge=1)
    max_parquet_row_groups: int = Field(default=4096, ge=1)
    max_tensor_count: int = Field(default=512, ge=1)
    max_tensor_header_bytes: int = Field(default=1024 * 1024, ge=8)


DEFAULT_READER_LIMITS = ReaderLimits()


class SourceState(StrictFrozenModel):
    """Exact source revision and dirty state captured at operation start."""

    source_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    source_dirty: bool


class ParentLink(StrictFrozenModel):
    """One typed edge from an artifact to a direct parent digest."""

    role: ParentRole
    artifact_type: ArtifactType
    sha256: Sha256


class LineageNode(VersionedModel):
    """Portable artifact identity and all direct typed parent edges."""

    artifact_type: ArtifactType
    artifact_id: ContentId
    sha256: Sha256
    parents: tuple[ParentLink, ...] = Field(default=(), max_length=4096)

    @model_validator(mode="after")
    def _identity_and_roles_are_coherent(self) -> LineageNode:
        if self.artifact_id != self.sha256[:20]:
            raise ValueError("artifact_id must be the 20-character prefix of sha256")
        roles = [parent.role for parent in self.parents]
        if len(set(roles)) != len(roles):
            raise ValueError("parent roles must be unique within one lineage node")
        return self


class ParentTypeRule(StrictFrozenModel):
    """Required and allowed direct parent artifact types for one child type."""

    required: tuple[ArtifactType, ...] = ()
    allowed: tuple[ArtifactType, ...] = ()

    @model_validator(mode="after")
    def _required_types_are_allowed(self) -> ParentTypeRule:
        if len(set(self.required)) != len(self.required):
            raise ValueError("required parent types must be unique")
        if len(set(self.allowed)) != len(self.allowed):
            raise ValueError("allowed parent types must be unique")
        missing = sorted(set(self.required) - set(self.allowed))
        if missing:
            raise ValueError(f"required parent types must also be allowed: {', '.join(missing)}")
        return self


@dataclass(frozen=True, slots=True)
class LineagePolicy:
    """Immutable type rules and graph-size bound for lineage validation."""

    rules: Mapping[str, ParentTypeRule]
    max_nodes: int = 10_000

    def __post_init__(self) -> None:
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        object.__setattr__(self, "rules", MappingProxyType(dict(self.rules)))


DEFAULT_LINEAGE_POLICY = LineagePolicy(
    rules={
        "run": ParentTypeRule(),
        "dataset": ParentTypeRule(required=("run",), allowed=("run",)),
        "model": ParentTypeRule(required=("dataset",), allowed=("dataset",)),
        "baseline": ParentTypeRule(required=("dataset",), allowed=("dataset",)),
        "report": ParentTypeRule(
            required=("dataset", "model"),
            allowed=("dataset", "model", "baseline"),
        ),
    }
)


def _assert_logical_metadata(value: JsonValue, *, path: str = "logical_metadata") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.casefold()
            if normalized in _LOGICAL_METADATA_FORBIDDEN_KEYS:
                raise ArtifactIntegrityError(
                    "artifact identity metadata must not contain physical/mutable field "
                    f"{path}.{key}"
                )
            _assert_logical_metadata(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_logical_metadata(child, path=f"{path}[{index}]")


def _validate_artifact_key_text(key: str) -> PurePosixPath:
    if len(key) > 1024 or "\x00" in key or "\\" in key or ":" in key:
        raise ArtifactIntegrityError("artifact key must be a bounded relative POSIX path")
    path = PurePosixPath(key)
    if (
        path.is_absolute()
        or str(path) != key
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ArtifactIntegrityError("artifact key must be a normalized root-relative POSIX path")
    return path


def _provenance_identity(provenance: Provenance) -> dict[str, JsonValue]:
    value = provenance.model_dump(mode="python", exclude={"started_at", "completed_at"})
    return cast(dict[str, JsonValue], value)


def artifact_content_sha256(
    *,
    artifact_type: str,
    logical_metadata: Mapping[str, JsonValue],
    member_sha256: Mapping[str, str],
    provenance: Provenance,
) -> str:
    """Hash logical content while excluding location and wall-clock timestamps."""

    if re.fullmatch(r"^[a-z][a-z0-9_-]{0,63}$", artifact_type) is None:
        raise ArtifactIntegrityError("invalid artifact type")
    metadata = dict(logical_metadata)
    _assert_logical_metadata(metadata)
    members: dict[str, str] = {}
    for key, digest in member_sha256.items():
        _validate_artifact_key_text(key)
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ArtifactIntegrityError(f"invalid SHA-256 for artifact member {key!r}")
        members[key] = digest
    if not members:
        raise ArtifactIntegrityError("artifact identity requires at least one member digest")
    return canonical_sha256(
        {
            "schema_version": 1,
            "artifact_type": artifact_type,
            "logical_metadata": metadata,
            "member_sha256": members,
            "provenance": _provenance_identity(provenance),
        }
    )


class ArtifactEnvelope(VersionedModel):
    """Content identity plus mutable location/time outside the digest boundary."""

    artifact_type: ArtifactType
    artifact_id: ContentId
    sha256: Sha256
    logical_metadata: dict[str, PydanticJsonValue]
    member_sha256: dict[ArtifactKeyText, Sha256] = Field(min_length=1, max_length=4096)
    provenance: Provenance
    created_at: datetime
    uri: ArtifactKeyText

    @model_validator(mode="after")
    def _envelope_is_coherent(self) -> ArtifactEnvelope:
        if self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        ArtifactRef(
            artifact_type=self.artifact_type,
            artifact_id=self.artifact_id,
            sha256=self.sha256,
            size_bytes=0,
            uri=self.uri,
        )
        expected = artifact_content_sha256(
            artifact_type=self.artifact_type,
            logical_metadata=self.logical_metadata,
            member_sha256=self.member_sha256,
            provenance=self.provenance,
        )
        if self.sha256 != expected:
            raise ValueError("artifact sha256 does not match logical content and member digests")
        return self

    def lineage_node(self, *, parent_types: Mapping[str, str]) -> LineageNode:
        """Build a typed lineage node from provenance parent roles/digests."""

        if set(parent_types) != set(self.provenance.parent_sha256):
            raise ArtifactIntegrityError(
                "parent type roles must exactly match provenance parent digest roles"
            )
        return LineageNode(
            artifact_type=self.artifact_type,
            artifact_id=self.artifact_id,
            sha256=self.sha256,
            parents=tuple(
                ParentLink(
                    role=role,
                    artifact_type=parent_types[role],
                    sha256=digest,
                )
                for role, digest in sorted(self.provenance.parent_sha256.items())
            ),
        )


def capture_source_state(source_root: Path) -> SourceState:
    """Capture a full Git revision and tracked/untracked dirty state."""

    root = source_root.resolve(strict=True)
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status_output = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ArtifactIntegrityError("unable to capture Git source provenance") from error
    if _SOURCE_REVISION_PATTERN.fullmatch(revision) is None:
        raise ArtifactIntegrityError("Git source revision must be a full lowercase SHA-1")
    return SourceState(source_revision=revision, source_dirty=bool(status_output.strip()))


def capture_provenance(
    *,
    source_root: Path,
    config: object,
    parent_sha256: Mapping[str, str],
    package_names: Sequence[str],
    device_class: str,
    dtype_policy: str,
    seeds: Sequence[int],
    deterministic: bool,
    started_at: datetime,
    completed_at: datetime,
    gpu_seconds: float,
    lock_key: str = "uv.lock",
    source_state: SourceState | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> Provenance:
    """Capture reproducibility evidence without reading process environment values."""

    if not package_names:
        raise ArtifactIntegrityError("provenance requires an explicit package allowlist")
    if len(set(package_names)) != len(package_names):
        raise ArtifactIntegrityError("provenance package names must be unique")
    root = source_root.resolve(strict=True)
    source = source_state or capture_source_state(root)
    lock = safe_read_bytes(root, lock_key, max_bytes=16 * 1024 * 1024)
    if package_versions is None:
        versions: dict[str, str] = {}
        for name in package_names:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as error:
                raise ArtifactIntegrityError(
                    f"required provenance package {name!r} is not installed"
                ) from error
    else:
        versions = dict(package_versions)
        if set(versions) != set(package_names):
            raise ArtifactIntegrityError(
                "captured package versions must exactly match the package allowlist"
            )
    return Provenance(
        source_revision=source.source_revision,
        source_dirty=source.source_dirty,
        python_version=platform.python_version(),
        lock_sha256=sha256_bytes(lock),
        packages=versions,
        os=platform.system().casefold(),
        architecture=platform.machine(),
        device_class=device_class,
        dtype_policy=dtype_policy,
        config_sha256=canonical_sha256(config),
        parent_sha256=dict(parent_sha256),
        seeds=tuple(seeds),
        deterministic=deterministic,
        started_at=started_at,
        completed_at=completed_at,
        gpu_seconds=gpu_seconds,
    )


def validate_release_provenance(
    provenance: Provenance,
    *,
    expected_source_revision: str,
    expected_lock_sha256: str,
    expected_config_sha256: str,
    expected_packages: Mapping[str, str],
    required_parent_sha256: Mapping[str, str],
    require_deterministic: bool = True,
) -> None:
    """Fail release publication unless source, packages, and parents are exact."""

    if provenance.source_dirty:
        raise ArtifactIntegrityError("release provenance requires clean source")
    if (
        _SOURCE_REVISION_PATTERN.fullmatch(expected_source_revision) is None
        or provenance.source_revision != expected_source_revision
    ):
        raise ArtifactIntegrityError(
            "release provenance source revision does not match the reviewed revision"
        )
    if provenance.lock_sha256 != expected_lock_sha256:
        raise ArtifactIntegrityError(
            "release provenance lock digest does not match the reviewed lockfile"
        )
    if provenance.config_sha256 != expected_config_sha256:
        raise ArtifactIntegrityError(
            "release provenance config digest does not match the reviewed configuration"
        )
    if provenance.packages != dict(expected_packages):
        raise ArtifactIntegrityError(
            "release provenance packages must exactly match the reviewed allowlist"
        )
    if provenance.parent_sha256 != dict(required_parent_sha256):
        raise ArtifactIntegrityError(
            "release provenance parent evidence is missing, extra, or mismatched"
        )
    if require_deterministic and not provenance.deterministic:
        raise ArtifactIntegrityError("release provenance requires deterministic execution")


def verify_lineage(
    nodes: Iterable[LineageNode],
    *,
    policy: LineagePolicy = DEFAULT_LINEAGE_POLICY,
) -> tuple[LineageNode, ...]:
    """Verify existence, parent types, required edges, collisions, and acyclicity."""

    materialized = tuple(nodes)
    if not materialized:
        raise ArtifactIntegrityError("lineage graph must not be empty")
    if len(materialized) > policy.max_nodes:
        raise ArtifactIntegrityError(f"lineage graph exceeds {policy.max_nodes} nodes")
    by_digest: dict[str, LineageNode] = {}
    by_id: dict[str, str] = {}
    for node in materialized:
        if node.sha256 in by_digest:
            raise ArtifactIntegrityError(f"duplicate lineage digest {node.sha256}")
        previous = by_id.get(node.artifact_id)
        if previous is not None and previous != node.sha256:
            raise ArtifactIntegrityError(f"artifact ID prefix collision for {node.artifact_id}")
        by_digest[node.sha256] = node
        by_id[node.artifact_id] = node.sha256

    children: dict[str, set[str]] = {digest: set() for digest in by_digest}
    indegree: dict[str, int] = {digest: 0 for digest in by_digest}
    for node in materialized:
        rule = policy.rules.get(node.artifact_type)
        if rule is None:
            raise ArtifactIntegrityError(
                f"no lineage parent-type rule for artifact type {node.artifact_type!r}"
            )
        observed_types: set[str] = set()
        unique_parent_digests: set[str] = set()
        for link in node.parents:
            parent = by_digest.get(link.sha256)
            if parent is None:
                raise ArtifactIntegrityError(
                    f"lineage parent {link.sha256} for {node.artifact_id} is missing"
                )
            if parent.artifact_type != link.artifact_type:
                raise ArtifactIntegrityError(
                    f"lineage parent type mismatch for role {link.role!r}: "
                    f"declared {link.artifact_type}, actual {parent.artifact_type}"
                )
            if link.artifact_type not in rule.allowed:
                raise ArtifactIntegrityError(
                    f"artifact type {node.artifact_type!r} cannot depend on {link.artifact_type!r}"
                )
            observed_types.add(link.artifact_type)
            unique_parent_digests.add(link.sha256)
        missing_types = sorted(set(rule.required) - observed_types)
        if missing_types:
            raise ArtifactIntegrityError(
                f"artifact {node.artifact_id} lacks required parent types: "
                f"{', '.join(missing_types)}"
            )
        indegree[node.sha256] = len(unique_parent_digests)
        for parent_digest in unique_parent_digests:
            children[parent_digest].add(node.sha256)

    ready = [digest for digest, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[LineageNode] = []
    while ready:
        digest = heapq.heappop(ready)
        ordered.append(by_digest[digest])
        for child in sorted(children[digest]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(ordered) != len(materialized):
        raise ArtifactIntegrityError("artifact lineage graph contains a cycle")
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class VerifiedConsumerArtifacts:
    """Exact dataset/model/report nodes accepted at a consumer boundary."""

    dataset: LineageNode
    model: LineageNode
    report: LineageNode


def verify_consumer_identities(
    nodes: Iterable[LineageNode],
    *,
    dataset_id: str,
    model_id: str,
    report_id: str,
    policy: LineagePolicy = DEFAULT_LINEAGE_POLICY,
    dataset_type: str = "dataset",
    model_type: str = "model",
    report_type: str = "report",
) -> VerifiedConsumerArtifacts:
    """Bind a consumer to a report, model, and dataset in one verified DAG."""

    ordered = verify_lineage(nodes, policy=policy)
    by_id = {node.artifact_id: node for node in ordered}
    try:
        dataset = by_id[dataset_id]
        model = by_id[model_id]
        report = by_id[report_id]
    except KeyError as error:
        raise ArtifactIntegrityError(f"consumer identity {error.args[0]!r} is missing") from error
    for label, node, expected_type in (
        ("dataset", dataset, dataset_type),
        ("model", model, model_type),
        ("report", report, report_type),
    ):
        if node.artifact_type != expected_type:
            raise ArtifactIntegrityError(
                f"consumer {label} identity has type {node.artifact_type!r}, "
                f"expected {expected_type!r}"
            )
    model_parents = {parent.sha256 for parent in model.parents}
    report_parents = {parent.sha256 for parent in report.parents}
    if dataset.sha256 not in model_parents:
        raise ArtifactIntegrityError("model lineage does not reference the selected dataset")
    if dataset.sha256 not in report_parents or model.sha256 not in report_parents:
        raise ArtifactIntegrityError(
            "report lineage does not reference the selected model and dataset"
        )
    return VerifiedConsumerArtifacts(dataset=dataset, model=model, report=report)


def resolve_artifact_key(root: Path, key: str) -> Path:
    """Resolve a normalized key beneath an existing root and reject symlinks."""

    relative = _validate_artifact_key_text(key)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ArtifactIntegrityError("artifact root does not exist") from error
    if not resolved_root.is_dir():
        raise ArtifactIntegrityError("artifact root must be a directory")
    current = resolved_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactIntegrityError(f"artifact key {key!r} traverses a symbolic link")
    resolved = current.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ArtifactIntegrityError(f"artifact key {key!r} escapes its root") from error
    return resolved


def safe_read_bytes(
    root: Path,
    key: str,
    *,
    max_bytes: int,
    expected_sha256: str | None = None,
) -> bytes:
    """Read one regular root-constrained file with a pre-allocation size cap."""

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if expected_sha256 is not None and _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    path = resolve_artifact_key(root, key)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactIntegrityError(f"unable to open artifact key {key!r}") from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ArtifactIntegrityError(f"artifact key {key!r} is not a regular file")
        if details.st_size > max_bytes:
            raise ArtifactIntegrityError(f"artifact key {key!r} exceeds the {max_bytes}-byte limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(max_bytes + 1)
        if len(content) > max_bytes or len(content) != details.st_size:
            raise ArtifactIntegrityError(
                f"artifact key {key!r} changed size or exceeded its byte limit"
            )
    finally:
        os.close(descriptor)
    if expected_sha256 is not None and sha256_bytes(content) != expected_sha256:
        raise ArtifactIntegrityError(f"artifact key {key!r} failed SHA-256 verification")
    return content


class _DuplicateJsonKey(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _bounded_json_loads(content: bytes, *, limits: ReaderLimits) -> JsonValue:
    try:
        decoded = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise ArtifactIntegrityError(
            "artifact JSON is malformed, duplicate, or non-finite"
        ) from error

    keys = 0
    stack: list[tuple[JsonValue, int]] = [(cast(JsonValue, decoded), 1)]
    while stack:
        value, depth = stack.pop()
        if depth > limits.max_json_depth:
            raise ArtifactIntegrityError(
                f"artifact JSON exceeds maximum depth {limits.max_json_depth}"
            )
        if isinstance(value, dict):
            keys += len(value)
            if keys > limits.max_json_keys:
                raise ArtifactIntegrityError(
                    f"artifact JSON exceeds maximum key count {limits.max_json_keys}"
                )
            stack.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value)
    return cast(JsonValue, decoded)


def safe_read_json(
    root: Path,
    key: str,
    *,
    model: type[TModel],
    limits: ReaderLimits = DEFAULT_READER_LIMITS,
    expected_sha256: str | None = None,
    require_schema_version: bool = True,
) -> TModel:
    """Bound and validate JSON before returning a strict typed model."""

    maximum = min(limits.max_file_bytes, limits.max_json_bytes)
    content = safe_read_bytes(root, key, max_bytes=maximum, expected_sha256=expected_sha256)
    decoded = _bounded_json_loads(content, limits=limits)
    if require_schema_version:
        if not isinstance(decoded, dict) or "schema_version" not in decoded:
            raise ArtifactIntegrityError("artifact JSON requires an explicit schema_version")
        validate_schema_version(decoded["schema_version"])
    try:
        return model.model_validate_json(content)
    except SchemaVersionError:
        raise
    except ValidationError as error:
        raise ArtifactIntegrityError(
            f"artifact JSON at key {key!r} does not match {model.__name__}"
        ) from error


def _safe_zip_member(info: zipfile.ZipInfo, *, limits: ReaderLimits) -> None:
    path = _validate_artifact_key_text(info.filename)
    if len(path.parts) != 1 or path.suffix != _NPY_SUFFIX or info.is_dir():
        raise ArtifactIntegrityError(f"unsafe NPZ member name {info.filename!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ArtifactIntegrityError(f"NPZ member {info.filename!r} is a symbolic link")
    if mode & 0o111:
        raise ArtifactIntegrityError(f"NPZ member {info.filename!r} is executable")
    if info.flag_bits & 0x1:
        raise ArtifactIntegrityError(f"encrypted NPZ member {info.filename!r} is forbidden")
    if info.compress_type not in _SAFE_ZIP_COMPRESSION:
        raise ArtifactIntegrityError(f"unsupported NPZ compression for {info.filename!r}")
    if info.file_size > limits.max_member_bytes:
        raise ArtifactIntegrityError(f"NPZ member {info.filename!r} exceeds its byte cap")
    ratio = info.file_size / max(1, info.compress_size)
    if ratio > limits.max_compression_ratio:
        raise ArtifactIntegrityError(f"NPZ member {info.filename!r} exceeds compression ratio cap")


def _read_npy_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    limits: ReaderLimits,
) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    try:
        with archive.open(info, "r") as handle:
            version = np.lib.format.read_magic(handle)  # type: ignore[no-untyped-call]
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    handle, max_header_size=limits.max_npy_header_bytes
                )  # type: ignore[no-untyped-call,call-arg]
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    handle, max_header_size=limits.max_npy_header_bytes
                )  # type: ignore[no-untyped-call,call-arg]
            else:
                raise ArtifactIntegrityError(
                    f"NPZ member {info.filename!r} uses unsupported NPY version {version}"
                )
            data_offset = handle.tell()
    except ArtifactIntegrityError:
        raise
    except (EOFError, OSError, ValueError) as error:
        raise ArtifactIntegrityError(f"invalid NPY header in {info.filename!r}") from error
    return tuple(shape), bool(fortran_order), np.dtype(dtype), data_offset


def safe_read_npz(
    root: Path,
    key: str,
    *,
    expected: Mapping[str, ArrayDescriptor],
    limits: ReaderLimits = DEFAULT_READER_LIMITS,
    expected_sha256: str | None = None,
) -> dict[str, npt.NDArray[Any]]:
    """Preflight an NPZ central directory and NPY headers before allocation."""

    if not expected:
        raise ValueError("expected NPZ descriptors must not be empty")
    content = safe_read_bytes(
        root,
        key,
        max_bytes=min(limits.max_file_bytes, limits.max_total_uncompressed_bytes),
        expected_sha256=expected_sha256,
    )
    try:
        with zipfile.ZipFile(io.BytesIO(content), "r") as archive:
            infos = archive.infolist()
            if len(infos) > limits.max_archive_members:
                raise ArtifactIntegrityError("NPZ archive exceeds member count cap")
            names = [info.filename for info in infos]
            if len(set(names)) != len(names):
                raise ArtifactIntegrityError("NPZ archive contains duplicate member names")
            expected_names = {f"{name}{_NPY_SUFFIX}" for name in expected}
            if set(names) != expected_names:
                raise ArtifactIntegrityError("NPZ members do not exactly match descriptors")
            total_uncompressed = 0
            for info in infos:
                _safe_zip_member(info, limits=limits)
                total_uncompressed += info.file_size
                if total_uncompressed > limits.max_total_uncompressed_bytes:
                    raise ArtifactIntegrityError("NPZ archive exceeds uncompressed byte cap")
                name = info.filename.removesuffix(_NPY_SUFFIX)
                descriptor = expected[name]
                shape, fortran_order, dtype, data_offset = _read_npy_header(
                    archive, info, limits=limits
                )
                if dtype.hasobject:
                    raise ArtifactIntegrityError(
                        f"DM-5 NO_PICKLE: NPZ member {name!r} has object dtype"
                    )
                if shape != descriptor.shape or dtype != np.dtype(descriptor.dtype):
                    raise ArtifactIntegrityError(
                        f"NPZ member {name!r} does not match its dtype/shape descriptor"
                    )
                if fortran_order or descriptor.order != "C" or descriptor.allow_pickle:
                    raise ArtifactIntegrityError(
                        f"NPZ member {name!r} violates C-order/no-pickle contract"
                    )
                expected_data_bytes = math.prod(shape) * dtype.itemsize
                if info.file_size != data_offset + expected_data_bytes:
                    raise ArtifactIntegrityError(
                        f"NPZ member {name!r} byte size does not match its descriptor"
                    )
    except ArtifactIntegrityError:
        raise
    except (zipfile.BadZipFile, OSError, RuntimeError, ValueError) as error:
        raise ArtifactIntegrityError("invalid or unsupported NPZ archive") from error

    arrays: dict[str, npt.NDArray[Any]] = {}
    try:
        with np.load(
            io.BytesIO(content),
            allow_pickle=False,
            max_header_size=limits.max_npy_header_bytes,
        ) as archive:
            for name, descriptor in expected.items():
                array = np.asarray(archive[name])
                validate_array(
                    array,
                    name=name,
                    dtype=np.dtype(descriptor.dtype),
                    shape=descriptor.shape,
                    finite=descriptor.finite,
                )
                array.setflags(write=False)
                arrays[name] = array
    except ArtifactIntegrityError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ArtifactIntegrityError("NPZ payload failed safe NumPy loading") from error
    return arrays


def safe_read_parquet(
    root: Path,
    key: str,
    *,
    expected_columns: Mapping[str, str],
    limits: ReaderLimits = DEFAULT_READER_LIMITS,
    expected_sha256: str | None = None,
) -> pa.Table:
    """Preflight Parquet metadata, then project an exact primitive schema."""

    if not expected_columns:
        raise ValueError("expected Parquet columns must not be empty")
    if len(expected_columns) > limits.max_parquet_columns:
        raise ValueError("expected Parquet columns exceed configured cap")
    unsupported = sorted(set(expected_columns.values()) - _SAFE_PARQUET_TYPES)
    if unsupported:
        raise ValueError(f"unsupported expected Parquet types: {', '.join(unsupported)}")
    content = safe_read_bytes(
        root,
        key,
        max_bytes=limits.max_file_bytes,
        expected_sha256=expected_sha256,
    )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        reader = pq.ParquetFile(pa.BufferReader(content))
        metadata = reader.metadata
        if metadata.num_rows > limits.max_parquet_rows:
            raise ArtifactIntegrityError("Parquet row count exceeds configured cap")
        if metadata.num_columns > limits.max_parquet_columns:
            raise ArtifactIntegrityError("Parquet column count exceeds configured cap")
        if metadata.num_row_groups > limits.max_parquet_row_groups:
            raise ArtifactIntegrityError("Parquet row-group count exceeds configured cap")
        total_uncompressed = sum(
            metadata.row_group(index).total_byte_size for index in range(metadata.num_row_groups)
        )
        if total_uncompressed > limits.max_total_uncompressed_bytes:
            raise ArtifactIntegrityError("Parquet metadata exceeds uncompressed byte cap")
        schema = reader.schema_arrow
        if set(schema.names) != set(expected_columns):
            raise ArtifactIntegrityError("Parquet columns do not exactly match declared schema")
        for name, expected_type in expected_columns.items():
            actual_type = str(schema.field(name).type)
            if actual_type != expected_type:
                raise ArtifactIntegrityError(
                    f"Parquet column {name!r} has type {actual_type}, expected {expected_type}"
                )
        table = reader.read(columns=list(expected_columns), use_threads=False)
        if table.num_rows > limits.max_parquet_rows:
            raise ArtifactIntegrityError("Parquet decoded row count exceeds configured cap")
        if table.nbytes > limits.max_total_uncompressed_bytes:
            raise ArtifactIntegrityError("Parquet decoded table exceeds byte cap")
        return table
    except ArtifactIntegrityError:
        raise
    except Exception as error:
        raise ArtifactIntegrityError("invalid or unsupported Parquet artifact") from error


def _tensor_header(content: bytes, *, limits: ReaderLimits) -> tuple[dict[str, JsonValue], int]:
    if len(content) < 8:
        raise ArtifactIntegrityError("safetensor file is shorter than its header prefix")
    header_size = struct.unpack("<Q", content[:8])[0]
    if header_size < 2 or header_size > limits.max_tensor_header_bytes:
        raise ArtifactIntegrityError("safetensor header exceeds configured cap")
    data_start = 8 + header_size
    if data_start > len(content):
        raise ArtifactIntegrityError("safetensor header is truncated")
    header = _bounded_json_loads(content[8:data_start], limits=limits)
    if not isinstance(header, dict):
        raise ArtifactIntegrityError("safetensor header must be a JSON object")
    return header, data_start


def safe_read_tensors(
    root: Path,
    key: str,
    *,
    expected: Mapping[str, ArrayDescriptor],
    limits: ReaderLimits = DEFAULT_READER_LIMITS,
    expected_sha256: str | None = None,
) -> dict[str, npt.NDArray[Any]]:
    """Preflight safetensor names, shapes, dtypes, and offsets before allocation."""

    if not expected:
        raise ValueError("expected tensor descriptors must not be empty")
    if len(expected) > limits.max_tensor_count:
        raise ValueError("expected tensor descriptors exceed configured cap")
    content = safe_read_bytes(
        root,
        key,
        max_bytes=limits.max_file_bytes,
        expected_sha256=expected_sha256,
    )
    header, data_start = _tensor_header(content, limits=limits)
    tensor_names = set(header) - {"__metadata__"}
    if tensor_names != set(expected):
        raise ArtifactIntegrityError("safetensor names do not exactly match descriptors")
    if len(tensor_names) > limits.max_tensor_count:
        raise ArtifactIntegrityError("safetensor count exceeds configured cap")
    metadata = header.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise ArtifactIntegrityError("safetensor metadata must contain only string pairs")

    intervals: list[tuple[int, int, str]] = []
    data_bytes = len(content) - data_start
    for name, descriptor in expected.items():
        entry = header[name]
        if not isinstance(entry, dict):
            raise ArtifactIntegrityError(f"safetensor descriptor for {name!r} is not an object")
        dtype_name = entry.get("dtype")
        shape_value = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype_name, str) or dtype_name not in _SAFETENSOR_DTYPES:
            raise ArtifactIntegrityError(f"unsupported safetensor dtype for {name!r}")
        if not isinstance(shape_value, list) or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape_value
        ):
            raise ArtifactIntegrityError(f"invalid safetensor shape for {name!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in offsets)
        ):
            raise ArtifactIntegrityError(f"invalid safetensor offsets for {name!r}")
        shape_items = cast(list[int], shape_value)
        offset_items = cast(list[int], offsets)
        start, end = offset_items
        dtype = _SAFETENSOR_DTYPES[dtype_name]
        shape = tuple(shape_items)
        if dtype != np.dtype(descriptor.dtype) or shape != descriptor.shape:
            raise ArtifactIntegrityError(
                f"safetensor {name!r} does not match its dtype/shape descriptor"
            )
        if descriptor.allow_pickle or descriptor.order != "C":
            raise ArtifactIntegrityError(f"safetensor {name!r} violates C-order/no-pickle contract")
        expected_bytes = math.prod(shape) * dtype.itemsize
        if start < 0 or end < start or end - start != expected_bytes or end > data_bytes:
            raise ArtifactIntegrityError(f"safetensor byte range is invalid for {name!r}")
        if expected_bytes > limits.max_member_bytes:
            raise ArtifactIntegrityError(f"safetensor {name!r} exceeds member byte cap")
        intervals.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise ArtifactIntegrityError(
                f"safetensor ranges overlap or contain a gap before {name!r}"
            )
        cursor = end
    if cursor != data_bytes or cursor > limits.max_total_uncompressed_bytes:
        raise ArtifactIntegrityError("safetensor data region is trailing or oversized")

    try:
        from safetensors.numpy import load

        loaded = load(content)
    except Exception as error:
        raise ArtifactIntegrityError("safetensor payload failed safe loading") from error
    arrays: dict[str, npt.NDArray[Any]] = {}
    for name, descriptor in expected.items():
        array = np.asarray(loaded[name])
        validate_array(
            array,
            name=name,
            dtype=np.dtype(descriptor.dtype),
            shape=descriptor.shape,
            finite=descriptor.finite,
        )
        array.setflags(write=False)
        arrays[name] = array
    return arrays


ARTIFACT_SCHEMA_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "artifact-envelope": ArtifactEnvelope,
        "lineage-node": LineageNode,
        "reader-limits": ReaderLimits,
    }
)


def artifact_schema_documents() -> dict[str, dict[str, object]]:
    """Generate schema-v1 JSON documents for safe artifact contracts."""

    result: dict[str, dict[str, object]] = {}
    for name, model in ARTIFACT_SCHEMA_MODELS.items():
        document = cast(dict[str, object], model.model_json_schema())
        document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        document["$id"] = f"https://github.com/AbdelStark/soufflerie/schemas/v1/{name}.json"
        result[name] = document
    return result


def rendered_artifact_schema_documents() -> dict[str, str]:
    """Render safe artifact schemas in the checked-in canonical format."""

    return {
        f"{name}.json": json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        for name, document in artifact_schema_documents().items()
    }


__all__ = [
    "ARTIFACT_SCHEMA_MODELS",
    "DEFAULT_LINEAGE_POLICY",
    "DEFAULT_READER_LIMITS",
    "ArtifactEnvelope",
    "LineageNode",
    "LineagePolicy",
    "ParentLink",
    "ParentTypeRule",
    "ReaderLimits",
    "SourceState",
    "VerifiedConsumerArtifacts",
    "artifact_content_sha256",
    "artifact_schema_documents",
    "capture_provenance",
    "capture_source_state",
    "rendered_artifact_schema_documents",
    "resolve_artifact_key",
    "safe_read_bytes",
    "safe_read_json",
    "safe_read_npz",
    "safe_read_parquet",
    "safe_read_tensors",
    "validate_release_provenance",
    "verify_consumer_identities",
    "verify_lineage",
]
