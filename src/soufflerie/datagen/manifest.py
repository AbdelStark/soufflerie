"""Immutable Parquet dataset manifests and atomic local publication."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from errno import EEXIST, ENOTEMPTY
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Literal, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from soufflerie.artifacts import (
    DEFAULT_READER_LIMITS,
    ReaderLimits,
    safe_read_bytes,
    safe_read_json,
    safe_read_parquet,
)
from soufflerie.config import SweepConfig
from soufflerie.datagen._local_files import ensure_real_directory, fsync_directory, fsync_file
from soufflerie.datagen.design import (
    DESIGN_SAMPLE_COUNT,
    DesignPoint,
    case_config_for_point,
    design_sha256,
    sample_design,
    split_sha256,
)
from soufflerie.datagen.run_artifact import (
    LocalRunArtifactStore,
    RunArtifact,
    RunMetadata,
)
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import (
    ArtifactRef,
    ContentId,
    JsonValue,
    Sha256,
    Split,
    StrictFrozenModel,
    VersionedModel,
    canonical_sha256,
    sha256_bytes,
)

MANIFEST_ROW_GROUP_SIZE = 256
DATASET_PAYLOAD_LIMIT_BYTES = 2 * 1024**3
MANIFEST_MAX_FILE_BYTES = 16 * 1024 * 1024
DATASET_ROOT_PREFIX = "datasets"
DATASET_MANIFEST_NAME = "manifest.parquet"
DATASET_METADATA_NAME = "metadata.json"
DATASET_STATISTICS_NAME = "statistics.json"
DATASET_COMMIT_NAME = "COMMITTED"

_CONTENT_ID_PATTERN = re.compile(r"^[0-9a-f]{20}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")

MANIFEST_COLUMN_TYPES: Mapping[str, str] = MappingProxyType(
    {
        "schema_version": "int16",
        "dataset_id": "string",
        "case_id": "string",
        "design_id": "string",
        "split": "string",
        "aspect_ratio": "double",
        "rotation_deg": "double",
        "scale": "double",
        "reynolds": "double",
        "run_uri": "string",
        "run_digest": "string",
        "bytes": "int64",
        "cd": "double",
        "cl_mean": "double",
        "strouhal": "double",
        "solver_valid": "bool",
    }
)
MANIFEST_SCHEMA_DESCRIPTOR: tuple[dict[str, JsonValue], ...] = tuple(
    {
        "name": name,
        "arrow_type": arrow_type,
        "nullable": name == "strouhal",
    }
    for name, arrow_type in MANIFEST_COLUMN_TYPES.items()
)
MANIFEST_SCHEMA_SHA256 = canonical_sha256(MANIFEST_SCHEMA_DESCRIPTOR)

_PARQUET_METADATA_KEYS = frozenset(
    {
        "soufflerie.artifact_type",
        "soufflerie.config_sha256",
        "soufflerie.dataset_id",
        "soufflerie.dataset_sha256",
        "soufflerie.design_sha256",
        "soufflerie.lock_sha256",
        "soufflerie.row_group_size",
        "soufflerie.schema_sha256",
        "soufflerie.schema_version",
        "soufflerie.source_revision",
        "soufflerie.split_sha256",
        "soufflerie.writer",
        "soufflerie.writer_version",
    }
)


def _arrow_schema(*, metadata: Mapping[bytes, bytes] | None = None) -> pa.Schema:
    fields = (
        pa.field("schema_version", pa.int16(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("design_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("aspect_ratio", pa.float64(), nullable=False),
        pa.field("rotation_deg", pa.float64(), nullable=False),
        pa.field("scale", pa.float64(), nullable=False),
        pa.field("reynolds", pa.float64(), nullable=False),
        pa.field("run_uri", pa.string(), nullable=False),
        pa.field("run_digest", pa.string(), nullable=False),
        pa.field("bytes", pa.int64(), nullable=False),
        pa.field("cd", pa.float64(), nullable=False),
        pa.field("cl_mean", pa.float64(), nullable=False),
        pa.field("strouhal", pa.float64(), nullable=True),
        pa.field("solver_valid", pa.bool_(), nullable=False),
    )
    return pa.schema(fields, metadata=metadata)


class ManifestSplitCounts(StrictFrozenModel):
    train: Literal[600] = 600
    validation: Literal[200] = 200
    test: Literal[200] = 200


class ManifestRow(VersionedModel):
    """One authoritative training-membership row in the Parquet manifest."""

    dataset_id: ContentId
    case_id: ContentId
    design_id: ContentId
    split: Split
    aspect_ratio: float = Field(ge=0.5, le=1.0, allow_inf_nan=False)
    rotation_deg: float = Field(ge=0.0, le=30.0, allow_inf_nan=False)
    scale: float = Field(ge=0.75, le=1.25, allow_inf_nan=False)
    reynolds: float = Field(ge=40.0, le=300.0, allow_inf_nan=False)
    run_uri: str = Field(min_length=1, max_length=512)
    run_digest: Sha256
    bytes: int = Field(gt=0, lt=DATASET_PAYLOAD_LIMIT_BYTES)
    cd: float = Field(allow_inf_nan=False)
    cl_mean: float = Field(allow_inf_nan=False)
    strouhal: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    solver_valid: Literal[True] = True

    @model_validator(mode="after")
    def _run_reference_is_canonical(self) -> ManifestRow:
        expected = f"runs/{self.case_id}/{self.run_digest}"
        if self.run_uri != expected:
            raise ValueError("run_uri must be the canonical artifact-root-relative run key")
        return self

    def logical_identity(self) -> dict[str, JsonValue]:
        """Return row content that determines dataset identity."""

        return {
            "schema_version": 1,
            "case_id": self.case_id,
            "design_id": self.design_id,
            "split": self.split,
            "aspect_ratio": self.aspect_ratio,
            "rotation_deg": self.rotation_deg,
            "scale": self.scale,
            "reynolds": self.reynolds,
            "run_digest": self.run_digest,
            "bytes": self.bytes,
            "cd": self.cd,
            "cl_mean": self.cl_mean,
            "strouhal": self.strouhal,
            "solver_valid": True,
        }


class NumericColumnStatistics(StrictFrozenModel):
    """Finite fp64 summary for one manifest scalar column."""

    count: int = Field(ge=0, le=DESIGN_SAMPLE_COUNT)
    null_count: int = Field(ge=0, le=DESIGN_SAMPLE_COUNT)
    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)
    mean: float | None = Field(default=None, allow_inf_nan=False)
    standard_deviation: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    q05: float | None = Field(default=None, allow_inf_nan=False)
    q50: float | None = Field(default=None, allow_inf_nan=False)
    q95: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def _statistics_are_coherent(self) -> NumericColumnStatistics:
        values = (
            self.minimum,
            self.maximum,
            self.mean,
            self.standard_deviation,
            self.q05,
            self.q50,
            self.q95,
        )
        if self.count + self.null_count != DESIGN_SAMPLE_COUNT:
            raise ValueError("column count and null_count must total 1,000")
        if self.count == 0:
            if any(value is not None for value in values):
                raise ValueError("an empty numeric column cannot have scalar statistics")
            return self
        if any(value is None for value in values):
            raise ValueError("a populated numeric column requires complete statistics")
        minimum = cast(float, self.minimum)
        maximum = cast(float, self.maximum)
        mean = cast(float, self.mean)
        q05 = cast(float, self.q05)
        q50 = cast(float, self.q50)
        q95 = cast(float, self.q95)
        if not minimum <= q05 <= q50 <= q95 <= maximum:
            raise ValueError("numeric quantiles must be ordered within the observed range")
        if not minimum <= mean <= maximum:
            raise ValueError("numeric mean must lie within the observed range")
        return self


class DatasetColumnStatistics(StrictFrozenModel):
    aspect_ratio: NumericColumnStatistics
    rotation_deg: NumericColumnStatistics
    scale: NumericColumnStatistics
    reynolds: NumericColumnStatistics
    bytes: NumericColumnStatistics
    cd: NumericColumnStatistics
    cl_mean: NumericColumnStatistics
    strouhal: NumericColumnStatistics


class DatasetStatistics(VersionedModel):
    """Content-derived statistics for one complete manifest."""

    dataset_id: ContentId
    case_count: Literal[1000] = 1000
    split_counts: ManifestSplitCounts
    total_payload_bytes: int = Field(ge=1, lt=DATASET_PAYLOAD_LIMIT_BYTES)
    columns: DatasetColumnStatistics


class DatasetManifestMetadata(VersionedModel):
    """Identity, lineage, and physical codec metadata for a dataset manifest."""

    artifact_type: Literal["dataset"] = "dataset"
    dataset_id: ContentId
    dataset_sha256: Sha256
    config_sha256: Sha256
    design_sha256: Sha256
    split_sha256: Sha256
    manifest_schema_sha256: Sha256
    manifest_sha256: Sha256
    statistics_sha256: Sha256
    writer: Literal["pyarrow"] = "pyarrow"
    writer_version: str = Field(min_length=1, max_length=64)
    row_group_size: Literal[256] = 256
    case_count: Literal[1000] = 1000
    split_counts: ManifestSplitCounts
    total_payload_bytes: int = Field(ge=1, lt=DATASET_PAYLOAD_LIMIT_BYTES)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: Sha256
    parent_run_sha256: tuple[Sha256, ...] = Field(min_length=1000, max_length=1000)

    @field_validator("parent_run_sha256", mode="before")
    @classmethod
    def _json_parent_array_to_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _metadata_is_coherent(self) -> DatasetManifestMetadata:
        if self.dataset_id != self.dataset_sha256[:20]:
            raise ValueError("dataset_id must prefix the full logical dataset digest")
        if self.manifest_schema_sha256 != MANIFEST_SCHEMA_SHA256:
            raise ValueError("manifest schema digest does not match the v1 Arrow contract")
        if len(set(self.parent_run_sha256)) != DESIGN_SAMPLE_COUNT:
            raise ValueError("parent run digests must be unique")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedRunRecord:
    """Small manifest input obtained only after a full run-artifact open."""

    reference: ArtifactRef
    metadata: RunMetadata

    def __post_init__(self) -> None:
        try:
            reference = ArtifactRef.model_validate(self.reference.model_dump(mode="python"))
            metadata = RunMetadata.model_validate(self.metadata.model_dump(mode="python"))
        except Exception as error:
            raise ArtifactIntegrityError(
                "MANIFEST-1 PARENT: run reference or metadata failed strict validation"
            ) from error
        if reference != self.reference or metadata != self.metadata:
            raise ArtifactIntegrityError("MANIFEST-1 PARENT: normalized parent record changed")
        if self.reference.artifact_type != "run":
            raise ArtifactIntegrityError("MANIFEST-1 PARENT: expected a run ArtifactRef")
        if self.reference.sha256 != self.metadata.artifact_digest:
            raise ArtifactIntegrityError("MANIFEST-1 PARENT: run reference and metadata disagree")
        expected_uri = f"runs/{self.metadata.case_id}/{self.metadata.artifact_digest}"
        if self.reference.uri != expected_uri:
            raise ArtifactIntegrityError("MANIFEST-1 PARENT: run URI is not canonical")

    @classmethod
    def from_artifact(cls, artifact: RunArtifact) -> VerifiedRunRecord:
        if not isinstance(artifact, RunArtifact):
            raise TypeError("artifact must be a RunArtifact")
        return cls(reference=artifact.reference, metadata=artifact.metadata)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Validated logical rows plus deterministic Parquet and sidecar records."""

    rows: tuple[ManifestRow, ...]
    metadata: DatasetManifestMetadata
    statistics: DatasetStatistics
    parquet_bytes: bytes

    def __post_init__(self) -> None:
        if len(self.rows) != DESIGN_SAMPLE_COUNT:
            raise ArtifactIntegrityError("MANIFEST-2 COUNT: manifest requires exactly 1,000 rows")
        if self.metadata.manifest_sha256 != sha256_bytes(self.parquet_bytes):
            raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: manifest byte digest mismatch")
        if self.metadata.statistics_sha256 != sha256_bytes(_json_bytes(self.statistics)):
            raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: statistics byte digest mismatch")
        if self.statistics.dataset_id != self.metadata.dataset_id:
            raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: statistics dataset ID mismatch")
        if tuple(row.run_digest for row in self.rows) != self.metadata.parent_run_sha256:
            raise ArtifactIntegrityError("MANIFEST-7 LINEAGE: parent run digest order mismatch")


def _json_bytes(model: VersionedModel) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _numeric_statistics(values: Sequence[float | int | None]) -> NumericColumnStatistics:
    present = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    null_count = len(values) - int(present.size)
    if present.size == 0:
        return NumericColumnStatistics(count=0, null_count=null_count)
    quantiles = np.quantile(present, (0.05, 0.5, 0.95), method="linear")
    return NumericColumnStatistics(
        count=int(present.size),
        null_count=null_count,
        minimum=float(np.min(present)),
        maximum=float(np.max(present)),
        mean=float(np.mean(present, dtype=np.float64)),
        standard_deviation=float(np.std(present, dtype=np.float64)),
        q05=float(quantiles[0]),
        q50=float(quantiles[1]),
        q95=float(quantiles[2]),
    )


def _statistics(rows: Sequence[ManifestRow], dataset_id: str) -> DatasetStatistics:
    total_payload_bytes = sum(row.bytes for row in rows)
    return DatasetStatistics(
        dataset_id=dataset_id,
        split_counts=ManifestSplitCounts(),
        total_payload_bytes=total_payload_bytes,
        columns=DatasetColumnStatistics(
            aspect_ratio=_numeric_statistics([row.aspect_ratio for row in rows]),
            rotation_deg=_numeric_statistics([row.rotation_deg for row in rows]),
            scale=_numeric_statistics([row.scale for row in rows]),
            reynolds=_numeric_statistics([row.reynolds for row in rows]),
            bytes=_numeric_statistics([row.bytes for row in rows]),
            cd=_numeric_statistics([row.cd for row in rows]),
            cl_mean=_numeric_statistics([row.cl_mean for row in rows]),
            strouhal=_numeric_statistics([row.strouhal for row in rows]),
        ),
    )


def _dataset_digest(
    rows: Sequence[ManifestRow],
    *,
    config_sha256: str,
    design_digest: str,
    split_digest: str,
) -> str:
    logical_rows: list[JsonValue] = [row.logical_identity() for row in rows]
    return canonical_sha256(
        {
            "schema_version": 1,
            "manifest_schema": MANIFEST_SCHEMA_DESCRIPTOR,
            "manifest_schema_sha256": MANIFEST_SCHEMA_SHA256,
            "config_sha256": config_sha256,
            "design_sha256": design_digest,
            "split_sha256": split_digest,
            "rows": logical_rows,
        }
    )


def _parquet_metadata(
    *,
    dataset_sha256: str,
    config_sha256: str,
    design_digest: str,
    split_digest: str,
    source_revision: str,
    lock_sha256: str,
    writer_version: str,
) -> dict[bytes, bytes]:
    values = {
        "soufflerie.artifact_type": "dataset-manifest",
        "soufflerie.config_sha256": config_sha256,
        "soufflerie.dataset_id": dataset_sha256[:20],
        "soufflerie.dataset_sha256": dataset_sha256,
        "soufflerie.design_sha256": design_digest,
        "soufflerie.lock_sha256": lock_sha256,
        "soufflerie.row_group_size": str(MANIFEST_ROW_GROUP_SIZE),
        "soufflerie.schema_sha256": MANIFEST_SCHEMA_SHA256,
        "soufflerie.schema_version": "1",
        "soufflerie.source_revision": source_revision,
        "soufflerie.split_sha256": split_digest,
        "soufflerie.writer": "pyarrow",
        "soufflerie.writer_version": writer_version,
    }
    return {key.encode("ascii"): value.encode("ascii") for key, value in values.items()}


def _encode_parquet(
    rows: Sequence[ManifestRow],
    *,
    metadata: Mapping[bytes, bytes],
) -> bytes:
    values = {name: [getattr(row, name) for row in rows] for name in MANIFEST_COLUMN_TYPES}
    schema = _arrow_schema(metadata=metadata)
    arrays = [pa.array(values[field.name], type=field.type) for field in schema]
    table = pa.Table.from_arrays(arrays, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        row_group_size=MANIFEST_ROW_GROUP_SIZE,
        version="2.6",
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        store_schema=True,
    )
    content = cast(bytes, sink.getvalue().to_pybytes())
    if len(content) > MANIFEST_MAX_FILE_BYTES:
        raise ArtifactIntegrityError("MANIFEST-5 SIZE: Parquet manifest exceeds its byte cap")
    return content


def _validate_row_set(rows: Sequence[ManifestRow]) -> None:
    if len(rows) != DESIGN_SAMPLE_COUNT:
        raise ArtifactIntegrityError("MANIFEST-2 COUNT: manifest requires exactly 1,000 rows")
    if tuple(row.design_id for row in rows) != tuple(sorted(row.design_id for row in rows)):
        raise ArtifactIntegrityError("MANIFEST-2 ORDER: rows must sort by design_id")
    for name, values in (
        ("case_id", [row.case_id for row in rows]),
        ("design_id", [row.design_id for row in rows]),
        ("run_digest", [row.run_digest for row in rows]),
    ):
        if len(set(values)) != DESIGN_SAMPLE_COUNT:
            raise ArtifactIntegrityError(f"MANIFEST-2 UNIQUE: {name} values must be unique")
    counts = Counter(row.split for row in rows)
    if counts != Counter({"train": 600, "validation": 200, "test": 200}):
        raise ArtifactIntegrityError("MANIFEST-6 SPLIT: expected exact 600/200/200 membership")
    dataset_ids = {row.dataset_id for row in rows}
    if len(dataset_ids) != 1:
        raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: rows must share one dataset ID")
    total_payload_bytes = sum(row.bytes for row in rows)
    if total_payload_bytes >= DATASET_PAYLOAD_LIMIT_BYTES:
        raise ArtifactIntegrityError("MANIFEST-5 SIZE: total run payload must be below 2 GiB")


def _assemble_manifest(
    runs: Sequence[VerifiedRunRecord],
    *,
    config: SweepConfig,
    points: Sequence[DesignPoint] | None = None,
) -> DatasetManifest:
    if not isinstance(config, SweepConfig):
        raise TypeError("config must be a SweepConfig")
    if len(runs) != DESIGN_SAMPLE_COUNT:
        raise ArtifactIntegrityError("MANIFEST-2 COUNT: exactly 1,000 verified runs are required")
    if any(not isinstance(run, VerifiedRunRecord) for run in runs):
        raise TypeError("runs must contain VerifiedRunRecord instances")
    design_points = tuple(points) if points is not None else sample_design(config)
    if len(design_points) != DESIGN_SAMPLE_COUNT:
        raise ArtifactIntegrityError("MANIFEST-2 COUNT: canonical design must contain 1,000 points")
    expected_by_design = {point.design_id: point for point in design_points}
    if len(expected_by_design) != DESIGN_SAMPLE_COUNT:
        raise ArtifactIntegrityError("MANIFEST-2 UNIQUE: canonical design IDs must be unique")

    by_design: dict[str, VerifiedRunRecord] = {}
    for run in runs:
        design_id = run.metadata.design_id
        if design_id in by_design:
            raise ArtifactIntegrityError("MANIFEST-2 UNIQUE: duplicate design run")
        by_design[design_id] = run
    if set(by_design) != set(expected_by_design):
        raise ArtifactIntegrityError("MANIFEST-6 DESIGN: runs do not match the canonical design")

    source_revisions = {run.metadata.provenance.source_revision for run in runs}
    lock_digests = {run.metadata.provenance.lock_sha256 for run in runs}
    if len(source_revisions) != 1 or any(run.metadata.provenance.source_dirty for run in runs):
        raise ArtifactIntegrityError(
            "MANIFEST-7 PROVENANCE: runs require one clean source revision"
        )
    if len(lock_digests) != 1:
        raise ArtifactIntegrityError("MANIFEST-7 PROVENANCE: runs require one lock digest")
    source_revision = next(iter(source_revisions))
    lock_sha256 = next(iter(lock_digests))

    row_values: list[dict[str, object]] = []
    for design_id, point in sorted(expected_by_design.items()):
        run = by_design[design_id]
        run_metadata = run.metadata
        expected_case = case_config_for_point(point, config)
        if run_metadata.case != expected_case or run_metadata.case_id != expected_case.case_id:
            raise ArtifactIntegrityError(
                "MANIFEST-6 DESIGN: run case does not match canonical numerical controls"
            )
        if run_metadata.split != point.split:
            raise ArtifactIntegrityError("MANIFEST-6 SPLIT: run split does not match design point")
        row_values.append(
            {
                "schema_version": 1,
                "case_id": run_metadata.case_id,
                "design_id": run_metadata.design_id,
                "split": run_metadata.split,
                "aspect_ratio": run_metadata.case.shape.aspect_ratio,
                "rotation_deg": run_metadata.case.shape.rotation_deg,
                "scale": run_metadata.case.shape.scale,
                "reynolds": run_metadata.case.reynolds,
                "run_uri": run.reference.uri,
                "run_digest": run.reference.sha256,
                "bytes": run.reference.size_bytes,
                "cd": run_metadata.cd,
                "cl_mean": run_metadata.cl_mean,
                "strouhal": run_metadata.strouhal,
                "solver_valid": True,
            }
        )

    design_digest = design_sha256(design_points)
    split_digest = split_sha256(design_points)
    provisional = tuple(
        ManifestRow(dataset_id="0" * 20, **values)  # type: ignore[arg-type]
        for values in row_values
    )
    dataset_sha256 = _dataset_digest(
        provisional,
        config_sha256=config.config_digest,
        design_digest=design_digest,
        split_digest=split_digest,
    )
    rows = tuple(
        ManifestRow(dataset_id=dataset_sha256[:20], **values)  # type: ignore[arg-type]
        for values in row_values
    )
    _validate_row_set(rows)
    writer_version = str(pa.__version__)
    parquet_bytes = _encode_parquet(
        rows,
        metadata=_parquet_metadata(
            dataset_sha256=dataset_sha256,
            config_sha256=config.config_digest,
            design_digest=design_digest,
            split_digest=split_digest,
            source_revision=source_revision,
            lock_sha256=lock_sha256,
            writer_version=writer_version,
        ),
    )
    statistics = _statistics(rows, dataset_sha256[:20])
    dataset_metadata = DatasetManifestMetadata(
        dataset_id=dataset_sha256[:20],
        dataset_sha256=dataset_sha256,
        config_sha256=config.config_digest,
        design_sha256=design_digest,
        split_sha256=split_digest,
        manifest_schema_sha256=MANIFEST_SCHEMA_SHA256,
        manifest_sha256=sha256_bytes(parquet_bytes),
        statistics_sha256=sha256_bytes(_json_bytes(statistics)),
        writer_version=writer_version,
        split_counts=ManifestSplitCounts(),
        total_payload_bytes=sum(row.bytes for row in rows),
        source_revision=source_revision,
        lock_sha256=lock_sha256,
        parent_run_sha256=tuple(row.run_digest for row in rows),
    )
    return DatasetManifest(
        rows=rows,
        metadata=dataset_metadata,
        statistics=statistics,
        parquet_bytes=parquet_bytes,
    )


def build_manifest(
    run_root: Path,
    *,
    config: SweepConfig,
    run_references: Sequence[ArtifactRef],
) -> DatasetManifest:
    """Open exactly the supplied parents, then build the canonical manifest."""

    if len(run_references) != DESIGN_SAMPLE_COUNT:
        raise ArtifactIntegrityError("MANIFEST-2 COUNT: exactly 1,000 run references are required")
    if any(not isinstance(reference, ArtifactRef) for reference in run_references):
        raise TypeError("run_references must contain ArtifactRef instances")
    if len({reference.sha256 for reference in run_references}) != DESIGN_SAMPLE_COUNT:
        raise ArtifactIntegrityError("MANIFEST-2 UNIQUE: run references must be unique")
    store = LocalRunArtifactStore(run_root)
    runs = tuple(
        VerifiedRunRecord.from_artifact(store.open_run(reference)) for reference in run_references
    )
    return _assemble_manifest(runs, config=config)


def _decode_parquet_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, str]:
    if metadata is None:
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: Parquet metadata is missing")
    try:
        decoded = {key.decode("ascii"): value.decode("ascii") for key, value in metadata.items()}
    except UnicodeDecodeError as error:
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: metadata must be ASCII") from error
    if set(decoded) != _PARQUET_METADATA_KEYS:
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: metadata keys do not match v1")
    return decoded


def load_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
    limits: ReaderLimits = DEFAULT_READER_LIMITS,
) -> DatasetManifest:
    """Safely validate one standalone Parquet manifest and its logical identity."""

    root = path.parent.resolve()
    key = path.name
    manifest_limits = limits.model_copy(
        update={
            "max_file_bytes": min(limits.max_file_bytes, MANIFEST_MAX_FILE_BYTES),
            "max_parquet_rows": min(limits.max_parquet_rows, DESIGN_SAMPLE_COUNT),
            "max_parquet_columns": min(limits.max_parquet_columns, len(MANIFEST_COLUMN_TYPES)),
            "max_parquet_row_groups": min(limits.max_parquet_row_groups, 4),
            "max_total_uncompressed_bytes": min(
                limits.max_total_uncompressed_bytes,
                MANIFEST_MAX_FILE_BYTES,
            ),
        }
    )
    content = safe_read_bytes(
        root,
        key,
        max_bytes=manifest_limits.max_file_bytes,
        expected_sha256=expected_sha256,
    )
    table = safe_read_parquet(
        root,
        key,
        expected_columns=MANIFEST_COLUMN_TYPES,
        limits=manifest_limits,
        expected_sha256=expected_sha256,
    )
    reader = pq.ParquetFile(pa.BufferReader(content))
    if reader.schema_arrow.remove_metadata() != _arrow_schema().remove_metadata():
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: Arrow field contract mismatch")
    expected_group_sizes = (256, 256, 256, 232)
    actual_group_sizes = tuple(
        reader.metadata.row_group(index).num_rows for index in range(reader.metadata.num_row_groups)
    )
    if actual_group_sizes != expected_group_sizes:
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: row groups must be 256/256/256/232")
    metadata_values = _decode_parquet_metadata(reader.schema_arrow.metadata)
    if metadata_values["soufflerie.schema_version"] != "1":
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: unsupported manifest schema version")
    if metadata_values["soufflerie.artifact_type"] != "dataset-manifest":
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: wrong artifact type")
    if metadata_values["soufflerie.schema_sha256"] != MANIFEST_SCHEMA_SHA256:
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: schema fingerprint mismatch")
    if metadata_values["soufflerie.row_group_size"] != str(MANIFEST_ROW_GROUP_SIZE):
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: row-group policy mismatch")
    if metadata_values["soufflerie.writer"] != "pyarrow":
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: unsupported writer")

    try:
        rows = tuple(ManifestRow.model_validate(item) for item in table.to_pylist())
    except Exception as error:
        raise ArtifactIntegrityError("MANIFEST-1 SCHEMA: row validation failed") from error
    _validate_row_set(rows)
    dataset_sha256 = _dataset_digest(
        rows,
        config_sha256=metadata_values["soufflerie.config_sha256"],
        design_digest=metadata_values["soufflerie.design_sha256"],
        split_digest=metadata_values["soufflerie.split_sha256"],
    )
    if metadata_values["soufflerie.dataset_sha256"] != dataset_sha256:
        raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: logical dataset digest mismatch")
    if metadata_values["soufflerie.dataset_id"] != dataset_sha256[:20]:
        raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: dataset ID metadata mismatch")
    if {row.dataset_id for row in rows} != {dataset_sha256[:20]}:
        raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: row dataset IDs mismatch")
    source_revision = metadata_values["soufflerie.source_revision"]
    lock_sha256 = metadata_values["soufflerie.lock_sha256"]
    if _SOURCE_REVISION_PATTERN.fullmatch(source_revision) is None:
        raise ArtifactIntegrityError("MANIFEST-7 PROVENANCE: invalid source revision")
    if _SHA256_PATTERN.fullmatch(lock_sha256) is None:
        raise ArtifactIntegrityError("MANIFEST-7 PROVENANCE: invalid lock digest")
    for name in ("config_sha256", "design_sha256", "split_sha256"):
        if _SHA256_PATTERN.fullmatch(metadata_values[f"soufflerie.{name}"]) is None:
            raise ArtifactIntegrityError(f"MANIFEST-4 IDENTITY: invalid {name}")

    statistics = _statistics(rows, dataset_sha256[:20])
    metadata = DatasetManifestMetadata(
        dataset_id=dataset_sha256[:20],
        dataset_sha256=dataset_sha256,
        config_sha256=metadata_values["soufflerie.config_sha256"],
        design_sha256=metadata_values["soufflerie.design_sha256"],
        split_sha256=metadata_values["soufflerie.split_sha256"],
        manifest_schema_sha256=MANIFEST_SCHEMA_SHA256,
        manifest_sha256=sha256_bytes(content),
        statistics_sha256=sha256_bytes(_json_bytes(statistics)),
        writer_version=metadata_values["soufflerie.writer_version"],
        split_counts=ManifestSplitCounts(),
        total_payload_bytes=sum(row.bytes for row in rows),
        source_revision=source_revision,
        lock_sha256=lock_sha256,
        parent_run_sha256=tuple(row.run_digest for row in rows),
    )
    return DatasetManifest(
        rows=rows,
        metadata=metadata,
        statistics=statistics,
        parquet_bytes=content,
    )


class LocalDatasetArtifactStore:
    """Stage-verify-commit local publication for complete dataset manifests."""

    expected_files: ClassVar[frozenset[str]] = frozenset(
        {
            DATASET_MANIFEST_NAME,
            DATASET_METADATA_NAME,
            DATASET_STATISTICS_NAME,
            DATASET_COMMIT_NAME,
        }
    )

    def __init__(
        self,
        root: Path,
        *,
        limits: ReaderLimits = DEFAULT_READER_LIMITS,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.limits = limits
        self._fault_injector = fault_injector
        self.root.mkdir(parents=True, exist_ok=True)

    def _inject(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)

    def _committed_size(self, uri: str) -> int:
        return sum(
            len(
                safe_read_bytes(
                    self.root,
                    f"{uri}/{name}",
                    max_bytes=(65 if name == DATASET_COMMIT_NAME else MANIFEST_MAX_FILE_BYTES),
                )
            )
            for name in self.expected_files
        )

    def publish(self, manifest: DatasetManifest) -> ArtifactRef:
        if not isinstance(manifest, DatasetManifest):
            raise TypeError("manifest must be a DatasetManifest")
        metadata_bytes = _json_bytes(manifest.metadata)
        statistics_bytes = _json_bytes(manifest.statistics)
        metadata_sha256 = sha256_bytes(metadata_bytes)
        uri = f"{DATASET_ROOT_PREFIX}/{manifest.metadata.dataset_id}"
        reference = ArtifactRef(
            artifact_type="dataset",
            artifact_id=manifest.metadata.dataset_id,
            sha256=manifest.metadata.dataset_sha256,
            size_bytes=(
                len(manifest.parquet_bytes) + len(metadata_bytes) + len(statistics_bytes) + 65
            ),
            uri=uri,
        )
        target = self.root / uri
        if target.exists() or target.is_symlink():
            return self.open(
                reference.model_copy(update={"size_bytes": self._committed_size(uri)})
            ).reference

        staging_parent = ensure_real_directory(self.root, ".staging", "datasets")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f"{manifest.metadata.dataset_id}-",
                dir=staging_parent,
            )
        )
        try:
            files = {
                DATASET_MANIFEST_NAME: manifest.parquet_bytes,
                DATASET_METADATA_NAME: metadata_bytes,
                DATASET_STATISTICS_NAME: statistics_bytes,
            }
            for name, content in files.items():
                path = staging / name
                path.write_bytes(content)
                fsync_file(path)
            self._inject("members_written")

            loaded = load_manifest(
                staging / DATASET_MANIFEST_NAME,
                expected_sha256=manifest.metadata.manifest_sha256,
                limits=self.limits,
            )
            loaded_metadata = safe_read_json(
                staging,
                DATASET_METADATA_NAME,
                model=DatasetManifestMetadata,
                expected_sha256=metadata_sha256,
            )
            loaded_statistics = safe_read_json(
                staging,
                DATASET_STATISTICS_NAME,
                model=DatasetStatistics,
                expected_sha256=manifest.metadata.statistics_sha256,
            )
            if loaded.metadata != loaded_metadata or loaded.statistics != loaded_statistics:
                raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: staged sidecars disagree")
            self._inject("verified")

            commit = staging / DATASET_COMMIT_NAME
            commit.write_text(metadata_sha256 + "\n", encoding="ascii")
            fsync_file(commit)
            fsync_directory(staging)
            self._inject("committed")

            parent = ensure_real_directory(self.root, DATASET_ROOT_PREFIX)
            try:
                os.rename(staging, target)
            except OSError as error:
                if error.errno not in {EEXIST, ENOTEMPTY}:
                    raise
                reference = self.open(
                    reference.model_copy(update={"size_bytes": self._committed_size(uri)})
                ).reference
            fsync_directory(parent)
            self._inject("published")
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return reference

    def open(self, reference: ArtifactRef) -> PublishedDataset:
        if not isinstance(reference, ArtifactRef) or reference.artifact_type != "dataset":
            raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: expected a dataset ArtifactRef")
        expected_uri = f"{DATASET_ROOT_PREFIX}/{reference.artifact_id}"
        if reference.uri != expected_uri or reference.sha256[:20] != reference.artifact_id:
            raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: dataset reference is incoherent")
        marker = safe_read_bytes(
            self.root,
            f"{reference.uri}/{DATASET_COMMIT_NAME}",
            max_bytes=65,
        )
        try:
            metadata_digest = marker.decode("ascii").removesuffix("\n")
        except UnicodeDecodeError as error:
            raise ArtifactIntegrityError("MANIFEST-3 COMMIT: marker is not ASCII") from error
        if len(marker) != 65 or _SHA256_PATTERN.fullmatch(metadata_digest) is None:
            raise ArtifactIntegrityError("MANIFEST-3 COMMIT: marker must contain a metadata digest")
        metadata = safe_read_json(
            self.root,
            f"{reference.uri}/{DATASET_METADATA_NAME}",
            model=DatasetManifestMetadata,
            expected_sha256=metadata_digest,
        )
        manifest = load_manifest(
            self.root / reference.uri / DATASET_MANIFEST_NAME,
            expected_sha256=metadata.manifest_sha256,
            limits=self.limits,
        )
        statistics = safe_read_json(
            self.root,
            f"{reference.uri}/{DATASET_STATISTICS_NAME}",
            model=DatasetStatistics,
            expected_sha256=metadata.statistics_sha256,
        )
        if manifest.metadata != metadata or manifest.statistics != statistics:
            raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: published sidecars disagree")
        actual_size = self._committed_size(reference.uri)
        if actual_size != reference.size_bytes:
            raise ArtifactIntegrityError("MANIFEST-5 SIZE: reference byte count mismatch")
        if reference.sha256 != metadata.dataset_sha256:
            raise ArtifactIntegrityError("MANIFEST-4 IDENTITY: reference digest mismatch")
        return PublishedDataset(reference=reference, manifest=manifest)


@dataclass(frozen=True, slots=True)
class PublishedDataset:
    reference: ArtifactRef
    manifest: DatasetManifest


__all__ = [
    "DATASET_PAYLOAD_LIMIT_BYTES",
    "MANIFEST_COLUMN_TYPES",
    "MANIFEST_ROW_GROUP_SIZE",
    "MANIFEST_SCHEMA_SHA256",
    "DatasetManifest",
    "DatasetManifestMetadata",
    "DatasetStatistics",
    "LocalDatasetArtifactStore",
    "ManifestRow",
    "NumericColumnStatistics",
    "PublishedDataset",
    "VerifiedRunRecord",
    "build_manifest",
    "load_manifest",
]
