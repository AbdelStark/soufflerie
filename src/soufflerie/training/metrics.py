"""Authoritative schema-bound epoch JSONL evidence."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from soufflerie.config import Seed
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ContentId, Sha256, StrictFrozenModel, VersionedModel, canonical_json

MAX_EPOCH_JSONL_BYTES = 8 * 1024 * 1024
MAX_EPOCH_RECORD_BYTES = 16 * 1024
FiniteNonnegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
FinitePositive = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]


class EpochLossMetrics(StrictFrozenModel):
    """Fp64 epoch means for every raw term and the weighted objective."""

    u: FiniteNonnegative
    v: FiniteNonnegative
    rho: FiniteNonnegative
    obstacle: FiniteNonnegative
    cd: FiniteNonnegative
    total: FiniteNonnegative


class TrainingEpochRecord(VersionedModel):
    """One complete canonical training epoch; JSONL is the source of truth."""

    record_type: Literal["training-epoch"] = "training-epoch"
    experiment_id: ContentId
    dataset_id: ContentId
    config_digest: Sha256
    seed: Seed
    epoch: int = Field(ge=1)
    batches: int = Field(ge=1)
    samples: int = Field(ge=1)
    global_step: int = Field(ge=1)
    learning_rate: FinitePositive
    precision: Literal["bf16", "fp16"]
    reduction_dtype: Literal["float32"] = "float32"
    reporting_dtype: Literal["float64"] = "float64"
    device: Annotated[str, Field(pattern=r"^cuda:[0-9]+$")]
    device_name: Annotated[str, Field(min_length=1, max_length=256)]
    compute_capability: Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+$")]
    loss: EpochLossMetrics
    wall_seconds: FiniteNonnegative
    compute_seconds: FiniteNonnegative
    io_seconds: FiniteNonnegative
    gpu_seconds: FiniteNonnegative
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _accounting_is_coherent(self) -> Self:
        tolerance = max(1e-9, self.wall_seconds * 1e-9)
        if self.compute_seconds + self.io_seconds > self.wall_seconds + tolerance:
            raise ValueError("compute and I/O time cannot exceed epoch wall time")
        if not math.isclose(self.gpu_seconds, self.compute_seconds, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("single-GPU accounting must equal synchronized compute time")
        if self.peak_allocated_bytes > self.peak_reserved_bytes:
            raise ValueError("allocated CUDA memory cannot exceed reserved CUDA memory")
        if self.global_step < self.batches:
            raise ValueError("global_step cannot be smaller than completed batch count")
        return self


class EpochJsonlWriter:
    """Append-only validator that rejects identity drift, gaps, and partial JSONL."""

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a Path")
        self.path = path

    def read(self) -> tuple[TrainingEpochRecord, ...]:
        if self.path.is_symlink():
            raise ArtifactIntegrityError("TRAIN-8 JSONL: epoch log must not be a symlink")
        if not self.path.exists():
            return ()
        if not self.path.is_file():
            raise ArtifactIntegrityError("TRAIN-8 JSONL: epoch log must be one regular file")
        try:
            size = self.path.stat().st_size
            if size > MAX_EPOCH_JSONL_BYTES:
                raise ArtifactIntegrityError("TRAIN-8 JSONL: epoch log exceeds the byte cap")
            payload = self.path.read_text(encoding="utf-8")
        except ArtifactIntegrityError:
            raise
        except (OSError, UnicodeError) as error:
            raise ArtifactIntegrityError("TRAIN-8 JSONL: epoch log cannot be read") from error
        if payload and not payload.endswith("\n"):
            raise ArtifactIntegrityError("TRAIN-8 JSONL: epoch log has a partial final record")
        records: list[TrainingEpochRecord] = []
        for line in payload.splitlines():
            if not line or len(line.encode("utf-8")) > MAX_EPOCH_RECORD_BYTES:
                raise ArtifactIntegrityError("TRAIN-8 JSONL: epoch record is empty or oversized")
            try:
                records.append(TrainingEpochRecord.model_validate_json(line))
            except (TypeError, ValueError) as error:
                raise ArtifactIntegrityError("TRAIN-8 JSONL: epoch record is invalid") from error
        self._validate_sequence(tuple(records))
        return tuple(records)

    @staticmethod
    def _validate_sequence(records: tuple[TrainingEpochRecord, ...]) -> None:
        if not records:
            return
        first = records[0]
        identity = (first.experiment_id, first.dataset_id, first.config_digest, first.seed)
        previous_step = 0
        for expected_epoch, record in enumerate(records, start=1):
            if (
                record.experiment_id,
                record.dataset_id,
                record.config_digest,
                record.seed,
            ) != identity:
                raise ArtifactIntegrityError(
                    "TRAIN-8 JSONL: training identity changed within one log"
                )
            if record.epoch != expected_epoch:
                raise ArtifactIntegrityError("TRAIN-8 JSONL: epochs must be contiguous from one")
            if record.global_step != previous_step + record.batches:
                raise ArtifactIntegrityError(
                    "TRAIN-8 JSONL: global step accounting is not contiguous"
                )
            previous_step = record.global_step

    def append(self, record: TrainingEpochRecord) -> None:
        if not isinstance(record, TrainingEpochRecord):
            raise TypeError("record must be a TrainingEpochRecord")
        existing = self.read()
        self._validate_sequence((*existing, record))
        line = canonical_json(record)
        if len(line.encode("utf-8")) > MAX_EPOCH_RECORD_BYTES:
            raise ArtifactIntegrityError(
                "TRAIN-8 JSONL: rendered epoch record exceeds the byte cap"
            )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.parent.is_symlink() or not self.path.parent.is_dir():
                raise ArtifactIntegrityError("TRAIN-8 JSONL: epoch log parent must be a directory")
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{line}\n")
                handle.flush()
                os.fsync(handle.fileno())
        except ArtifactIntegrityError:
            raise
        except OSError as error:
            raise ArtifactIntegrityError("TRAIN-8 JSONL: epoch record append failed") from error


__all__ = [
    "MAX_EPOCH_JSONL_BYTES",
    "MAX_EPOCH_RECORD_BYTES",
    "EpochJsonlWriter",
    "EpochLossMetrics",
    "TrainingEpochRecord",
]
