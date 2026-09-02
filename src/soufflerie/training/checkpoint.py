"""Private epoch checkpoint publication, verified resume, and validation selection."""

from __future__ import annotations

import io
import json
import math
import os
import random
import re
import shutil
import stat
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Literal, Protocol, Self, cast

import numpy as np
from pydantic import Field, StringConstraints, model_validator

from soufflerie.config import Seed
from soufflerie.datagen._local_files import fsync_directory, fsync_file
from soufflerie.errors import ArtifactIntegrityError, InternalInvariantError
from soufflerie.schemas import (
    ArtifactRef,
    ContentId,
    Sha256,
    StrictFrozenModel,
    VersionedModel,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
    verify_sha256,
)
from soufflerie.surrogate.bundle import (
    LocalModelBundleStore,
    ModelCardMetadata,
    build_model_bundle,
    snapshot_fno_weights,
)
from soufflerie.surrogate.fno import FnoPredictor
from soufflerie.surrogate.preprocessing import PreprocessingStatistics
from soufflerie.training.loop import TrainingSession

CHECKPOINT_ROOT = "training"
MODEL_STATE_NAME = "model.pt"
OPTIMIZER_STATE_NAME = "optimizer.pt"
SCHEDULER_STATE_NAME = "scheduler.json"
SCALER_STATE_NAME = "scaler.pt"
RNG_JSON_NAME = "rng.json"
RNG_TORCH_NAME = "rng.pt"
METADATA_NAME = "metadata.json"
COMMIT_NAME = "COMMITTED"
MAX_METADATA_BYTES = 64 * 1024
MAX_JSON_STATE_BYTES = 256 * 1024
MAX_MODEL_STATE_BYTES = 256 * 1024 * 1024
MAX_OPTIMIZER_STATE_BYTES = 1024 * 1024 * 1024
MAX_SCALER_STATE_BYTES = 1024 * 1024
MAX_RNG_STATE_BYTES = 16 * 1024 * 1024
SourceRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Device = Annotated[str, StringConstraints(pattern=r"^cuda:[0-9]+$")]
FiniteNonnegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class SchedulerState(StrictFrozenModel):
    """Formula scheduler state required to continue at one epoch boundary."""

    completed_epoch: int = Field(ge=1, le=100)
    next_epoch: int = Field(ge=2, le=101)
    learning_rate: float = Field(gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _next_is_contiguous(self) -> Self:
        if self.next_epoch != self.completed_epoch + 1:
            raise ValueError("scheduler next_epoch must follow completed_epoch")
        return self


class RngStateMetadata(StrictFrozenModel):
    """JSON-safe Python and NumPy RNG state; framework tensors are separate."""

    python_version: int
    python_state: tuple[int, ...] = Field(min_length=1, max_length=4096)
    python_gaussian: float | None = Field(default=None, allow_inf_nan=False)
    numpy_kind: Literal["MT19937"]
    numpy_state: tuple[int, ...] = Field(min_length=624, max_length=624)
    numpy_position: int = Field(ge=0, le=624)
    numpy_has_gaussian: Literal[0, 1]
    numpy_cached_gaussian: float = Field(allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _normalize_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("python_state", "numpy_state"):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized


class TrainingCheckpointMetadata(VersionedModel):
    """Commit-bound identity and member digests for private trusted state."""

    artifact_type: Literal["training-checkpoint"] = "training-checkpoint"
    checkpoint_id: ContentId
    checkpoint_sha256: Sha256
    experiment_id: ContentId
    dataset_id: ContentId
    dataset_sha256: Sha256
    architecture_sha256: Sha256
    config_digest: Sha256
    code_revision: SourceRevision
    lock_digest: Sha256
    seed: Seed
    completed_epoch: int = Field(ge=1, le=100)
    global_step: int = Field(ge=1)
    device: Device
    device_name: Annotated[str, Field(min_length=1, max_length=256)]
    compute_capability: Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+$")]
    precision: Literal["bf16", "fp16"]
    model_digest: Sha256
    model_bytes: int = Field(ge=1, le=MAX_MODEL_STATE_BYTES)
    optimizer_digest: Sha256
    optimizer_bytes: int = Field(ge=1, le=MAX_OPTIMIZER_STATE_BYTES)
    scheduler_digest: Sha256
    scheduler_bytes: int = Field(ge=1, le=MAX_JSON_STATE_BYTES)
    scaler_digest: Sha256 | None
    scaler_bytes: int | None = Field(default=None, ge=1, le=MAX_SCALER_STATE_BYTES)
    rng_state_digest: Sha256
    rng_json_digest: Sha256
    rng_json_bytes: int = Field(ge=1, le=MAX_JSON_STATE_BYTES)
    rng_torch_digest: Sha256
    rng_torch_bytes: int = Field(ge=1, le=MAX_RNG_STATE_BYTES)

    def logical_identity(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"checkpoint_id", "checkpoint_sha256"})

    @model_validator(mode="after")
    def _identity_is_coherent(self) -> Self:
        if self.dataset_id != self.dataset_sha256[:20]:
            raise ValueError("dataset_id must prefix dataset_sha256")
        if (self.scaler_digest is None) != (self.scaler_bytes is None):
            raise ValueError("scaler digest and size must be present together")
        if self.precision == "fp16" and self.scaler_digest is None:
            raise ValueError("fp16 checkpoints require scaler state")
        if self.precision == "bf16" and self.scaler_digest is not None:
            raise ValueError("bf16 checkpoints must not contain scaler state")
        expected_rng = canonical_sha256(
            {"json": self.rng_json_digest, "torch": self.rng_torch_digest}
        )
        if self.rng_state_digest != expected_rng:
            raise ValueError("rng_state_digest does not bind both RNG members")
        expected = canonical_sha256(self.logical_identity())
        if self.checkpoint_sha256 != expected or self.checkpoint_id != expected[:20]:
            raise ValueError("checkpoint identity does not bind all state members")
        return self

    @classmethod
    def create(cls, **values: object) -> TrainingCheckpointMetadata:
        logical = {"schema_version": 1, "artifact_type": "training-checkpoint", **values}
        digest = canonical_sha256(logical)
        return cls.model_validate(
            {"checkpoint_id": digest[:20], "checkpoint_sha256": digest, **logical}
        )


@dataclass(frozen=True, slots=True)
class CheckpointPayload:
    metadata: TrainingCheckpointMetadata
    model_state: bytes
    optimizer_state: bytes
    scheduler_state: bytes
    rng_json: bytes
    rng_torch_state: bytes
    scaler_state: bytes | None = None

    def members(self) -> dict[str, bytes]:
        result = {
            MODEL_STATE_NAME: self.model_state,
            OPTIMIZER_STATE_NAME: self.optimizer_state,
            SCHEDULER_STATE_NAME: self.scheduler_state,
            RNG_JSON_NAME: self.rng_json,
            RNG_TORCH_NAME: self.rng_torch_state,
            METADATA_NAME: canonical_json_bytes(self.metadata),
        }
        if self.scaler_state is not None:
            result[SCALER_STATE_NAME] = self.scaler_state
        return result


@dataclass(frozen=True, slots=True)
class PublishedCheckpoint:
    root: Path
    metadata: TrainingCheckpointMetadata
    model_state: bytes
    optimizer_state: bytes
    scheduler: SchedulerState
    rng: RngStateMetadata
    rng_torch_state: bytes
    scaler_state: bytes | None


class StateCodec(Protocol):
    def encode(self, value: object) -> bytes: ...

    def decode(self, content: bytes, *, map_location: str) -> object: ...


@dataclass(frozen=True, slots=True)
class TorchStateCodec:
    torch: ModuleType

    def encode(self, value: object) -> bytes:
        output = io.BytesIO()
        try:
            cast(Any, self.torch).save(value, output)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ArtifactIntegrityError("TRAIN-11 CODEC: trusted state encoding failed") from error
        return output.getvalue()

    def decode(self, content: bytes, *, map_location: str) -> object:
        try:
            return cast(Any, self.torch).load(
                io.BytesIO(content), map_location=map_location, weights_only=True
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise ArtifactIntegrityError("TRAIN-11 CODEC: trusted state decoding failed") from error


def _rng_metadata() -> RngStateMetadata:
    python_version, python_state, python_gaussian = random.getstate()
    numpy_kind, numpy_state, position, has_gaussian, cached = np.random.get_state()
    return RngStateMetadata(
        python_version=python_version,
        python_state=tuple(int(value) for value in python_state),
        python_gaussian=python_gaussian,
        numpy_kind=cast(Literal["MT19937"], numpy_kind),
        numpy_state=tuple(int(value) for value in numpy_state),
        numpy_position=int(position),
        numpy_has_gaussian=cast(Literal[0, 1], int(has_gaussian)),
        numpy_cached_gaussian=float(cached),
    )


def _checked_member(content: bytes, *, maximum: int, name: str) -> bytes:
    if not isinstance(content, bytes) or not 0 < len(content) <= maximum:
        raise ArtifactIntegrityError(f"TRAIN-11 MEMBER: {name} has an invalid size")
    return content


def capture_training_checkpoint(
    session: TrainingSession,
    *,
    completed_epoch: int,
    dataset_sha256: str,
    architecture_sha256: str,
    code_revision: str,
    lock_digest: str,
    codec: StateCodec | None = None,
) -> CheckpointPayload:
    """Snapshot complete state only after a finished epoch."""

    if not isinstance(session, TrainingSession):
        raise TypeError("session must be a TrainingSession")
    if not 1 <= completed_epoch <= session.config.epochs:
        raise ArtifactIntegrityError("TRAIN-11 EPOCH: completed epoch is outside the config")
    torch = cast(Any, session.torch)
    resolved = codec or TorchStateCodec(session.torch)
    try:
        model_state = resolved.encode(cast(Any, session.model).state_dict())
        optimizer_state = resolved.encode(session.optimizer.state_dict())
        scaler_state = (
            None if session.scaler is None else resolved.encode(session.scaler.state_dict())
        )
        rng_torch = resolved.encode(
            {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise InternalInvariantError("TRAIN-11 SNAPSHOT: training state is unavailable") from error
    scheduler = SchedulerState(
        completed_epoch=completed_epoch,
        next_epoch=completed_epoch + 1,
        learning_rate=float(session.optimizer.param_groups[0]["lr"]),
    )
    rng = _rng_metadata()
    scheduler_bytes = canonical_json_bytes(scheduler)
    rng_json = canonical_json_bytes(rng)
    model_state = _checked_member(model_state, maximum=MAX_MODEL_STATE_BYTES, name=MODEL_STATE_NAME)
    optimizer_state = _checked_member(
        optimizer_state, maximum=MAX_OPTIMIZER_STATE_BYTES, name=OPTIMIZER_STATE_NAME
    )
    rng_torch = _checked_member(rng_torch, maximum=MAX_RNG_STATE_BYTES, name=RNG_TORCH_NAME)
    if scaler_state is not None:
        scaler_state = _checked_member(
            scaler_state, maximum=MAX_SCALER_STATE_BYTES, name=SCALER_STATE_NAME
        )
    rng_json_digest = sha256_bytes(rng_json)
    rng_torch_digest = sha256_bytes(rng_torch)
    values: dict[str, object] = {
        "experiment_id": session.experiment_id,
        "dataset_id": session.config.dataset_id,
        "dataset_sha256": dataset_sha256,
        "architecture_sha256": architecture_sha256,
        "config_digest": session.config.config_digest,
        "code_revision": code_revision,
        "lock_digest": lock_digest,
        "seed": session.seed,
        "completed_epoch": completed_epoch,
        "global_step": session.global_step,
        "device": session.policy.device,
        "device_name": session.policy.device_name,
        "compute_capability": session.policy.compute_capability,
        "precision": session.policy.precision,
        "model_digest": sha256_bytes(model_state),
        "model_bytes": len(model_state),
        "optimizer_digest": sha256_bytes(optimizer_state),
        "optimizer_bytes": len(optimizer_state),
        "scheduler_digest": sha256_bytes(scheduler_bytes),
        "scheduler_bytes": len(scheduler_bytes),
        "scaler_digest": None if scaler_state is None else sha256_bytes(scaler_state),
        "scaler_bytes": None if scaler_state is None else len(scaler_state),
        "rng_state_digest": canonical_sha256({"json": rng_json_digest, "torch": rng_torch_digest}),
        "rng_json_digest": rng_json_digest,
        "rng_json_bytes": len(rng_json),
        "rng_torch_digest": rng_torch_digest,
        "rng_torch_bytes": len(rng_torch),
    }
    return CheckpointPayload(
        metadata=TrainingCheckpointMetadata.create(**values),
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_bytes,
        rng_json=rng_json,
        rng_torch_state=rng_torch,
        scaler_state=scaler_state,
    )


def _real_directory(root: Path, *parts: str) -> Path:
    current = root
    for part in parts:
        candidate = current / part
        with suppress(FileExistsError):
            candidate.mkdir()
        try:
            mode = candidate.lstat().st_mode
        except OSError as error:
            raise ArtifactIntegrityError("TRAIN-12 STORE: directory inspection failed") from error
        if not stat.S_ISDIR(mode):
            raise ArtifactIntegrityError("TRAIN-12 STORE: path component is not a real directory")
        current = candidate
    return current


class LocalTrainingCheckpointStore:
    """Atomic immutable checkpoint store with explicit latest/best pointers."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError("root must be a Path")
        self.root = root

    def _seed_root(self, metadata: TrainingCheckpointMetadata) -> Path:
        return self.root / CHECKPOINT_ROOT / metadata.experiment_id / str(metadata.seed)

    @staticmethod
    def _validate_path_identity(
        experiment_id: object,
        seed: object,
        checkpoint_id: object | None = None,
    ) -> None:
        malformed = (
            not isinstance(experiment_id, str)
            or re.fullmatch(r"[0-9a-f]{20}", experiment_id) is None
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 0 <= seed < 2**64
        )
        if checkpoint_id is not None:
            malformed = (
                malformed
                or not isinstance(checkpoint_id, str)
                or re.fullmatch(r"[0-9a-f]{20}", checkpoint_id) is None
            )
        if malformed:
            raise ArtifactIntegrityError("TRAIN-12 STORE: checkpoint identity is malformed")

    def publish(self, payload: CheckpointPayload, *, best: bool = False) -> PublishedCheckpoint:
        if not isinstance(payload, CheckpointPayload):
            raise TypeError("payload must be a CheckpointPayload")
        self._verify_payload(payload)
        metadata = payload.metadata
        seed_root = _real_directory(
            self.root, CHECKPOINT_ROOT, metadata.experiment_id, str(metadata.seed)
        )
        checkpoints = _real_directory(seed_root, "checkpoints")
        target = checkpoints / metadata.checkpoint_id
        if target.is_symlink():
            raise ArtifactIntegrityError("TRAIN-12 STORE: checkpoint target must not be a symlink")
        if not target.exists():
            staging_parent = _real_directory(seed_root, ".staging")
            staging = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=staging_parent))
            try:
                members = payload.members()
                for name, content in members.items():
                    path = staging / name
                    path.write_bytes(content)
                    fsync_file(path)
                marker = staging / COMMIT_NAME
                marker.write_text(sha256_bytes(members[METADATA_NAME]) + "\n", encoding="ascii")
                fsync_file(marker)
                fsync_directory(staging)
                os.rename(staging, target)
                fsync_directory(checkpoints)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        published = self.open(metadata.experiment_id, metadata.seed, metadata.checkpoint_id)
        self._write_pointer(seed_root, "latest", metadata.checkpoint_id)
        if best:
            self._write_pointer(seed_root, "best", metadata.checkpoint_id)
        self._prune_checkpoints(seed_root)
        return published

    def open_pointer(
        self,
        experiment_id: str,
        seed: int,
        kind: Literal["latest", "best"],
    ) -> PublishedCheckpoint:
        """Resolve one bounded pointer, then fully reopen its immutable target."""

        if kind not in {"latest", "best"}:
            raise ArtifactIntegrityError("TRAIN-12 STORE: pointer kind is unsupported")
        self._validate_path_identity(experiment_id, seed)
        seed_root = self.root / CHECKPOINT_ROOT / experiment_id / str(seed)
        checkpoint_id = self._read_pointer_id(seed_root, kind)
        return self.open(experiment_id, seed, checkpoint_id)

    def _read_pointer_id(self, seed_root: Path, kind: Literal["latest", "best"]) -> str:
        content = self._read(seed_root / f"{kind}.json", 128)
        try:
            value = json.loads(content)
            checkpoint_id = value["checkpoint_id"]
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
            raise ArtifactIntegrityError("TRAIN-12 STORE: checkpoint pointer is invalid") from error
        if (
            not isinstance(checkpoint_id, str)
            or re.fullmatch(r"[0-9a-f]{20}", checkpoint_id) is None
        ):
            raise ArtifactIntegrityError("TRAIN-12 STORE: checkpoint pointer ID is invalid")
        return checkpoint_id

    def _prune_checkpoints(self, seed_root: Path) -> None:
        """Retain only immutable targets referenced by latest and current best."""

        retained = {self._read_pointer_id(seed_root, "latest")}
        best_pointer = seed_root / "best.json"
        if best_pointer.exists():
            if best_pointer.is_symlink():
                raise ArtifactIntegrityError("TRAIN-12 STORE: pointer must not be a symlink")
            retained.add(self._read_pointer_id(seed_root, "best"))
        checkpoints = seed_root / "checkpoints"
        changed = False
        for candidate in checkpoints.iterdir():
            if re.fullmatch(r"[0-9a-f]{20}", candidate.name) is None:
                raise ArtifactIntegrityError("TRAIN-12 STORE: checkpoint entry is malformed")
            if candidate.is_symlink() or not candidate.is_dir():
                raise ArtifactIntegrityError(
                    "TRAIN-12 STORE: checkpoint entry is not a real directory"
                )
            if candidate.name not in retained:
                shutil.rmtree(candidate)
                changed = True
        if changed:
            fsync_directory(checkpoints)

    @staticmethod
    def _write_pointer(seed_root: Path, name: str, checkpoint_id: str) -> None:
        content = canonical_json_bytes({"checkpoint_id": checkpoint_id})
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}-", dir=seed_root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, seed_root / f"{name}.json")
            fsync_directory(seed_root)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _verify_payload(self, payload: CheckpointPayload) -> None:
        metadata = payload.metadata
        members = payload.members()
        contracts: list[tuple[str, str, int | None]] = [
            (MODEL_STATE_NAME, metadata.model_digest, metadata.model_bytes),
            (OPTIMIZER_STATE_NAME, metadata.optimizer_digest, metadata.optimizer_bytes),
            (SCHEDULER_STATE_NAME, metadata.scheduler_digest, metadata.scheduler_bytes),
            (RNG_JSON_NAME, metadata.rng_json_digest, metadata.rng_json_bytes),
            (RNG_TORCH_NAME, metadata.rng_torch_digest, metadata.rng_torch_bytes),
        ]
        if metadata.scaler_digest is not None:
            contracts.append((SCALER_STATE_NAME, metadata.scaler_digest, metadata.scaler_bytes))
        for name, digest, size in contracts:
            content = members.get(name)
            if content is None or size is None or len(content) != size:
                raise ArtifactIntegrityError(f"TRAIN-12 STORE: {name} size mismatch")
            verify_sha256(content, digest)

    def open(self, experiment_id: str, seed: int, checkpoint_id: str) -> PublishedCheckpoint:
        self._validate_path_identity(experiment_id, seed, checkpoint_id)
        directory = (
            self.root / CHECKPOINT_ROOT / experiment_id / str(seed) / "checkpoints" / checkpoint_id
        )
        if directory.is_symlink() or not directory.is_dir():
            raise ArtifactIntegrityError("TRAIN-12 STORE: checkpoint directory is unavailable")
        names = {entry.name for entry in directory.iterdir()}
        required = {
            MODEL_STATE_NAME,
            OPTIMIZER_STATE_NAME,
            SCHEDULER_STATE_NAME,
            RNG_JSON_NAME,
            RNG_TORCH_NAME,
            METADATA_NAME,
            COMMIT_NAME,
        }
        if names not in (required, required | {SCALER_STATE_NAME}):
            raise ArtifactIntegrityError("TRAIN-12 STORE: checkpoint members are not allowlisted")
        for name in names:
            if not stat.S_ISREG((directory / name).lstat().st_mode):
                raise ArtifactIntegrityError("TRAIN-12 STORE: checkpoint member is not regular")
        metadata_bytes = self._read(directory / METADATA_NAME, MAX_METADATA_BYTES)
        marker = self._read(directory / COMMIT_NAME, 65)
        if marker != (sha256_bytes(metadata_bytes) + "\n").encode("ascii"):
            raise ArtifactIntegrityError("TRAIN-12 STORE: commit marker mismatch")
        try:
            metadata = TrainingCheckpointMetadata.model_validate_json(metadata_bytes)
        except (TypeError, ValueError) as error:
            raise ArtifactIntegrityError(
                "TRAIN-12 STORE: checkpoint metadata is invalid"
            ) from error
        if metadata.checkpoint_id != checkpoint_id or metadata.experiment_id != experiment_id:
            raise ArtifactIntegrityError("TRAIN-12 STORE: path and metadata identities differ")
        payload = CheckpointPayload(
            metadata=metadata,
            model_state=self._read(directory / MODEL_STATE_NAME, MAX_MODEL_STATE_BYTES),
            optimizer_state=self._read(directory / OPTIMIZER_STATE_NAME, MAX_OPTIMIZER_STATE_BYTES),
            scheduler_state=self._read(directory / SCHEDULER_STATE_NAME, MAX_JSON_STATE_BYTES),
            rng_json=self._read(directory / RNG_JSON_NAME, MAX_JSON_STATE_BYTES),
            rng_torch_state=self._read(directory / RNG_TORCH_NAME, MAX_RNG_STATE_BYTES),
            scaler_state=(
                self._read(directory / SCALER_STATE_NAME, MAX_SCALER_STATE_BYTES)
                if SCALER_STATE_NAME in names
                else None
            ),
        )
        self._verify_payload(payload)
        try:
            scheduler = SchedulerState.model_validate_json(payload.scheduler_state)
            rng = RngStateMetadata.model_validate_json(payload.rng_json)
        except (TypeError, ValueError) as error:
            raise ArtifactIntegrityError("TRAIN-12 STORE: JSON state member is invalid") from error
        if scheduler.completed_epoch != metadata.completed_epoch:
            raise ArtifactIntegrityError("TRAIN-12 STORE: scheduler epoch differs from metadata")
        return PublishedCheckpoint(
            root=directory,
            metadata=metadata,
            model_state=payload.model_state,
            optimizer_state=payload.optimizer_state,
            scheduler=scheduler,
            rng=rng,
            rng_torch_state=payload.rng_torch_state,
            scaler_state=payload.scaler_state,
        )

    @staticmethod
    def _read(path: Path, maximum: int) -> bytes:
        try:
            status = path.lstat()
            if not stat.S_ISREG(status.st_mode):
                raise ArtifactIntegrityError("TRAIN-12 STORE: member is not a regular file")
            size = status.st_size
            if not 0 < size <= maximum:
                raise ArtifactIntegrityError("TRAIN-12 STORE: member exceeds its byte contract")
            with path.open("rb") as handle:
                return handle.read()
        except ArtifactIntegrityError:
            raise
        except OSError as error:
            raise ArtifactIntegrityError("TRAIN-12 STORE: member read failed") from error


class ResumeIdentity(StrictFrozenModel):
    experiment_id: ContentId
    dataset_id: ContentId
    dataset_sha256: Sha256
    architecture_sha256: Sha256
    config_digest: Sha256
    code_revision: SourceRevision
    lock_digest: Sha256
    seed: Seed
    device: Device
    device_name: str
    compute_capability: str
    precision: Literal["bf16", "fp16"]


def restore_training_checkpoint(
    checkpoint: PublishedCheckpoint,
    session: TrainingSession,
    expected: ResumeIdentity,
    *,
    codec: StateCodec | None = None,
) -> int:
    """Verify complete identity, restore state/RNG, and return the next epoch."""

    if not isinstance(checkpoint, PublishedCheckpoint):
        raise TypeError("checkpoint must be a PublishedCheckpoint")
    metadata = checkpoint.metadata
    observed = ResumeIdentity(
        **metadata.model_dump(include=set(ResumeIdentity.model_fields), mode="python")
    )
    if observed != expected:
        raise ArtifactIntegrityError("TRAIN-13 RESUME: checkpoint identity does not match runtime")
    if (
        session.experiment_id != expected.experiment_id
        or session.config.dataset_id != expected.dataset_id
        or session.config.config_digest != expected.config_digest
        or session.seed != expected.seed
        or session.policy.device != expected.device
        or session.policy.device_name != expected.device_name
        or session.policy.compute_capability != expected.compute_capability
        or session.policy.precision != expected.precision
    ):
        raise ArtifactIntegrityError("TRAIN-13 RESUME: session identity does not match checkpoint")
    if (session.scaler is None) != (checkpoint.scaler_state is None):
        raise ArtifactIntegrityError("TRAIN-13 RESUME: session scaler contract differs")
    resolved = codec or TorchStateCodec(session.torch)
    model_state = resolved.decode(checkpoint.model_state, map_location=expected.device)
    optimizer_state = resolved.decode(checkpoint.optimizer_state, map_location=expected.device)
    torch_rng = resolved.decode(checkpoint.rng_torch_state, map_location="cpu")
    scaler_state = (
        None
        if checkpoint.scaler_state is None
        else resolved.decode(checkpoint.scaler_state, map_location=expected.device)
    )
    try:
        cast(Any, session.model).load_state_dict(model_state, strict=True)
        session.optimizer.load_state_dict(optimizer_state)
        if session.scaler is not None:
            session.scaler.load_state_dict(scaler_state)
        random.setstate(
            (
                checkpoint.rng.python_version,
                tuple(checkpoint.rng.python_state),
                checkpoint.rng.python_gaussian,
            )
        )
        np.random.set_state(
            (
                checkpoint.rng.numpy_kind,
                np.asarray(checkpoint.rng.numpy_state, dtype=np.uint32),
                checkpoint.rng.numpy_position,
                checkpoint.rng.numpy_has_gaussian,
                checkpoint.rng.numpy_cached_gaussian,
            )
        )
        torch = cast(Any, session.torch)
        torch.set_rng_state(cast(Mapping[str, Any], torch_rng)["cpu"])
        torch.cuda.set_rng_state_all(cast(Mapping[str, Any], torch_rng)["cuda"])
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("TRAIN-13 RESUME: decoded state cannot be restored") from error
    session.global_step = metadata.global_step
    learning_rates = {float(group["lr"]) for group in session.optimizer.param_groups}
    if learning_rates != {checkpoint.scheduler.learning_rate}:
        raise ArtifactIntegrityError("TRAIN-13 RESUME: optimizer and scheduler rates differ")
    return checkpoint.scheduler.next_epoch


class ValidationCheckpointMetric(VersionedModel):
    """Validation-only checkpoint score; test metrics are structurally impossible."""

    split: Literal["validation"] = "validation"
    experiment_id: ContentId
    dataset_id: ContentId
    config_digest: Sha256
    checkpoint_id: ContentId
    seed: Seed
    epoch: int = Field(ge=1, le=100)
    median_velocity_relative_l2: FiniteNonnegative
    median_cd_head_relative_error: FiniteNonnegative
    score: FiniteNonnegative

    @model_validator(mode="after")
    def _score_is_exact(self) -> Self:
        expected = self.median_velocity_relative_l2 + self.median_cd_head_relative_error
        if not math.isclose(self.score, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("validation score must equal velocity plus Cd error")
        return self


class SelectedCheckpoint(StrictFrozenModel):
    seed: Seed
    epoch: int = Field(ge=1, le=100)
    checkpoint_id: ContentId
    validation_score: FiniteNonnegative


class FrozenTrainingSelection(VersionedModel):
    """Immutable all-seed validation decision required before test evaluation."""

    selection_id: ContentId
    selection_sha256: Sha256
    experiment_id: ContentId
    dataset_id: ContentId
    config_digest: Sha256
    selected: tuple[SelectedCheckpoint, SelectedCheckpoint, SelectedCheckpoint]
    deployable_seed: Seed
    test_metrics_read: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_selected(cls, value: object) -> object:
        if isinstance(value, Mapping) and isinstance(value.get("selected"), list):
            return {**value, "selected": tuple(value["selected"])}
        return value

    @model_validator(mode="after")
    def _identity_is_coherent(self) -> Self:
        if (
            tuple(item.seed for item in self.selected)
            != tuple(sorted(item.seed for item in self.selected))
            or len({item.seed for item in self.selected}) != 3
        ):
            raise ValueError("selection must contain three distinct sorted seeds")
        best = min(self.selected, key=lambda item: (item.validation_score, item.seed))
        if self.deployable_seed != best.seed:
            raise ValueError("deployable seed must minimize validation score")
        logical = self.model_dump(mode="json", exclude={"selection_id", "selection_sha256"})
        digest = canonical_sha256(logical)
        if self.selection_sha256 != digest or self.selection_id != digest[:20]:
            raise ValueError("selection identity does not bind the frozen decision")
        return self


def freeze_validation_selection(
    metrics: Sequence[ValidationCheckpointMetric],
    *,
    expected_seeds: tuple[int, int, int],
) -> FrozenTrainingSelection:
    """Select earlier epoch ties, then lower-seed ties, without test inputs."""

    if not metrics or any(not isinstance(item, ValidationCheckpointMetric) for item in metrics):
        raise ArtifactIntegrityError("TRAIN-14 SELECT: validation metrics are required")
    if len(set(expected_seeds)) != 3:
        raise ArtifactIntegrityError("TRAIN-14 SELECT: exactly three distinct seeds are required")
    identities = {(item.experiment_id, item.dataset_id, item.config_digest) for item in metrics}
    if len(identities) != 1:
        raise ArtifactIntegrityError("TRAIN-14 SELECT: validation identities differ")
    grouped: dict[int, list[ValidationCheckpointMetric]] = defaultdict(list)
    for item in metrics:
        grouped[item.seed].append(item)
    if set(grouped) != set(expected_seeds):
        raise ArtifactIntegrityError("TRAIN-14 SELECT: validation evidence must cover every seed")
    selected: list[SelectedCheckpoint] = []
    for seed in sorted(expected_seeds):
        candidates = grouped[seed]
        if len({item.epoch for item in candidates}) != len(candidates):
            raise ArtifactIntegrityError("TRAIN-14 SELECT: duplicate epoch evidence")
        best = min(candidates, key=lambda item: (item.score, item.epoch))
        selected.append(
            SelectedCheckpoint(
                seed=seed,
                epoch=best.epoch,
                checkpoint_id=best.checkpoint_id,
                validation_score=best.score,
            )
        )
    experiment_id, dataset_id, config_digest = next(iter(identities))
    deployable = min(selected, key=lambda item: (item.validation_score, item.seed)).seed
    logical = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "config_digest": config_digest,
        "selected": [item.model_dump(mode="json") for item in selected],
        "deployable_seed": deployable,
        "test_metrics_read": False,
    }
    digest = canonical_sha256(logical)
    return FrozenTrainingSelection(
        selection_id=digest[:20],
        selection_sha256=digest,
        **logical,  # type: ignore[arg-type]
    )


def export_selected_checkpoint_bundle(
    checkpoint: PublishedCheckpoint,
    selection: FrozenTrainingSelection,
    predictor: FnoPredictor,
    preprocessing: PreprocessingStatistics,
    *,
    dataset_sha256: str,
    code_revision: str,
    lock_digest: str,
    model_card: ModelCardMetadata,
    store: LocalModelBundleStore,
    codec: StateCodec | None = None,
) -> ArtifactRef:
    """Export only safe weights for one validation-frozen selected checkpoint."""

    metadata = checkpoint.metadata
    candidates = tuple(item for item in selection.selected if item.seed == metadata.seed)
    if len(candidates) != 1:
        raise ArtifactIntegrityError("TRAIN-14 EXPORT: checkpoint seed is not frozen")
    selected = candidates[0]
    if (
        selected.checkpoint_id != metadata.checkpoint_id
        or selected.epoch != metadata.completed_epoch
        or selection.experiment_id != metadata.experiment_id
        or selection.dataset_id != metadata.dataset_id
        or selection.config_digest != metadata.config_digest
        or dataset_sha256 != metadata.dataset_sha256
        or code_revision != metadata.code_revision
        or lock_digest != metadata.lock_digest
    ):
        raise ArtifactIntegrityError("TRAIN-14 EXPORT: selected checkpoint identity differs")
    resolved = codec or TorchStateCodec(predictor._runtime.torch)
    state = resolved.decode(checkpoint.model_state, map_location="cpu")
    try:
        predictor.load_state_dict(cast(Mapping[str, object], state), strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("TRAIN-14 EXPORT: selected model state is invalid") from error
    bundle = build_model_bundle(
        weights=snapshot_fno_weights(predictor),
        preprocessing=preprocessing,
        dataset_sha256=dataset_sha256,
        experiment_id=selection.experiment_id,
        seed=metadata.seed,
        selected_epoch=metadata.completed_epoch,
        code_revision=code_revision,
        lock_digest=lock_digest,
        model_card=model_card,
    )
    return store.publish(bundle)


__all__ = [
    "CheckpointPayload",
    "FrozenTrainingSelection",
    "LocalTrainingCheckpointStore",
    "PublishedCheckpoint",
    "ResumeIdentity",
    "RngStateMetadata",
    "SchedulerState",
    "SelectedCheckpoint",
    "StateCodec",
    "TorchStateCodec",
    "TrainingCheckpointMetadata",
    "ValidationCheckpointMetric",
    "capture_training_checkpoint",
    "export_selected_checkpoint_bundle",
    "freeze_validation_selection",
    "restore_training_checkpoint",
]
