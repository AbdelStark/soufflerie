"""Provider-neutral identity and persistence boundaries for train/validate jobs."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar, cast

from pydantic import Field, StringConstraints, field_validator, model_validator

from infra.remote_execution import MAX_REMOTE_INPUT_BYTES, DeviceClass, parse_remote_model
from infra.runtime_manifest import RuntimeBuildManifest
from soufflerie.artifacts import safe_read_bytes
from soufflerie.config import TrainingConfig, ValidationConfig
from soufflerie.datagen._local_files import ensure_real_directory, fsync_directory, fsync_file
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import (
    ArtifactRef,
    ContentId,
    Sha256,
    VersionedModel,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)
from soufflerie.surrogate.architecture import FnoArchitecture
from soufflerie.validation.gates import ValidationReport
from soufflerie.validation.reporting import report_parent_sha256

WorkflowKind = Literal["training", "validation"]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
FiniteNonnegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
_ModelT = TypeVar("_ModelT", bound=VersionedModel)


def fno_architecture_sha256() -> str:
    """Return the digest of the one release-eligible architecture."""

    return sha256_bytes(canonical_json_bytes(FnoArchitecture()))


def training_experiment_sha256(
    *,
    config: TrainingConfig,
    dataset_sha256: str,
    architecture_sha256: str,
    source_revision: str,
    lock_sha256: str,
) -> str:
    """Bind every value whose change creates a new training experiment."""

    return canonical_sha256(
        {
            "contract": "training-experiment-v1",
            "dataset_sha256": dataset_sha256,
            "architecture_sha256": architecture_sha256,
            "training_config_sha256": config.config_digest,
            "seeds": list(config.seeds),
            "source_revision": source_revision,
            "lock_sha256": lock_sha256,
        }
    )


class RemoteTrainingRequest(VersionedModel):
    """Immutable config and artifact identity for all three seed workers."""

    config: TrainingConfig
    dataset: ArtifactRef
    requested_device_class: DeviceClass
    source_revision: Revision
    lock_sha256: Sha256
    architecture_sha256: Sha256
    experiment_id: ContentId
    request_digest: Sha256

    @model_validator(mode="before")
    @classmethod
    def _normalize_config_json_tuples(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        config = normalized.get("config")
        if isinstance(config, Mapping):
            normalized_config = dict(config)
            for name in ("seeds", "field_weights"):
                item = normalized_config.get(name)
                if isinstance(item, list):
                    normalized_config[name] = tuple(item)
            normalized["config"] = normalized_config
        return normalized

    def logical_identity(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"request_digest"})

    @model_validator(mode="after")
    def _identity_is_coherent(self) -> Self:
        if self.dataset.artifact_type != "dataset":
            raise ValueError("training request requires a dataset ArtifactRef")
        if self.config.dataset_id != self.dataset.artifact_id:
            raise ValueError("training config and dataset reference IDs differ")
        if self.config.epochs > 100:
            raise ValueError("remote training supports at most 100 checkpointed epochs")
        if self.architecture_sha256 != fno_architecture_sha256():
            raise ValueError("training request architecture is not fno2d-v1")
        experiment_sha256 = training_experiment_sha256(
            config=self.config,
            dataset_sha256=self.dataset.sha256,
            architecture_sha256=self.architecture_sha256,
            source_revision=self.source_revision,
            lock_sha256=self.lock_sha256,
        )
        if self.experiment_id != experiment_sha256[:20]:
            raise ValueError("experiment_id does not bind the complete training identity")
        if self.request_digest != canonical_sha256(self.logical_identity()):
            raise ValueError("request_digest does not match the training request")
        return self

    @classmethod
    def create(
        cls,
        *,
        config: TrainingConfig,
        dataset: ArtifactRef,
        requested_device_class: DeviceClass,
        source_revision: str,
        lock_sha256: str,
    ) -> Self:
        architecture_sha256 = fno_architecture_sha256()
        experiment_sha256 = training_experiment_sha256(
            config=config,
            dataset_sha256=dataset.sha256,
            architecture_sha256=architecture_sha256,
            source_revision=source_revision,
            lock_sha256=lock_sha256,
        )
        values: dict[str, object] = {
            "schema_version": 1,
            "config": config,
            "dataset": dataset,
            "requested_device_class": requested_device_class,
            "source_revision": source_revision,
            "lock_sha256": lock_sha256,
            "architecture_sha256": architecture_sha256,
            "experiment_id": experiment_sha256[:20],
        }
        return cls.model_validate({**values, "request_digest": canonical_sha256(values)})


class RemoteValidationRequest(VersionedModel):
    """Immutable dataset/model/baseline identity for one report evaluation."""

    config: ValidationConfig
    dataset: ArtifactRef
    models: tuple[ArtifactRef, ArtifactRef, ArtifactRef]
    baseline_sha256s: tuple[Sha256, Sha256]
    selected_model_id: ContentId
    solver_sha256: Sha256
    requested_device_class: DeviceClass
    precision: Literal["bf16", "fp16"]
    model_source_revision: Revision
    source_revision: Revision
    lock_sha256: Sha256
    request_digest: Sha256

    @model_validator(mode="before")
    @classmethod
    def _normalize_config_json_tuples(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        config = normalized.get("config")
        if isinstance(config, Mapping):
            normalized_config = dict(config)
            for name in ("ensemble_model_ids", "baseline_ids"):
                item = normalized_config.get(name)
                if isinstance(item, list):
                    normalized_config[name] = tuple(item)
            normalized["config"] = normalized_config
        return normalized

    @field_validator("models", "baseline_sha256s", mode="before")
    @classmethod
    def _json_arrays_to_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @property
    def expected_parent_sha256(self) -> dict[str, str]:
        return {
            "baseline_0": self.baseline_sha256s[0],
            "baseline_1": self.baseline_sha256s[1],
            "dataset": self.dataset.sha256,
            "ensemble_model_0": self.models[0].sha256,
            "ensemble_model_1": self.models[1].sha256,
            "ensemble_model_2": self.models[2].sha256,
            "solver": self.solver_sha256,
        }

    def logical_identity(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"request_digest"})

    @model_validator(mode="after")
    def _identity_is_coherent(self) -> Self:
        if self.dataset.artifact_type != "dataset":
            raise ValueError("validation request requires a dataset ArtifactRef")
        if self.config.dataset_id != self.dataset.artifact_id:
            raise ValueError("validation config and dataset reference IDs differ")
        if any(reference.artifact_type != "model" for reference in self.models):
            raise ValueError("validation request requires three model ArtifactRefs")
        if tuple(reference.artifact_id for reference in self.models) != tuple(
            self.config.ensemble_model_ids
        ):
            raise ValueError("validation model references differ from the config")
        if tuple(value[:20] for value in self.baseline_sha256s) != tuple(self.config.baseline_ids):
            raise ValueError("validation baseline digests differ from the config")
        if self.selected_model_id not in self.config.ensemble_model_ids:
            raise ValueError("selected model is not in the validation ensemble")
        if self.request_digest != canonical_sha256(self.logical_identity()):
            raise ValueError("request_digest does not match the validation request")
        return self

    @classmethod
    def create(
        cls,
        *,
        config: ValidationConfig,
        dataset: ArtifactRef,
        models: tuple[ArtifactRef, ArtifactRef, ArtifactRef],
        baseline_sha256s: tuple[str, str],
        selected_model_id: str,
        solver_sha256: str,
        requested_device_class: DeviceClass,
        source_revision: str,
        lock_sha256: str,
        precision: Literal["bf16", "fp16"] = "bf16",
        model_source_revision: str | None = None,
    ) -> Self:
        values: dict[str, object] = {
            "schema_version": 1,
            "config": config,
            "dataset": dataset,
            "models": models,
            "baseline_sha256s": baseline_sha256s,
            "selected_model_id": selected_model_id,
            "solver_sha256": solver_sha256,
            "requested_device_class": requested_device_class,
            "precision": precision,
            "model_source_revision": model_source_revision or source_revision,
            "source_revision": source_revision,
            "lock_sha256": lock_sha256,
        }
        return cls.model_validate({**values, "request_digest": canonical_sha256(values)})


class ExecutionAccounting(VersionedModel):
    """Common bounded resource evidence for a terminal remote operation."""

    started_at: datetime
    completed_at: datetime
    wall_seconds: FiniteNonnegative
    gpu_seconds: FiniteNonnegative
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)
    device_class: DeviceClass
    device_name: str = Field(min_length=1, max_length=256)
    precision: Literal["bf16", "fp16"]
    source_revision: Revision
    lock_sha256: Sha256

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_timestamps(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("started_at", "completed_at"):
            timestamp = normalized.get(name)
            if isinstance(timestamp, str):
                with suppress(ValueError):
                    normalized[name] = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return normalized

    @model_validator(mode="after")
    def _accounting_is_coherent(self) -> Self:
        if self.started_at.utcoffset() is None or self.completed_at.utcoffset() is None:
            raise ValueError("execution timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("execution completion precedes its start")
        if self.peak_allocated_bytes > self.peak_reserved_bytes:
            raise ValueError("allocated memory cannot exceed reserved memory")
        elapsed = (self.completed_at - self.started_at).total_seconds()
        tolerance = max(1e-6, elapsed * 1e-6)
        if abs(self.wall_seconds - elapsed) > tolerance:
            raise ValueError("wall_seconds does not match the execution timestamps")
        if self.gpu_seconds > self.wall_seconds + tolerance:
            raise ValueError("single-GPU seconds cannot exceed wall time")
        return self


class TrainingSeedReceipt(VersionedModel):
    """Small terminal receipt for one selected per-seed model bundle."""

    experiment_id: ContentId
    seed: int = Field(ge=0, le=2**64 - 1)
    completed_epochs: int = Field(ge=1, le=100)
    model: ArtifactRef
    best_checkpoint_id: ContentId
    validation_score: FiniteNonnegative
    selection_id: ContentId
    selection_sha256: Sha256
    deployable_seed: int = Field(ge=0, le=2**64 - 1)
    baseline_sha256s: tuple[Sha256, Sha256]
    solver_sha256: Sha256
    accounting: ExecutionAccounting
    parent_sha256: dict[str, Sha256]
    final_state: Literal["succeeded"] = "succeeded"
    evidence_sha256: Sha256

    @field_validator("baseline_sha256s", mode="before")
    @classmethod
    def _json_baselines_to_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _receipt_is_coherent(self) -> Self:
        if self.model.artifact_type != "model":
            raise ValueError("training receipt requires a model ArtifactRef")
        expected_roles = {
            "architecture",
            "best_checkpoint",
            "config",
            "dataset",
            "model",
        }
        if set(self.parent_sha256) != expected_roles:
            raise ValueError("training receipt parent roles are incomplete")
        if self.parent_sha256["model"] != self.model.sha256:
            raise ValueError("training receipt model parent differs")
        if self.parent_sha256["best_checkpoint"][:20] != self.best_checkpoint_id:
            raise ValueError("training receipt checkpoint parent differs")
        if self.selection_id != self.selection_sha256[:20]:
            raise ValueError("training receipt selection ID does not prefix its digest")
        baseline_ids = tuple(value[:20] for value in self.baseline_sha256s)
        if len(set(baseline_ids)) != 2 or self.model.artifact_id in baseline_ids:
            raise ValueError("training receipt baseline identities are incoherent")
        payload = self.model_dump(mode="json", exclude={"evidence_sha256"})
        if self.evidence_sha256 != canonical_sha256(payload):
            raise ValueError("evidence_sha256 does not match the training receipt")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        payload: dict[str, object] = {
            "schema_version": 1,
            "final_state": "succeeded",
            **values,
        }
        temporary = cls.model_construct(**cast(Any, payload), evidence_sha256="0" * 64)
        identity = temporary.model_dump(mode="json", exclude={"evidence_sha256"})
        return cls.model_validate({**payload, "evidence_sha256": canonical_sha256(identity)})


class ValidationReceipt(VersionedModel):
    """Small terminal receipt for a committed, lineage-checked report."""

    report: ArtifactRef
    overall_status: Literal["green", "red"]
    accounting: ExecutionAccounting
    parent_sha256: dict[str, Sha256]
    final_state: Literal["succeeded"] = "succeeded"
    evidence_sha256: Sha256

    @model_validator(mode="after")
    def _receipt_is_coherent(self) -> Self:
        if self.report.artifact_type != "validation_report":
            raise ValueError("validation receipt requires a report ArtifactRef")
        if set(self.parent_sha256) != {
            "baseline_0",
            "baseline_1",
            "dataset",
            "ensemble_model_0",
            "ensemble_model_1",
            "ensemble_model_2",
            "solver",
        }:
            raise ValueError("validation receipt parent roles are incomplete")
        payload = self.model_dump(mode="json", exclude={"evidence_sha256"})
        if self.evidence_sha256 != canonical_sha256(payload):
            raise ValueError("evidence_sha256 does not match the validation receipt")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        payload: dict[str, object] = {
            "schema_version": 1,
            "final_state": "succeeded",
            **values,
        }
        temporary = cls.model_construct(**cast(Any, payload), evidence_sha256="0" * 64)
        identity = temporary.model_dump(mode="json", exclude={"evidence_sha256"})
        return cls.model_validate({**payload, "evidence_sha256": canonical_sha256(identity)})


class TrainingRunIndex(VersionedModel):
    """Complete small handoff from remote training to immutable validation."""

    request: RemoteTrainingRequest
    receipts: tuple[TrainingSeedReceipt, TrainingSeedReceipt, TrainingSeedReceipt]
    selected_model_id: ContentId
    evidence_sha256: Sha256

    @field_validator("receipts", mode="before")
    @classmethod
    def _json_receipts_to_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _index_is_coherent(self) -> Self:
        if tuple(item.seed for item in self.receipts) != self.request.config.seeds:
            raise ValueError("training index receipts do not follow configured seed order")
        if any(
            item.experiment_id != self.request.experiment_id
            or item.completed_epochs != self.request.config.epochs
            or item.accounting.source_revision != self.request.source_revision
            or item.accounting.lock_sha256 != self.request.lock_sha256
            or item.accounting.device_class != self.request.requested_device_class
            or item.accounting.precision != self.request.config.precision
            or item.parent_sha256["dataset"] != self.request.dataset.sha256
            or item.parent_sha256["config"] != self.request.config.config_digest
            or item.parent_sha256["architecture"] != self.request.architecture_sha256
            for item in self.receipts
        ):
            raise ValueError("training index receipt identity differs from its request")
        if len({item.model.artifact_id for item in self.receipts}) != 3:
            raise ValueError("training index requires three distinct models")
        if len({item.selection_sha256 for item in self.receipts}) != 1:
            raise ValueError("training index receipts differ on frozen selection")
        if len({item.baseline_sha256s for item in self.receipts}) != 1:
            raise ValueError("training index receipts differ on baseline identities")
        if len({item.solver_sha256 for item in self.receipts}) != 1:
            raise ValueError("training index receipts differ on solver lineage")
        deployable_seeds = {item.deployable_seed for item in self.receipts}
        if len(deployable_seeds) != 1:
            raise ValueError("training index receipts differ on deployable seed")
        deployable_seed = next(iter(deployable_seeds))
        expected_deployable = min(
            self.receipts,
            key=lambda item: (item.validation_score, item.seed),
        ).seed
        if deployable_seed != expected_deployable:
            raise ValueError("deployable seed does not minimize validation score")
        selected = tuple(item for item in self.receipts if item.seed == deployable_seed)
        if len(selected) != 1 or self.selected_model_id != selected[0].model.artifact_id:
            raise ValueError("selected model does not match the frozen deployable seed")
        payload = self.model_dump(mode="json", exclude={"evidence_sha256"})
        if self.evidence_sha256 != canonical_sha256(payload):
            raise ValueError("evidence_sha256 does not match the training index")
        return self

    @classmethod
    def create(
        cls,
        *,
        request: RemoteTrainingRequest,
        receipts: tuple[TrainingSeedReceipt, TrainingSeedReceipt, TrainingSeedReceipt],
    ) -> Self:
        deployable_seed = receipts[0].deployable_seed
        selected = tuple(item for item in receipts if item.seed == deployable_seed)
        if len(selected) != 1:
            raise ArtifactIntegrityError(
                "REMOTE_TRAIN_INDEX_MISMATCH: deployable seed is not represented once"
            )
        payload: dict[str, object] = {
            "schema_version": 1,
            "request": request,
            "receipts": receipts,
            "selected_model_id": selected[0].model.artifact_id,
        }
        temporary = cls.model_construct(**cast(Any, payload), evidence_sha256="0" * 64)
        identity = temporary.model_dump(mode="json", exclude={"evidence_sha256"})
        return cls.model_validate({**payload, "evidence_sha256": canonical_sha256(identity)})


def assert_request_matches_build(
    request: RemoteTrainingRequest | RemoteValidationRequest,
    build: RuntimeBuildManifest,
) -> None:
    """Reject execution under a different or dirty immutable image identity."""

    if build.source_dirty:
        raise RuntimeError("train/validate requires an image built from clean source")
    if request.source_revision != build.source_revision:
        raise RuntimeError("request source revision does not match the image")
    if request.lock_sha256 != build.lock_sha256:
        raise RuntimeError("request lock digest does not match the image")


def assert_training_seed(request: RemoteTrainingRequest, seed: int) -> None:
    """Reject undeclared or type-coerced seed identities before GPU work."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in request.config.seeds:
        raise ArtifactIntegrityError("REMOTE_TRAIN_SEED_MISMATCH: seed is not declared by config")


def assert_report_matches_request(
    report: ValidationReport,
    request: RemoteValidationRequest,
) -> None:
    """Refuse every report whose model, dataset, config, or parent lineage drifted."""

    if report.dataset_id != request.dataset.artifact_id:
        raise ArtifactIntegrityError("REMOTE_VALIDATE_DATASET_MISMATCH: report dataset differs")
    if report.ensemble_model_ids != request.config.ensemble_model_ids:
        raise ArtifactIntegrityError("REMOTE_VALIDATE_MODEL_MISMATCH: report ensemble differs")
    if report.selected_model_id != request.selected_model_id:
        raise ArtifactIntegrityError("REMOTE_VALIDATE_MODEL_MISMATCH: selected model differs")
    if report.baseline_ids != request.config.baseline_ids:
        raise ArtifactIntegrityError("REMOTE_VALIDATE_BASELINE_MISMATCH: report baselines differ")
    if report.provenance.config_sha256 != request.config.config_digest:
        raise ArtifactIntegrityError("REMOTE_VALIDATE_CONFIG_MISMATCH: report config differs")
    if report.provenance.source_revision != request.source_revision:
        raise ArtifactIntegrityError("REMOTE_VALIDATE_SOURCE_MISMATCH: report source differs")
    if report.provenance.lock_sha256 != request.lock_sha256:
        raise ArtifactIntegrityError("REMOTE_VALIDATE_LOCK_MISMATCH: report lock differs")
    if report_parent_sha256(report) != request.expected_parent_sha256:
        raise ArtifactIntegrityError("REMOTE_VALIDATE_PARENT_MISMATCH: report parents differ")


def workflow_request_reference(content: bytes, *, kind: WorkflowKind) -> ArtifactRef:
    """Return the immutable volume key for one already-validated request."""

    digest = sha256_bytes(content)
    return ArtifactRef(
        artifact_type=f"{kind}_request",
        artifact_id=digest[:20],
        sha256=digest,
        size_bytes=len(content),
        uri=f"requests/{kind}/{digest}.json",
    )


def publish_workflow_request(
    root: Path,
    content: bytes,
    *,
    kind: WorkflowKind,
    model: type[_ModelT],
) -> ArtifactRef:
    """Parse, atomically publish, or verify one bounded workflow request."""

    parse_remote_model(content, model)
    reference = workflow_request_reference(content, kind=kind)
    parent = ensure_real_directory(root.resolve(), "requests", kind)
    target = parent / f"{reference.sha256}.json"
    if target.exists() or target.is_symlink():
        existing = safe_read_bytes(
            root,
            reference.uri,
            max_bytes=MAX_REMOTE_INPUT_BYTES,
            expected_sha256=reference.sha256,
        )
        if existing != content:
            raise ArtifactIntegrityError("content-addressed workflow request diverged")
        return reference

    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".request-", suffix=".tmp", dir=parent)
        temporary = Path(name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_file(temporary)
        os.replace(temporary, target)
        temporary = None
        fsync_directory(parent)
    except OSError as error:
        raise ArtifactIntegrityError("atomic workflow request publication failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    stored = safe_read_bytes(
        root,
        reference.uri,
        max_bytes=MAX_REMOTE_INPUT_BYTES,
        expected_sha256=reference.sha256,
    )
    if stored != content:
        raise ArtifactIntegrityError("published workflow request bytes diverged")
    return reference


def load_workflow_request(
    root: Path,
    reference: ArtifactRef,
    *,
    kind: WorkflowKind,
    model: type[_ModelT],
) -> _ModelT:
    """Reload one request only through its exact type, path, size, and digest."""

    if reference.artifact_type != f"{kind}_request":
        raise ArtifactIntegrityError(f"remote {kind} requires a {kind}_request ArtifactRef")
    expected = f"requests/{kind}/{reference.sha256}.json"
    if reference.uri != expected or reference.artifact_id != reference.sha256[:20]:
        raise ArtifactIntegrityError(f"remote {kind} request reference is incoherent")
    content = safe_read_bytes(
        root,
        reference.uri,
        max_bytes=MAX_REMOTE_INPUT_BYTES,
        expected_sha256=reference.sha256,
    )
    if len(content) != reference.size_bytes:
        raise ArtifactIntegrityError(f"remote {kind} request size does not match its reference")
    return parse_remote_model(content, model)


def utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "ExecutionAccounting",
    "RemoteTrainingRequest",
    "RemoteValidationRequest",
    "TrainingRunIndex",
    "TrainingSeedReceipt",
    "ValidationReceipt",
    "assert_report_matches_request",
    "assert_request_matches_build",
    "assert_training_seed",
    "fno_architecture_sha256",
    "load_workflow_request",
    "publish_workflow_request",
    "training_experiment_sha256",
    "utc_now",
    "workflow_request_reference",
]
