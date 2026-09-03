"""Provider-neutral, resumable single-seed training execution."""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

import numpy as np
from pydantic import Field, model_validator

from infra.runtime_manifest import RuntimeBuildManifest
from infra.train_validate_execution import (
    ExecutionAccounting,
    RemoteTrainingRequest,
    TrainingSeedReceipt,
    assert_request_matches_build,
    assert_training_seed,
    utc_now,
)
from soufflerie.artifacts import ReaderLimits, safe_read_bytes, safe_read_json
from soufflerie.datagen._local_files import fsync_directory
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import Sha256, VersionedModel, canonical_json_bytes, canonical_sha256
from soufflerie.surrogate import (
    LocalModelBundleStore,
    ModelCardGate,
    ModelCardMetadata,
    denormalize_fields,
    fit_preprocessing_statistics,
)
from soufflerie.surrogate.fno import FnoPredictor
from soufflerie.training import (
    EpochJsonlWriter,
    LocalTrainingCheckpointStore,
    ResumeIdentity,
    TrainingEpochRecord,
    ValidationCheckpointMetric,
    capture_training_checkpoint,
    export_selected_checkpoint_bundle,
    fit_baselines,
    freeze_validation_selection,
    open_manifest_dataset,
    prepare_training_session,
    restore_training_checkpoint,
    run_training_epoch,
    training_batch_to_torch,
)

_MAX_STATE_BYTES = 2 * 1024 * 1024
_SELECTION_WAIT_SECONDS = 15 * 60
_POLL_SECONDS = 5.0
FiniteNonnegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class SeedCompletion(VersionedModel):
    """Immutable validation-only checkpoint evidence used at the seed barrier."""

    experiment_id: str = Field(pattern=r"^[0-9a-f]{20}$")
    dataset_sha256: Sha256
    config_sha256: Sha256
    architecture_sha256: Sha256
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: Sha256
    seed: int = Field(ge=0, le=2**64 - 1)
    metrics: tuple[ValidationCheckpointMetric, ...] = Field(min_length=1, max_length=100)
    epoch_records: tuple[TrainingEpochRecord, ...] = Field(min_length=1, max_length=100)
    validation_wall_seconds: tuple[FiniteNonnegative, ...] = Field(min_length=1, max_length=100)
    validation_gpu_seconds: tuple[FiniteNonnegative, ...] = Field(min_length=1, max_length=100)
    best_checkpoint_id: str = Field(pattern=r"^[0-9a-f]{20}$")
    best_checkpoint_sha256: Sha256
    device_class: Literal["L40S", "A10G"]
    device_name: str = Field(min_length=1, max_length=256)
    precision: Literal["bf16", "fp16"]
    evidence_sha256: Sha256

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if isinstance(value, Mapping):
            normalized = dict(value)
            for name in (
                "metrics",
                "epoch_records",
                "validation_wall_seconds",
                "validation_gpu_seconds",
            ):
                item = normalized.get(name)
                if isinstance(item, list):
                    normalized[name] = tuple(item)
            return normalized
        return value

    @model_validator(mode="after")
    def _completion_is_coherent(self) -> Self:
        if not (
            len(self.metrics)
            == len(self.epoch_records)
            == len(self.validation_wall_seconds)
            == len(self.validation_gpu_seconds)
        ):
            raise ValueError("completion metrics and epoch records must align")
        if any(
            gpu > wall
            for gpu, wall in zip(
                self.validation_gpu_seconds,
                self.validation_wall_seconds,
                strict=True,
            )
        ):
            raise ValueError("validation GPU seconds cannot exceed wall seconds")
        if tuple(item.epoch for item in self.metrics) != tuple(
            item.epoch for item in self.epoch_records
        ):
            raise ValueError("completion metric epochs differ from training records")
        if any(
            item.experiment_id != self.experiment_id
            or item.dataset_id != self.dataset_sha256[:20]
            or item.config_digest != self.config_sha256
            or item.seed != self.seed
            for item in self.metrics
        ):
            raise ValueError("completion validation metric identity differs")
        if any(
            item.experiment_id != self.experiment_id
            or item.config_digest != self.config_sha256
            or item.seed != self.seed
            for item in self.epoch_records
        ):
            raise ValueError("completion epoch record identity differs")
        best = min(self.metrics, key=lambda item: (item.score, item.epoch))
        if best.checkpoint_id != self.best_checkpoint_id:
            raise ValueError("completion best checkpoint does not minimize validation score")
        if self.best_checkpoint_sha256[:20] != self.best_checkpoint_id:
            raise ValueError("completion checkpoint ID does not prefix its full digest")
        payload = self.model_dump(mode="json", exclude={"evidence_sha256"})
        if self.evidence_sha256 != canonical_sha256(payload):
            raise ValueError("completion evidence digest differs")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        payload: dict[str, object] = {"schema_version": 1, **values}
        temporary = cls.model_construct(**cast(Any, payload), evidence_sha256="0" * 64)
        identity = temporary.model_dump(mode="json", exclude={"evidence_sha256"})
        return cls.model_validate({**payload, "evidence_sha256": canonical_sha256(identity)})


def _seed_root(root: Path, request: RemoteTrainingRequest, seed: int) -> Path:
    return root / "training" / request.experiment_id / str(seed)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise ArtifactIntegrityError("REMOTE_TRAIN_STORE_UNSAFE: symbolic links are forbidden")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except OSError as error:
        raise ArtifactIntegrityError("REMOTE_TRAIN_STORE_FAILED: atomic write failed") from error
    finally:
        temporary.unlink(missing_ok=True)


def _read_completion(path: Path) -> SeedCompletion:
    return safe_read_json(
        path.parent,
        path.name,
        model=SeedCompletion,
        limits=ReaderLimits(max_file_bytes=_MAX_STATE_BYTES, max_json_bytes=_MAX_STATE_BYTES),
    )


def _read_receipt(path: Path) -> TrainingSeedReceipt:
    return safe_read_json(
        path.parent,
        path.name,
        model=TrainingSeedReceipt,
        limits=ReaderLimits(max_file_bytes=_MAX_STATE_BYTES, max_json_bytes=_MAX_STATE_BYTES),
    )


def _publish_immutable(path: Path, content: bytes, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        existing = safe_read_bytes(path.parent, path.name, max_bytes=_MAX_STATE_BYTES)
        if existing != content:
            raise ArtifactIntegrityError(
                f"REMOTE_TRAIN_{label}_MISMATCH: committed evidence differs"
            )
        return
    _atomic_write(path, content)


def _validation_metrics(
    request: RemoteTrainingRequest,
    *,
    seed: int,
    epoch: int,
    checkpoint_id: str,
    session: Any,
    dataset: Any,
    statistics: Any,
) -> tuple[ValidationCheckpointMetric, float, float]:
    """Evaluate only the validation split through physical velocity/Cd metrics."""

    velocity: list[float] = []
    drag: list[float] = []
    torch = cast(Any, session.torch)
    was_training = bool(getattr(session.model, "training", True))
    session.model.train(False)
    validation_started = time.perf_counter()
    gpu_seconds = 0.0
    try:
        with torch.inference_mode():
            for batch in dataset.iter_batches(
                statistics,
                "validation",
                batch_size=request.config.batch_size,
                seed=seed,
                epoch=epoch - 1,
            ):
                tensors = training_batch_to_torch(
                    batch.data,
                    device=session.policy.device,
                    torch_module=session.torch,
                )
                session.synchronize()
                gpu_started = time.perf_counter()
                with torch.autocast(
                    device_type="cuda",
                    dtype=session.policy.autocast_dtype,
                    enabled=True,
                ):
                    predicted = session.model(tensors.inputs)
                session.synchronize()
                gpu_seconds += time.perf_counter() - gpu_started
                predicted_fields = np.ascontiguousarray(
                    predicted.fields_normalized.detach().cpu().numpy(),
                    dtype=np.float32,
                )
                target_fields = np.ascontiguousarray(
                    cast(Any, tensors.target.fields_normalized).detach().cpu().numpy(),
                    dtype=np.float32,
                )
                predicted_physical = denormalize_fields(predicted_fields, statistics)
                target_physical = denormalize_fields(target_fields, statistics)
                masks = np.ascontiguousarray(batch.data.fluid_mask[:, 0], dtype=np.bool_)
                predicted_cd = np.asarray(
                    predicted.cd_head.detach().cpu().numpy(), dtype=np.float64
                )
                target_cd = np.asarray(batch.data.cd, dtype=np.float64)
                for index, fluid in enumerate(masks):
                    delta = predicted_physical[index, :2].astype(np.float64) - target_physical[
                        index, :2
                    ].astype(np.float64)
                    target = target_physical[index, :2].astype(np.float64)
                    numerator = math.sqrt(float(np.sum(delta[:, fluid] ** 2, dtype=np.float64)))
                    denominator = math.sqrt(float(np.sum(target[:, fluid] ** 2, dtype=np.float64)))
                    velocity.append(numerator / max(denominator, 1e-8))
                    drag.append(
                        abs(float(predicted_cd[index] - target_cd[index]))
                        / max(abs(float(target_cd[index])), 0.1)
                    )
    finally:
        session.model.train(was_training)
    if len(velocity) != 200 or len(drag) != 200:
        raise ArtifactIntegrityError(
            "REMOTE_TRAIN_VALIDATION_INCOMPLETE: validation membership changed"
        )
    median_velocity = float(np.median(np.asarray(velocity, dtype=np.float64)))
    median_drag = float(np.median(np.asarray(drag, dtype=np.float64)))
    wall_seconds = time.perf_counter() - validation_started
    if min(wall_seconds, gpu_seconds) < 0.0 or gpu_seconds > wall_seconds:
        raise RuntimeError("REMOTE_TRAIN_CLOCK_ERROR: validation timing is incoherent")
    return (
        ValidationCheckpointMetric(
            experiment_id=request.experiment_id,
            dataset_id=request.dataset.artifact_id,
            config_digest=request.config.config_digest,
            checkpoint_id=checkpoint_id,
            seed=seed,
            epoch=epoch,
            median_velocity_relative_l2=median_velocity,
            median_cd_head_relative_error=median_drag,
            score=median_velocity + median_drag,
        ),
        wall_seconds,
        gpu_seconds,
    )


def _model_card() -> ModelCardMetadata:
    return ModelCardMetadata(
        display_name="Soufflerie FNO v0.1 candidate",
        summary="A deterministic three-seed FNO candidate trained on the frozen synthetic dataset.",
        intended_uses=(
            "RFC-0008 validation and bounded surrogate flow prediction after report review.",
        ),
        limitations=(
            "Scientific validation is not established by training; consult the bound report.",
            "OOD variance is a heuristic and does not establish calibrated uncertainty.",
        ),
        gates=(
            ModelCardGate(
                name="RFC-0008 release validation",
                status="not_evaluated",
                threshold="Every required validation gate must be evaluated explicitly",
            ),
        ),
    )


def _assert_device(request: RemoteTrainingRequest, device_name: str) -> None:
    expected = request.requested_device_class.casefold()
    if expected not in device_name.casefold().replace(" ", ""):
        raise ArtifactIntegrityError(
            "REMOTE_TRAIN_DEVICE_MISMATCH: allocated GPU differs from the request"
        )


def _await_completions(
    request: RemoteTrainingRequest,
    *,
    root: Path,
    reload_volume: Callable[[], None],
    clock: Callable[[], float],
) -> tuple[SeedCompletion, SeedCompletion, SeedCompletion]:
    deadline = clock() + _SELECTION_WAIT_SECONDS
    while True:
        reload_volume()
        paths = tuple(
            _seed_root(root, request, seed) / "completion.json" for seed in request.config.seeds
        )
        if all(path.is_file() and not path.is_symlink() for path in paths):
            completions = tuple(_read_completion(path) for path in paths)
            break
        if clock() >= deadline:
            raise RuntimeError(
                "REMOTE_TRAIN_BARRIER_TIMEOUT: all seed completions were not committed"
            )
        time.sleep(_POLL_SECONDS)
    identities = {
        (
            item.experiment_id,
            item.dataset_sha256,
            item.config_sha256,
            item.architecture_sha256,
            item.source_revision,
            item.lock_sha256,
            item.device_class,
            item.precision,
        )
        for item in completions
    }
    expected_identity = {
        (
            request.experiment_id,
            request.dataset.sha256,
            request.config.config_digest,
            request.architecture_sha256,
            request.source_revision,
            request.lock_sha256,
            request.requested_device_class,
            request.config.precision,
        )
    }
    if (
        identities != expected_identity
        or tuple(item.seed for item in completions) != request.config.seeds
    ):
        raise ArtifactIntegrityError(
            "REMOTE_TRAIN_BARRIER_MISMATCH: seed completion identities differ"
        )
    return cast(tuple[SeedCompletion, SeedCompletion, SeedCompletion], completions)


def execute_training_seed(
    request: RemoteTrainingRequest,
    *,
    seed: int,
    root: Path,
    build: RuntimeBuildManifest,
    reload_volume: Callable[[], None],
    commit_volume: Callable[[], None],
    clock: Callable[[], float] = time.monotonic,
) -> TrainingSeedReceipt:
    """Train/resume one seed, freeze all-seed selection, and export its safe bundle."""

    if not isinstance(request, RemoteTrainingRequest):
        raise TypeError("request must be a RemoteTrainingRequest")
    assert_request_matches_build(request, build)
    assert_training_seed(request, seed)
    reload_volume()
    seed_root = _seed_root(root, request, seed)
    receipt_path = seed_root / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _read_receipt(receipt_path)
        if (
            receipt.experiment_id != request.experiment_id
            or receipt.seed != seed
            or receipt.completed_epochs != request.config.epochs
            or receipt.accounting.device_class != request.requested_device_class
            or receipt.accounting.precision != request.config.precision
            or receipt.accounting.source_revision != request.source_revision
            or receipt.accounting.lock_sha256 != request.lock_sha256
            or receipt.parent_sha256["dataset"] != request.dataset.sha256
            or receipt.parent_sha256["config"] != request.config.config_digest
            or receipt.parent_sha256["architecture"] != request.architecture_sha256
        ):
            raise ArtifactIntegrityError(
                "REMOTE_TRAIN_RECEIPT_MISMATCH: committed receipt differs from request"
            )
        LocalModelBundleStore(root).open(receipt.model)
        return receipt
    dataset = open_manifest_dataset(root, request.dataset)
    if dataset.dataset_sha256 != request.dataset.sha256:
        raise ArtifactIntegrityError("REMOTE_TRAIN_DATASET_MISMATCH: dataset digest differs")
    cached_samples = dataset.preload_splits(("train", "validation"))
    if cached_samples != 800:
        raise ArtifactIntegrityError("REMOTE_TRAIN_CACHE_INCOMPLETE: expected 800 cached samples")
    statistics = fit_preprocessing_statistics(dataset.iter_samples("train"))
    session = prepare_training_session(
        request.config,
        experiment_id=request.experiment_id,
        seed=seed,
        device="cuda:0",
    )
    _assert_device(request, session.policy.device_name)
    writer = EpochJsonlWriter(seed_root / "epochs.jsonl")
    checkpoint_store = LocalTrainingCheckpointStore(root)
    existing_records = writer.read()
    next_epoch = 1
    checkpoint_completed_epoch: int | None = None
    latest_pointer = seed_root / "latest.json"
    if latest_pointer.exists() or latest_pointer.is_symlink():
        checkpoint = checkpoint_store.open_pointer(request.experiment_id, seed, "latest")
        expected = ResumeIdentity(
            experiment_id=request.experiment_id,
            dataset_id=request.dataset.artifact_id,
            dataset_sha256=request.dataset.sha256,
            architecture_sha256=request.architecture_sha256,
            config_digest=request.config.config_digest,
            code_revision=request.source_revision,
            lock_digest=request.lock_sha256,
            seed=seed,
            device=session.policy.device,
            device_name=session.policy.device_name,
            compute_capability=session.policy.compute_capability,
            precision=session.policy.precision,
        )
        next_epoch = restore_training_checkpoint(checkpoint, session, expected)
        checkpoint_completed_epoch = checkpoint.metadata.completed_epoch
    elif existing_records:
        raise ArtifactIntegrityError(
            "REMOTE_TRAIN_RESUME_MISMATCH: epoch log exists without a checkpoint"
        )

    metrics_path = seed_root / "validation-metrics.json"
    if metrics_path.exists() or metrics_path.is_symlink():
        metrics_payload = safe_read_json(
            metrics_path.parent,
            metrics_path.name,
            model=_ValidationMetrics,
            limits=ReaderLimits(
                max_file_bytes=_MAX_STATE_BYTES,
                max_json_bytes=_MAX_STATE_BYTES,
            ),
        )
        metrics = list(metrics_payload.metrics)
        validation_wall_seconds = list(metrics_payload.validation_wall_seconds)
        validation_gpu_seconds = list(metrics_payload.validation_gpu_seconds)
    else:
        metrics = []
        validation_wall_seconds = []
        validation_gpu_seconds = []
    if (
        checkpoint_completed_epoch is not None
        and len(existing_records) == checkpoint_completed_epoch + 1
        and len(metrics) == checkpoint_completed_epoch
        and len(validation_wall_seconds) == checkpoint_completed_epoch
        and len(validation_gpu_seconds) == checkpoint_completed_epoch
    ):
        orphan = existing_records[-1]
        recovery_path = (
            seed_root
            / "recovery"
            / f"orphaned-epoch-{orphan.epoch}-{canonical_sha256(orphan)}.json"
        )
        _publish_immutable(
            recovery_path,
            canonical_json_bytes(orphan),
            label="RECOVERY",
        )
        recovered = writer.rollback_uncheckpointed_tail(checkpoint_completed_epoch)
        if recovered != orphan:
            raise ArtifactIntegrityError(
                "REMOTE_TRAIN_RECOVERY_MISMATCH: preserved and removed records differ"
            )
        commit_volume()
        existing_records = writer.read()
    if not (
        len(metrics)
        == len(validation_wall_seconds)
        == len(validation_gpu_seconds)
        == len(existing_records)
        == (checkpoint_completed_epoch or 0)
    ):
        raise ArtifactIntegrityError(
            "REMOTE_TRAIN_RESUME_MISMATCH: validation metrics and epoch log differ"
        )

    for epoch in range(next_epoch, request.config.epochs + 1):
        record = run_training_epoch(
            session,
            dataset,
            statistics,
            epoch=epoch,
            writer=writer,
        )
        payload = capture_training_checkpoint(
            session,
            completed_epoch=epoch,
            dataset_sha256=request.dataset.sha256,
            architecture_sha256=request.architecture_sha256,
            code_revision=request.source_revision,
            lock_digest=request.lock_sha256,
        )
        metric, validation_wall, validation_gpu = _validation_metrics(
            request,
            seed=seed,
            epoch=epoch,
            checkpoint_id=payload.metadata.checkpoint_id,
            session=session,
            dataset=dataset,
            statistics=statistics,
        )
        is_best = not metrics or (metric.score, metric.epoch) < min(
            (item.score, item.epoch) for item in metrics
        )
        checkpoint_store.publish(payload, best=is_best)
        metrics.append(metric)
        validation_wall_seconds.append(validation_wall)
        validation_gpu_seconds.append(validation_gpu)
        _atomic_write(
            metrics_path,
            canonical_json_bytes(
                _ValidationMetrics(
                    metrics=tuple(metrics),
                    validation_wall_seconds=tuple(validation_wall_seconds),
                    validation_gpu_seconds=tuple(validation_gpu_seconds),
                )
            ),
        )
        commit_volume()
        existing_records = (*existing_records, record)

    records = writer.read()
    if len(records) != request.config.epochs or len(metrics) != request.config.epochs:
        raise ArtifactIntegrityError("REMOTE_TRAIN_INCOMPLETE: configured epochs did not finish")
    best_checkpoint = checkpoint_store.open_pointer(request.experiment_id, seed, "best")
    completion = SeedCompletion.create(
        experiment_id=request.experiment_id,
        dataset_sha256=request.dataset.sha256,
        config_sha256=request.config.config_digest,
        architecture_sha256=request.architecture_sha256,
        source_revision=request.source_revision,
        lock_sha256=request.lock_sha256,
        seed=seed,
        metrics=tuple(metrics),
        epoch_records=records,
        validation_wall_seconds=tuple(validation_wall_seconds),
        validation_gpu_seconds=tuple(validation_gpu_seconds),
        best_checkpoint_id=best_checkpoint.metadata.checkpoint_id,
        best_checkpoint_sha256=best_checkpoint.metadata.checkpoint_sha256,
        device_class=request.requested_device_class,
        device_name=session.policy.device_name,
        precision=session.policy.precision,
    )
    _publish_immutable(
        seed_root / "completion.json",
        canonical_json_bytes(completion),
        label="COMPLETION",
    )
    commit_volume()

    completions = _await_completions(
        request,
        root=root,
        reload_volume=reload_volume,
        clock=clock,
    )
    selection = freeze_validation_selection(
        tuple(metric for item in completions for metric in item.metrics),
        expected_seeds=request.config.seeds,
    )
    selection_path = root / "training" / request.experiment_id / "selection.json"
    if selection_path.exists() or selection_path.is_symlink():
        observed = safe_read_json(
            selection_path.parent,
            selection_path.name,
            model=type(selection),
            limits=ReaderLimits(
                max_file_bytes=_MAX_STATE_BYTES,
                max_json_bytes=_MAX_STATE_BYTES,
            ),
        )
        if observed != selection:
            raise ArtifactIntegrityError(
                "REMOTE_TRAIN_SELECTION_MISMATCH: committed selection differs"
            )
    else:
        _publish_immutable(
            selection_path,
            canonical_json_bytes(selection),
            label="SELECTION",
        )

    best_checkpoint = checkpoint_store.open_pointer(request.experiment_id, seed, "best")
    predictor = FnoPredictor().to(device="cpu")
    model_reference = export_selected_checkpoint_bundle(
        best_checkpoint,
        selection,
        predictor,
        statistics,
        dataset_sha256=request.dataset.sha256,
        code_revision=request.source_revision,
        lock_digest=request.lock_sha256,
        model_card=_model_card(),
        store=LocalModelBundleStore(root),
    )
    commit_volume()
    reload_volume()
    LocalModelBundleStore(root).open(model_reference)

    baseline_sha256s = tuple(
        baseline.metadata.baseline_sha256 for baseline in fit_baselines(dataset, statistics)
    )
    solver_sha256 = canonical_sha256(list(dataset.parent_run_sha256))
    selected_checkpoint = tuple(item for item in selection.selected if item.seed == seed)
    if len(selected_checkpoint) != 1:
        raise ArtifactIntegrityError(
            "REMOTE_TRAIN_SELECTION_MISMATCH: seed is absent from frozen selection"
        )

    total_wall = math.fsum(item.wall_seconds for item in records) + math.fsum(
        validation_wall_seconds
    )
    total_gpu = math.fsum(item.gpu_seconds for item in records) + math.fsum(validation_gpu_seconds)
    completed_at = utc_now()
    accounting = ExecutionAccounting(
        started_at=completed_at - timedelta(seconds=total_wall),
        completed_at=completed_at,
        wall_seconds=total_wall,
        gpu_seconds=total_gpu,
        peak_allocated_bytes=max(item.peak_allocated_bytes for item in records),
        peak_reserved_bytes=max(item.peak_reserved_bytes for item in records),
        device_class=request.requested_device_class,
        device_name=session.policy.device_name,
        precision=session.policy.precision,
        source_revision=request.source_revision,
        lock_sha256=request.lock_sha256,
    )
    receipt = TrainingSeedReceipt.create(
        experiment_id=request.experiment_id,
        seed=seed,
        completed_epochs=request.config.epochs,
        model=model_reference,
        best_checkpoint_id=best_checkpoint.metadata.checkpoint_id,
        validation_score=selected_checkpoint[0].validation_score,
        selection_id=selection.selection_id,
        selection_sha256=selection.selection_sha256,
        deployable_seed=selection.deployable_seed,
        baseline_sha256s=cast(tuple[str, str], baseline_sha256s),
        solver_sha256=solver_sha256,
        accounting=accounting,
        parent_sha256={
            "architecture": request.architecture_sha256,
            "best_checkpoint": best_checkpoint.metadata.checkpoint_sha256,
            "config": request.config.config_digest,
            "dataset": request.dataset.sha256,
            "model": model_reference.sha256,
        },
    )
    _publish_immutable(
        receipt_path,
        canonical_json_bytes(receipt),
        label="RECEIPT",
    )
    commit_volume()
    return receipt


class _ValidationMetrics(VersionedModel):
    metrics: tuple[ValidationCheckpointMetric, ...]
    validation_wall_seconds: tuple[FiniteNonnegative, ...]
    validation_gpu_seconds: tuple[FiniteNonnegative, ...]

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_array(cls, value: object) -> object:
        if isinstance(value, Mapping):
            normalized = dict(value)
            for name in (
                "metrics",
                "validation_wall_seconds",
                "validation_gpu_seconds",
            ):
                item = normalized.get(name)
                if isinstance(item, list):
                    normalized[name] = tuple(item)
            return normalized
        return value

    @model_validator(mode="after")
    def _timings_are_coherent(self) -> Self:
        if not (
            len(self.metrics)
            == len(self.validation_wall_seconds)
            == len(self.validation_gpu_seconds)
        ):
            raise ValueError("validation timing arrays must align with metrics")
        if any(
            gpu > wall
            for gpu, wall in zip(
                self.validation_gpu_seconds,
                self.validation_wall_seconds,
                strict=True,
            )
        ):
            raise ValueError("validation GPU seconds cannot exceed wall seconds")
        return self


__all__ = ["SeedCompletion", "execute_training_seed"]
