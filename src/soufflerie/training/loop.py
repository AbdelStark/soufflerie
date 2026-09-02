"""Single-GPU deterministic mixed-precision optimization loop."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, cast

import numpy as np

from soufflerie.config import TrainingConfig
from soufflerie.errors import ArtifactIntegrityError, InternalInvariantError
from soufflerie.surrogate.fno import FnoPredictor
from soufflerie.surrogate.preprocessing import (
    PredictionBatch,
    PredictionBatchResult,
    PreprocessedBatch,
    PreprocessingStatistics,
)
from soufflerie.training.data import ManifestDataset
from soufflerie.training.loss import LossScalars, masked_training_loss
from soufflerie.training.metrics import (
    EpochJsonlWriter,
    EpochLossMetrics,
    TrainingEpochRecord,
)
from soufflerie.training.runtime import (
    PrecisionPolicy,
    apply_learning_rate,
    build_adamw,
    configure_determinism,
    learning_rate_for_epoch,
    preflight_precision,
    require_torch,
)


class TrainingModel(Protocol):
    def __call__(self, batch: PredictionBatch) -> PredictionBatchResult: ...

    def train(self, mode: bool = True) -> TrainingModel: ...

    def to(self, *args: object, **kwargs: object) -> TrainingModel: ...

    def parameters(self, *, recurse: bool = True) -> Iterable[Any]: ...

    def named_parameters(self, *, recurse: bool = True) -> Iterable[tuple[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class TrainingTensorBatch:
    """Device-resident model inputs and supervised targets."""

    inputs: PredictionBatch
    target: PredictionBatchResult

    @property
    def sample_count(self) -> int:
        return int(cast(tuple[int, ...], self.inputs.inputs.shape)[0])


def training_batch_to_torch(
    batch: PreprocessedBatch,
    *,
    device: str,
    torch_module: ModuleType | None = None,
) -> TrainingTensorBatch:
    """Own and explicitly transfer every NumPy training member to one device."""

    if not isinstance(batch, PreprocessedBatch):
        raise TypeError("batch must be a PreprocessedBatch")
    torch = cast(Any, torch_module or require_torch())

    def convert(value: np.ndarray[Any, Any]) -> Any:
        owned = np.array(value, copy=True, order="C")
        try:
            return torch.from_numpy(owned).to(device=device)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise ArtifactIntegrityError(
                f"TRAIN-9 BATCH: unable to transfer a training tensor to {device}"
            ) from error

    inputs = PredictionBatch(
        inputs=convert(batch.inputs),
        fluid_mask=convert(batch.fluid_mask),
        design_params=convert(batch.design_params),
    )
    target = PredictionBatchResult(
        fields_normalized=convert(batch.fields_normalized),
        cd_head=convert(batch.cd),
    )
    return TrainingTensorBatch(inputs=inputs, target=target)


@dataclass(slots=True)
class TrainingSession:
    """Initialized model/optimizer/scaler state for one canonical seed."""

    experiment_id: str
    config: TrainingConfig
    seed: int
    policy: PrecisionPolicy
    model: TrainingModel
    optimizer: Any
    torch: ModuleType
    scaler: Any | None = None
    global_step: int = 0

    def synchronize(self) -> None:
        try:
            cast(Any, self.torch).cuda.synchronize(self.policy.device_index)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise InternalInvariantError("TRAIN-10 RUNTIME: CUDA synchronization failed") from error

    def reset_peak_memory(self) -> None:
        try:
            cast(Any, self.torch).cuda.reset_peak_memory_stats(self.policy.device_index)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise InternalInvariantError("TRAIN-10 RUNTIME: CUDA memory reset failed") from error

    def peak_memory(self) -> tuple[int, int]:
        try:
            torch = cast(Any, self.torch)
            allocated = int(torch.cuda.max_memory_allocated(self.policy.device_index))
            reserved = int(torch.cuda.max_memory_reserved(self.policy.device_index))
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise InternalInvariantError(
                "TRAIN-10 RUNTIME: CUDA memory accounting failed"
            ) from error
        if allocated < 0 or reserved < allocated:
            raise InternalInvariantError("TRAIN-10 RUNTIME: incoherent CUDA memory accounting")
        return allocated, reserved

    def train_batch(self, batch: TrainingTensorBatch) -> LossScalars:
        """Run one autocast forward and fp32 loss/backward/update step."""

        if not isinstance(batch, TrainingTensorBatch):
            raise TypeError("batch must be a TrainingTensorBatch")
        torch = cast(Any, self.torch)
        try:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=self.policy.autocast_dtype,
                enabled=True,
            ):
                prediction = self.model(batch.inputs)
                loss = masked_training_loss(
                    prediction,
                    batch.target,
                    batch.inputs.fluid_mask,
                    self.config,
                    torch_module=self.torch,
                )
            if self.scaler is None:
                loss.total.backward()
            else:
                self.scaler.scale(loss.total).backward()
                self.scaler.unscale_(self.optimizer)
            parameters = tuple(self.model.parameters(recurse=True))
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float(self.config.gradient_clip_norm),
                error_if_nonfinite=True,
            )
            norm_value = float(gradient_norm.detach().item())
            if not math.isfinite(norm_value):
                raise ArtifactIntegrityError("TRAIN-6 OPTIMIZER: gradient norm is non-finite")
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            values = loss.detached()
        except RuntimeError as error:
            if "determin" in str(error).casefold():
                raise ArtifactIntegrityError(
                    "TRAIN-4 DETERMINISM: framework rejected a nondeterministic operation"
                ) from error
            raise
        self.global_step += 1
        return values


def prepare_training_session(
    config: TrainingConfig,
    *,
    experiment_id: str,
    seed: int,
    device: str,
    model_factory: Callable[[], TrainingModel] = FnoPredictor,
    torch_module: ModuleType | None = None,
) -> TrainingSession:
    """Preflight, seed, then initialize a model so weights reproduce by seed."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if seed not in config.seeds:
        raise ArtifactIntegrityError("TRAIN-4 SEED: seed is not declared by the training config")
    if (
        not isinstance(experiment_id, str)
        or len(experiment_id) != 20
        or any(character not in "0123456789abcdef" for character in experiment_id)
    ):
        raise ArtifactIntegrityError("TRAIN-10 IDENTITY: experiment_id must be a content ID")
    torch = torch_module or require_torch()
    policy = preflight_precision(device, config.precision, torch_module=torch)
    configure_determinism(seed, torch_module=torch)
    try:
        model = model_factory().to(device=policy.device)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise InternalInvariantError("TRAIN-10 RUNTIME: model initialization failed") from error
    optimizer = build_adamw(model, config, torch_module=torch)
    scaler: Any | None = None
    if policy.precision == "fp16":
        try:
            scaler = cast(Any, torch).amp.GradScaler("cuda", enabled=True)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise InternalInvariantError(
                "TRAIN-10 RUNTIME: fp16 scaler construction failed"
            ) from error
    session = TrainingSession(
        experiment_id=experiment_id,
        config=config,
        seed=seed,
        policy=policy,
        model=model,
        optimizer=optimizer,
        torch=torch,
        scaler=scaler,
    )
    session.reset_peak_memory()
    return session


def _mean_losses(weighted: list[tuple[int, LossScalars]], samples: int) -> EpochLossMetrics:
    if samples <= 0 or not weighted:
        raise InternalInvariantError("TRAIN-10 EPOCH: no training samples were optimized")

    def mean(name: str) -> float:
        return math.fsum(count * getattr(values, name) for count, values in weighted) / samples

    return EpochLossMetrics(
        u=mean("u"),
        v=mean("v"),
        rho=mean("rho"),
        obstacle=mean("obstacle"),
        cd=mean("cd"),
        total=mean("total"),
    )


def run_training_epoch(
    session: TrainingSession,
    dataset: ManifestDataset,
    statistics: PreprocessingStatistics,
    *,
    epoch: int,
    writer: EpochJsonlWriter,
    clock: Callable[[], float] = time.perf_counter,
) -> TrainingEpochRecord:
    """Optimize every train row once and atomically append complete epoch evidence."""

    if not isinstance(session, TrainingSession):
        raise TypeError("session must be a TrainingSession")
    if not isinstance(dataset, ManifestDataset):
        raise TypeError("dataset must be a ManifestDataset")
    if not isinstance(statistics, PreprocessingStatistics):
        raise TypeError("statistics must be PreprocessingStatistics")
    if not isinstance(writer, EpochJsonlWriter):
        raise TypeError("writer must be an EpochJsonlWriter")
    if dataset.reference.artifact_id != session.config.dataset_id:
        raise ArtifactIntegrityError("TRAIN-10 IDENTITY: config and dataset identities differ")
    existing = writer.read()
    if epoch != len(existing) + 1:
        raise ArtifactIntegrityError("TRAIN-8 JSONL: requested epoch is not the next log record")
    if existing:
        last = existing[-1]
        if (
            last.experiment_id != session.experiment_id
            or last.dataset_id != dataset.reference.artifact_id
            or last.config_digest != session.config.config_digest
            or last.seed != session.seed
            or last.global_step != session.global_step
        ):
            raise ArtifactIntegrityError(
                "TRAIN-10 IDENTITY: session does not continue the epoch log"
            )
    elif session.global_step != 0:
        raise ArtifactIntegrityError("TRAIN-10 IDENTITY: a new epoch log requires global_step zero")

    learning_rate = learning_rate_for_epoch(session.config, epoch)
    apply_learning_rate(session.optimizer, learning_rate)
    session.model.train(True)
    session.reset_peak_memory()
    wall_started = clock()
    io_seconds = 0.0
    compute_seconds = 0.0
    samples = 0
    batch_count = 0
    losses: list[tuple[int, LossScalars]] = []
    iterator = iter(
        dataset.iter_batches(
            statistics,
            "train",
            batch_size=session.config.batch_size,
            seed=session.seed,
            epoch=epoch - 1,
        )
    )
    while True:
        io_started = clock()
        try:
            manifest_batch = next(iterator)
        except StopIteration:
            io_seconds += clock() - io_started
            break
        tensor_batch = training_batch_to_torch(
            manifest_batch.data,
            device=session.policy.device,
            torch_module=session.torch,
        )
        io_seconds += clock() - io_started
        session.synchronize()
        compute_started = clock()
        values = session.train_batch(tensor_batch)
        session.synchronize()
        compute_seconds += clock() - compute_started
        count = tensor_batch.sample_count
        samples += count
        batch_count += 1
        losses.append((count, values))
    wall_seconds = clock() - wall_started
    if samples != len(dataset.split_rows("train")):
        raise InternalInvariantError("TRAIN-10 EPOCH: optimized sample count changed")
    if min(wall_seconds, io_seconds, compute_seconds) < 0.0:
        raise InternalInvariantError("TRAIN-10 EPOCH: monotonic clock moved backwards")
    peak_allocated, peak_reserved = session.peak_memory()
    record = TrainingEpochRecord(
        experiment_id=session.experiment_id,
        dataset_id=dataset.reference.artifact_id,
        config_digest=session.config.config_digest,
        seed=session.seed,
        epoch=epoch,
        batches=batch_count,
        samples=samples,
        global_step=session.global_step,
        learning_rate=learning_rate,
        precision=session.policy.precision,
        device=session.policy.device,
        device_name=session.policy.device_name,
        compute_capability=session.policy.compute_capability,
        loss=_mean_losses(losses, samples),
        wall_seconds=wall_seconds,
        compute_seconds=compute_seconds,
        io_seconds=io_seconds,
        gpu_seconds=compute_seconds,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )
    writer.append(record)
    return record


__all__ = [
    "TrainingModel",
    "TrainingSession",
    "TrainingTensorBatch",
    "prepare_training_session",
    "run_training_epoch",
    "training_batch_to_torch",
]
