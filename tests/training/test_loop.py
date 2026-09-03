from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Self, cast

import numpy as np
import pytest
from pydantic import ValidationError

import soufflerie.training.loop as loop_module
from soufflerie.config import TrainingConfig
from soufflerie.errors import ArtifactIntegrityError, InternalInvariantError
from soufflerie.surrogate.preprocessing import (
    MODEL_SPATIAL_SHAPE,
    PredictionBatch,
    PredictionBatchResult,
    PreprocessedBatch,
)
from soufflerie.training import (
    EpochJsonlWriter,
    EpochLossMetrics,
    LossScalars,
    PrecisionPolicy,
    TrainingEpochRecord,
    TrainingSession,
    TrainingTensorBatch,
    prepare_training_session,
    run_training_epoch,
    training_batch_to_torch,
)
from soufflerie.training.data import ManifestDataset
from tests.training.helpers import build_harness


class ShapeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: str = "torch.float32",
        device: str = "cuda:0",
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def is_contiguous(self) -> bool:
        return True

    def isfinite(self) -> ShapeTensor:
        return ShapeTensor((), dtype="torch.bool", device=self.device)

    def all(self) -> ShapeTensor:
        return self

    def item(self) -> object:
        return True


class DeviceTensor:
    def __init__(self, value: np.ndarray[Any, Any], *, device: str = "cpu") -> None:
        self.value = np.ascontiguousarray(value)
        self.device = device
        self.dtype = "torch.bool" if self.value.dtype == np.dtype(np.bool_) else "torch.float32"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def to(self, *, device: str) -> DeviceTensor:
        return DeviceTensor(self.value, device=device)

    def is_contiguous(self) -> bool:
        return True

    def isfinite(self) -> DeviceTensor:
        return DeviceTensor(np.isfinite(self.value), device=self.device)

    def all(self) -> DeviceTensor:
        return DeviceTensor(np.asarray(self.value.all(), dtype=np.bool_), device=self.device)

    def item(self) -> object:
        return self.value.item()


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value
        self.dtype = "torch.float32"
        self.backward_calls = 0

    def backward(self) -> None:
        self.backward_calls += 1

    def detach(self) -> Self:
        return self

    def item(self) -> float:
        return self.value

    def new_tensor(self, value: float) -> Scalar:
        return Scalar(value)

    def to(self, *, dtype: str) -> Self:
        assert dtype == self.dtype
        return self


class FakeCuda:
    def synchronize(self, index: int) -> None:
        assert index == 0

    def reset_peak_memory_stats(self, index: int) -> None:
        assert index == 0

    def max_memory_allocated(self, index: int) -> int:
        assert index == 0
        return 100

    def max_memory_reserved(self, index: int) -> int:
        assert index == 0
        return 120


class FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups: list[dict[str, object]] = [{"lr": 1e-3}]
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none
        self.zero_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class FakeModel:
    def __init__(self) -> None:
        self.training = False
        self.parameter = SimpleNamespace(ndim=2, requires_grad=True)
        self.target: PredictionBatchResult | None = None
        self.device: str | None = None

    def __call__(self, batch: PredictionBatch) -> PredictionBatchResult:
        assert isinstance(batch, PredictionBatch)
        assert self.target is not None
        return self.target

    def train(self, mode: bool = True) -> Self:
        self.training = mode
        return self

    def to(self, *args: object, **kwargs: object) -> Self:
        self.device = cast(str, kwargs.get("device", args[0] if args else None))
        return self

    def parameters(self, *, recurse: bool = True) -> tuple[object, ...]:
        assert recurse
        return (self.parameter,)

    def named_parameters(self, *, recurse: bool = True) -> tuple[tuple[str, object], ...]:
        return (("core.weight", self.parameter),)


class FakeTorch(ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.cuda = FakeCuda()
        self.nn = SimpleNamespace(
            utils=SimpleNamespace(clip_grad_norm_=lambda *_args, **_kwargs: Scalar(0.5))
        )
        self.amp = SimpleNamespace(GradScaler=lambda *_args, **_kwargs: FakeScaler())

    @staticmethod
    def from_numpy(value: np.ndarray[Any, Any]) -> DeviceTensor:
        return DeviceTensor(value)

    @staticmethod
    def autocast(**kwargs: object) -> Any:
        assert kwargs == {"device_type": "cuda", "dtype": "bf16", "enabled": True}
        return nullcontext()


class FakeScaler:
    def __init__(self) -> None:
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, value: Scalar) -> Scalar:
        return value

    def unscale_(self, optimizer: FakeOptimizer) -> None:
        assert isinstance(optimizer, FakeOptimizer)
        self.unscale_calls += 1

    def step(self, optimizer: FakeOptimizer) -> None:
        optimizer.step()
        self.step_calls += 1

    def update(self) -> None:
        self.update_calls += 1


def _config(*, dataset_id: str = "a" * 20, precision: str = "bf16") -> TrainingConfig:
    return TrainingConfig(
        dataset_id=dataset_id,
        seeds=(17, 23, 31),
        precision=cast(Any, precision),
    )


def _policy(*, precision: str = "bf16") -> PrecisionPolicy:
    return PrecisionPolicy(
        precision=cast(Any, precision),
        device="cuda:0",
        device_index=0,
        autocast_dtype="bf16",
        device_name="Contract GPU",
        compute_capability="8.9",
    )


def _readonly(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    value.flags.writeable = False
    return value


def _preprocessed_batch() -> PreprocessedBatch:
    return PreprocessedBatch(
        inputs=_readonly(np.zeros((1, 2, *MODEL_SPATIAL_SHAPE), dtype=np.float32)),
        fields_normalized=_readonly(np.zeros((1, 3, *MODEL_SPATIAL_SHAPE), dtype=np.float32)),
        fluid_mask=_readonly(np.ones((1, 1, *MODEL_SPATIAL_SHAPE), dtype=np.bool_)),
        design_params=_readonly(np.zeros((1, 4), dtype=np.float32)),
        cd=_readonly(np.zeros((1,), dtype=np.float32)),
    )


def _tensor_batch(samples: int = 1) -> TrainingTensorBatch:
    return TrainingTensorBatch(
        inputs=PredictionBatch(
            inputs=cast(Any, ShapeTensor((samples, 2, *MODEL_SPATIAL_SHAPE))),
            fluid_mask=cast(
                Any,
                ShapeTensor(
                    (samples, 1, *MODEL_SPATIAL_SHAPE),
                    dtype="torch.bool",
                ),
            ),
            design_params=cast(Any, ShapeTensor((samples, 4))),
        ),
        target=PredictionBatchResult(
            fields_normalized=cast(
                Any,
                ShapeTensor((samples, 3, *MODEL_SPATIAL_SHAPE)),
            ),
            cd_head=cast(Any, ShapeTensor((samples,))),
        ),
    )


def _record(*, epoch: int, global_step: int, seed: int = 17) -> TrainingEpochRecord:
    return TrainingEpochRecord(
        experiment_id="b" * 20,
        dataset_id="a" * 20,
        config_digest="c" * 64,
        seed=seed,
        epoch=epoch,
        batches=2,
        samples=8,
        global_step=global_step,
        learning_rate=1e-3,
        precision="bf16",
        device="cuda:0",
        device_name="Contract GPU",
        compute_capability="8.9",
        loss=EpochLossMetrics(u=1, v=2, rho=3, obstacle=4, cd=5, total=6),
        wall_seconds=3.0,
        compute_seconds=2.0,
        io_seconds=0.5,
        gpu_seconds=2.0,
        peak_allocated_bytes=100,
        peak_reserved_bytes=120,
    )


def test_training_batch_transfer_is_explicit_complete_and_single_device() -> None:
    converted = training_batch_to_torch(
        _preprocessed_batch(),
        device="cuda:0",
        torch_module=FakeTorch(),
    )
    assert converted.sample_count == 1
    assert converted.inputs.inputs.device == "cuda:0"
    assert converted.target.fields_normalized.device == "cuda:0"


def test_training_session_clips_and_updates_bf16_and_fp16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _tensor_batch()
    fake_loss = SimpleNamespace(
        u=Scalar(1),
        v=Scalar(2),
        rho=Scalar(3),
        obstacle=Scalar(4),
        cd=Scalar(5),
        total=Scalar(6),
        detached=lambda: LossScalars(u=1, v=2, rho=3, obstacle=4, cd=5, total=6),
    )
    monkeypatch.setattr(loop_module, "masked_training_loss", lambda *_args, **_kwargs: fake_loss)
    for scaler in (None, FakeScaler()):
        model = FakeModel()
        model.target = batch.target
        optimizer = FakeOptimizer()
        session = TrainingSession(
            experiment_id="b" * 20,
            config=_config(precision="fp16" if scaler else "bf16"),
            seed=17,
            policy=_policy(precision="fp16" if scaler else "bf16"),
            model=model,
            optimizer=optimizer,
            torch=FakeTorch(),
            scaler=scaler,
        )
        assert session.train_batch(batch).total == 6
        assert optimizer.zero_calls == optimizer.step_calls == 1
        assert session.global_step == 1
        if scaler is not None:
            assert scaler.unscale_calls == scaler.step_calls == scaler.update_calls == 1


def test_preparation_preflights_and_seeds_before_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    model = FakeModel()
    optimizer = FakeOptimizer()

    def preflight(*_args: object, **_kwargs: object) -> PrecisionPolicy:
        events.append("preflight")
        return _policy()

    def seed(*_args: object, **_kwargs: object) -> None:
        events.append("seed")

    def optimizer_factory(*_args: object, **_kwargs: object) -> FakeOptimizer:
        events.append("optimizer")
        return optimizer

    monkeypatch.setattr(
        loop_module,
        "preflight_precision",
        preflight,
    )
    monkeypatch.setattr(
        loop_module,
        "configure_determinism",
        seed,
    )
    monkeypatch.setattr(
        loop_module,
        "build_adamw",
        optimizer_factory,
    )

    def factory() -> FakeModel:
        events.append("model")
        return model

    session = prepare_training_session(
        _config(),
        experiment_id="b" * 20,
        seed=17,
        device="cuda:0",
        model_factory=factory,
        torch_module=FakeTorch(),
    )
    assert events == ["preflight", "seed", "model", "optimizer"]
    assert session.model is model
    assert model.device == "cuda:0"


def test_epoch_jsonl_is_append_only_contiguous_and_schema_bound(tmp_path: Path) -> None:
    path = tmp_path / "epoch.jsonl"
    writer = EpochJsonlWriter(path)
    first = _record(epoch=1, global_step=2)
    second = _record(epoch=2, global_step=4)
    writer.append(first)
    writer.append(second)
    assert writer.read() == (first, second)
    assert path.read_text(encoding="utf-8").count("\n") == 2

    with pytest.raises(ArtifactIntegrityError, match="identity changed"):
        writer.append(_record(epoch=3, global_step=6, seed=23))
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="partial final"):
        writer.read()
    with pytest.raises(ValidationError, match="single-GPU accounting"):
        _record(epoch=1, global_step=2).model_copy(
            update={"gpu_seconds": 1.0}
        ).__class__.model_validate(
            {**_record(epoch=1, global_step=2).model_dump(), "gpu_seconds": 1.0}
        )


def test_epoch_jsonl_rolls_back_exactly_one_uncheckpointed_tail(tmp_path: Path) -> None:
    writer = EpochJsonlWriter(tmp_path / "epochs.jsonl")
    records = tuple(_record(epoch=epoch, global_step=epoch * 2) for epoch in range(1, 4))
    for record in records:
        writer.append(record)

    assert writer.rollback_uncheckpointed_tail(2) == records[-1]
    assert writer.read() == records[:2]
    assert writer.path.read_text(encoding="utf-8").count("\n") == 2

    with pytest.raises(ArtifactIntegrityError, match="exactly one"):
        writer.rollback_uncheckpointed_tail(2)
    with pytest.raises(ArtifactIntegrityError, match="must be positive"):
        writer.rollback_uncheckpointed_tail(True)


def test_epoch_schema_and_writer_reject_incoherent_or_corrupt_evidence(tmp_path: Path) -> None:
    base = _record(epoch=1, global_step=2).model_dump()
    for update, message in (
        ({"wall_seconds": 1.0}, "cannot exceed epoch wall"),
        ({"peak_allocated_bytes": 121}, "cannot exceed reserved"),
        ({"global_step": 1}, "smaller than completed"),
    ):
        with pytest.raises(ValidationError, match=message):
            TrainingEpochRecord.model_validate({**base, **update})

    with pytest.raises(TypeError, match="path must"):
        EpochJsonlWriter(cast(Path, "epochs.jsonl"))
    writer = EpochJsonlWriter(tmp_path / "epochs.jsonl")
    with pytest.raises(TypeError, match="record must"):
        writer.append(cast(TrainingEpochRecord, object()))

    writer.path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="record is invalid"):
        writer.read()
    writer.path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="empty or oversized"):
        writer.read()
    writer.path.unlink()
    writer.path.mkdir()
    with pytest.raises(ArtifactIntegrityError, match="regular file"):
        writer.read()
    writer.path.rmdir()
    (tmp_path / "target.jsonl").write_text("", encoding="utf-8")
    writer.path.symlink_to(tmp_path / "target.jsonl")
    with pytest.raises(ArtifactIntegrityError, match="must not be a symlink"):
        writer.read()

    with pytest.raises(ArtifactIntegrityError, match="epochs must be contiguous"):
        EpochJsonlWriter._validate_sequence((_record(epoch=2, global_step=2),))
    with pytest.raises(ArtifactIntegrityError, match="global step accounting"):
        EpochJsonlWriter._validate_sequence((_record(epoch=1, global_step=3),))


def test_session_runtime_guards_conversion_memory_and_nondeterminism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="PreprocessedBatch"):
        training_batch_to_torch(cast(PreprocessedBatch, object()), device="cuda:0")

    class BrokenTorch(FakeTorch):
        @staticmethod
        def from_numpy(value: np.ndarray[Any, Any]) -> DeviceTensor:
            raise TypeError("malformed tensor runtime")

    broken_torch = BrokenTorch()
    with pytest.raises(ArtifactIntegrityError, match="unable to transfer"):
        training_batch_to_torch(
            _preprocessed_batch(),
            device="cuda:0",
            torch_module=broken_torch,
        )

    batch = _tensor_batch()
    model = FakeModel()
    model.target = batch.target
    session = TrainingSession(
        experiment_id="b" * 20,
        config=_config(),
        seed=17,
        policy=_policy(),
        model=model,
        optimizer=FakeOptimizer(),
        torch=FakeTorch(),
    )
    with pytest.raises(TypeError, match="TrainingTensorBatch"):
        session.train_batch(cast(TrainingTensorBatch, object()))

    class IncoherentCuda(FakeCuda):
        def max_memory_allocated(self, index: int) -> int:
            assert index == 0
            return 121

    cast(FakeTorch, session.torch).cuda = IncoherentCuda()
    with pytest.raises(InternalInvariantError, match="incoherent CUDA memory"):
        session.peak_memory()

    def nondeterministic(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("operation has no deterministic implementation")

    monkeypatch.setattr(loop_module, "masked_training_loss", nondeterministic)
    with pytest.raises(ArtifactIntegrityError, match="nondeterministic operation"):
        session.train_batch(batch)


def test_run_epoch_weights_partial_batches_and_records_gpu_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_harness()
    dataset = harness.open(tmp_path, monkeypatch)
    config = _config(dataset_id=dataset.reference.artifact_id)
    model = FakeModel()
    session = TrainingSession(
        experiment_id="b" * 20,
        config=config,
        seed=17,
        policy=_policy(),
        model=model,
        optimizer=FakeOptimizer(),
        torch=FakeTorch(),
    )
    batches = (SimpleNamespace(data=object()), SimpleNamespace(data=object()))
    monkeypatch.setattr(ManifestDataset, "iter_batches", lambda *_args, **_kwargs: iter(batches))
    tensor_batches = iter((_tensor_batch(400), _tensor_batch(200)))
    monkeypatch.setattr(
        loop_module,
        "training_batch_to_torch",
        lambda *_args, **_kwargs: next(tensor_batches),
    )
    values = iter(
        (
            LossScalars(u=1, v=1, rho=1, obstacle=1, cd=1, total=1),
            LossScalars(u=4, v=4, rho=4, obstacle=4, cd=4, total=4),
        )
    )

    def train_batch(active: TrainingSession, _batch: TrainingTensorBatch) -> LossScalars:
        active.global_step += 1
        return next(values)

    monkeypatch.setattr(TrainingSession, "train_batch", train_batch)
    writer = EpochJsonlWriter(tmp_path / "metrics" / "epochs.jsonl")
    record = run_training_epoch(
        session,
        dataset,
        harness.statistics,
        epoch=1,
        writer=writer,
    )
    assert record.samples == 600
    assert record.batches == record.global_step == 2
    assert record.loss.total == pytest.approx(2.0)
    assert record.gpu_seconds == record.compute_seconds
    assert record.peak_allocated_bytes == 100
    assert writer.read() == (record,)
    with pytest.raises(ArtifactIntegrityError, match="not the next"):
        run_training_epoch(
            session,
            dataset,
            harness.statistics,
            epoch=1,
            writer=writer,
        )
