from __future__ import annotations

import json
import random
from pathlib import Path
from types import ModuleType
from typing import Any, Self, cast

import numpy as np
import pytest

import soufflerie.training.checkpoint as checkpoint_module
from soufflerie.config import TrainingConfig
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ArtifactRef
from soufflerie.training import (
    LocalTrainingCheckpointStore,
    PrecisionPolicy,
    ResumeIdentity,
    StateCodec,
    TrainingSession,
    ValidationCheckpointMetric,
    capture_training_checkpoint,
    export_selected_checkpoint_bundle,
    freeze_validation_selection,
    restore_training_checkpoint,
)


class JsonCodec(StateCodec):
    def encode(self, value: object) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(self, content: bytes, *, map_location: str) -> object:
        assert map_location in {"cpu", "cuda:0"}
        return json.loads(content)


class FakeModel:
    def __init__(self, weight: float) -> None:
        self.weight = weight

    def state_dict(self) -> dict[str, float]:
        return {"weight": self.weight}

    def load_state_dict(self, state: object, *, strict: bool) -> None:
        assert strict
        self.weight = float(cast(dict[str, float], state)["weight"])

    def train(self, mode: bool = True) -> Self:
        return self

    def to(self, *args: object, **kwargs: object) -> Self:
        return self

    def parameters(self, *, recurse: bool = True) -> tuple[object, ...]:
        return ()

    def named_parameters(self, *, recurse: bool = True) -> tuple[tuple[str, object], ...]:
        return ()

    def __call__(self, batch: object) -> object:
        return batch


class FakeOptimizer:
    def __init__(self, momentum: float, learning_rate: float = 5e-4) -> None:
        self.momentum = momentum
        self.param_groups: list[dict[str, float]] = [{"lr": learning_rate}]

    def state_dict(self) -> dict[str, object]:
        return {"momentum": self.momentum, "param_groups": self.param_groups}

    def load_state_dict(self, state: object) -> None:
        values = cast(dict[str, Any], state)
        self.momentum = float(values["momentum"])
        self.param_groups = cast(list[dict[str, float]], values["param_groups"])


class FakeCuda:
    def __init__(self) -> None:
        self.state: list[int] = [2, 3]

    def get_rng_state_all(self) -> list[int]:
        return list(self.state)

    def set_rng_state_all(self, state: object) -> None:
        self.state = [int(value) for value in cast(list[int], state)]


class FakeTorch(ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.cpu_state: list[int] = [1]
        self.cuda = FakeCuda()

    def get_rng_state(self) -> list[int]:
        return list(self.cpu_state)

    def set_rng_state(self, state: object) -> None:
        self.cpu_state = [int(value) for value in cast(list[int], state)]


def _config() -> TrainingConfig:
    return TrainingConfig(dataset_id="a" * 20, seeds=(17, 23, 31))


def _policy() -> PrecisionPolicy:
    return PrecisionPolicy(
        precision="bf16",
        device="cuda:0",
        device_index=0,
        autocast_dtype="torch.bfloat16",
        device_name="Contract GPU",
        compute_capability="8.9",
    )


def _session() -> TrainingSession:
    return TrainingSession(
        experiment_id="e" * 20,
        config=_config(),
        seed=17,
        policy=_policy(),
        model=cast(Any, FakeModel(3.5)),
        optimizer=FakeOptimizer(0.75),
        torch=FakeTorch(),
        global_step=75,
    )


def _capture(session: TrainingSession) -> Any:
    return capture_training_checkpoint(
        session,
        completed_epoch=1,
        dataset_sha256="a" * 64,
        architecture_sha256="b" * 64,
        code_revision="c" * 40,
        lock_digest="d" * 64,
        codec=JsonCodec(),
    )


def _identity(payload: Any) -> ResumeIdentity:
    metadata = payload.metadata
    return ResumeIdentity(
        **metadata.model_dump(include=set(ResumeIdentity.model_fields), mode="python")
    )


def test_checkpoint_round_trip_restores_exact_next_epoch_state_and_rng(tmp_path: Path) -> None:
    random.seed(123)
    np.random.seed(123)
    session = _session()
    payload = _capture(session)
    expected_random = (random.random(), float(np.random.random()))
    store = LocalTrainingCheckpointStore(tmp_path)
    published = store.publish(payload, best=True)
    assert store.open_pointer("e" * 20, 17, "latest").metadata == payload.metadata
    assert store.open_pointer("e" * 20, 17, "best").metadata == payload.metadata

    cast(FakeModel, session.model).weight = -1.0
    cast(FakeOptimizer, session.optimizer).momentum = -1.0
    cast(FakeOptimizer, session.optimizer).param_groups[0]["lr"] = 9e-3
    cast(FakeTorch, session.torch).cpu_state = [99]
    random.seed(999)
    np.random.seed(999)
    session.global_step = 0

    next_epoch = restore_training_checkpoint(
        published,
        session,
        _identity(payload),
        codec=JsonCodec(),
    )
    assert next_epoch == 2
    assert session.global_step == 75
    assert cast(FakeModel, session.model).weight == 3.5
    assert cast(FakeOptimizer, session.optimizer).momentum == 0.75
    assert cast(FakeTorch, session.torch).cpu_state == [1]
    assert (random.random(), float(np.random.random())) == expected_random


def test_checkpoint_store_retains_only_latest_and_current_best(tmp_path: Path) -> None:
    session = _session()
    store = LocalTrainingCheckpointStore(tmp_path)
    first = _capture(session)
    store.publish(first, best=True)

    session.global_step = 150
    second = capture_training_checkpoint(
        session,
        completed_epoch=2,
        dataset_sha256="a" * 64,
        architecture_sha256="b" * 64,
        code_revision="c" * 40,
        lock_digest="d" * 64,
        codec=JsonCodec(),
    )
    store.publish(second)
    checkpoints = tmp_path / "training" / ("e" * 20) / "17" / "checkpoints"
    assert {path.name for path in checkpoints.iterdir()} == {
        first.metadata.checkpoint_id,
        second.metadata.checkpoint_id,
    }

    session.global_step = 225
    third = capture_training_checkpoint(
        session,
        completed_epoch=3,
        dataset_sha256="a" * 64,
        architecture_sha256="b" * 64,
        code_revision="c" * 40,
        lock_digest="d" * 64,
        codec=JsonCodec(),
    )
    store.publish(third, best=True)
    assert {path.name for path in checkpoints.iterdir()} == {third.metadata.checkpoint_id}
    assert store.open_pointer("e" * 20, 17, "latest").metadata == third.metadata
    assert store.open_pointer("e" * 20, 17, "best").metadata == third.metadata


def test_checkpoint_store_rejects_path_traversal_and_pointer_symlink(tmp_path: Path) -> None:
    store = LocalTrainingCheckpointStore(tmp_path)
    payload = _capture(_session())
    store.publish(payload)

    with pytest.raises(ArtifactIntegrityError, match="identity is malformed"):
        store.open_pointer(cast(Any, "../../outside"), 17, "latest")
    with pytest.raises(ArtifactIntegrityError, match="identity is malformed"):
        store.open("e" * 20, cast(Any, "../../outside"), payload.metadata.checkpoint_id)

    pointer = tmp_path / "training" / ("e" * 20) / "17" / "latest.json"
    target = tmp_path / "pointer-target.json"
    target.write_bytes(pointer.read_bytes())
    pointer.unlink()
    pointer.symlink_to(target)
    with pytest.raises(ArtifactIntegrityError, match="not a regular file"):
        store.open_pointer("e" * 20, 17, "latest")


@pytest.mark.parametrize(
    "member",
    [
        "model.pt",
        "optimizer.pt",
        "scheduler.json",
        "rng.json",
        "rng.pt",
        "metadata.json",
        "COMMITTED",
    ],
)
def test_every_corrupt_checkpoint_member_refuses_open(tmp_path: Path, member: str) -> None:
    payload = _capture(_session())
    store = LocalTrainingCheckpointStore(tmp_path)
    published = store.publish(payload)
    path = published.root / member
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ArtifactIntegrityError):
        store.open("e" * 20, 17, payload.metadata.checkpoint_id)


def test_every_resume_identity_dimension_is_mandatory(tmp_path: Path) -> None:
    session = _session()
    payload = _capture(session)
    checkpoint = LocalTrainingCheckpointStore(tmp_path).publish(payload)
    expected = _identity(payload)
    updates: tuple[dict[str, object], ...] = (
        {"experiment_id": "f" * 20},
        {"dataset_id": "0" * 20},
        {"dataset_sha256": "0" * 64},
        {"architecture_sha256": "0" * 64},
        {"config_digest": "0" * 64},
        {"code_revision": "0" * 40},
        {"lock_digest": "0" * 64},
        {"seed": 23},
        {"device": "cuda:1"},
        {"device_name": "Different GPU"},
        {"compute_capability": "9.0"},
        {"precision": "fp16"},
    )
    for update in updates:
        with pytest.raises(ArtifactIntegrityError, match="identity does not match"):
            restore_training_checkpoint(
                checkpoint,
                session,
                expected.model_copy(update=update),
                codec=JsonCodec(),
            )


def test_capture_rejects_mid_epoch_or_missing_epoch_state() -> None:
    with pytest.raises(ArtifactIntegrityError, match="completed epoch"):
        capture_training_checkpoint(
            _session(),
            completed_epoch=0,
            dataset_sha256="a" * 64,
            architecture_sha256="b" * 64,
            code_revision="c" * 40,
            lock_digest="d" * 64,
            codec=JsonCodec(),
        )


def test_validation_frozen_export_crosses_only_the_safe_bundle_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _capture(_session())
    checkpoint = LocalTrainingCheckpointStore(tmp_path).publish(payload)

    def metric(seed: int, checkpoint_id: str) -> ValidationCheckpointMetric:
        return ValidationCheckpointMetric(
            experiment_id="e" * 20,
            dataset_id="a" * 20,
            config_digest=payload.metadata.config_digest,
            checkpoint_id=checkpoint_id,
            seed=seed,
            epoch=1,
            median_velocity_relative_l2=0.1,
            median_cd_head_relative_error=0.1,
            score=0.2,
        )

    selection = freeze_validation_selection(
        (
            metric(17, payload.metadata.checkpoint_id),
            metric(23, "2" * 20),
            metric(31, "3" * 20),
        ),
        expected_seeds=(17, 23, 31),
    )

    class Predictor:
        def __init__(self) -> None:
            self.loaded: object | None = None

        def load_state_dict(self, state: object, *, strict: bool) -> None:
            assert strict
            self.loaded = state

    class BundleStore:
        def publish(self, bundle: object) -> ArtifactRef:
            assert bundle == "safe-bundle"
            return ArtifactRef(
                artifact_type="model",
                artifact_id="4" * 20,
                sha256="4" * 64,
                size_bytes=1,
                uri=f"models/{'4' * 20}",
            )

    predictor = Predictor()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        checkpoint_module,
        "snapshot_fno_weights",
        lambda _predictor: observed.setdefault("weights", {"weight": 3.5}),
    )

    def build_bundle(**values: object) -> str:
        observed.update(values)
        return "safe-bundle"

    monkeypatch.setattr(checkpoint_module, "build_model_bundle", build_bundle)
    reference = export_selected_checkpoint_bundle(
        checkpoint,
        selection,
        cast(Any, predictor),
        cast(Any, object()),
        dataset_sha256="a" * 64,
        code_revision="c" * 40,
        lock_digest="d" * 64,
        model_card=cast(Any, object()),
        store=cast(Any, BundleStore()),
        codec=JsonCodec(),
    )
    assert reference.artifact_type == "model"
    assert predictor.loaded == {"weight": 3.5}
    assert observed["selected_epoch"] == 1
    assert "optimizer_state" not in observed
    assert "rng_torch_state" not in observed
