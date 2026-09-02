from __future__ import annotations

from types import ModuleType
from typing import Any, Self, cast

import numpy as np
import pytest

from soufflerie.config import TrainingConfig
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.surrogate.preprocessing import MODEL_SPATIAL_SHAPE, PredictionBatchResult
from soufflerie.training import (
    LossScalars,
    masked_training_loss,
    reference_training_loss,
)
from tests.training.helpers import ArrayTensor


def _config() -> TrainingConfig:
    return TrainingConfig(dataset_id="a" * 20, seeds=(17, 23, 31))


class FakeTensor:
    def __init__(
        self,
        value: Any,
        *,
        dtype: str = "torch.float32",
        device: str = "cuda:0",
    ) -> None:
        self.value: Any = np.asarray(
            value,
            dtype=np.bool_ if dtype == "torch.bool" else np.float32,
        )
        self.dtype = dtype
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], self.value.shape)

    def is_contiguous(self) -> bool:
        return bool(self.value.flags.c_contiguous)

    def isfinite(self) -> FakeTensor:
        return FakeTensor(np.isfinite(self.value), dtype="torch.bool", device=self.device)

    def all(self) -> FakeTensor:
        return FakeTensor(bool(self.value.all()), dtype="torch.bool", device=self.device)

    def item(self) -> object:
        return self.value.item()

    def to(self, *, dtype: str) -> FakeTensor:
        return FakeTensor(self.value, dtype=dtype, device=self.device)

    def detach(self) -> Self:
        return self

    def new_tensor(self, value: float) -> FakeTensor:
        return FakeTensor(value, dtype=self.dtype, device=self.device)

    def __getitem__(self, key: object) -> FakeTensor:
        return FakeTensor(self.value[key], dtype=self.dtype, device=self.device)

    def __add__(self, other: object) -> FakeTensor:
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value + value, device=self.device)

    def __radd__(self, other: object) -> FakeTensor:
        return self + other

    def __sub__(self, other: object) -> FakeTensor:
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value - value, device=self.device)

    def __mul__(self, other: object) -> FakeTensor:
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value * value, device=self.device)

    def __rmul__(self, other: object) -> FakeTensor:
        return self * other

    def __truediv__(self, other: object) -> FakeTensor:
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value / value, device=self.device)


class FakeTorch(ModuleType):
    Tensor = FakeTensor
    float32 = "torch.float32"

    @staticmethod
    def mean(value: FakeTensor, *, dtype: str) -> FakeTensor:
        assert dtype == "torch.float32"
        return FakeTensor(np.mean(value.value, dtype=np.float32), device=value.device)

    @staticmethod
    def clamp(value: FakeTensor, *, min: float) -> FakeTensor:
        return FakeTensor(np.maximum(value.value, np.float32(min)), device=value.device)

    @staticmethod
    def logical_not(value: FakeTensor) -> FakeTensor:
        return FakeTensor(np.logical_not(value.value), dtype="torch.bool", device=value.device)

    @staticmethod
    def abs(value: FakeTensor) -> FakeTensor:
        return FakeTensor(np.abs(value.value), device=value.device)

    @staticmethod
    def isfinite(value: FakeTensor) -> FakeTensor:
        return value.isfinite().all()


def _tensor_result(
    fields: np.ndarray[Any, Any],
    cd: np.ndarray[Any, Any],
) -> PredictionBatchResult:
    return PredictionBatchResult(
        fields_normalized=FakeTensor(fields),
        cd_head=FakeTensor(cd),
    )


def test_reference_loss_matches_hand_computed_masks_floors_and_weights() -> None:
    predicted = np.asarray([[[[3.0, 2.0]], [[1.0, 3.0]], [[1.0, 9.0]]]], dtype=np.float32)
    target = np.asarray([[[[2.0, 8.0]], [[0.0, 8.0]], [[1.0, 8.0]]]], dtype=np.float32)
    fluid = np.asarray([[[[True, False]]]])

    loss = reference_training_loss(
        predicted,
        target,
        fluid,
        np.asarray([0.15], dtype=np.float32),
        np.asarray([0.05], dtype=np.float32),
        _config(),
    )

    assert loss.u == pytest.approx(0.25)
    assert loss.v == pytest.approx(500_000.0)
    assert loss.rho == 0.0
    assert loss.obstacle == pytest.approx(6.5)
    assert loss.cd == pytest.approx(1.0)
    assert loss.total == pytest.approx(500_002.375)


def test_differentiable_loss_matches_independent_numpy_oracle() -> None:
    shape = (1, 3, *MODEL_SPATIAL_SHAPE)
    predicted = np.zeros(shape, dtype=np.float32)
    target = np.zeros(shape, dtype=np.float32)
    predicted[:, 0].fill(0.25)
    predicted[:, 1].fill(-0.5)
    target[:, 0].fill(0.5)
    target[:, 2].fill(1.0)
    mask = np.ones((1, 1, *MODEL_SPATIAL_SHAPE), dtype=np.bool_)
    mask[..., :10, :] = False
    cd_hat = np.asarray([1.5], dtype=np.float32)
    cd = np.asarray([1.0], dtype=np.float32)
    expected = reference_training_loss(predicted, target, mask, cd_hat, cd, _config())
    torch = FakeTorch("torch")

    observed = masked_training_loss(
        _tensor_result(predicted, cd_hat),
        _tensor_result(target, cd),
        cast(Any, FakeTensor(mask, dtype="torch.bool")),
        _config(),
        torch_module=torch,
    ).detached()

    assert observed == pytest.approx(expected)


def test_loss_contract_rejects_malformed_or_nonfinite_inputs() -> None:
    config = _config()
    fields = np.zeros((1, 3, 2, 2), dtype=np.float32)
    mask = np.ones((1, 1, 2, 2), dtype=np.bool_)
    cd = np.zeros((1,), dtype=np.float32)
    with pytest.raises(ArtifactIntegrityError, match="fields must share shape"):
        reference_training_loss(fields[:, :2], fields[:, :2], mask, cd, cd, config)
    with pytest.raises(ArtifactIntegrityError, match="mask contract"):
        reference_training_loss(fields, fields, mask.astype(np.uint8), cd, cd, config)
    with pytest.raises(ArtifactIntegrityError, match="drag shape"):
        reference_training_loss(fields, fields, mask, cd.reshape(1, 1), cd, config)
    fields[0, 0, 0, 0] = np.nan
    with pytest.raises(ArtifactIntegrityError, match="must be finite"):
        reference_training_loss(fields, fields, mask, cd, cd, config)
    with pytest.raises(ArtifactIntegrityError, match="reference velocity"):
        reference_training_loss(
            np.zeros((1, 3, 2, 2), dtype=np.float32),
            np.zeros((1, 3, 2, 2), dtype=np.float32),
            mask,
            cd,
            cd,
            config,
            u_ref_normalized=0.0,
        )
    with pytest.raises(ArtifactIntegrityError):
        LossScalars(u=-1.0, v=0.0, rho=0.0, obstacle=0.0, cd=0.0, total=0.0)


def test_differentiable_loss_rejects_tensor_identity_mask_and_device_drift() -> None:
    torch = FakeTorch("torch")
    fields = np.zeros((1, 3, *MODEL_SPATIAL_SHAPE), dtype=np.float32)
    cd = np.zeros((1,), dtype=np.float32)
    result = _tensor_result(fields, cd)
    mask = FakeTensor(
        np.ones((1, 1, *MODEL_SPATIAL_SHAPE), dtype=np.bool_),
        dtype="torch.bool",
    )
    with pytest.raises(TypeError, match="prediction must"):
        masked_training_loss(cast(Any, object()), result, cast(Any, mask), _config())
    with pytest.raises(TypeError, match="target must"):
        masked_training_loss(result, cast(Any, object()), cast(Any, mask), _config())
    with pytest.raises(TypeError, match="config must"):
        masked_training_loss(result, result, cast(Any, mask), cast(Any, object()))
    with pytest.raises(ArtifactIntegrityError, match="reference velocity"):
        masked_training_loss(
            result,
            result,
            cast(Any, mask),
            _config(),
            u_ref_normalized=float("nan"),
            torch_module=torch,
        )

    array_result = PredictionBatchResult(
        fields_normalized=ArrayTensor(fields, device="cuda:0"),
        cd_head=ArrayTensor(cd, device="cuda:0"),
    )
    with pytest.raises(ArtifactIntegrityError, match="fields must"):
        masked_training_loss(
            array_result,
            array_result,
            cast(Any, mask),
            _config(),
            torch_module=torch,
        )
    mixed_drag = PredictionBatchResult(
        fields_normalized=FakeTensor(fields, device="cpu"),
        cd_head=ArrayTensor(cd, device="cpu"),
    )
    with pytest.raises(ArtifactIntegrityError, match="drag values"):
        masked_training_loss(
            mixed_drag,
            mixed_drag,
            cast(Any, FakeTensor(mask.value, dtype="torch.bool", device="cpu")),
            _config(),
            torch_module=torch,
        )
    larger = _tensor_result(
        np.zeros((2, 3, *MODEL_SPATIAL_SHAPE), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    )
    with pytest.raises(ArtifactIntegrityError, match="field shapes differ"):
        masked_training_loss(result, larger, cast(Any, mask), _config(), torch_module=torch)
    other_device = PredictionBatchResult(
        fields_normalized=FakeTensor(fields, device="cuda:1"),
        cd_head=FakeTensor(cd, device="cuda:1"),
    )
    with pytest.raises(ArtifactIntegrityError, match="share one device"):
        masked_training_loss(
            result,
            other_device,
            cast(Any, mask),
            _config(),
            torch_module=torch,
        )
    for bad_mask in (
        ArrayTensor(mask.value, device="cuda:0"),
        FakeTensor(np.ones((1, 1, 2, 2), dtype=np.bool_), dtype="torch.bool"),
        FakeTensor(mask.value),
        FakeTensor(mask.value, dtype="torch.bool", device="cuda:1"),
    ):
        with pytest.raises(ArtifactIntegrityError, match=r"fluid_mask|share one device"):
            masked_training_loss(
                result,
                result,
                cast(Any, bad_mask),
                _config(),
                torch_module=torch,
            )
