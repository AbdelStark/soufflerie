"""Fixed masked multi-term optimization objective from RFC-0007."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from soufflerie.config import TrainingConfig
from soufflerie.errors import ArtifactIntegrityError, DependencyUnavailableError
from soufflerie.surrogate.preprocessing import PredictionBatchResult, TensorLike

RELATIVE_ENERGY_FLOOR = 1e-6
CD_RELATIVE_FLOOR = 0.1
NORMALIZED_REFERENCE_VELOCITY = 1.0


@dataclass(frozen=True, slots=True)
class LossScalars:
    """Detached finite values for reporting and deterministic accumulation."""

    u: float
    v: float
    rho: float
    obstacle: float
    cd: float
    total: float

    def __post_init__(self) -> None:
        values = (self.u, self.v, self.rho, self.obstacle, self.cd, self.total)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ArtifactIntegrityError(
                "TRAIN-3 LOSS: loss scalars must be finite and nonnegative"
            )


@dataclass(frozen=True, slots=True)
class DifferentiableLoss:
    """Autograd-connected fp32 terms plus their weighted total."""

    u: Any
    v: Any
    rho: Any
    obstacle: Any
    cd: Any
    total: Any

    def detached(self) -> LossScalars:
        """Copy scalar evidence to Python floats without retaining the graph."""

        def value(term: Any, *, name: str) -> float:
            try:
                result = float(term.detach().to(dtype=term.new_tensor(0.0).dtype).item())
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise ArtifactIntegrityError(
                    f"TRAIN-3 LOSS: {name} is not a detachable scalar tensor"
                ) from error
            return result

        return LossScalars(
            u=value(self.u, name="u"),
            v=value(self.v, name="v"),
            rho=value(self.rho, name="rho"),
            obstacle=value(self.obstacle, name="obstacle"),
            cd=value(self.cd, name="cd"),
            total=value(self.total, name="total"),
        )


def _require_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise DependencyUnavailableError(
            "training loss evaluation requires the locked 'ml' extra"
        ) from error


def _validate_mask(mask: TensorLike, prediction: PredictionBatchResult, torch: Any) -> None:
    batch_size = int(cast(tuple[int, ...], prediction.fields_normalized.shape)[0])
    expected = (batch_size, 1, *cast(tuple[int, ...], prediction.fields_normalized.shape)[-2:])
    if not isinstance(mask, torch.Tensor):
        raise ArtifactIntegrityError("TRAIN-3 LOSS: fluid_mask must be a PyTorch tensor")
    if tuple(int(value) for value in cast(Any, mask).shape) != expected:
        raise ArtifactIntegrityError(f"TRAIN-3 LOSS: fluid_mask must have shape {expected}")
    if str(cast(Any, mask).dtype) != "torch.bool":
        raise ArtifactIntegrityError("TRAIN-3 LOSS: fluid_mask must have dtype torch.bool")
    if cast(Any, mask).is_contiguous() is not True:
        raise ArtifactIntegrityError("TRAIN-3 LOSS: fluid_mask must be contiguous")
    if str(cast(Any, mask).device) != str(prediction.fields_normalized.device):
        raise ArtifactIntegrityError("TRAIN-3 LOSS: loss tensors must share one device")


def masked_training_loss(
    prediction: PredictionBatchResult,
    target: PredictionBatchResult,
    fluid_mask: TensorLike,
    config: TrainingConfig,
    *,
    u_ref_normalized: float = NORMALIZED_REFERENCE_VELOCITY,
    torch_module: ModuleType | None = None,
) -> DifferentiableLoss:
    """Compute the differentiable RFC objective with every reduction in fp32."""

    if not isinstance(prediction, PredictionBatchResult):
        raise TypeError("prediction must be a PredictionBatchResult")
    if not isinstance(target, PredictionBatchResult):
        raise TypeError("target must be a PredictionBatchResult")
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if (
        isinstance(u_ref_normalized, bool)
        or not isinstance(u_ref_normalized, (int, float))
        or not math.isfinite(float(u_ref_normalized))
        or float(u_ref_normalized) <= 0.0
    ):
        raise ArtifactIntegrityError(
            "TRAIN-3 LOSS: normalized reference velocity must be finite and positive"
        )
    torch = cast(Any, torch_module or _require_torch())
    if not isinstance(prediction.fields_normalized, torch.Tensor) or not isinstance(
        target.fields_normalized, torch.Tensor
    ):
        raise ArtifactIntegrityError("TRAIN-3 LOSS: fields must be PyTorch tensors")
    if not isinstance(prediction.cd_head, torch.Tensor) or not isinstance(
        target.cd_head, torch.Tensor
    ):
        raise ArtifactIntegrityError("TRAIN-3 LOSS: drag values must be PyTorch tensors")
    if tuple(prediction.fields_normalized.shape) != tuple(target.fields_normalized.shape):
        raise ArtifactIntegrityError("TRAIN-3 LOSS: predicted and target field shapes differ")
    if tuple(prediction.cd_head.shape) != tuple(target.cd_head.shape):
        raise ArtifactIntegrityError("TRAIN-3 LOSS: predicted and target drag shapes differ")
    devices = {
        str(prediction.fields_normalized.device),
        str(prediction.cd_head.device),
        str(target.fields_normalized.device),
        str(target.cd_head.device),
    }
    if len(devices) != 1:
        raise ArtifactIntegrityError("TRAIN-3 LOSS: loss tensors must share one device")
    _validate_mask(fluid_mask, prediction, torch)

    fields = cast(Any, prediction.fields_normalized).to(dtype=torch.float32)
    targets = cast(Any, target.fields_normalized).to(dtype=torch.float32)
    fluid = cast(Any, fluid_mask).to(dtype=torch.float32)
    obstacle = torch.logical_not(fluid_mask).to(dtype=torch.float32)
    channel_terms: list[Any] = []
    for channel in range(3):
        error = fields[:, channel] - targets[:, channel]
        target_channel = targets[:, channel]
        numerator = torch.mean(fluid[:, 0] * error * error, dtype=torch.float32)
        energy = torch.mean(
            fluid[:, 0] * target_channel * target_channel,
            dtype=torch.float32,
        )
        channel_terms.append(numerator / torch.clamp(energy, min=RELATIVE_ENERGY_FLOOR))

    obstacle_term = (
        torch.mean(
            obstacle[:, 0] * (fields[:, 0] * fields[:, 0] + fields[:, 1] * fields[:, 1]),
            dtype=torch.float32,
        )
        / float(u_ref_normalized) ** 2
    )
    predicted_cd = cast(Any, prediction.cd_head).to(dtype=torch.float32)
    target_cd = cast(Any, target.cd_head).to(dtype=torch.float32)
    cd_scale = torch.clamp(torch.abs(target_cd), min=CD_RELATIVE_FLOOR)
    cd_error = (predicted_cd - target_cd) / cd_scale
    cd_term = torch.mean(cd_error * cd_error, dtype=torch.float32)

    u_term, v_term, rho_term = channel_terms
    total = (
        float(config.field_weights[0]) * u_term
        + float(config.field_weights[1]) * v_term
        + float(config.field_weights[2]) * rho_term
        + float(config.obstacle_weight) * obstacle_term
        + float(config.cd_weight) * cd_term
    )
    if not bool(torch.isfinite(total).item()):
        raise ArtifactIntegrityError("TRAIN-3 LOSS: weighted loss is non-finite")
    return DifferentiableLoss(
        u=u_term,
        v=v_term,
        rho=rho_term,
        obstacle=obstacle_term,
        cd=cd_term,
        total=total,
    )


def reference_training_loss(
    predicted_fields: npt.NDArray[np.generic],
    target_fields: npt.NDArray[np.generic],
    fluid_mask: npt.NDArray[np.generic],
    predicted_cd: npt.NDArray[np.generic],
    target_cd: npt.NDArray[np.generic],
    config: TrainingConfig,
    *,
    u_ref_normalized: float = NORMALIZED_REFERENCE_VELOCITY,
) -> LossScalars:
    """Independent NumPy oracle for hand-computed loss acceptance tests."""

    fields = np.asarray(predicted_fields, dtype=np.float32)
    targets = np.asarray(target_fields, dtype=np.float32)
    mask = np.asarray(fluid_mask)
    cd_hat = np.asarray(predicted_cd, dtype=np.float32)
    cd = np.asarray(target_cd, dtype=np.float32)
    if fields.ndim != 4 or fields.shape[1] != 3 or fields.shape != targets.shape:
        raise ArtifactIntegrityError("TRAIN-3 LOSS: fields must share shape [batch,3,height,width]")
    expected_mask = (fields.shape[0], 1, fields.shape[2], fields.shape[3])
    if mask.dtype != np.dtype(np.bool_) or mask.shape != expected_mask:
        raise ArtifactIntegrityError("TRAIN-3 LOSS: NumPy fluid mask contract changed")
    if cd_hat.shape != (fields.shape[0],) or cd.shape != cd_hat.shape:
        raise ArtifactIntegrityError("TRAIN-3 LOSS: NumPy drag shape contract changed")
    if not all(np.isfinite(value).all() for value in (fields, targets, cd_hat, cd)):
        raise ArtifactIntegrityError("TRAIN-3 LOSS: NumPy loss inputs must be finite")
    if not math.isfinite(u_ref_normalized) or u_ref_normalized <= 0.0:
        raise ArtifactIntegrityError(
            "TRAIN-3 LOSS: normalized reference velocity must be finite and positive"
        )
    fluid = mask[:, 0].astype(np.float32)
    obstacle = np.logical_not(mask[:, 0]).astype(np.float32)
    channels: list[float] = []
    for channel in range(3):
        error = fields[:, channel] - targets[:, channel]
        numerator = float(np.mean(fluid * error * error, dtype=np.float32))
        energy = float(np.mean(fluid * targets[:, channel] * targets[:, channel], dtype=np.float32))
        channels.append(numerator / max(energy, RELATIVE_ENERGY_FLOOR))
    obstacle_value = (
        float(
            np.mean(
                obstacle * (fields[:, 0] * fields[:, 0] + fields[:, 1] * fields[:, 1]),
                dtype=np.float32,
            )
        )
        / u_ref_normalized**2
    )
    scale = np.maximum(np.abs(cd), np.float32(CD_RELATIVE_FLOOR))
    cd_value = float(np.mean(((cd_hat - cd) / scale) ** 2, dtype=np.float32))
    total = float(
        np.float32(config.field_weights[0]) * np.float32(channels[0])
        + np.float32(config.field_weights[1]) * np.float32(channels[1])
        + np.float32(config.field_weights[2]) * np.float32(channels[2])
        + np.float32(config.obstacle_weight) * np.float32(obstacle_value)
        + np.float32(config.cd_weight) * np.float32(cd_value)
    )
    return LossScalars(
        u=channels[0],
        v=channels[1],
        rho=channels[2],
        obstacle=obstacle_value,
        cd=cd_value,
        total=total,
    )


__all__ = [
    "CD_RELATIVE_FLOOR",
    "NORMALIZED_REFERENCE_VELOCITY",
    "RELATIVE_ENERGY_FLOOR",
    "DifferentiableLoss",
    "LossScalars",
    "masked_training_loss",
    "reference_training_loss",
]
