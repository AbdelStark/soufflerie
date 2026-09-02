"""Strict mixed-precision, RNG, optimizer, and schedule policy."""

from __future__ import annotations

import hashlib
import importlib
import math
import random
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, Protocol, cast

import numpy as np

from soufflerie.config import UINT64_MAX, TrainingConfig
from soufflerie.errors import (
    ArtifactIntegrityError,
    DependencyUnavailableError,
    DeviceUnavailableError,
)

Precision = Literal["bf16", "fp16"]
_CUDA_DEVICE = re.compile(r"^cuda(?::([0-9]+))?$")
_NORM_SEGMENT = re.compile(r"^(?:norm|normalization|layernorm|batchnorm|groupnorm)[0-9_]*$")


class TrainableModel(Protocol):
    def named_parameters(self, *, recurse: bool = True) -> Iterable[tuple[str, Any]]: ...

    def parameters(self, *, recurse: bool = True) -> Iterable[Any]: ...


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Preflighted CUDA autocast policy with no fallback semantics."""

    precision: Precision
    device: str
    device_index: int
    autocast_dtype: object
    device_name: str
    compute_capability: str


@dataclass(frozen=True, slots=True)
class OptimizerGroups:
    """Complete non-overlapping AdamW parameter partition."""

    groups: tuple[dict[str, object], dict[str, object]]
    decay_names: tuple[str, ...]
    no_decay_names: tuple[str, ...]


def require_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise DependencyUnavailableError("training requires the locked 'ml' extra") from error


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= UINT64_MAX:
        raise ArtifactIntegrityError("TRAIN-4 SEED: seed must be an unsigned 64-bit integer")


def configure_determinism(seed: int, *, torch_module: ModuleType | None = None) -> None:
    """Seed every RNG and enable framework error-on-nondeterminism mode."""

    _validate_seed(seed)
    torch = cast(Any, torch_module or require_torch())
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        torch.manual_seed(seed)
        if bool(torch.cuda.is_available()):
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_deterministic_debug_mode("error")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise DeviceUnavailableError(
            "TRAIN-4 DETERMINISM: runtime cannot enforce the canonical deterministic policy"
        ) from error


def deterministic_worker_seed(seed: int, *, worker_id: int, epoch: int) -> int:
    """Derive one stable uint32 child seed without consuming parent RNG state."""

    _validate_seed(seed)
    if isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0:
        raise ArtifactIntegrityError("TRAIN-4 SEED: worker_id must be nonnegative")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ArtifactIntegrityError("TRAIN-4 SEED: epoch must be nonnegative")
    payload = f"soufflerie-worker-v1\0{seed}\0{epoch}\0{worker_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def worker_seed_initializer(
    seed: int,
    *,
    epoch: int,
    torch_module: ModuleType | None = None,
) -> Callable[[int], None]:
    """Return the deterministic worker callback used by framework data loaders."""

    _validate_seed(seed)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ArtifactIntegrityError("TRAIN-4 SEED: epoch must be nonnegative")
    torch = cast(Any, torch_module or require_torch())

    def initialize(worker_id: int) -> None:
        child = deterministic_worker_seed(seed, worker_id=worker_id, epoch=epoch)
        random.seed(child)
        np.random.seed(child)
        torch.manual_seed(child)

    return initialize


def preflight_precision(
    device: str,
    precision: Precision,
    *,
    torch_module: ModuleType | None = None,
) -> PrecisionPolicy:
    """Resolve the requested CUDA dtype or fail before model construction."""

    if not isinstance(device, str) or (match := _CUDA_DEVICE.fullmatch(device)) is None:
        raise DeviceUnavailableError(
            "TRAIN-5 PRECISION: canonical mixed-precision training requires cuda[:index]"
        )
    if precision not in {"bf16", "fp16"}:
        raise DeviceUnavailableError("TRAIN-5 PRECISION: precision must be bf16 or fp16")
    torch = cast(Any, torch_module or require_torch())
    try:
        if not bool(torch.cuda.is_available()):
            raise DeviceUnavailableError("TRAIN-5 PRECISION: CUDA is unavailable")
        index = int(match.group(1) or torch.cuda.current_device())
        count = int(torch.cuda.device_count())
        if not 0 <= index < count:
            raise DeviceUnavailableError(
                f"TRAIN-5 PRECISION: CUDA device index {index} is unavailable"
            )
        if precision == "bf16" and not bool(
            torch.cuda.is_bf16_supported(including_emulation=False)
        ):
            raise DeviceUnavailableError(
                f"TRAIN-5 PRECISION: {device} does not support native bf16"
            )
        properties = torch.cuda.get_device_properties(index)
        major = int(properties.major)
        minor = int(properties.minor)
        name = str(properties.name)
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    except DeviceUnavailableError:
        raise
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise DeviceUnavailableError(
            f"TRAIN-5 PRECISION: unable to prove {precision} support on {device}"
        ) from error
    if not name:
        raise DeviceUnavailableError("TRAIN-5 PRECISION: CUDA device name is unavailable")
    return PrecisionPolicy(
        precision=precision,
        device=f"cuda:{index}",
        device_index=index,
        autocast_dtype=dtype,
        device_name=name,
        compute_capability=f"{major}.{minor}",
    )


def _no_weight_decay(name: str) -> bool:
    segments = name.casefold().split(".")
    return name.endswith(".bias") or any(
        _NORM_SEGMENT.fullmatch(segment) is not None for segment in segments
    )


def adamw_parameter_groups(model: TrainableModel, *, weight_decay: float) -> OptimizerGroups:
    """Partition each trainable parameter exactly once; biases/norms never decay."""

    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ArtifactIntegrityError(
            "TRAIN-6 OPTIMIZER: weight_decay must be finite and nonnegative"
        )
    try:
        named = tuple(model.named_parameters(recurse=True))
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            "TRAIN-6 OPTIMIZER: model parameters are unavailable"
        ) from error
    if not named:
        raise ArtifactIntegrityError("TRAIN-6 OPTIMIZER: model has no named parameters")
    names = tuple(name for name, _parameter in named)
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ArtifactIntegrityError(
            "TRAIN-6 OPTIMIZER: parameter names must be non-empty and unique"
        )
    decay: list[Any] = []
    no_decay: list[Any] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    identities: set[int] = set()
    for name, parameter in named:
        if not bool(getattr(parameter, "requires_grad", True)):
            continue
        identity = id(parameter)
        if identity in identities:
            raise ArtifactIntegrityError(
                "TRAIN-6 OPTIMIZER: a trainable parameter appears under multiple names"
            )
        identities.add(identity)
        destination = no_decay if _no_weight_decay(name) else decay
        destination_names = no_decay_names if destination is no_decay else decay_names
        destination.append(parameter)
        destination_names.append(name)
    if not identities:
        raise ArtifactIntegrityError("TRAIN-6 OPTIMIZER: model has no trainable parameters")
    return OptimizerGroups(
        groups=(
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        decay_names=tuple(decay_names),
        no_decay_names=tuple(no_decay_names),
    )


def build_adamw(
    model: TrainableModel,
    config: TrainingConfig,
    *,
    torch_module: ModuleType | None = None,
) -> Any:
    """Construct the only accepted optimizer with explicit fixed hyperparameters."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    torch = cast(Any, torch_module or require_torch())
    groups = adamw_parameter_groups(model, weight_decay=float(config.weight_decay))
    try:
        return torch.optim.AdamW(
            groups.groups,
            lr=float(config.learning_rate),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("TRAIN-6 OPTIMIZER: AdamW construction failed") from error


def learning_rate_for_epoch(config: TrainingConfig, epoch: int) -> float:
    """Five-epoch 10%-to-full warmup followed by decay to the declared floor."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= config.epochs:
        raise ArtifactIntegrityError("TRAIN-7 SCHEDULE: epoch is outside the configured run")
    maximum = float(config.learning_rate)
    minimum = float(config.min_learning_rate)
    if epoch <= 5:
        return maximum * (0.1 + 0.9 * (epoch - 1) / 4.0)
    if config.epochs <= 5:
        return maximum
    progress = (epoch - 5) / (config.epochs - 5)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def apply_learning_rate(optimizer: Any, learning_rate: float) -> None:
    """Set every optimizer group to one finite schedule value."""

    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ArtifactIntegrityError("TRAIN-7 SCHEDULE: learning rate must be finite and positive")
    try:
        groups = cast(list[Mapping[str, object]], optimizer.param_groups)
        if not groups:
            raise ArtifactIntegrityError("TRAIN-7 SCHEDULE: optimizer has no parameter groups")
        for group in groups:
            cast(dict[str, object], group)["lr"] = float(learning_rate)
    except ArtifactIntegrityError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("TRAIN-7 SCHEDULE: optimizer groups are malformed") from error


__all__ = [
    "OptimizerGroups",
    "Precision",
    "PrecisionPolicy",
    "TrainableModel",
    "adamw_parameter_groups",
    "apply_learning_rate",
    "build_adamw",
    "configure_determinism",
    "deterministic_worker_seed",
    "learning_rate_for_epoch",
    "preflight_precision",
    "require_torch",
    "worker_seed_initializer",
]
