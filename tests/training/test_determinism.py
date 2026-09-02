from __future__ import annotations

import random
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from soufflerie.config import TrainingConfig
from soufflerie.errors import ArtifactIntegrityError, DeviceUnavailableError
from soufflerie.training import (
    adamw_parameter_groups,
    apply_learning_rate,
    build_adamw,
    configure_determinism,
    deterministic_worker_seed,
    learning_rate_for_epoch,
    preflight_precision,
    worker_seed_initializer,
)


@dataclass(slots=True)
class FakeParameter:
    ndim: int
    requires_grad: bool = True


class FakeCuda:
    def __init__(self) -> None:
        self.available = True
        self.bf16 = True
        self.seed: int | None = None

    def is_available(self) -> bool:
        return self.available

    def manual_seed_all(self, seed: int) -> None:
        self.seed = seed

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def is_bf16_supported(self, *, including_emulation: bool) -> bool:
        assert including_emulation is False
        return self.bf16

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(major=8, minor=9, name="Contract GPU")


class FakeAdamW:
    def __init__(
        self,
        groups: tuple[dict[str, object], dict[str, object]],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        self.param_groups = list(groups)
        for group in self.param_groups:
            group["lr"] = lr
        self.betas = betas
        self.eps = eps


class FakeTorch(ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.cuda = FakeCuda()
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        )
        self.optim = SimpleNamespace(AdamW=FakeAdamW)
        self.bfloat16 = "torch.bfloat16"
        self.float16 = "torch.float16"
        self.seed: int | None = None
        self.deterministic: bool | None = None
        self.debug_mode: str | None = None

    def manual_seed(self, seed: int) -> None:
        self.seed = seed

    def use_deterministic_algorithms(self, enabled: bool) -> None:
        self.deterministic = enabled

    def set_deterministic_debug_mode(self, mode: str) -> None:
        self.debug_mode = mode


class FakeModel:
    def __init__(self) -> None:
        self.weight = FakeParameter(2)
        self.scale = FakeParameter(1)
        self.bias = FakeParameter(1)
        self.norm = FakeParameter(2)
        self.frozen = FakeParameter(2, requires_grad=False)

    def named_parameters(self, *, recurse: bool = True) -> tuple[tuple[str, FakeParameter], ...]:
        assert recurse
        return (
            ("core.weight", self.weight),
            ("core.scale", self.scale),
            ("core.bias", self.bias),
            ("core.layernorm.weight", self.norm),
            ("core.frozen", self.frozen),
        )

    def parameters(self, *, recurse: bool = True) -> tuple[FakeParameter, ...]:
        return tuple(parameter for _name, parameter in self.named_parameters(recurse=recurse))


def _config() -> TrainingConfig:
    return TrainingConfig(dataset_id="a" * 20, seeds=(17, 23, 31))


def test_seed_policy_reproduces_initialization_and_disables_nondeterminism() -> None:
    torch = FakeTorch()

    configure_determinism(17, torch_module=torch)
    first = (random.random(), float(np.random.random()))
    configure_determinism(17, torch_module=torch)
    second = (random.random(), float(np.random.random()))

    assert first == second
    assert torch.seed == torch.cuda.seed == 17
    assert torch.deterministic is True
    assert torch.debug_mode == "error"
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    with pytest.raises(ArtifactIntegrityError, match="unsigned 64-bit"):
        configure_determinism(-1, torch_module=torch)


def test_worker_child_seeds_are_stable_distinct_and_initialize_all_cpu_rngs() -> None:
    torch = FakeTorch()
    first = deterministic_worker_seed(17, worker_id=0, epoch=4)
    assert first == deterministic_worker_seed(17, worker_id=0, epoch=4)
    assert first != deterministic_worker_seed(17, worker_id=1, epoch=4)
    initialize = worker_seed_initializer(17, epoch=4, torch_module=torch)
    initialize(0)
    observed = (random.random(), float(np.random.random()), torch.seed)
    initialize(0)
    assert (random.random(), float(np.random.random()), torch.seed) == observed
    with pytest.raises(ArtifactIntegrityError, match="worker_id"):
        deterministic_worker_seed(17, worker_id=-1, epoch=0)


def test_precision_preflight_never_falls_back() -> None:
    torch = FakeTorch()
    bf16 = preflight_precision("cuda", "bf16", torch_module=torch)
    fp16 = preflight_precision("cuda:0", "fp16", torch_module=torch)
    assert (bf16.device, bf16.autocast_dtype, bf16.compute_capability) == (
        "cuda:0",
        "torch.bfloat16",
        "8.9",
    )
    assert fp16.autocast_dtype == "torch.float16"

    torch.cuda.bf16 = False
    with pytest.raises(DeviceUnavailableError, match="does not support native bf16"):
        preflight_precision("cuda:0", "bf16", torch_module=torch)
    torch.cuda.available = False
    with pytest.raises(DeviceUnavailableError, match="CUDA is unavailable"):
        preflight_precision("cuda:0", "fp16", torch_module=torch)
    with pytest.raises(DeviceUnavailableError, match="requires cuda"):
        preflight_precision("cpu", "bf16", torch_module=torch)
    with pytest.raises(DeviceUnavailableError, match="must be bf16 or fp16"):
        preflight_precision("cuda:0", cast_precision("tf32"), torch_module=torch)


def cast_precision(value: Any) -> Any:
    return value


def test_adamw_groups_and_schedule_match_the_fixed_policy() -> None:
    model = FakeModel()
    groups = adamw_parameter_groups(model, weight_decay=1e-4)
    assert groups.decay_names == ("core.weight", "core.scale")
    assert groups.no_decay_names == ("core.bias", "core.layernorm.weight")
    assert groups.groups[0]["weight_decay"] == 1e-4
    assert groups.groups[1]["weight_decay"] == 0.0

    optimizer = build_adamw(model, _config(), torch_module=FakeTorch())
    assert optimizer.betas == (0.9, 0.999)
    assert optimizer.eps == 1e-8
    config = _config()
    assert learning_rate_for_epoch(config, 1) == pytest.approx(1e-4)
    assert learning_rate_for_epoch(config, 5) == pytest.approx(1e-3)
    assert learning_rate_for_epoch(config, 100) == pytest.approx(1e-5)
    midpoint = learning_rate_for_epoch(config, 52)
    assert 1e-5 < midpoint < 1e-3
    apply_learning_rate(optimizer, midpoint)
    assert all(group["lr"] == midpoint for group in optimizer.param_groups)
    with pytest.raises(ArtifactIntegrityError, match="outside"):
        learning_rate_for_epoch(config, 0)
    with pytest.raises(ArtifactIntegrityError, match="weight_decay"):
        adamw_parameter_groups(model, weight_decay=-1.0)


def test_optimizer_rejects_duplicate_parameter_aliases() -> None:
    parameter = FakeParameter(2)

    class AliasedModel:
        def named_parameters(
            self, *, recurse: bool = True
        ) -> tuple[tuple[str, FakeParameter], ...]:
            return (("first", parameter), ("second", parameter))

    with pytest.raises(ArtifactIntegrityError, match="multiple names"):
        adamw_parameter_groups(AnyModel(AliasedModel()), weight_decay=0.0)


def test_runtime_and_optimizer_malformed_contracts_fail_closed() -> None:
    torch = FakeTorch()
    torch.cuda.device_count = lambda: 0  # type: ignore[method-assign]
    with pytest.raises(DeviceUnavailableError, match="index 0 is unavailable"):
        preflight_precision("cuda:0", "fp16", torch_module=torch)

    class NamelessCuda(FakeCuda):
        def get_device_properties(self, index: int) -> SimpleNamespace:
            assert index == 0
            return SimpleNamespace(major=8, minor=9, name="")

    torch = FakeTorch()
    torch.cuda = NamelessCuda()
    with pytest.raises(DeviceUnavailableError, match="name is unavailable"):
        preflight_precision("cuda:0", "fp16", torch_module=torch)

    with pytest.raises(DeviceUnavailableError, match="cannot enforce"):
        configure_determinism(17, torch_module=ModuleType("broken"))
    with pytest.raises(ArtifactIntegrityError, match="epoch must"):
        deterministic_worker_seed(17, worker_id=0, epoch=-1)
    with pytest.raises(ArtifactIntegrityError, match="epoch must"):
        worker_seed_initializer(17, epoch=-1, torch_module=torch)

    class EmptyModel:
        def named_parameters(self, *, recurse: bool = True) -> tuple[tuple[str, object], ...]:
            return ()

    with pytest.raises(ArtifactIntegrityError, match="no named parameters"):
        adamw_parameter_groups(AnyModel(EmptyModel()), weight_decay=0.0)

    class FrozenModel:
        def named_parameters(
            self, *, recurse: bool = True
        ) -> tuple[tuple[str, FakeParameter], ...]:
            return (("weight", FakeParameter(2, requires_grad=False)),)

    with pytest.raises(ArtifactIntegrityError, match="no trainable parameters"):
        adamw_parameter_groups(AnyModel(FrozenModel()), weight_decay=0.0)

    with pytest.raises(ArtifactIntegrityError, match="finite and positive"):
        apply_learning_rate(SimpleNamespace(param_groups=[{}]), 0.0)
    with pytest.raises(ArtifactIntegrityError, match="no parameter groups"):
        apply_learning_rate(SimpleNamespace(param_groups=[]), 1e-3)


def AnyModel(value: Any) -> Any:
    return value
