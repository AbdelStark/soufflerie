from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable, Mapping
from types import ModuleType, SimpleNamespace
from typing import Any, Self, cast

import pytest
from pydantic import ValidationError

from soufflerie.errors import (
    ArtifactIntegrityError,
    DependencyUnavailableError,
    DeviceUnavailableError,
)
from soufflerie.surrogate import FlowPredictor, FnoArchitecture
from soufflerie.surrogate import fno as fno_module
from soufflerie.surrogate.fno import FnoPredictor
from soufflerie.surrogate.preprocessing import (
    MODEL_SPATIAL_SHAPE,
    PredictionBatch,
    PredictionBatchResult,
    TensorLike,
)


class FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: str = "torch.float32",
        device: str = "cpu",
        lineage: str = "fixture",
        item: bool = True,
        empty_fluid: bool = False,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.lineage = lineage
        self._item = item
        self.empty_fluid = empty_fluid

    def is_contiguous(self) -> bool:
        return True

    def isfinite(self) -> Self:
        return self

    def all(self) -> Self:
        return self

    def item(self) -> object:
        return self._item

    def to(self, *, dtype: str) -> FakeTensor:
        return FakeTensor(
            self.shape,
            dtype=dtype,
            device=self.device,
            lineage=self.lineage,
            empty_fluid=self.empty_fluid,
        )

    def sum(self, *, dim: tuple[int, int]) -> FakeTensor:
        assert dim == (-2, -1)
        return FakeTensor(
            self.shape[:-2],
            dtype=self.dtype,
            device=self.device,
            lineage=f"sum({self.lineage})",
            empty_fluid=self.empty_fluid,
        )

    def reshape(self, *shape: int) -> FakeTensor:
        return FakeTensor(
            tuple(shape),
            dtype=self.dtype,
            device=self.device,
            lineage=self.lineage,
        )

    def contiguous(self) -> Self:
        return self

    def __mul__(self, other: object) -> FakeTensor:
        assert isinstance(other, FakeTensor)
        return FakeTensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            lineage=f"mul({self.lineage})",
        )

    def __truediv__(self, other: object) -> FakeTensor:
        assert isinstance(other, FakeTensor)
        return FakeTensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            lineage=f"div({self.lineage})",
        )

    def __le__(self, other: object) -> FakeTensor:
        assert other == 0
        return FakeTensor(
            self.shape,
            dtype="torch.bool",
            device=self.device,
            item=self.empty_fluid,
        )


class FakeModule:
    def __init__(self) -> None:
        self.training = True
        self._children: dict[str, FakeModule] = {}
        self._parameters: list[FakeTensor] = []

    def add_module(self, name: str, module: FakeModule) -> None:
        self._children[name] = module

    def train(self, mode: bool = True) -> Self:
        self.training = mode
        for child in self._children.values():
            child.train(mode)
        return self

    def to(self, *args: object, **kwargs: object) -> Self:
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, str):
            for parameter in self.parameters():
                parameter.device = device
        return self

    def parameters(self, *, recurse: bool = True) -> Iterable[FakeTensor]:
        yield from self._parameters
        if recurse:
            for child in self._children.values():
                yield from child.parameters(recurse=True)

    def named_parameters(self, *, recurse: bool = True) -> Iterable[tuple[str, FakeTensor]]:
        for index, parameter in enumerate(self._parameters):
            yield (f"parameter_{index}", parameter)
        if recurse:
            for name, child in self._children.items():
                for child_name, parameter in child.named_parameters(recurse=True):
                    yield (f"{name}.{child_name}", parameter)

    def state_dict(self) -> Mapping[str, FakeTensor]:
        return dict(self.named_parameters())

    def load_state_dict(self, state_dict: Mapping[str, object], *, strict: bool = True) -> object:
        return (tuple(state_dict), strict)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        assert set_to_none


class FakeConv2d(FakeModule):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._parameters.append(FakeTensor((out_channels, in_channels, 1, 1)))


class FakeLinear(FakeModule):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._parameters.append(FakeTensor((out_features, in_features)))

    def __call__(self, value: FakeTensor) -> FakeTensor:
        return FakeTensor(
            (*value.shape[:-1], self.out_features),
            dtype=value.dtype,
            device=value.device,
            lineage=value.lineage,
        )


class FakeGelu(FakeModule):
    def __call__(self, value: FakeTensor) -> FakeTensor:
        return value


class FakeSequential(FakeModule):
    def __init__(self, *modules: FakeModule) -> None:
        super().__init__()
        self.modules = modules
        for index, module in enumerate(modules):
            self.add_module(str(index), module)

    def __call__(self, value: FakeTensor) -> FakeTensor:
        for module in self.modules:
            value = cast(Any, module)(value)
        return value


class FakeSpecEncoder(FakeModule):
    def __init__(self, *, latent_channels: int) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.lift_network: FakeModule = FakeModule()
        self._parameters.append(FakeTensor((1,)))

    def __call__(self, value: FakeTensor) -> FakeTensor:
        return FakeTensor(
            (value.shape[0], self.latent_channels, *value.shape[-2:]),
            dtype=value.dtype,
            device=value.device,
            lineage="latent",
        )

    def grid_to_points(self, value: FakeTensor) -> tuple[FakeTensor, list[int]]:
        batch, channels, height, width = value.shape
        return (
            FakeTensor(
                (batch * height * width, channels),
                dtype=value.dtype,
                device=value.device,
                lineage=value.lineage,
            ),
            list(value.shape),
        )

    def points_to_grid(self, value: FakeTensor, shape: list[int]) -> FakeTensor:
        return FakeTensor(
            (shape[0], value.shape[-1], shape[2], shape[3]),
            dtype=value.dtype,
            device=value.device,
            lineage="decoded-grid",
        )


class FakeDecoder(FakeModule):
    def __init__(self, *, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self._parameters.append(FakeTensor((1,)))

    def __call__(self, value: FakeTensor) -> FakeTensor:
        return FakeTensor(
            (*value.shape[:-1], self.out_channels),
            dtype=value.dtype,
            device=value.device,
            lineage="decoded-points",
        )


class FakeFno(FakeModule):
    def __init__(self, **arguments: object) -> None:
        super().__init__()
        self.arguments = arguments
        self.spec_encoder = FakeSpecEncoder(latent_channels=cast(int, arguments["latent_channels"]))
        self.decoder_net = FakeDecoder(out_channels=cast(int, arguments["out_channels"]))
        self.add_module("spec_encoder", self.spec_encoder)
        self.add_module("decoder_net", self.decoder_net)


class FakeInferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


class FakeTorchModule(ModuleType):
    Tensor: type[FakeTensor]
    float32: str
    nn: SimpleNamespace
    cat: Any
    any: Any
    autocast: Any
    autocast_calls: list[dict[str, object]]
    inference_mode: type[FakeInferenceMode]


def _fake_runtime() -> fno_module._MlRuntime:
    torch = FakeTorchModule("torch")
    torch.Tensor = FakeTensor
    torch.float32 = "torch.float32"
    torch.nn = SimpleNamespace(
        Module=FakeModule,
        Conv2d=FakeConv2d,
        Linear=FakeLinear,
        GELU=FakeGelu,
        Sequential=FakeSequential,
    )

    def concatenate(values: tuple[FakeTensor, FakeTensor], *, dim: int) -> FakeTensor:
        assert dim == 1
        first, second = values
        return FakeTensor(
            (first.shape[0], first.shape[1] + second.shape[1]),
            dtype=first.dtype,
            device=first.device,
            lineage="cd-inputs",
        )

    torch.cat = concatenate
    torch.any = lambda value: value
    torch.autocast_calls = []

    def autocast(**kwargs: object) -> FakeInferenceMode:
        torch.autocast_calls.append(kwargs)
        return FakeInferenceMode()

    torch.autocast = autocast
    torch.inference_mode = FakeInferenceMode
    return fno_module._MlRuntime(
        torch=torch,
        fno_type=FakeFno,
        physicsnemo_version="2.2.1",
    )


def _fake_batch(*, empty_fluid: bool = False) -> PredictionBatch:
    return PredictionBatch(
        inputs=cast(
            TensorLike,
            FakeTensor((2, 2, *MODEL_SPATIAL_SHAPE)),
        ),
        fluid_mask=cast(
            TensorLike,
            FakeTensor(
                (2, 1, *MODEL_SPATIAL_SHAPE),
                dtype="torch.bool",
                empty_fluid=empty_fluid,
            ),
        ),
        design_params=cast(TensorLike, FakeTensor((2, 4))),
    )


def test_architecture_is_exact_immutable_and_json_round_trippable() -> None:
    architecture = FnoArchitecture()

    assert FnoArchitecture.model_validate_json(architecture.model_dump_json()) == architecture
    assert dict(architecture.physicsnemo_arguments) == {
        "in_channels": 2,
        "out_channels": 3,
        "decoder_layers": 1,
        "decoder_layer_size": 128,
        "decoder_activation_fn": "gelu",
        "dimension": 2,
        "latent_channels": 64,
        "num_fno_layers": 4,
        "num_fno_modes": [24, 24],
        "padding": 8,
        "padding_type": "constant",
        "activation_fn": "gelu",
        "coord_features": False,
    }
    payload = architecture.model_dump(mode="python")
    payload["coordinate_features"] = True
    with pytest.raises(ValidationError):
        FnoArchitecture.model_validate(payload)


def test_framework_free_adapter_builds_exact_modules_and_raw_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fno_module, "_require_ml_runtime", _fake_runtime)
    predictor = FnoPredictor()

    core = cast(FakeFno, predictor._core)
    assert core.arguments == dict(predictor.architecture.physicsnemo_arguments)
    lift = cast(FakeConv2d, core.spec_encoder.lift_network)
    assert (lift.in_channels, lift.out_channels, lift.kernel_size) == (2, 64, 1)
    head = cast(FakeSequential, predictor._cd_head)
    assert [
        (module.in_features, module.out_features)
        for module in head.modules
        if isinstance(module, FakeLinear)
    ] == [(68, 64), (64, 32), (32, 1)]

    result = predictor(_fake_batch())

    assert cast(FakeTorchModule, predictor._runtime.torch).autocast_calls == [
        {"device_type": "cpu", "enabled": False}
    ]
    assert result.fields_normalized.shape == (2, 3, *MODEL_SPATIAL_SHAPE)
    assert result.cd_head.shape == (2,)
    assert cast(FakeTensor, result.fields_normalized).lineage == "decoded-grid"
    assert predictor.training
    contract: FlowPredictor = predictor
    predicted = contract.predict(_fake_batch())
    assert cast(FakeTorchModule, predictor._runtime.torch).autocast_calls == [
        {"device_type": "cpu", "enabled": False},
        {"device_type": "cpu", "enabled": False},
    ]
    assert predicted.fields_normalized.shape == result.fields_normalized.shape
    assert predictor.training
    assert predictor.eval().training is False
    assert predictor.train().training is True
    assert predictor.to("cpu") is predictor
    assert tuple(predictor.named_parameters())
    assert predictor.state_dict()
    assert predictor.load_state_dict(predictor.state_dict())
    predictor.zero_grad()


def test_prediction_result_rejects_shape_dtype_and_device_drift() -> None:
    valid_fields = cast(
        TensorLike,
        FakeTensor((2, 3, *MODEL_SPATIAL_SHAPE)),
    )
    valid_cd = cast(TensorLike, FakeTensor((2,)))
    PredictionBatchResult(fields_normalized=valid_fields, cd_head=valid_cd)

    for fields, cd_head in (
        (cast(TensorLike, FakeTensor((2, 2, *MODEL_SPATIAL_SHAPE))), valid_cd),
        (
            cast(
                TensorLike,
                FakeTensor((2, 3, *MODEL_SPATIAL_SHAPE), dtype="torch.float64"),
            ),
            valid_cd,
        ),
        (
            valid_fields,
            cast(TensorLike, FakeTensor((2,), device="cuda:0")),
        ),
    ):
        with pytest.raises(ArtifactIntegrityError):
            PredictionBatchResult(fields_normalized=fields, cd_head=cd_head)


def test_adapter_rejects_missing_runtime_version_device_and_empty_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> ModuleType:
        raise ImportError("not installed")

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(DependencyUnavailableError, match=r"ml.*extra"):
        fno_module._require_ml_runtime()

    runtime = _fake_runtime()
    wrong_version = fno_module._MlRuntime(
        torch=runtime.torch,
        fno_type=runtime.fno_type,
        physicsnemo_version="9.0.0",
    )
    monkeypatch.setattr(fno_module, "_require_ml_runtime", lambda: wrong_version)
    with pytest.raises(DependencyUnavailableError, match=r"2\.2\.1"):
        FnoPredictor()

    monkeypatch.setattr(fno_module, "_require_ml_runtime", _fake_runtime)
    predictor = FnoPredictor()
    with pytest.raises(TypeError, match="batch must"):
        predictor(cast(PredictionBatch, object()))
    with pytest.raises(TypeError, match="boolean"):
        predictor.train(cast(bool, 1))
    with pytest.raises(ArtifactIntegrityError, match="fluid cell"):
        predictor(_fake_batch(empty_fluid=True))
    first_parameter = next(iter(predictor.parameters()))
    first_parameter.dtype = "torch.float16"
    with pytest.raises(ArtifactIntegrityError, match="parameters must remain float32"):
        predictor(_fake_batch())
    first_parameter.dtype = "torch.float32"
    second_parameter = tuple(predictor.parameters())[1]
    second_parameter.device = "cuda:0"
    with pytest.raises(ArtifactIntegrityError, match="must share one device"):
        predictor(_fake_batch())
    second_parameter.device = "cpu"
    predictor.to("cuda:0")
    with pytest.raises(DeviceUnavailableError, match="batch is on cpu"):
        predictor(_fake_batch())


def test_real_physicsnemo_shapes_parameters_gradients_and_unmasked_output() -> None:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("physicsnemo") is None:
        pytest.skip("the optional ml runtime is not installed")
    torch = cast(Any, importlib.import_module("torch"))
    torch.manual_seed(17)
    predictor = FnoPredictor()
    inputs = torch.zeros((1, 2, *MODEL_SPATIAL_SHAPE), dtype=torch.float32)
    fluid_mask = torch.ones((1, 1, *MODEL_SPATIAL_SHAPE), dtype=torch.bool)
    fluid_mask[..., 0, 0] = False
    batch = PredictionBatch(
        inputs=inputs,
        fluid_mask=fluid_mask,
        design_params=torch.zeros((1, 4), dtype=torch.float32),
    )

    result = predictor(batch)
    fields = cast(Any, result.fields_normalized)
    cd_head = cast(Any, result.cd_head)
    loss = fields.square().mean() + cd_head.square().mean()
    loss.backward()

    parameters = tuple(predictor.parameters())
    assert sum(parameter.numel() for parameter in parameters) == 37_780_804
    assert all(parameter.grad is not None for parameter in parameters)
    assert result.fields_normalized.shape == (1, 3, *MODEL_SPATIAL_SHAPE)
    assert result.cd_head.shape == (1,)
    assert result.fields_normalized.dtype == torch.float32
    assert result.cd_head.dtype == torch.float32
    assert bool(torch.any(fields[..., 0, 0] != 0).item())
    assert not any(
        isinstance(module, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm))
        for module in predictor._root.modules()
    )
