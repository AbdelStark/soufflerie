"""Optional PhysicsNeMo FNO adapter with an exact RFC-0006 architecture."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Self, cast

from soufflerie.errors import (
    ArtifactIntegrityError,
    DependencyUnavailableError,
    DeviceUnavailableError,
    InternalInvariantError,
)
from soufflerie.surrogate.architecture import FnoArchitecture
from soufflerie.surrogate.preprocessing import (
    MODEL_SPATIAL_SHAPE,
    PredictionBatch,
    PredictionBatchResult,
)


@dataclass(frozen=True, slots=True)
class _MlRuntime:
    torch: ModuleType
    fno_type: type[Any]
    physicsnemo_version: str


def _require_ml_runtime() -> _MlRuntime:
    try:
        torch = importlib.import_module("torch")
        physicsnemo = importlib.import_module("physicsnemo")
        fno_module = importlib.import_module("physicsnemo.models.fno")
    except ImportError as error:
        raise DependencyUnavailableError(
            "FNO construction requires the locked 'ml' extra"
        ) from error
    fno_type = getattr(fno_module, "FNO", None)
    version = getattr(physicsnemo, "__version__", None)
    if not isinstance(fno_type, type) or not isinstance(version, str):
        raise DependencyUnavailableError("the installed PhysicsNeMo runtime is malformed")
    return _MlRuntime(torch=torch, fno_type=fno_type, physicsnemo_version=version)


class FnoPredictor:
    """Trainable FNO and inference adapter without importing ML frameworks at module import."""

    def __init__(self, architecture: FnoArchitecture | None = None) -> None:
        self.architecture = architecture or FnoArchitecture()
        self._runtime = _require_ml_runtime()
        if self._runtime.physicsnemo_version != self.architecture.framework_version:
            raise DependencyUnavailableError(
                "FNO architecture requires PhysicsNeMo "
                f"{self.architecture.framework_version}, got {self._runtime.physicsnemo_version}"
            )

        torch = cast(Any, self._runtime.torch)
        core = self._runtime.fno_type(**dict(self.architecture.physicsnemo_arguments))
        # PhysicsNeMo's stock lift is 2 -> 32 -> 64. RFC-0006 instead freezes one
        # pointwise 2 -> 64 projection, while retaining PhysicsNeMo's spectral blocks.
        core.spec_encoder.lift_network = torch.nn.Conv2d(
            self.architecture.lifting_channels[0],
            self.architecture.lifting_channels[1],
            kernel_size=1,
        )
        cd_channels = self.architecture.cd_head_channels
        cd_head = torch.nn.Sequential(
            torch.nn.Linear(cd_channels[0], cd_channels[1]),
            torch.nn.GELU(),
            torch.nn.Linear(cd_channels[1], cd_channels[2]),
            torch.nn.GELU(),
            torch.nn.Linear(cd_channels[2], cd_channels[3]),
        )
        root = torch.nn.Module()
        root.add_module("core", core)
        root.add_module("cd_head", cd_head)
        self._root = root
        self._core = core
        self._cd_head = cd_head

    @property
    def training(self) -> bool:
        return bool(self._root.training)

    def train(self, mode: bool = True) -> Self:
        if not isinstance(mode, bool):
            raise TypeError("mode must be a boolean")
        self._root.train(mode)
        return self

    def eval(self) -> Self:
        return self.train(False)

    def to(self, *args: object, **kwargs: object) -> Self:
        self._root.to(*args, **kwargs)
        return self

    def parameters(self, *, recurse: bool = True) -> Iterable[Any]:
        return cast(Iterable[Any], self._root.parameters(recurse=recurse))

    def named_parameters(self, *, recurse: bool = True) -> Iterable[tuple[str, Any]]:
        return cast(
            Iterable[tuple[str, Any]],
            self._root.named_parameters(recurse=recurse),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self._root.state_dict())

    def load_state_dict(self, state_dict: Mapping[str, object], *, strict: bool = True) -> object:
        return self._root.load_state_dict(state_dict, strict=strict)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self._root.zero_grad(set_to_none=set_to_none)

    def __call__(self, batch: PredictionBatch) -> PredictionBatchResult:
        return self.forward(batch)

    def _checked_tensors(self, batch: PredictionBatch) -> tuple[Any, Any, Any]:
        if not isinstance(batch, PredictionBatch):
            raise TypeError("batch must be a PredictionBatch")
        torch = cast(Any, self._runtime.torch)
        tensors = (batch.inputs, batch.fluid_mask, batch.design_params)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ArtifactIntegrityError(
                "FNO-1 FRAMEWORK: prediction batch must contain PyTorch tensors"
            )
        parameters = tuple(self.parameters())
        if not parameters:
            raise InternalInvariantError("FNO model has no parameters")
        if any(str(parameter.dtype) != "torch.float32" for parameter in parameters):
            raise ArtifactIntegrityError("FNO-1 DTYPE: model parameters must remain float32")
        parameter_devices = {str(parameter.device) for parameter in parameters}
        if len(parameter_devices) != 1:
            raise ArtifactIntegrityError("FNO-1 DEVICE: model parameters must share one device")
        model_device = next(iter(parameter_devices))
        batch_device = str(batch.inputs.device)
        if model_device != batch_device:
            raise DeviceUnavailableError(
                f"FNO parameters are on {model_device}, but the batch is on {batch_device}"
            )
        return cast(tuple[Any, Any, Any], tensors)

    def forward(self, batch: PredictionBatch) -> PredictionBatchResult:
        """Run one differentiable forward pass and preserve raw obstacle predictions."""

        inputs, fluid_mask, design_params = self._checked_tensors(batch)
        torch = cast(Any, self._runtime.torch)
        batch_size = int(inputs.shape[0])
        # cuFFT does not accept bf16/fp16 inputs. CUDA autocast can cast the
        # lift convolution before PhysicsNeMo reaches its first rfft2, so keep
        # the complete spectral encoder in fp32 while allowing the projection
        # and Cd heads to use the caller's mixed-precision context.
        device_type = str(inputs.device).split(":", maxsplit=1)[0]
        with torch.autocast(device_type=device_type, enabled=False):
            latent = self._core.spec_encoder(inputs.to(dtype=torch.float32))
        expected_latent_shape = (
            batch_size,
            self.architecture.latent_channels,
            *MODEL_SPATIAL_SHAPE,
        )
        if tuple(int(item) for item in latent.shape) != expected_latent_shape:
            raise InternalInvariantError(
                "FNO latent shape changed: "
                f"expected {expected_latent_shape}, got {tuple(latent.shape)}"
            )

        mask = fluid_mask.to(dtype=latent.dtype)
        fluid_counts = mask.sum(dim=(-2, -1))
        if bool(torch.any(fluid_counts <= 0).item()):
            raise ArtifactIntegrityError(
                "FNO-2 MASK: every sample requires at least one fluid cell"
            )
        pooled = (latent * mask).sum(dim=(-2, -1)) / fluid_counts
        design_for_head = design_params.to(dtype=latent.dtype)
        cd_inputs = torch.cat((pooled, design_for_head), dim=1)
        cd_head = self._cd_head(cd_inputs)

        points, latent_shape = self._core.spec_encoder.grid_to_points(latent)
        decoded = self._core.decoder_net(points)
        fields = self._core.spec_encoder.points_to_grid(decoded, latent_shape)

        fields_float32 = fields.to(dtype=torch.float32).contiguous()
        cd_float32 = cd_head.reshape(batch_size).to(dtype=torch.float32).contiguous()
        return PredictionBatchResult(
            fields_normalized=fields_float32,
            cd_head=cd_float32,
        )

    def predict(self, batch: PredictionBatch) -> PredictionBatchResult:
        """Run deterministic evaluation without retaining an autograd graph."""

        torch = cast(Any, self._runtime.torch)
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                return self.forward(batch)
        finally:
            self.train(was_training)


__all__ = ["FnoPredictor"]
