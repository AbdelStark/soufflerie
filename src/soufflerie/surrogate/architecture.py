"""Frozen FNO architecture metadata independent of the optional ML runtime."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import model_validator

from soufflerie.schemas import VersionedModel


class FnoArchitecture(VersionedModel):
    """Exact RFC-0006 architecture serialized into every model bundle."""

    architecture: Literal["fno2d-v1"] = "fno2d-v1"
    framework: Literal["nvidia-physicsnemo"] = "nvidia-physicsnemo"
    framework_version: Literal["2.2.1"] = "2.2.1"
    spatial_shape: tuple[Literal[320], Literal[256]] = (320, 256)
    input_channels: tuple[
        Literal["sdf_over_reference_diameter"],
        Literal["reynolds_affine"],
    ] = ("sdf_over_reference_diameter", "reynolds_affine")
    output_channels: tuple[
        Literal["u_mean_normalized"],
        Literal["v_mean_normalized"],
        Literal["rho_delta_normalized"],
    ] = ("u_mean_normalized", "v_mean_normalized", "rho_delta_normalized")
    model_dtype: Literal["float32"] = "float32"
    lifting_channels: tuple[Literal[2], Literal[64]] = (2, 64)
    latent_channels: Literal[64] = 64
    spectral_blocks: Literal[4] = 4
    retained_modes: tuple[Literal[24], Literal[24]] = (24, 24)
    padding: tuple[Literal[8], Literal[8]] = (8, 8)
    padding_type: Literal["constant"] = "constant"
    coordinate_features: Literal[False] = False
    spectral_activation: Literal["gelu"] = "gelu"
    projection_channels: tuple[Literal[64], Literal[128], Literal[3]] = (64, 128, 3)
    projection_activation: Literal["gelu"] = "gelu"
    cd_pooling: Literal["fluid_masked_mean"] = "fluid_masked_mean"
    cd_head_channels: tuple[Literal[68], Literal[64], Literal[32], Literal[1]] = (
        68,
        64,
        32,
        1,
    )
    cd_activation: Literal["gelu"] = "gelu"
    dropout: Literal[0] = 0
    batch_normalization: Literal[False] = False
    outputs_masked: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in (
            "spatial_shape",
            "input_channels",
            "output_channels",
            "lifting_channels",
            "retained_modes",
            "padding",
            "projection_channels",
            "cd_head_channels",
        ):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized

    @property
    def physicsnemo_arguments(self) -> Mapping[str, object]:
        """Return every resolved argument passed to PhysicsNeMo's FNO wrapper."""

        return MappingProxyType(
            {
                "in_channels": self.lifting_channels[0],
                "out_channels": self.projection_channels[-1],
                "decoder_layers": 1,
                "decoder_layer_size": self.projection_channels[1],
                "decoder_activation_fn": self.projection_activation,
                "dimension": 2,
                "latent_channels": self.latent_channels,
                "num_fno_layers": self.spectral_blocks,
                "num_fno_modes": list(self.retained_modes),
                "padding": self.padding[0],
                "padding_type": self.padding_type,
                "activation_fn": self.spectral_activation,
                "coord_features": self.coordinate_features,
            }
        )


__all__ = ["FnoArchitecture"]
