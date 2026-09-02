"""Leakage-safe output statistics and strict model tensor adapters."""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, Protocol, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from soufflerie.errors import (
    ArtifactIntegrityError,
    DependencyUnavailableError,
    DeviceUnavailableError,
)
from soufflerie.geometry import OUTPUT_GRID_NX, OUTPUT_GRID_NY, reference_diameter_lu
from soufflerie.schemas import ContentId, GridSpec, Split, StrictFrozenModel, VersionedModel
from soufflerie.schemas import validate_array as validate_numpy_array

Float16Array = npt.NDArray[np.float16]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
UInt8Array = npt.NDArray[np.uint8]

MODEL_SPATIAL_SHAPE = (OUTPUT_GRID_NY, OUTPUT_GRID_NX)
MODEL_GRID = GridSpec(nx=OUTPUT_GRID_NX, ny=OUTPUT_GRID_NY)
MODEL_REFERENCE_DIAMETER_LU = reference_diameter_lu(MODEL_GRID)
MODEL_CELL_COUNT = OUTPUT_GRID_NY * OUTPUT_GRID_NX
STANDARD_DEVIATION_FLOOR = 1e-6
REYNOLDS_MIN = 40.0
REYNOLDS_MAX = 300.0
_CONTENT_ID_PATTERN = re.compile(r"^[0-9a-f]{20}$")
_DEVICE_PATTERN = re.compile(r"^(?:cpu|cuda(?::[0-9]+)?)$")
_INPUT_CHANNELS = ("sdf_over_reference_diameter", "reynolds_affine")
_OUTPUT_CHANNELS = ("u_mean", "v_mean", "rho_delta")
_DESIGN_PARAMETERS = ("aspect_ratio", "rotation_deg", "scale", "reynolds")


class OutputChannelStatistics(StrictFrozenModel):
    """One fp64-fitted scalar normalization contract."""

    mean: float = Field(allow_inf_nan=False)
    raw_standard_deviation: float = Field(ge=0.0, allow_inf_nan=False)
    standard_deviation: float = Field(ge=STANDARD_DEVIATION_FLOOR, allow_inf_nan=False)
    floored: bool

    @model_validator(mode="after")
    def _floor_is_coherent(self) -> Self:
        expected_floored = self.raw_standard_deviation < STANDARD_DEVIATION_FLOOR
        expected_standard_deviation = (
            STANDARD_DEVIATION_FLOOR if expected_floored else self.raw_standard_deviation
        )
        if self.floored != expected_floored:
            raise ValueError("floored must record whether the raw deviation is below 1e-6")
        if not math.isclose(
            self.standard_deviation,
            expected_standard_deviation,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("standard_deviation does not match the declared floor policy")
        return self


class OutputNormalizationStatistics(StrictFrozenModel):
    """Ordered output-channel statistics for u, v, and density delta."""

    u_mean: OutputChannelStatistics
    v_mean: OutputChannelStatistics
    rho_delta: OutputChannelStatistics


class PreprocessingStatistics(VersionedModel):
    """Durable training-only preprocessing record embedded in model bundles."""

    dataset_id: ContentId
    training_split: Literal["train"] = "train"
    training_case_count: int = Field(ge=1, le=600)
    training_cell_count: int = Field(ge=MODEL_CELL_COUNT, le=600 * MODEL_CELL_COUNT)
    spatial_shape: tuple[Literal[320], Literal[256]] = (320, 256)
    input_channels: tuple[Literal["sdf_over_reference_diameter"], Literal["reynolds_affine"]] = (
        "sdf_over_reference_diameter",
        "reynolds_affine",
    )
    output_channels: tuple[Literal["u_mean"], Literal["v_mean"], Literal["rho_delta"]] = (
        "u_mean",
        "v_mean",
        "rho_delta",
    )
    design_parameters: tuple[
        Literal["aspect_ratio"],
        Literal["rotation_deg"],
        Literal["scale"],
        Literal["reynolds"],
    ] = ("aspect_ratio", "rotation_deg", "scale", "reynolds")
    stored_field_dtype: Literal["float16"] = "float16"
    fit_dtype: Literal["float64"] = "float64"
    model_dtype: Literal["float32"] = "float32"
    sdf_reference_diameter_lu: float = Field(
        default=16.0,
        allow_inf_nan=False,
        json_schema_extra={"const": 16.0},
    )
    sdf_clip: tuple[float, float] = Field(
        default=(-1.0, 1.0),
        json_schema_extra={"const": [-1.0, 1.0]},
    )
    reynolds_range: tuple[float, float] = Field(
        default=(40.0, 300.0),
        json_schema_extra={"const": [40.0, 300.0]},
    )
    standard_deviation_floor: float = Field(
        default=1e-6,
        allow_inf_nan=False,
        json_schema_extra={"const": 1e-6},
    )
    outputs: OutputNormalizationStatistics

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
            "design_parameters",
            "sdf_clip",
            "reynolds_range",
        ):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized

    @model_validator(mode="after")
    def _contract_is_coherent(self) -> Self:
        if self.training_cell_count != self.training_case_count * MODEL_CELL_COUNT:
            raise ValueError("training_cell_count must equal cases times the fixed grid size")
        expected_values: tuple[tuple[object, object], ...] = (
            (self.spatial_shape, MODEL_SPATIAL_SHAPE),
            (self.input_channels, _INPUT_CHANNELS),
            (self.output_channels, _OUTPUT_CHANNELS),
            (self.design_parameters, _DESIGN_PARAMETERS),
            (self.sdf_clip, (-1.0, 1.0)),
            (self.reynolds_range, (REYNOLDS_MIN, REYNOLDS_MAX)),
        )
        if any(actual != expected for actual, expected in expected_values):
            raise ValueError("preprocessing fixed channel/range contract changed")
        if not math.isclose(
            self.sdf_reference_diameter_lu,
            MODEL_REFERENCE_DIAMETER_LU,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("sdf_reference_diameter_lu must match the fixed model grid")
        if not math.isclose(
            self.standard_deviation_floor,
            STANDARD_DEVIATION_FLOOR,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("standard_deviation_floor must remain 1e-6")
        return self


def _validate_scalar(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ArtifactIntegrityError(f"PRE-1 SCALAR: {name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ArtifactIntegrityError(f"PRE-1 SCALAR: {name} must be finite")
    if minimum is not None and numeric < minimum:
        raise ArtifactIntegrityError(f"PRE-1 SCALAR: {name} is below {minimum:g}")
    if maximum is not None and numeric > maximum:
        raise ArtifactIntegrityError(f"PRE-1 SCALAR: {name} exceeds {maximum:g}")


@dataclass(frozen=True, slots=True)
class PreprocessingSample:
    """One manifest/run pair at the fp16 artifact boundary."""

    dataset_id: str
    case_id: str
    split: Split
    aspect_ratio: float
    rotation_deg: float
    scale: float
    reynolds: float
    cd: float
    u_mean: Float16Array
    v_mean: Float16Array
    rho_mean: Float16Array
    sdf: Float16Array
    obstacle_mask: UInt8Array

    def __post_init__(self) -> None:
        for identity_name, identity_value in (
            ("dataset_id", self.dataset_id),
            ("case_id", self.case_id),
        ):
            if (
                not isinstance(identity_value, str)
                or _CONTENT_ID_PATTERN.fullmatch(identity_value) is None
            ):
                raise ArtifactIntegrityError(
                    f"PRE-1 IDENTITY: {identity_name} must be 20 lowercase hexadecimal digits"
                )
        if self.split not in {"train", "validation", "test"}:
            raise ArtifactIntegrityError("PRE-1 SPLIT: unsupported dataset split")
        for scalar_name, scalar_value, minimum, maximum in (
            ("aspect_ratio", self.aspect_ratio, 0.5, 1.0),
            ("rotation_deg", self.rotation_deg, 0.0, 30.0),
            ("scale", self.scale, 0.75, 1.25),
            ("reynolds", self.reynolds, REYNOLDS_MIN, REYNOLDS_MAX),
            ("cd", self.cd, None, None),
        ):
            _validate_scalar(
                scalar_name,
                scalar_value,
                minimum=minimum,
                maximum=maximum,
            )
        for name in ("u_mean", "v_mean", "rho_mean", "sdf"):
            array = getattr(self, name)
            validate_numpy_array(
                array,
                name=name,
                dtype=np.dtype(np.float16),
                shape=MODEL_SPATIAL_SHAPE,
            )
            if array.flags.writeable:
                raise ArtifactIntegrityError(f"PRE-1 MUTABLE: {name} must be read-only")
        validate_numpy_array(
            self.obstacle_mask,
            name="obstacle_mask",
            dtype=np.dtype(np.uint8),
            shape=MODEL_SPATIAL_SHAPE,
            finite=False,
        )
        if np.any(self.obstacle_mask > np.uint8(1)):
            raise ArtifactIntegrityError("PRE-1 MASK: obstacle_mask values must be zero or one")
        if self.obstacle_mask.flags.writeable:
            raise ArtifactIntegrityError("PRE-1 MUTABLE: obstacle_mask must be read-only")
        if not np.all(self.rho_mean > np.float16(0.0)):
            raise ArtifactIntegrityError("PRE-1 DENSITY: rho_mean must be strictly positive")


@dataclass(slots=True)
class _StreamingMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: npt.NDArray[np.generic]) -> None:
        chunk = np.asarray(values, dtype=np.float64).reshape(-1)
        chunk_count = int(chunk.size)
        chunk_mean = float(np.mean(chunk, dtype=np.float64))
        centered = chunk - chunk_mean
        chunk_m2 = float(np.sum(centered * centered, dtype=np.float64))
        if self.count == 0:
            self.count = chunk_count
            self.mean = chunk_mean
            self.m2 = chunk_m2
            return
        total = self.count + chunk_count
        delta = chunk_mean - self.mean
        self.mean += delta * chunk_count / total
        self.m2 += chunk_m2 + delta * delta * self.count * chunk_count / total
        self.count = total

    def finish(self) -> OutputChannelStatistics:
        if self.count <= 0:
            raise ArtifactIntegrityError("PRE-2 FIT: output statistics require training cells")
        variance = max(0.0, self.m2 / self.count)
        raw = math.sqrt(variance)
        floored = raw < STANDARD_DEVIATION_FLOOR
        return OutputChannelStatistics(
            mean=self.mean,
            raw_standard_deviation=raw,
            standard_deviation=STANDARD_DEVIATION_FLOOR if floored else raw,
            floored=floored,
        )


def fit_preprocessing_statistics(
    samples: Iterable[PreprocessingSample],
) -> PreprocessingStatistics:
    """Fit deterministic fp64 output moments from training samples only."""

    materialized = tuple(samples)
    if not materialized:
        raise ArtifactIntegrityError("PRE-2 FIT: at least one preprocessing sample is required")
    if any(not isinstance(sample, PreprocessingSample) for sample in materialized):
        raise TypeError("samples must contain PreprocessingSample instances")
    dataset_ids = {sample.dataset_id for sample in materialized}
    if len(dataset_ids) != 1:
        raise ArtifactIntegrityError("PRE-2 FIT: all samples must share one dataset identity")
    case_ids = [sample.case_id for sample in materialized]
    if len(case_ids) != len(set(case_ids)):
        raise ArtifactIntegrityError("PRE-2 FIT: sample case identities must be unique")
    training = tuple(
        sorted(
            (sample for sample in materialized if sample.split == "train"),
            key=lambda sample: sample.case_id,
        )
    )
    if not training:
        raise ArtifactIntegrityError("PRE-2 FIT: no training-split samples were supplied")

    u_moments = _StreamingMoments()
    v_moments = _StreamingMoments()
    rho_delta_moments = _StreamingMoments()
    for sample in training:
        u_moments.update(sample.u_mean)
        v_moments.update(sample.v_mean)
        rho_delta = sample.rho_mean.astype(np.float64) - np.float64(1.0)
        rho_delta_moments.update(rho_delta)

    return PreprocessingStatistics(
        dataset_id=next(iter(dataset_ids)),
        training_case_count=len(training),
        training_cell_count=len(training) * MODEL_CELL_COUNT,
        outputs=OutputNormalizationStatistics(
            u_mean=u_moments.finish(),
            v_mean=v_moments.finish(),
            rho_delta=rho_delta_moments.finish(),
        ),
    )


def _readonly(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    array.flags.writeable = False
    return array


def _normalize_range(value: float, minimum: float, maximum: float) -> np.float32:
    return np.float32(2.0 * (value - minimum) / (maximum - minimum) - 1.0)


@dataclass(frozen=True, slots=True)
class PreprocessedSample:
    """One immutable NumPy model sample before batching or device transfer."""

    inputs: Float32Array
    fields_normalized: Float32Array
    fluid_mask: BoolArray
    design_params: Float32Array
    cd: np.float32

    def __post_init__(self) -> None:
        validate_numpy_array(
            self.inputs,
            name="inputs",
            dtype=np.dtype(np.float32),
            shape=(2, *MODEL_SPATIAL_SHAPE),
        )
        validate_numpy_array(
            self.fields_normalized,
            name="fields_normalized",
            dtype=np.dtype(np.float32),
            shape=(3, *MODEL_SPATIAL_SHAPE),
        )
        validate_numpy_array(
            self.fluid_mask,
            name="fluid_mask",
            dtype=np.dtype(np.bool_),
            shape=(1, *MODEL_SPATIAL_SHAPE),
            finite=False,
        )
        validate_numpy_array(
            self.design_params,
            name="design_params",
            dtype=np.dtype(np.float32),
            shape=(4,),
        )
        if not isinstance(self.cd, np.float32) or not np.isfinite(self.cd):
            raise ArtifactIntegrityError("PRE-3 BATCH: cd must be one finite float32 scalar")
        for name in ("inputs", "fields_normalized", "fluid_mask", "design_params"):
            if getattr(self, name).flags.writeable:
                raise ArtifactIntegrityError(f"PRE-3 BATCH: {name} must be read-only")


def _normalize_output(
    values: Float16Array,
    channel: OutputChannelStatistics,
    *,
    subtract_one: bool = False,
) -> Float32Array:
    restored = values.astype(np.float32)
    if subtract_one:
        restored -= np.float32(1.0)
    result = np.ascontiguousarray(
        (restored - np.float32(channel.mean)) / np.float32(channel.standard_deviation),
        dtype=np.float32,
    )
    if not np.isfinite(result).all():
        raise ArtifactIntegrityError("PRE-3 FINITE: normalized output contains NaN or infinity")
    return result


def preprocess_sample(
    sample: PreprocessingSample,
    statistics: PreprocessingStatistics,
) -> PreprocessedSample:
    """Map one curated sample to the fixed NumPy model contract."""

    if not isinstance(sample, PreprocessingSample):
        raise TypeError("sample must be a PreprocessingSample")
    if not isinstance(statistics, PreprocessingStatistics):
        raise TypeError("statistics must be PreprocessingStatistics")
    if sample.dataset_id != statistics.dataset_id:
        raise ArtifactIntegrityError("PRE-3 IDENTITY: sample and statistics dataset IDs differ")

    inputs = np.empty((2, *MODEL_SPATIAL_SHAPE), dtype=np.float32)
    np.divide(
        sample.sdf.astype(np.float32),
        np.float32(MODEL_REFERENCE_DIAMETER_LU),
        out=inputs[0],
    )
    np.clip(inputs[0], np.float32(-1.0), np.float32(1.0), out=inputs[0])
    inputs[1].fill(_normalize_range(float(sample.reynolds), REYNOLDS_MIN, REYNOLDS_MAX))

    outputs = statistics.outputs
    fields_normalized = np.stack(
        (
            _normalize_output(sample.u_mean, outputs.u_mean),
            _normalize_output(sample.v_mean, outputs.v_mean),
            _normalize_output(sample.rho_mean, outputs.rho_delta, subtract_one=True),
        ),
        axis=0,
    ).astype(np.float32, copy=False)
    fluid_mask = np.ascontiguousarray((sample.sdf > np.float16(0.0))[None, ...])
    design_params = np.asarray(
        (
            _normalize_range(float(sample.aspect_ratio), 0.5, 1.0),
            _normalize_range(float(sample.rotation_deg), 0.0, 30.0),
            _normalize_range(float(sample.scale), 0.75, 1.25),
            _normalize_range(float(sample.reynolds), REYNOLDS_MIN, REYNOLDS_MAX),
        ),
        dtype=np.float32,
    )
    return PreprocessedSample(
        inputs=cast(Float32Array, _readonly(inputs)),
        fields_normalized=cast(
            Float32Array,
            _readonly(np.ascontiguousarray(fields_normalized, dtype=np.float32)),
        ),
        fluid_mask=cast(BoolArray, _readonly(fluid_mask)),
        design_params=cast(Float32Array, _readonly(design_params)),
        cd=np.float32(sample.cd),
    )


@dataclass(frozen=True, slots=True)
class PreprocessedBatch:
    """Batch-first NumPy tensors used by training and Torch conversion."""

    inputs: Float32Array
    fields_normalized: Float32Array
    fluid_mask: BoolArray
    design_params: Float32Array
    cd: Float32Array

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, np.ndarray) or self.inputs.ndim != 4:
            raise ArtifactIntegrityError("PRE-3 BATCH: inputs must be a four-dimensional array")
        batch_size = int(self.inputs.shape[0])
        if batch_size <= 0:
            raise ArtifactIntegrityError("PRE-3 BATCH: batch size must be positive")
        contracts = (
            (
                "inputs",
                self.inputs,
                np.dtype(np.float32),
                (batch_size, 2, *MODEL_SPATIAL_SHAPE),
                True,
            ),
            (
                "fields_normalized",
                self.fields_normalized,
                np.dtype(np.float32),
                (batch_size, 3, *MODEL_SPATIAL_SHAPE),
                True,
            ),
            (
                "fluid_mask",
                self.fluid_mask,
                np.dtype(np.bool_),
                (batch_size, 1, *MODEL_SPATIAL_SHAPE),
                False,
            ),
            (
                "design_params",
                self.design_params,
                np.dtype(np.float32),
                (batch_size, 4),
                True,
            ),
            ("cd", self.cd, np.dtype(np.float32), (batch_size,), True),
        )
        for name, array, dtype, shape, finite in contracts:
            validate_numpy_array(
                array,
                name=name,
                dtype=dtype,
                shape=shape,
                finite=finite,
            )
            if array.flags.writeable:
                raise ArtifactIntegrityError(f"PRE-3 BATCH: {name} must be read-only")


def preprocess_batch(
    samples: Sequence[PreprocessingSample],
    statistics: PreprocessingStatistics,
) -> PreprocessedBatch:
    """Stack one non-empty homogeneous batch without implicit casts."""

    if not samples:
        raise ArtifactIntegrityError("PRE-3 BATCH: at least one sample is required")
    prepared = tuple(preprocess_sample(sample, statistics) for sample in samples)
    arrays = {
        "inputs": np.stack([sample.inputs for sample in prepared], axis=0),
        "fields_normalized": np.stack([sample.fields_normalized for sample in prepared], axis=0),
        "fluid_mask": np.stack([sample.fluid_mask for sample in prepared], axis=0),
        "design_params": np.stack([sample.design_params for sample in prepared], axis=0),
        "cd": np.asarray([sample.cd for sample in prepared], dtype=np.float32),
    }
    for array in arrays.values():
        _readonly(array)
    return PreprocessedBatch(
        inputs=cast(Float32Array, arrays["inputs"]),
        fields_normalized=cast(Float32Array, arrays["fields_normalized"]),
        fluid_mask=cast(BoolArray, arrays["fluid_mask"]),
        design_params=cast(Float32Array, arrays["design_params"]),
        cd=cast(Float32Array, arrays["cd"]),
    )


def denormalize_fields(
    fields_normalized: Float32Array,
    statistics: PreprocessingStatistics,
) -> Float32Array:
    """Return public float32 u, v, rho fields from normalized batch output."""

    if not isinstance(statistics, PreprocessingStatistics):
        raise TypeError("statistics must be PreprocessingStatistics")
    if not isinstance(fields_normalized, np.ndarray) or fields_normalized.ndim != 4:
        raise ArtifactIntegrityError(
            "PRE-4 OUTPUT: normalized fields must be a four-dimensional NumPy array"
        )
    batch_size = int(fields_normalized.shape[0])
    validate_numpy_array(
        fields_normalized,
        name="fields_normalized",
        dtype=np.dtype(np.float32),
        shape=(batch_size, 3, *MODEL_SPATIAL_SHAPE),
    )
    result = np.empty_like(fields_normalized)
    for index, channel in enumerate(
        (statistics.outputs.u_mean, statistics.outputs.v_mean, statistics.outputs.rho_delta)
    ):
        np.multiply(
            fields_normalized[:, index],
            np.float32(channel.standard_deviation),
            out=result[:, index],
        )
        result[:, index] += np.float32(channel.mean)
    result[:, 2] += np.float32(1.0)
    if not np.isfinite(result).all():
        raise ArtifactIntegrityError("PRE-4 OUTPUT: de-normalized fields are non-finite")
    return cast(Float32Array, _readonly(np.ascontiguousarray(result, dtype=np.float32)))


class TensorLike(Protocol):
    """Structural subset used to validate Torch tensors without importing Torch."""

    @property
    def shape(self) -> object: ...

    @property
    def dtype(self) -> object: ...

    @property
    def device(self) -> object: ...

    def is_contiguous(self) -> bool: ...

    def isfinite(self) -> TensorLike: ...

    def all(self) -> TensorLike: ...

    def item(self) -> object: ...


def _tensor_shape(tensor: TensorLike, *, name: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(dimension) for dimension in cast(Iterable[int], tensor.shape))
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(f"PRE-5 TENSOR: {name} has an invalid shape") from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ArtifactIntegrityError(f"PRE-5 TENSOR: {name} dimensions must be positive")
    return shape


def _tensor_is_finite(tensor: TensorLike, *, name: str) -> None:
    try:
        finite = tensor.isfinite().all().item()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"PRE-5 TENSOR: {name} does not expose finite-value validation"
        ) from error
    if not isinstance(finite, (bool, np.bool_)) or not bool(finite):
        raise ArtifactIntegrityError(f"PRE-5 TENSOR: {name} contains NaN or infinity")


def _validate_tensor(
    tensor: TensorLike,
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    finite: bool,
) -> str:
    if _tensor_shape(tensor, name=name) != shape:
        raise ArtifactIntegrityError(
            f"PRE-5 TENSOR: {name} must have shape {shape}, got {_tensor_shape(tensor, name=name)}"
        )
    if str(tensor.dtype) != dtype:
        raise ArtifactIntegrityError(
            f"PRE-5 TENSOR: {name} must have dtype {dtype}, got {tensor.dtype}"
        )
    try:
        contiguous = tensor.is_contiguous()
    except (AttributeError, RuntimeError, TypeError) as error:
        raise ArtifactIntegrityError(f"PRE-5 TENSOR: {name} does not expose contiguity") from error
    if contiguous is not True:
        raise ArtifactIntegrityError(f"PRE-5 TENSOR: {name} must be contiguous")
    if finite:
        _tensor_is_finite(tensor, name=name)
    device = str(tensor.device)
    if _DEVICE_PATTERN.fullmatch(device) is None:
        raise ArtifactIntegrityError(f"PRE-5 TENSOR: {name} has unsupported device {device!r}")
    return device


@dataclass(frozen=True, slots=True)
class PredictionBatch:
    """Strict Torch-compatible inference inputs from RFC-0006."""

    inputs: TensorLike
    fluid_mask: TensorLike
    design_params: TensorLike

    def __post_init__(self) -> None:
        batch_shape = _tensor_shape(self.inputs, name="inputs")
        if len(batch_shape) != 4:
            raise ArtifactIntegrityError("PRE-5 TENSOR: inputs must be batch-first rank four")
        batch_size = batch_shape[0]
        devices = {
            _validate_tensor(
                self.inputs,
                name="inputs",
                dtype="torch.float32",
                shape=(batch_size, 2, *MODEL_SPATIAL_SHAPE),
                finite=True,
            ),
            _validate_tensor(
                self.fluid_mask,
                name="fluid_mask",
                dtype="torch.bool",
                shape=(batch_size, 1, *MODEL_SPATIAL_SHAPE),
                finite=False,
            ),
            _validate_tensor(
                self.design_params,
                name="design_params",
                dtype="torch.float32",
                shape=(batch_size, 4),
                finite=True,
            ),
        }
        if len(devices) != 1:
            raise ArtifactIntegrityError("PRE-5 TENSOR: batch tensors must share one device")


@dataclass(frozen=True, slots=True)
class PredictionBatchResult:
    """Strict raw normalized fields and learned drag returned by a predictor."""

    fields_normalized: TensorLike
    cd_head: TensorLike

    def __post_init__(self) -> None:
        fields_shape = _tensor_shape(self.fields_normalized, name="fields_normalized")
        if len(fields_shape) != 4:
            raise ArtifactIntegrityError(
                "PRE-6 RESULT: fields_normalized must be batch-first rank four"
            )
        batch_size = fields_shape[0]
        devices = {
            _validate_tensor(
                self.fields_normalized,
                name="fields_normalized",
                dtype="torch.float32",
                shape=(batch_size, 3, *MODEL_SPATIAL_SHAPE),
                finite=True,
            ),
            _validate_tensor(
                self.cd_head,
                name="cd_head",
                dtype="torch.float32",
                shape=(batch_size,),
                finite=True,
            ),
        }
        if len(devices) != 1:
            raise ArtifactIntegrityError("PRE-6 RESULT: result tensors must share one device")


class FlowPredictor(Protocol):
    """Framework-neutral structural contract shared by learned and baseline predictors."""

    def predict(self, batch: PredictionBatch) -> PredictionBatchResult: ...


def validate_prediction_batch(batch: PredictionBatch, *, expected_device: str) -> None:
    """Fail if an already-validated batch is not on the explicitly requested device."""

    if not isinstance(batch, PredictionBatch):
        raise TypeError("batch must be a PredictionBatch")
    if _DEVICE_PATTERN.fullmatch(expected_device) is None:
        raise DeviceUnavailableError("expected_device must be cpu or cuda[:index]")
    actual = str(batch.inputs.device)
    matches = actual == expected_device or (
        expected_device == "cuda" and actual.startswith("cuda:")
    )
    if not matches:
        raise DeviceUnavailableError(
            f"prediction batch is on {actual}, but {expected_device} was explicitly requested"
        )


def _require_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise DependencyUnavailableError(
            "prediction tensor conversion requires the 'ml' extra"
        ) from error


def prediction_batch_to_torch(
    batch: PreprocessedBatch,
    *,
    device: str,
) -> PredictionBatch:
    """Explicitly copy NumPy inputs into Torch and move them to one named device."""

    if not isinstance(batch, PreprocessedBatch):
        raise TypeError("batch must be a PreprocessedBatch")
    if _DEVICE_PATTERN.fullmatch(device) is None:
        raise DeviceUnavailableError("device must be cpu or cuda[:index]")
    torch = _require_torch()
    if device.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise DeviceUnavailableError(f"requested device {device} is unavailable")

    def convert(array: npt.NDArray[np.generic]) -> TensorLike:
        owned = np.array(array, copy=True, order="C")
        tensor = torch.from_numpy(owned)
        if device != "cpu":
            try:
                tensor = tensor.to(device=device)
            except (RuntimeError, ValueError) as error:
                raise DeviceUnavailableError(f"requested device {device} is unavailable") from error
        return cast(TensorLike, tensor)

    result = PredictionBatch(
        inputs=convert(batch.inputs),
        fluid_mask=convert(batch.fluid_mask),
        design_params=convert(batch.design_params),
    )
    validate_prediction_batch(result, expected_device=device)
    return result


__all__ = [
    "MODEL_CELL_COUNT",
    "MODEL_REFERENCE_DIAMETER_LU",
    "MODEL_SPATIAL_SHAPE",
    "STANDARD_DEVIATION_FLOOR",
    "FlowPredictor",
    "OutputChannelStatistics",
    "OutputNormalizationStatistics",
    "PredictionBatch",
    "PredictionBatchResult",
    "PreprocessedBatch",
    "PreprocessedSample",
    "PreprocessingSample",
    "PreprocessingStatistics",
    "TensorLike",
    "denormalize_fields",
    "fit_preprocessing_statistics",
    "prediction_batch_to_torch",
    "preprocess_batch",
    "preprocess_sample",
    "validate_prediction_batch",
]
