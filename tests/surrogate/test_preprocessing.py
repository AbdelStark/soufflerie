from __future__ import annotations

import importlib
from dataclasses import replace
from types import ModuleType
from typing import Self, cast

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from soufflerie.errors import (
    ArtifactIntegrityError,
    DependencyUnavailableError,
    DeviceUnavailableError,
)
from soufflerie.schemas import Split
from soufflerie.surrogate import preprocessing
from soufflerie.surrogate.preprocessing import (
    MODEL_CELL_COUNT,
    MODEL_REFERENCE_DIAMETER_LU,
    MODEL_SPATIAL_SHAPE,
    STANDARD_DEVIATION_FLOOR,
    OutputChannelStatistics,
    OutputNormalizationStatistics,
    PredictionBatch,
    PreprocessedBatch,
    PreprocessingSample,
    PreprocessingStatistics,
    TensorLike,
    denormalize_fields,
    fit_preprocessing_statistics,
    prediction_batch_to_torch,
    preprocess_batch,
    preprocess_sample,
    validate_prediction_batch,
)

DATASET_ID = "d" * 20
Float16Array = npt.NDArray[np.float16]
UInt8Array = npt.NDArray[np.uint8]


def _field(value: float) -> Float16Array:
    result = np.full(MODEL_SPATIAL_SHAPE, value, dtype=np.float16)
    result.flags.writeable = False
    return result


def _mask(*, obstacle: bool = False) -> UInt8Array:
    value = np.uint8(1 if obstacle else 0)
    result = np.full(MODEL_SPATIAL_SHAPE, value, dtype=np.uint8)
    result.flags.writeable = False
    return result


def _sample(
    index: int,
    *,
    split: str,
    u: float,
    v: float,
    rho: float,
    sdf: float = 8.0,
    reynolds: float = 170.0,
    aspect_ratio: float = 0.75,
    rotation_deg: float = 15.0,
    scale: float = 1.0,
    dataset_id: str = DATASET_ID,
) -> PreprocessingSample:
    assert split in {"train", "validation", "test"}
    return PreprocessingSample(
        dataset_id=dataset_id,
        case_id=f"{index:020x}",
        split=cast(Split, split),
        aspect_ratio=aspect_ratio,
        rotation_deg=rotation_deg,
        scale=scale,
        reynolds=reynolds,
        cd=1.0 + index / 100.0,
        u_mean=_field(u),
        v_mean=_field(v),
        rho_mean=_field(rho),
        sdf=_field(sdf),
        obstacle_mask=_mask(),
    )


def _training_pair() -> tuple[PreprocessingSample, PreprocessingSample]:
    return (
        _sample(
            1,
            split="train",
            u=1.0,
            v=-2.0,
            rho=0.75,
            sdf=-32.0,
            reynolds=40.0,
            aspect_ratio=0.5,
            rotation_deg=0.0,
            scale=0.75,
        ),
        _sample(
            2,
            split="train",
            u=3.0,
            v=2.0,
            rho=1.25,
            sdf=32.0,
            reynolds=300.0,
            aspect_ratio=1.0,
            rotation_deg=30.0,
            scale=1.25,
        ),
    )


def test_fit_uses_training_cells_only_and_matches_hand_computed_population_moments() -> None:
    first, second = _training_pair()
    sentinel = _sample(
        3,
        split="validation",
        u=1000.0,
        v=-1000.0,
        rho=10.0,
    )

    expected = fit_preprocessing_statistics((first, second))
    actual = fit_preprocessing_statistics((sentinel, second, first))

    assert actual == expected
    assert PreprocessingStatistics.model_validate_json(actual.model_dump_json()) == actual
    assert actual.dataset_id == DATASET_ID
    assert actual.training_case_count == 2
    assert actual.training_cell_count == 2 * MODEL_CELL_COUNT
    assert actual.outputs.u_mean.mean == pytest.approx(2.0)
    assert actual.outputs.u_mean.standard_deviation == pytest.approx(1.0)
    assert actual.outputs.v_mean.mean == pytest.approx(0.0)
    assert actual.outputs.v_mean.standard_deviation == pytest.approx(2.0)
    assert actual.outputs.rho_delta.mean == pytest.approx(0.0)
    assert actual.outputs.rho_delta.standard_deviation == pytest.approx(0.25)
    assert not any(
        channel.floored
        for channel in (
            actual.outputs.u_mean,
            actual.outputs.v_mean,
            actual.outputs.rho_delta,
        )
    )


def test_fit_records_the_exact_floor_for_constant_channels() -> None:
    statistics = fit_preprocessing_statistics((_sample(1, split="train", u=2.0, v=-1.0, rho=1.0),))

    for channel in (
        statistics.outputs.u_mean,
        statistics.outputs.v_mean,
        statistics.outputs.rho_delta,
    ):
        assert channel.raw_standard_deviation == 0.0
        assert channel.standard_deviation == STANDARD_DEVIATION_FLOOR
        assert channel.floored


def test_preprocess_and_denormalize_round_trip_fixed_float32_batch() -> None:
    first, second = _training_pair()
    first_mask = first.obstacle_mask.copy()
    first_mask[0, 0] = np.uint8(1)
    first_mask.flags.writeable = False
    first = replace(first, obstacle_mask=first_mask)
    statistics = fit_preprocessing_statistics((first, second))

    batch = preprocess_batch((first, second), statistics)

    assert batch.inputs.shape == (2, 2, *MODEL_SPATIAL_SHAPE)
    assert batch.fields_normalized.shape == (2, 3, *MODEL_SPATIAL_SHAPE)
    assert batch.fluid_mask.shape == (2, 1, *MODEL_SPATIAL_SHAPE)
    assert batch.design_params.shape == (2, 4)
    assert batch.cd.shape == (2,)
    assert batch.inputs.dtype == np.dtype(np.float32)
    assert batch.fields_normalized.dtype == np.dtype(np.float32)
    assert batch.fluid_mask.dtype == np.dtype(np.bool_)
    assert batch.design_params.dtype == np.dtype(np.float32)
    assert batch.cd.dtype == np.dtype(np.float32)
    assert all(
        not array.flags.writeable
        for array in (
            batch.inputs,
            batch.fields_normalized,
            batch.fluid_mask,
            batch.design_params,
            batch.cd,
        )
    )
    assert np.all(batch.inputs[0, 0] == np.float32(-1.0))
    assert np.all(batch.inputs[1, 0] == np.float32(1.0))
    assert np.all(batch.inputs[0, 1] == np.float32(-1.0))
    assert np.all(batch.inputs[1, 1] == np.float32(1.0))
    np.testing.assert_array_equal(batch.design_params[0], np.full(4, -1.0, np.float32))
    np.testing.assert_array_equal(batch.design_params[1], np.full(4, 1.0, np.float32))
    # The model mask follows SDF sign even when the separately persisted mask
    # differs at rasterized boundary cells.
    assert not np.any(batch.fluid_mask[0])
    assert np.all(batch.fluid_mask[1])

    restored = denormalize_fields(batch.fields_normalized, statistics)
    expected = np.stack(
        [
            np.stack(
                (
                    first.u_mean.astype(np.float32),
                    first.v_mean.astype(np.float32),
                    first.rho_mean.astype(np.float32),
                )
            ),
            np.stack(
                (
                    second.u_mean.astype(np.float32),
                    second.v_mean.astype(np.float32),
                    second.rho_mean.astype(np.float32),
                )
            ),
        ]
    )
    np.testing.assert_allclose(restored, expected, rtol=0.0, atol=2e-7)
    assert restored.dtype == np.dtype(np.float32)
    assert restored.flags.c_contiguous
    assert not restored.flags.writeable


def test_sdf_input_uses_the_persisted_grid_reference_diameter_and_clips() -> None:
    sample = _sample(1, split="train", u=0.0, v=0.0, rho=1.0, sdf=0.0)
    sdf = sample.sdf.copy()
    sdf[0, 0] = np.float16(-2.0 * MODEL_REFERENCE_DIAMETER_LU)
    sdf[0, 1] = np.float16(0.5 * MODEL_REFERENCE_DIAMETER_LU)
    sdf[0, 2] = np.float16(2.0 * MODEL_REFERENCE_DIAMETER_LU)
    sdf.flags.writeable = False
    sample = replace(sample, sdf=sdf)
    statistics = fit_preprocessing_statistics((sample,))

    batch = preprocess_batch((sample,), statistics)

    np.testing.assert_array_equal(batch.inputs[0, 0, 0, :3], [-1.0, 0.5, 1.0])


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("u_mean", np.zeros(MODEL_SPATIAL_SHAPE, dtype=np.float32)),
        ("v_mean", np.zeros((MODEL_SPATIAL_SHAPE[0], 1), dtype=np.float16)),
        (
            "rho_mean",
            np.asfortranarray(np.ones(MODEL_SPATIAL_SHAPE, dtype=np.float16)),
        ),
        ("sdf", np.full(MODEL_SPATIAL_SHAPE, np.nan, dtype=np.float16)),
        ("obstacle_mask", np.full(MODEL_SPATIAL_SHAPE, 2, dtype=np.uint8)),
    ],
)
def test_stored_array_boundary_rejects_implicit_dtype_shape_layout_and_invalid_values(
    field: str,
    replacement: npt.NDArray[np.generic],
) -> None:
    sample = _sample(1, split="train", u=0.0, v=0.0, rho=1.0)

    with pytest.raises(ArtifactIntegrityError):
        if field == "u_mean":
            replace(sample, u_mean=cast(Float16Array, replacement))
        elif field == "v_mean":
            replace(sample, v_mean=cast(Float16Array, replacement))
        elif field == "rho_mean":
            replace(sample, rho_mean=cast(Float16Array, replacement))
        elif field == "sdf":
            replace(sample, sdf=cast(Float16Array, replacement))
        else:
            replace(sample, obstacle_mask=cast(UInt8Array, replacement))


def test_fit_rejects_empty_missing_training_duplicate_and_mixed_dataset_samples() -> None:
    first = _sample(1, split="train", u=0.0, v=0.0, rho=1.0)
    validation = _sample(2, split="validation", u=0.0, v=0.0, rho=1.0)
    other_dataset = _sample(
        3,
        split="train",
        u=0.0,
        v=0.0,
        rho=1.0,
        dataset_id="e" * 20,
    )

    with pytest.raises(ArtifactIntegrityError, match="at least one"):
        fit_preprocessing_statistics(())
    with pytest.raises(ArtifactIntegrityError, match="no training"):
        fit_preprocessing_statistics((validation,))
    with pytest.raises(ArtifactIntegrityError, match="unique"):
        fit_preprocessing_statistics((first, first))
    with pytest.raises(ArtifactIntegrityError, match="one dataset"):
        fit_preprocessing_statistics((first, other_dataset))


def test_statistics_schema_rejects_incoherent_floor_and_cell_count() -> None:
    valid_channel = OutputChannelStatistics(
        mean=0.0,
        raw_standard_deviation=1.0,
        standard_deviation=1.0,
        floored=False,
    )
    outputs = OutputNormalizationStatistics(
        u_mean=valid_channel,
        v_mean=valid_channel,
        rho_delta=valid_channel,
    )

    with pytest.raises(ValidationError, match="floor policy"):
        OutputChannelStatistics(
            mean=0.0,
            raw_standard_deviation=0.0,
            standard_deviation=1.0,
            floored=True,
        )
    with pytest.raises(ValidationError, match="whether the raw deviation"):
        OutputChannelStatistics(
            mean=0.0,
            raw_standard_deviation=1.0,
            standard_deviation=1.0,
            floored=True,
        )
    with pytest.raises(ValidationError, match="cases times"):
        PreprocessingStatistics(
            dataset_id=DATASET_ID,
            training_case_count=2,
            training_cell_count=MODEL_CELL_COUNT,
            outputs=outputs,
        )

    valid = PreprocessingStatistics(
        dataset_id=DATASET_ID,
        training_case_count=1,
        training_cell_count=MODEL_CELL_COUNT,
        outputs=outputs,
    )
    for field, value, message in (
        ("sdf_clip", (-2.0, 1.0), "fixed channel/range"),
        ("sdf_reference_diameter_lu", 8.0, "fixed model grid"),
        ("standard_deviation_floor", 1e-5, "remain 1e-6"),
    ):
        payload = valid.model_dump(mode="python")
        payload[field] = value
        with pytest.raises(ValidationError, match=message):
            PreprocessingStatistics.model_validate(payload)


def test_sample_and_numpy_records_reject_invalid_scalars_identity_and_mutability() -> None:
    sample = _sample(1, split="train", u=0.0, v=0.0, rho=1.0)
    for changes in (
        {"dataset_id": "bad"},
        {"split": cast(Split, "holdout")},
        {"aspect_ratio": cast(float, True)},
        {"rotation_deg": float("nan")},
        {"scale": 0.5},
        {"reynolds": 301.0},
    ):
        with pytest.raises(ArtifactIntegrityError):
            replace(sample, **changes)

    writable = sample.u_mean.copy()
    with pytest.raises(ArtifactIntegrityError, match="read-only"):
        replace(sample, u_mean=writable)
    writable_mask = sample.obstacle_mask.copy()
    with pytest.raises(ArtifactIntegrityError, match="read-only"):
        replace(sample, obstacle_mask=writable_mask)
    zero_density = _field(0.0)
    with pytest.raises(ArtifactIntegrityError, match="strictly positive"):
        replace(sample, rho_mean=zero_density)
    with pytest.raises(TypeError, match="contain PreprocessingSample"):
        fit_preprocessing_statistics((cast(PreprocessingSample, object()),))

    statistics = fit_preprocessing_statistics((sample,))
    prepared = preprocess_sample(sample, statistics)
    with pytest.raises(ArtifactIntegrityError, match="cd must"):
        replace(prepared, cd=cast(np.float32, np.float64(1.0)))
    mutable_inputs = prepared.inputs.copy()
    with pytest.raises(ArtifactIntegrityError, match="read-only"):
        replace(prepared, inputs=mutable_inputs)
    with pytest.raises(TypeError, match="sample must"):
        preprocess_sample(cast(PreprocessingSample, object()), statistics)
    with pytest.raises(TypeError, match="statistics must"):
        preprocess_sample(sample, cast(PreprocessingStatistics, object()))

    batch = preprocess_batch((sample,), statistics)
    with pytest.raises(ArtifactIntegrityError, match="four-dimensional"):
        replace(batch, inputs=batch.inputs[0])
    mutable_cd = batch.cd.copy()
    with pytest.raises(ArtifactIntegrityError, match="read-only"):
        replace(batch, cd=mutable_cd)
    with pytest.raises(TypeError, match="statistics must"):
        denormalize_fields(
            batch.fields_normalized,
            cast(PreprocessingStatistics, object()),
        )


def test_batch_and_denormalization_reject_implicit_dtype_and_dataset_changes() -> None:
    first, second = _training_pair()
    statistics = fit_preprocessing_statistics((first, second))
    batch = preprocess_batch((first,), statistics)

    with pytest.raises(ArtifactIntegrityError, match="dataset IDs differ"):
        preprocess_batch((replace(first, dataset_id="e" * 20),), statistics)
    with pytest.raises(ArtifactIntegrityError, match="at least one"):
        preprocess_batch((), statistics)
    with pytest.raises(ArtifactIntegrityError, match="dtype float32"):
        denormalize_fields(batch.fields_normalized.astype(np.float64), statistics)
    with pytest.raises(ArtifactIntegrityError, match="C-contiguous"):
        denormalize_fields(batch.fields_normalized[:, :, :, ::-1], statistics)


class FakeTensor:
    def __init__(
        self,
        shape: object,
        *,
        dtype: str,
        device: str = "cpu",
        contiguous: bool = True,
        finite: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        self._finite = finite

    def is_contiguous(self) -> bool:
        return self._contiguous

    def isfinite(self) -> Self:
        return self

    def all(self) -> Self:
        return self

    def item(self) -> object:
        return self._finite


def _tensor_batch(
    *,
    inputs: FakeTensor | None = None,
    fluid_mask: FakeTensor | None = None,
    design_params: FakeTensor | None = None,
) -> PredictionBatch:
    return PredictionBatch(
        inputs=inputs or FakeTensor((2, 2, *MODEL_SPATIAL_SHAPE), dtype="torch.float32"),
        fluid_mask=fluid_mask or FakeTensor((2, 1, *MODEL_SPATIAL_SHAPE), dtype="torch.bool"),
        design_params=design_params or FakeTensor((2, 4), dtype="torch.float32"),
    )


def test_prediction_tensor_contract_accepts_only_exact_matching_tensors() -> None:
    batch = _tensor_batch()
    validate_prediction_batch(batch, expected_device="cpu")

    invalid_inputs = (
        FakeTensor((2, 2, *MODEL_SPATIAL_SHAPE), dtype="torch.float64"),
        FakeTensor((2, 2, *MODEL_SPATIAL_SHAPE), dtype="torch.float32", contiguous=False),
        FakeTensor((2, 2, *MODEL_SPATIAL_SHAPE), dtype="torch.float32", finite=False),
        FakeTensor((2, 3, *MODEL_SPATIAL_SHAPE), dtype="torch.float32"),
    )
    for tensor in invalid_inputs:
        with pytest.raises(ArtifactIntegrityError):
            _tensor_batch(inputs=tensor)
    with pytest.raises(ArtifactIntegrityError, match="one device"):
        _tensor_batch(design_params=FakeTensor((2, 4), dtype="torch.float32", device="cuda:0"))
    with pytest.raises(DeviceUnavailableError, match="explicitly requested"):
        validate_prediction_batch(batch, expected_device="cuda:0")

    cuda_batch = _tensor_batch(
        inputs=FakeTensor(
            (2, 2, *MODEL_SPATIAL_SHAPE),
            dtype="torch.float32",
            device="cuda:0",
        ),
        fluid_mask=FakeTensor(
            (2, 1, *MODEL_SPATIAL_SHAPE),
            dtype="torch.bool",
            device="cuda:0",
        ),
        design_params=FakeTensor((2, 4), dtype="torch.float32", device="cuda:0"),
    )
    validate_prediction_batch(cuda_batch, expected_device="cuda")


class BrokenContiguityTensor(FakeTensor):
    def is_contiguous(self) -> bool:
        raise RuntimeError("unavailable")


class BrokenFiniteTensor(FakeTensor):
    def isfinite(self) -> Self:
        raise RuntimeError("unavailable")


def test_prediction_tensor_protocol_rejects_malformed_methods_shape_device_and_calls() -> None:
    invalid_inputs = (
        FakeTensor("bad", dtype="torch.float32"),
        FakeTensor((), dtype="torch.float32"),
        FakeTensor((2, 2, 320), dtype="torch.float32"),
        FakeTensor((2, 2, *MODEL_SPATIAL_SHAPE), dtype="torch.float32", device="mps"),
        BrokenContiguityTensor((2, 2, *MODEL_SPATIAL_SHAPE), dtype="torch.float32"),
        BrokenFiniteTensor((2, 2, *MODEL_SPATIAL_SHAPE), dtype="torch.float32"),
    )
    for tensor in invalid_inputs:
        with pytest.raises(ArtifactIntegrityError):
            _tensor_batch(inputs=tensor)

    batch = _tensor_batch()
    with pytest.raises(TypeError, match="batch must"):
        validate_prediction_batch(cast(PredictionBatch, object()), expected_device="cpu")
    with pytest.raises(DeviceUnavailableError, match="cpu or cuda"):
        validate_prediction_batch(batch, expected_device="mps")


class FakeCuda:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class FakeTorchTensor(FakeTensor):
    def __init__(self, array: npt.NDArray[np.generic]) -> None:
        dtype = "torch.bool" if array.dtype == np.dtype(np.bool_) else "torch.float32"
        super().__init__(array.shape, dtype=dtype)

    def to(self, *, device: str) -> Self:
        self.device = device
        return self


class FakeTorchModule:
    def __init__(self, *, available: bool = True) -> None:
        self.cuda = FakeCuda(available=available)

    def from_numpy(self, array: npt.NDArray[np.generic]) -> FakeTorchTensor:
        assert array.flags.c_contiguous
        assert array.flags.writeable
        return FakeTorchTensor(array)


def test_torch_adapter_is_lazy_and_performs_only_explicit_device_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _training_pair()
    statistics = fit_preprocessing_statistics((first, second))
    numpy_batch = preprocess_batch((first,), statistics)

    def fake_torch() -> ModuleType:
        return cast(ModuleType, FakeTorchModule())

    monkeypatch.setattr(preprocessing, "_require_torch", fake_torch)
    result = prediction_batch_to_torch(numpy_batch, device="cuda:0")

    assert str(result.inputs.dtype) == "torch.float32"
    assert str(result.fluid_mask.dtype) == "torch.bool"
    assert str(result.inputs.device) == "cuda:0"
    assert str(result.fluid_mask.device) == "cuda:0"
    assert str(result.design_params.device) == "cuda:0"


def test_torch_adapter_reports_the_missing_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _training_pair()
    statistics = fit_preprocessing_statistics((first, second))
    numpy_batch = preprocess_batch((first,), statistics)

    def unavailable(_name: str) -> ModuleType:
        raise ImportError("not installed")

    monkeypatch.setattr(importlib, "import_module", unavailable)

    with pytest.raises(DependencyUnavailableError, match=r"ml.*extra"):
        prediction_batch_to_torch(numpy_batch, device="cpu")

    with pytest.raises(TypeError, match="batch must"):
        prediction_batch_to_torch(cast(PreprocessedBatch, object()), device="cpu")
    with pytest.raises(DeviceUnavailableError, match="cpu or cuda"):
        prediction_batch_to_torch(numpy_batch, device="mps")

    def fake_unavailable_cuda() -> ModuleType:
        return cast(ModuleType, FakeTorchModule(available=False))

    monkeypatch.setattr(preprocessing, "_require_torch", fake_unavailable_cuda)
    with pytest.raises(DeviceUnavailableError, match="unavailable"):
        prediction_batch_to_torch(numpy_batch, device="cuda:0")


def test_public_tensor_protocol_is_structurally_satisfied() -> None:
    tensor: TensorLike = FakeTensor((1,), dtype="torch.float32")
    assert tensor.is_contiguous()
