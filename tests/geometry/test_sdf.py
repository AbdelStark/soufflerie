from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from soufflerie.errors import DomainError
from soufflerie.geometry import ellipse_sdf, normalized_sdf_input, obstacle_mask
from soufflerie.schemas import GridSpec, ShapeParams


def _shape(*, aspect: float = 0.75, rotation: float = 12.0, scale: float = 1.0) -> ShapeParams:
    return ShapeParams(aspect_ratio=aspect, rotation_deg=rotation, scale=scale)


def test_sdf_layout_sign_mask_and_zero_contour() -> None:
    grid = GridSpec(nx=101, ny=201)
    shape = _shape(aspect=1.0, rotation=0.0, scale=200.0 / 201.0)
    sdf = ellipse_sdf(shape, grid)
    mask = obstacle_mask(sdf)

    assert sdf.shape == grid.shape
    assert sdf.dtype == np.float32
    assert sdf.flags.c_contiguous
    assert not sdf.flags.writeable
    assert np.isfinite(sdf).all()
    assert sdf[100, 30] < 0.0
    assert sdf[0, 0] > 0.0
    assert sdf[100, 35] == pytest.approx(0.0, abs=1e-6)
    assert mask.dtype == np.bool_
    assert mask.flags.c_contiguous
    assert not mask.flags.writeable
    np.testing.assert_array_equal(mask, sdf <= 0.0)


@given(rotation=st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=20, deadline=None)
def test_circle_is_bitwise_rotation_invariant(rotation: float) -> None:
    grid = GridSpec(nx=101, ny=81)
    baseline = ellipse_sdf(_shape(aspect=1.0, rotation=0.0), grid)
    rotated = ellipse_sdf(_shape(aspect=1.0, rotation=rotation), grid)
    np.testing.assert_array_equal(rotated, baseline)


@given(
    aspect=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    rotation=st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, deadline=None)
def test_scale_monotonically_contains_smaller_mask(aspect: float, rotation: float) -> None:
    grid = GridSpec(nx=101, ny=81)
    small = obstacle_mask(ellipse_sdf(_shape(aspect=aspect, rotation=rotation, scale=0.75), grid))
    large = obstacle_mask(ellipse_sdf(_shape(aspect=aspect, rotation=rotation, scale=1.25), grid))
    assert np.all(~small | large)


def test_normalized_model_channel_is_clipped_read_only_fp32() -> None:
    grid = GridSpec(nx=101, ny=81)
    shape = _shape()
    channel = normalized_sdf_input(shape, grid)

    assert channel.dtype == np.float32
    assert channel.flags.c_contiguous
    assert not channel.flags.writeable
    assert float(channel.min()) >= -1.0
    assert float(channel.max()) <= 1.0
    expected = np.clip(ellipse_sdf(shape, grid) / np.float32(grid.ny / 20.0), -1.0, 1.0)
    np.testing.assert_array_equal(channel, expected)


@pytest.mark.parametrize(
    "sdf",
    [
        np.ones((4, 4), dtype=np.float64),
        np.ones((4, 8), dtype=np.float32)[:, ::2],
        np.array([[np.nan]], dtype=np.float32),
    ],
)
def test_mask_rejects_invalid_sdf_arrays(sdf: npt.NDArray[np.generic]) -> None:
    with pytest.raises(DomainError, match="GEO-1 SDF"):
        obstacle_mask(cast("npt.NDArray[np.float32]", sdf))
