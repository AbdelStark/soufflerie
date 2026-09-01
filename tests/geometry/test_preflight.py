from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import soufflerie.geometry as geometry_module
from soufflerie.errors import DomainError
from soufflerie.geometry import reference_diameter_lu, validate_geometry
from soufflerie.schemas import GridSpec, ShapeParams


def _shape(*, aspect: float = 0.75, rotation: float = 12.0, scale: float = 1.0) -> ShapeParams:
    return ShapeParams(aspect_ratio=aspect, rotation_deg=rotation, scale=scale)


@pytest.mark.parametrize(
    ("aspect", "rotation", "scale"),
    list(itertools.product((0.5, 1.0), (0.0, 30.0), (0.75, 1.25))),
)
def test_all_canonical_public_domain_corners_pass(
    aspect: float, rotation: float, scale: float
) -> None:
    diagnostics = validate_geometry(
        _shape(aspect=aspect, rotation=rotation, scale=scale),
        GridSpec(nx=512, ny=256),
    )
    assert diagnostics.inlet_outlet_connected
    assert diagnostics.scaled_minor_diameter_lu >= 12.0


@given(
    aspect=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    rotation=st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.75, max_value=1.25, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, deadline=None)
def test_canonical_public_domain_samples_pass(aspect: float, rotation: float, scale: float) -> None:
    diagnostics = validate_geometry(
        _shape(aspect=aspect, rotation=rotation, scale=scale),
        GridSpec(nx=512, ny=256),
    )
    assert diagnostics.inlet_outlet_connected
    assert diagnostics.scaled_minor_diameter_lu >= 12.0


def test_successful_diagnostics_are_coherent() -> None:
    grid = GridSpec(nx=512, ny=256)
    diagnostics = validate_geometry(_shape(), grid)

    assert reference_diameter_lu(grid) == 32.0
    assert diagnostics.grid_shape == (256, 512)
    assert diagnostics.reference_diameter_lu == 32.0
    assert diagnostics.sponge_columns == 64
    assert diagnostics.sponge_start_x_lu == 448.0
    assert diagnostics.obstacle_cell_count > 0
    assert diagnostics.fluid_cell_count > 0
    assert diagnostics.obstacle_cell_count + diagnostics.fluid_cell_count == 512 * 256


def test_underresolved_geometry_fails_before_raster_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_rasterized(*args: object, **kwargs: object) -> npt.NDArray[np.float32]:
        raise AssertionError("dense geometry allocation must not happen")

    monkeypatch.setattr(geometry_module, "_ellipse_sdf_unchecked", fail_if_rasterized)
    with pytest.raises(DomainError, match="GEO-2 RESOLUTION"):
        validate_geometry(
            _shape(aspect=0.5, rotation=0.0, scale=0.75),
            GridSpec(nx=128, ny=64),
        )


def test_boundary_clearance_fails_closed() -> None:
    with pytest.raises(DomainError, match="GEO-2 CLEARANCE: inlet"):
        validate_geometry(
            _shape(aspect=1.0, rotation=0.0, scale=1.25),
            GridSpec(nx=100, ny=96),
        )


def test_disconnected_fluid_domain_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def disconnected_raster(*args: object, **kwargs: object) -> npt.NDArray[np.float32]:
        sdf = np.ones((256, 512), dtype=np.float32)
        sdf[:, 256] = -1.0
        sdf.flags.writeable = False
        return sdf

    monkeypatch.setattr(geometry_module, "_ellipse_sdf_unchecked", disconnected_raster)
    with pytest.raises(DomainError, match="GEO-1 CONNECTIVITY"):
        validate_geometry(_shape(), GridSpec(nx=512, ny=256))


def test_sponge_cannot_occupy_entire_grid() -> None:
    with pytest.raises(DomainError, match="GEO-2 SPONGE"):
        validate_geometry(_shape(), GridSpec(nx=32, ny=128))
