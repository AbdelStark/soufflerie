from __future__ import annotations

import numpy as np
import pytest

from soufflerie.errors import DomainError
from soufflerie.schemas import FlowFields
from soufflerie.solver.diagnostics import field_drag_coefficient, select_control_surface


def _fields(
    *,
    u: float = 0.05,
    v: float = 0.0,
    rho: float = 1.0,
    center_x: float = 20.0,
    center_y: float = 24.0,
    radius: float = 4.0,
) -> FlowFields:
    ny, nx = 48, 64
    y, x = np.mgrid[:ny, :nx]
    sdf = np.ascontiguousarray(
        np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) - radius,
        dtype=np.float32,
    )
    return FlowFields(
        u=np.full((ny, nx), u, dtype=np.float32),
        v=np.full((ny, nx), v, dtype=np.float32),
        rho=np.full((ny, nx), rho, dtype=np.float32),
        sdf=sdf,
        obstacle_mask=np.ascontiguousarray(sdf <= np.float32(0.0)),
    )


def test_control_surface_is_tight_deterministic_and_eight_cells_clear() -> None:
    fields = _fields()

    first = select_control_surface(fields, sponge_start_x=56)
    second = select_control_surface(fields, sponge_start_x=56)

    assert first == second
    assert (first.left_x, first.right_x, first.bottom_y, first.top_y) == (8, 32, 12, 36)
    assert first.clearance_cells == 8
    assert first.minimum_sdf == pytest.approx(8.0)


def test_constant_fields_have_exactly_zero_pressure_and_convective_drag() -> None:
    estimate = field_drag_coefficient(
        _fields(v=0.0125),
        sponge_start_x=56,
        inlet_velocity_lu=0.05,
        reference_diameter_lu=8.0,
    )

    assert estimate.pressure_force_x_lu == 0.0
    assert estimate.convective_force_x_lu == 0.0
    assert estimate.force_x_lu == 0.0
    assert estimate.cd == 0.0


def test_manufactured_pressure_difference_has_declared_sign_and_normalization() -> None:
    fields = _fields(u=0.0)
    surface = select_control_surface(fields, sponge_start_x=56)
    rho = fields.rho.copy()
    rho[:, surface.left_x] = np.float32(1.03)
    rho[:, surface.right_x] = np.float32(0.97)
    manufactured = FlowFields(
        u=fields.u,
        v=fields.v,
        rho=rho,
        sdf=fields.sdf,
        obstacle_mask=fields.obstacle_mask,
    )

    estimate = field_drag_coefficient(
        manufactured,
        sponge_start_x=56,
        inlet_velocity_lu=0.05,
        reference_diameter_lu=8.0,
    )

    side_samples = surface.top_y - surface.bottom_y + 1
    expected_pressure_force = side_samples * (1.03 - 0.97) / 3.0
    expected_normalization = 0.5 * 0.05**2 * 8.0
    assert estimate.pressure_force_x_lu == pytest.approx(expected_pressure_force, rel=1e-6)
    assert estimate.convective_force_x_lu == 0.0
    assert estimate.normalization_lu == pytest.approx(expected_normalization)
    assert estimate.cd == pytest.approx(expected_pressure_force / expected_normalization, rel=1e-6)
    assert estimate.cd > 0.0


def test_symmetric_transverse_flux_cancels_in_fixed_surface_order() -> None:
    fields = _fields()
    surface = select_control_surface(fields, sponge_start_x=56)
    v = fields.v.copy()
    v[surface.bottom_y, :] = np.float32(0.02)
    v[surface.top_y, :] = np.float32(0.02)
    symmetric = FlowFields(
        u=fields.u,
        v=v,
        rho=fields.rho,
        sdf=fields.sdf,
        obstacle_mask=fields.obstacle_mask,
    )

    estimate = field_drag_coefficient(
        symmetric,
        sponge_start_x=56,
        inlet_velocity_lu=0.05,
        reference_diameter_lu=8.0,
    )

    assert estimate.convective_force_x_lu == 0.0
    assert estimate.cd == 0.0


def test_invalid_control_surface_and_missing_obstacle_fail_closed() -> None:
    fields = _fields()
    with pytest.raises(DomainError, match="no 8-cell surface"):
        select_control_surface(fields, sponge_start_x=32)

    no_obstacle_sdf = np.ones(fields.shape, dtype=np.float32)
    no_obstacle = FlowFields(
        u=fields.u,
        v=fields.v,
        rho=fields.rho,
        sdf=no_obstacle_sdf,
        obstacle_mask=np.zeros(fields.shape, dtype=np.bool_),
    )
    with pytest.raises(DomainError, match="no obstacle zero contour"):
        select_control_surface(no_obstacle, sponge_start_x=56)


@pytest.mark.parametrize(
    ("name", "value"),
    (("inlet_velocity_lu", 0.0), ("reference_diameter_lu", float("nan"))),
)
def test_field_drag_rejects_invalid_normalization(name: str, value: float) -> None:
    arguments = {"inlet_velocity_lu": 0.05, "reference_diameter_lu": 8.0}
    arguments[name] = value
    with pytest.raises(DomainError, match=name):
        field_drag_coefficient(_fields(), sponge_start_x=56, **arguments)
