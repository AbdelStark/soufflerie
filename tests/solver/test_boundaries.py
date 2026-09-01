from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from soufflerie.errors import DomainError
from soufflerie.geometry import validate_geometry
from soufflerie.schemas import GridSpec, ShapeParams
from soufflerie.solver.boundaries import (
    MAX_SPONGE_STRENGTH,
    BoundaryOwner,
    apply_sponge_numpy,
    channel_boundary_ownership,
    numpy_channel_step,
    pull_stream_channel_numpy,
    pull_stream_halfway_walls_numpy,
    sponge_columns,
    sponge_strengths,
    validate_sponge_mask,
)
from soufflerie.solver.kernels import WarpKernelAdapter
from soufflerie.solver.lattice import (
    D2Q9_OPPOSITE,
    D2Q9_VELOCITIES,
    D2Q9_WEIGHTS,
    DerivedLatticeConfig,
    LatticeConfig,
    equilibrium,
    macroscopic_moments,
    preflight_lattice,
)
from soufflerie.solver.lifecycle import NumpyChannelStepper
from soufflerie.solver.numpy_oracle import initialize_numpy


def _config() -> DerivedLatticeConfig:
    return preflight_lattice(
        LatticeConfig(
            nx=40,
            ny=16,
            steps=100,
            warmup_steps=20,
            sample_interval=10,
            inlet_velocity_lu=0.05,
            reynolds=100.0,
            reference_diameter_lu=4.0,
        )
    )


def _mask(config: DerivedLatticeConfig) -> npt.NDArray[np.bool_]:
    result = np.zeros(config.shape, dtype=np.bool_)
    result[0, :] = True
    result[-1, :] = True
    return result


def _populations(config: DerivedLatticeConfig) -> npt.NDArray[np.float32]:
    result = np.broadcast_to(D2Q9_WEIGHTS, (*config.shape, 9)).copy()
    result += np.linspace(-0.001, 0.001, result.size, dtype=np.float32).reshape(result.shape)
    return result


def _require_warp() -> None:
    if importlib.util.find_spec("warp") is None:
        pytest.skip("install the Soufflerie solver extra to run Warp boundary tests")


def test_boundary_ownership_is_exclusive_and_corner_fixture_backed() -> None:
    owners = channel_boundary_ownership(GridSpec(nx=40, ny=16))

    assert owners.shape == (16, 40, 9)
    assert owners.dtype == np.uint8
    assert not owners.flags.writeable
    assert set(np.unique(owners)) == set(BoundaryOwner)
    counts = np.bincount(owners.ravel(), minlength=len(BoundaryOwner))
    assert tuple(counts) == (4_648, 480, 480, 112, 40)
    assert np.all(owners[2:-2, 0, :] == BoundaryOwner.INLET)

    lower_left = owners[1, 0]
    assert lower_left[1] == BoundaryOwner.INLET
    assert lower_left[2] == BoundaryOwner.LOWER_WALL
    assert lower_left[5] == BoundaryOwner.LOWER_WALL
    assert lower_left[8] == BoundaryOwner.INLET

    upper_left = owners[-2, 0]
    assert upper_left[1] == BoundaryOwner.INLET
    assert upper_left[5] == BoundaryOwner.INLET
    assert upper_left[8] == BoundaryOwner.UPPER_WALL

    lower_right = owners[1, -1]
    assert lower_right[3] == BoundaryOwner.OUTLET
    assert lower_right[6] == BoundaryOwner.LOWER_WALL
    assert lower_right[7] == BoundaryOwner.OUTLET

    upper_right = owners[-2, -1]
    assert upper_right[3] == BoundaryOwner.OUTLET
    assert upper_right[6] == BoundaryOwner.OUTLET
    assert upper_right[7] == BoundaryOwner.UPPER_WALL


def test_halfway_wall_links_bounce_from_same_fluid_node() -> None:
    config = _config()
    post_collision = _populations(config)
    streamed = pull_stream_halfway_walls_numpy(post_collision)

    for direction in range(9):
        cy = int(D2Q9_VELOCITIES[direction, 1])
        opposite = int(D2Q9_OPPOSITE[direction])
        if cy == 1:
            np.testing.assert_array_equal(streamed[1, :, direction], post_collision[1, :, opposite])
        elif cy == -1:
            np.testing.assert_array_equal(
                streamed[-2, :, direction], post_collision[-2, :, opposite]
            )
    np.testing.assert_array_equal(streamed[0], post_collision[0])
    np.testing.assert_array_equal(streamed[-1], post_collision[-1])


def test_regularized_inlet_and_nonequilibrium_outlet_match_contract() -> None:
    config = _config()
    post_collision = _populations(config)
    wall_streamed = pull_stream_halfway_walls_numpy(post_collision)
    streamed = pull_stream_channel_numpy(post_collision, inlet_velocity_lu=0.03)
    _, velocity = macroscopic_moments(streamed)

    np.testing.assert_allclose(velocity[2:-2, 0, 0], 0.03, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(velocity[2:-2, 0, 1], 0.0, rtol=0.0, atol=1e-8)
    for direction in (3, 6, 7):
        np.testing.assert_array_equal(
            streamed[2:-2, -1, direction], wall_streamed[2:-2, -2, direction]
        )

    # Wall priority at the two inlet corners is not an accidental branch order.
    assert streamed[1, 0, 5] == post_collision[1, 0, 7]
    assert streamed[-2, 0, 8] == post_collision[-2, 0, 6]


def test_quadratic_sponge_has_exact_extent_endpoints_and_mask_exclusion() -> None:
    config = _config()
    strengths = sponge_strengths(config)

    assert sponge_columns(config) == 16
    assert np.count_nonzero(strengths) == 15
    assert np.all(strengths[: config.nx - 16] == 0.0)
    assert strengths[config.nx - 16] == 0.0
    assert strengths[-1] == np.float32(MAX_SPONGE_STRENGTH)
    assert np.all(np.diff(strengths) >= 0.0)

    populations = _populations(config)
    mask = _mask(config)
    damped = apply_sponge_numpy(populations, config, mask, inlet_velocity_lu=0.02)
    rho = np.ones((1, 1), dtype=np.float32)
    velocity = np.array([[[0.02, 0.0]]], dtype=np.float32)
    target = equilibrium(rho, velocity)[0, 0]

    np.testing.assert_array_equal(damped[0], populations[0])
    np.testing.assert_array_equal(damped[-1], populations[-1])
    expected = populations[7, -1] + np.float32(MAX_SPONGE_STRENGTH) * (target - populations[7, -1])
    np.testing.assert_array_equal(damped[7, -1], expected)

    mask[8, -1] = True
    with pytest.raises(DomainError, match="obstacle mask enters"):
        validate_sponge_mask(config, mask)


def test_geometry_preflight_places_obstacle_before_sponge() -> None:
    diagnostics = validate_geometry(
        ShapeParams(aspect_ratio=1.0, rotation_deg=0.0, scale=1.25),
        GridSpec(nx=512, ny=256),
    )
    obstacle_right = diagnostics.center_x_lu + diagnostics.semi_major_lu
    assert obstacle_right < diagnostics.sponge_start_x_lu


def test_numpy_channel_stepper_consumes_the_lifecycle_ramp_target() -> None:
    config = _config()
    mask = _mask(config)
    stepper = NumpyChannelStepper()
    state = stepper.initialize(config, mask)
    stepper.advance(
        state,
        config,
        mask,
        completed_step=1,
        inlet_velocity_lu=0.01,
    )
    snapshot = stepper.snapshot(state)
    np.testing.assert_allclose(snapshot.velocity[2:-2, 0, 0], 0.01, rtol=2e-6, atol=2e-7)


def test_boundary_preflight_rejects_impossible_sponge_and_velocity() -> None:
    config = _config()
    with pytest.raises(DomainError, match="inlet velocity"):
        pull_stream_channel_numpy(_populations(config), inlet_velocity_lu=np.nan)

    short = preflight_lattice(
        LatticeConfig(
            nx=16,
            ny=16,
            steps=100,
            warmup_steps=20,
            sample_interval=10,
            inlet_velocity_lu=0.05,
            reynolds=100.0,
            reference_diameter_lu=4.0,
        )
    )
    with pytest.raises(DomainError, match="entire streamwise"):
        sponge_columns(short)


def test_warp_channel_stage_is_bitwise_equal_to_numpy_oracle_on_cpu() -> None:
    _require_warp()
    config = _config()
    mask = _mask(config)
    initial = initialize_numpy(config, mask)
    expected = numpy_channel_step(
        initial.f,
        config,
        mask,
        inlet_velocity_lu=0.01,
    )

    adapter = WarpKernelAdapter("cpu")
    state = adapter.from_numpy(initial.f, config)
    adapter.step_channel(state, config, mask, inlet_velocity_lu=0.01)
    actual = adapter.snapshot(state)

    np.testing.assert_array_equal(actual.f, expected.f)
    np.testing.assert_array_equal(actual.rho, expected.rho)
    np.testing.assert_array_equal(actual.velocity, expected.velocity)


def test_channel_kernel_has_one_non_atomic_writer_per_population() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "soufflerie" / "solver" / "_warp_backend.py"
    ).read_text(encoding="utf-8")
    kernel = source.split("def channel_boundaries_kernel", maxsplit=1)[1].split(
        "def macroscopic_kernel", maxsplit=1
    )[0]
    assert "wp.atomic" not in kernel
    for direction in range(9):
        assert kernel.count(f"streamed[y, x, {direction}] =") == 1
    assert "dim=(ny, nx)" in source
