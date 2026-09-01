from __future__ import annotations

import numpy as np
import pytest

from soufflerie.errors import DomainError
from soufflerie.solver.lattice import (
    D2Q9_VELOCITIES,
    DerivedLatticeConfig,
    LatticeConfig,
    preflight_lattice,
)
from soufflerie.solver.numpy_oracle import (
    NumpyLatticeState,
    collide_numpy,
    initialize_numpy,
    numpy_periodic_step,
    pull_stream_periodic_numpy,
)


def _config(*, nx: int = 5, ny: int = 4) -> DerivedLatticeConfig:
    return preflight_lattice(
        LatticeConfig(
            nx=nx,
            ny=ny,
            steps=100,
            warmup_steps=20,
            sample_interval=10,
            inlet_velocity_lu=0.05,
            reynolds=100.0,
            reference_diameter_lu=32.0,
        )
    )


def _rng_state_equal(left: object, right: object) -> bool:
    if not isinstance(left, tuple) or not isinstance(right, tuple):
        return repr(left) == repr(right)
    return (
        left[0] == right[0]
        and np.array_equal(np.asarray(left[1]), np.asarray(right[1]))
        and left[2:] == right[2:]
    )


def test_initialization_sets_equilibrium_fluid_and_zero_obstacle_velocity() -> None:
    config = _config()
    mask = np.zeros(config.shape, dtype=np.bool_)
    mask[1:3, 2] = True
    state = initialize_numpy(config, mask)

    assert isinstance(state, NumpyLatticeState)
    assert state.shape == config.shape
    assert state.f.shape == (*config.shape, 9)
    assert state.f.dtype == np.float32
    np.testing.assert_array_equal(state.rho, np.ones(config.shape, dtype=np.float32))
    np.testing.assert_array_equal(state.velocity[mask], 0.0)
    np.testing.assert_array_equal(
        state.velocity[~mask, 0], np.full(np.count_nonzero(~mask), 0.05, dtype=np.float32)
    )
    np.testing.assert_array_equal(state.velocity[~mask, 1], 0.0)


def test_collision_preserves_an_equilibrium_state_with_fp32_tolerance() -> None:
    config = _config()
    state = initialize_numpy(config)
    collided = collide_numpy(state.f, omega=config.omega)

    assert collided.dtype == np.float32
    assert collided.flags.c_contiguous
    np.testing.assert_allclose(collided, state.f, rtol=2e-6, atol=2e-8)


def test_periodic_pull_stream_mapping_is_exhaustive() -> None:
    ny, nx = 3, 4
    post_collision = np.arange(ny * nx * 9, dtype=np.float32).reshape(ny, nx, 9)
    streamed = pull_stream_periodic_numpy(post_collision)

    for y in range(ny):
        for x in range(nx):
            for direction, (cx, cy) in enumerate(D2Q9_VELOCITIES):
                expected = post_collision[(y - int(cy)) % ny, (x - int(cx)) % nx, direction]
                assert streamed[y, x, direction] == expected
    np.testing.assert_array_equal(
        np.sort(streamed.reshape(-1)), np.sort(post_collision.reshape(-1))
    )


def test_one_step_is_deterministic_pure_and_mass_conserving() -> None:
    config = _config()
    initial = initialize_numpy(config)
    populations = initial.f.copy()
    populations[1, 2, 1] += np.float32(0.002)
    populations[2, 3, 6] -= np.float32(0.001)
    before = populations.copy()
    rng_before = np.random.get_state()

    first = numpy_periodic_step(populations, config)
    second = numpy_periodic_step(populations, config)
    rng_after = np.random.get_state()

    np.testing.assert_array_equal(populations, before)
    np.testing.assert_array_equal(first.f, second.f)
    np.testing.assert_array_equal(first.rho, second.rho)
    np.testing.assert_array_equal(first.velocity, second.velocity)
    assert _rng_state_equal(rng_before, rng_after)
    mass_before = float(np.sum(before, dtype=np.float64))
    mass_after = float(np.sum(first.rho, dtype=np.float64))
    assert abs(mass_after - mass_before) / mass_before < 1e-6


def test_oracle_step_equals_explicit_collision_then_streaming() -> None:
    config = _config(nx=4, ny=3)
    populations = initialize_numpy(config).f.copy()
    populations[0, 0, 5] += np.float32(0.005)

    expected = pull_stream_periodic_numpy(collide_numpy(populations, omega=config.omega))
    actual = numpy_periodic_step(populations, config)

    np.testing.assert_array_equal(actual.f, expected)


def test_oracle_rejects_bad_mask_grid_precision_and_relaxation() -> None:
    config = _config()
    with pytest.raises(DomainError, match="mask"):
        initialize_numpy(config, np.zeros((2, 2), dtype=np.bool_))
    with pytest.raises(DomainError, match="does not match"):
        numpy_periodic_step(initialize_numpy(_config(nx=6)).f, config)
    with pytest.raises(DomainError, match="omega"):
        collide_numpy(initialize_numpy(config).f, omega=2.0)
