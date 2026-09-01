from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from soufflerie.errors import DomainError
from soufflerie.solver.forces import (
    ForceHistoryRecorder,
    ObstacleForce,
    enumerate_obstacle_links,
    momentum_exchange_force,
)
from soufflerie.solver.lattice import (
    D2Q9_WEIGHTS,
    DerivedLatticeConfig,
    LatticeConfig,
    preflight_lattice,
)
from soufflerie.solver.lifecycle import NumpyObstacleStepper


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
    result[8, 10] = True
    return result


def _equilibrium_populations(config: DerivedLatticeConfig) -> npt.NDArray[np.float32]:
    return np.broadcast_to(D2Q9_WEIGHTS, (*config.shape, 9)).copy()


def test_no_obstacle_has_zero_force_and_channel_walls_are_excluded() -> None:
    config = _config()
    mask = np.zeros(config.shape, dtype=np.bool_)
    links = enumerate_obstacle_links(mask)
    force = momentum_exchange_force(_equilibrium_populations(config), links, config)

    assert links.count == 0
    assert force.fx_lu == 0.0
    assert force.fy_lu == 0.0
    assert force.cd == 0.0
    assert force.cl == 0.0


def test_symmetric_obstacle_force_cancels_exactly() -> None:
    config = _config()
    mask = _mask(config)
    links = enumerate_obstacle_links(mask)
    force = momentum_exchange_force(_equilibrium_populations(config), links, config)

    assert links.count == 8
    assert force.fx_lu == 0.0
    assert force.fy_lu == 0.0
    assert force.cd == 0.0
    assert force.cl == 0.0


def test_positive_drag_sign_and_declared_normalization() -> None:
    config = _config()
    mask = _mask(config)
    links = enumerate_obstacle_links(mask)
    populations = _equilibrium_populations(config)
    populations[8, 9, 1] += np.float32(0.2)
    force = momentum_exchange_force(populations, links, config)

    expected_normalization = 0.5 * 1.0 * 0.05**2 * 4.0
    assert force.fx_lu == pytest.approx(0.4)
    assert force.fy_lu == 0.0
    assert force.normalization_lu == pytest.approx(expected_normalization)
    assert force.cd == pytest.approx(0.4 / expected_normalization)
    assert force.cl == 0.0


def test_mirrored_lift_has_equal_magnitude_and_opposite_sign() -> None:
    config = _config()
    mask = _mask(config)
    links = enumerate_obstacle_links(mask)
    positive = _equilibrium_populations(config)
    negative = _equilibrium_populations(config)
    positive[7, 10, 2] += np.float32(0.125)
    negative[9, 10, 4] += np.float32(0.125)

    upward = momentum_exchange_force(positive, links, config)
    downward = momentum_exchange_force(negative, links, config)
    assert upward.fx_lu == downward.fx_lu == 0.0
    assert upward.fy_lu == pytest.approx(-downward.fy_lu)
    assert upward.cl == pytest.approx(-downward.cl)
    assert upward.cl > 0.0


def test_force_reduction_and_history_are_repeatable_and_persistence_typed() -> None:
    config = _config()
    mask = _mask(config)
    links = enumerate_obstacle_links(mask)
    populations = _equilibrium_populations(config)
    first = momentum_exchange_force(populations, links, config)
    second = momentum_exchange_force(populations, links, config)
    assert first == second

    recorder = ForceHistoryRecorder()
    recorder.record(30, first)
    recorder.record(40, ObstacleForce(8, 1.0, -0.5, 0.005, 200.0, -100.0))
    history = recorder.snapshot()
    np.testing.assert_array_equal(history.steps, np.array([30, 40], dtype=np.int64))
    assert history.cd.dtype == np.float32
    assert history.cl.dtype == np.float32
    assert history.fx_lu.dtype == np.float64
    assert not history.steps.flags.writeable
    assert not history.cd.flags.writeable
    with pytest.raises(DomainError, match="strictly increasing"):
        recorder.record(40, first)


def test_obstacle_stepper_samples_force_after_warmup_in_declared_cadence() -> None:
    config = _config()
    mask = _mask(config)
    stepper = NumpyObstacleStepper()
    state = stepper.initialize(config, mask)
    for completed_step in (20, 21, 30, 40):
        stepper.advance(
            state,
            config,
            mask,
            completed_step=completed_step,
            inlet_velocity_lu=0.01,
        )
    history = stepper.force_history()
    np.testing.assert_array_equal(history.steps, np.array([30, 40], dtype=np.int64))
    assert history.count == 2
