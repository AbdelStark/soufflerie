from __future__ import annotations

import importlib.util

import numpy as np
import numpy.typing as npt
import pytest

from soufflerie.errors import DomainError
from soufflerie.solver.boundaries import pull_stream_channel_numpy
from soufflerie.solver.forces import enumerate_obstacle_links
from soufflerie.solver.kernels import WarpKernelAdapter
from soufflerie.solver.lattice import (
    D2Q9_OPPOSITE,
    D2Q9_WEIGHTS,
    DerivedLatticeConfig,
    LatticeConfig,
    preflight_lattice,
)
from soufflerie.solver.numpy_oracle import initialize_numpy
from soufflerie.solver.obstacle import (
    apply_obstacle_bounce_back_numpy,
    numpy_obstacle_step,
)


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


def _populations(config: DerivedLatticeConfig) -> npt.NDArray[np.float32]:
    result = np.broadcast_to(D2Q9_WEIGHTS, (*config.shape, 9)).copy()
    result += np.linspace(-0.001, 0.001, result.size, dtype=np.float32).reshape(result.shape)
    return result


def _require_warp() -> None:
    if importlib.util.find_spec("warp") is None:
        pytest.skip("install the Soufflerie solver extra to run Warp obstacle tests")


def test_single_cell_link_enumeration_is_complete_unique_and_fixed_order() -> None:
    mask = np.zeros((7, 7), dtype=np.bool_)
    mask[3, 3] = True
    links = enumerate_obstacle_links(mask)

    assert links.count == 8
    assert not links.fluid_y.flags.writeable
    assert not links.fluid_x.flags.writeable
    assert not links.direction.flags.writeable
    assert list(zip(links.fluid_y, links.fluid_x, links.direction, strict=True)) == [
        (2, 2, 5),
        (2, 3, 2),
        (2, 4, 6),
        (3, 2, 1),
        (3, 4, 3),
        (4, 2, 8),
        (4, 3, 4),
        (4, 4, 7),
    ]
    destination_keys = {
        (int(y), int(x), int(D2Q9_OPPOSITE[int(direction)]))
        for y, x, direction in zip(
            links.fluid_y,
            links.fluid_x,
            links.direction,
            strict=True,
        )
    }
    assert len(destination_keys) == links.count


def test_halfway_obstacle_bounce_uses_outgoing_population_at_fluid_node() -> None:
    config = _config()
    mask = _mask(config)
    links = enumerate_obstacle_links(mask)
    post_collision = _populations(config)
    channel = pull_stream_channel_numpy(post_collision, inlet_velocity_lu=0.02)
    bounced = apply_obstacle_bounce_back_numpy(channel, post_collision, mask, links)

    for y, x, direction in zip(
        links.fluid_y,
        links.fluid_x,
        links.direction,
        strict=True,
    ):
        opposite = int(D2Q9_OPPOSITE[int(direction)])
        assert bounced[int(y), int(x), opposite] == post_collision[int(y), int(x), int(direction)]
    np.testing.assert_array_equal(bounced[mask], post_collision[mask])
    assert bounced[2, 2, 0] == channel[2, 2, 0]


def test_links_that_do_not_match_mask_fail_closed() -> None:
    config = _config()
    mask = _mask(config)
    links = enumerate_obstacle_links(mask)
    changed = mask.copy()
    changed[8, 10] = False
    populations = _populations(config)
    channel = pull_stream_channel_numpy(populations, inlet_velocity_lu=0.02)

    with pytest.raises(DomainError, match="links do not match"):
        apply_obstacle_bounce_back_numpy(channel, populations, changed, links)


def test_warp_obstacle_step_is_bitwise_equal_to_numpy_oracle_on_cpu() -> None:
    _require_warp()
    config = _config()
    mask = _mask(config)
    links = enumerate_obstacle_links(mask)
    initialization_mask = mask.copy()
    initialization_mask[0, :] = True
    initialization_mask[-1, :] = True
    initial = initialize_numpy(config, initialization_mask)
    expected = numpy_obstacle_step(
        initial.f,
        config,
        mask,
        links,
        inlet_velocity_lu=0.01,
    )

    adapter = WarpKernelAdapter("cpu")
    state = adapter.from_numpy(initial.f, config)
    force = adapter.step_obstacle(
        state,
        config,
        mask,
        links,
        inlet_velocity_lu=0.01,
    )
    actual = adapter.snapshot(state)

    np.testing.assert_array_equal(actual.f, expected.state.f)
    np.testing.assert_array_equal(actual.rho, expected.state.rho)
    np.testing.assert_array_equal(actual.velocity, expected.state.velocity)
    assert force == expected.force
