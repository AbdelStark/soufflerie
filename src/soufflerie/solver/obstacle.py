"""Half-way obstacle bounce-back composed with channel boundary stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from soufflerie.errors import DomainError, InternalInvariantError
from soufflerie.solver.boundaries import (
    apply_sponge_numpy,
    pull_stream_channel_numpy,
    validate_sponge_mask,
)
from soufflerie.solver.forces import ObstacleForce, ObstacleLinks, momentum_exchange_force
from soufflerie.solver.lattice import (
    D2Q9_OPPOSITE,
    D2Q9_VELOCITIES,
    DerivedLatticeConfig,
    macroscopic_moments,
    validate_populations,
)
from soufflerie.solver.numpy_oracle import NumpyLatticeState, collide_numpy

BoolArray = npt.NDArray[np.bool_]
Float32Array = npt.NDArray[np.float32]


def _validate_links_against_mask(mask: BoolArray, links: ObstacleLinks) -> None:
    if mask.shape != links.grid_shape:
        raise DomainError("BC-2 OBSTACLE: mask and obstacle-link grids must match")
    ny, nx = links.grid_shape
    for index in range(links.count):
        y = int(links.fluid_y[index])
        x = int(links.fluid_x[index])
        direction = int(links.direction[index])
        neighbor_y = y + int(D2Q9_VELOCITIES[direction, 1])
        neighbor_x = x + int(D2Q9_VELOCITIES[direction, 0])
        if (
            mask[y, x]
            or not 0 <= neighbor_y < ny
            or not 0 <= neighbor_x < nx
            or not mask[neighbor_y, neighbor_x]
        ):
            raise DomainError("BC-2 OBSTACLE: links do not match the declared obstacle mask")


def apply_obstacle_bounce_back_numpy(
    streamed: Float32Array,
    post_collision: Float32Array,
    mask: BoolArray,
    links: ObstacleLinks,
) -> Float32Array:
    """Apply each fixed fluid-solid link without competing destination writes."""

    validate_populations(streamed)
    validate_populations(post_collision)
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or not mask.flags.c_contiguous:
        raise DomainError("BC-2 OBSTACLE: mask must be C-contiguous bool")
    if streamed.shape != post_collision.shape or streamed.shape[:2] != mask.shape:
        raise DomainError("BC-2 OBSTACLE: streamed, post-collision, and mask grids must match")
    if not isinstance(links, ObstacleLinks):
        raise TypeError("links must be an ObstacleLinks instance")
    _validate_links_against_mask(mask, links)

    result = streamed.copy()
    result[mask, :] = post_collision[mask, :]
    for index in range(links.count):
        y = int(links.fluid_y[index])
        x = int(links.fluid_x[index])
        direction = int(links.direction[index])
        destination_direction = int(D2Q9_OPPOSITE[direction])
        result[y, x, destination_direction] = post_collision[y, x, direction]
    if not np.isfinite(result).all():
        raise InternalInvariantError("BC-2 OBSTACLE: bounce-back produced non-finite populations")
    return np.ascontiguousarray(result, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class NumpyObstacleStep:
    """One completed NumPy obstacle step and its pre-stream force reduction."""

    state: NumpyLatticeState
    force: ObstacleForce


def numpy_obstacle_step(
    populations: Float32Array,
    config: DerivedLatticeConfig,
    mask: BoolArray,
    links: ObstacleLinks,
    *,
    inlet_velocity_lu: float,
) -> NumpyObstacleStep:
    """Run collision, sponge, force, and obstacle-priority channel streaming."""

    validate_sponge_mask(config, mask)
    post_collision = collide_numpy(populations, omega=config.omega)
    damped = apply_sponge_numpy(
        post_collision,
        config,
        mask,
        inlet_velocity_lu=inlet_velocity_lu,
    )
    force = momentum_exchange_force(damped, links, config)
    channel_streamed = pull_stream_channel_numpy(
        damped,
        inlet_velocity_lu=inlet_velocity_lu,
    )
    streamed = apply_obstacle_bounce_back_numpy(
        channel_streamed,
        damped,
        mask,
        links,
    )
    rho, velocity = macroscopic_moments(streamed)
    return NumpyObstacleStep(
        state=NumpyLatticeState(f=streamed, rho=rho, velocity=velocity),
        force=force,
    )


__all__ = [
    "NumpyObstacleStep",
    "apply_obstacle_bounce_back_numpy",
    "numpy_obstacle_step",
]
