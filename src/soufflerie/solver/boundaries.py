"""Deterministic D2Q9 channel boundaries and outlet sponge oracle."""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from soufflerie.errors import DomainError, NumericalStabilityError
from soufflerie.geometry import MIN_SPONGE_COLUMNS, SPONGE_LENGTH_DIAMETERS
from soufflerie.schemas import GridSpec
from soufflerie.solver.lattice import (
    CS2,
    D2Q9_OPPOSITE,
    D2Q9_VELOCITIES,
    D2Q9_WEIGHTS,
    MAX_INLET_VELOCITY_LU,
    DerivedLatticeConfig,
    Q,
    macroscopic_moments,
    validate_populations,
)
from soufflerie.solver.numpy_oracle import NumpyLatticeState, collide_numpy

Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
UInt8Array = npt.NDArray[np.uint8]

MAX_SPONGE_STRENGTH = 0.15
MIN_CHANNEL_NX = 3
MIN_CHANNEL_NY = 5


class BoundaryOwner(IntEnum):
    """Exclusive owner of one streamed destination population."""

    INTERIOR = 0
    LOWER_WALL = 1
    UPPER_WALL = 2
    INLET = 3
    OUTLET = 4


def _validate_channel_shape(shape: tuple[int, int]) -> None:
    ny, nx = shape
    if ny < MIN_CHANNEL_NY or nx < MIN_CHANNEL_NX:
        raise DomainError(
            f"BC-1 OWNERSHIP: channel grid must be at least {MIN_CHANNEL_NX} x {MIN_CHANNEL_NY}"
        )


def _validate_wall_channel_shape(shape: tuple[int, int]) -> None:
    ny, nx = shape
    if ny < MIN_CHANNEL_NY or nx < 1:
        raise DomainError(
            f"BC-1 OWNERSHIP: wall channel grid must be at least 1 x {MIN_CHANNEL_NY}"
        )


def _validate_inlet_velocity(value: float) -> np.float32:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError("BC-1 INLET: inlet velocity must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > MAX_INLET_VELOCITY_LU:
        raise DomainError(f"BC-1 INLET: inlet velocity must be in [0, {MAX_INLET_VELOCITY_LU}]")
    return np.float32(result)


def _validate_mask(mask: BoolArray, shape: tuple[int, int]) -> None:
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.shape != shape
        or not mask.flags.c_contiguous
    ):
        raise DomainError("BC-1 SPONGE: mask must be C-contiguous bool with channel grid shape")


def channel_boundary_ownership(grid: GridSpec) -> UInt8Array:
    """Return the exclusive RFC-0003 owner of every destination population."""

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec instance")
    _validate_channel_shape(grid.shape)
    owners = np.full((*grid.shape, Q), BoundaryOwner.INTERIOR, dtype=np.uint8)
    owners[0, :, :] = BoundaryOwner.LOWER_WALL
    owners[-1, :, :] = BoundaryOwner.UPPER_WALL

    for direction in range(Q):
        cy = int(D2Q9_VELOCITIES[direction, 1])
        if cy == 1:
            owners[1, :, direction] = BoundaryOwner.LOWER_WALL
        elif cy == -1:
            owners[-2, :, direction] = BoundaryOwner.UPPER_WALL

    # Regularization reconstructs all nine populations at non-corner inlet
    # nodes. At corner-adjacent nodes, wall priority permits only the remaining
    # streamwise incoming links to be inlet-owned.
    owners[2:-2, 0, :] = BoundaryOwner.INLET
    for direction in range(Q):
        cx = int(D2Q9_VELOCITIES[direction, 0])
        if cx == 1:
            inlet = owners[1:-1, 0, direction]
            inlet[inlet == BoundaryOwner.INTERIOR] = BoundaryOwner.INLET
        elif cx == -1:
            outlet = owners[1:-1, -1, direction]
            outlet[outlet == BoundaryOwner.INTERIOR] = BoundaryOwner.OUTLET

    owners.flags.writeable = False
    return owners


def pull_stream_halfway_walls_numpy(post_collision: Float32Array) -> Float32Array:
    """Pull periodically in x and apply half-way bounce-back at channel walls."""

    validate_populations(post_collision)
    _validate_wall_channel_shape((post_collision.shape[0], post_collision.shape[1]))
    streamed = np.empty_like(post_collision)
    for direction in range(Q):
        cx = int(D2Q9_VELOCITIES[direction, 0])
        cy = int(D2Q9_VELOCITIES[direction, 1])
        streamed[..., direction] = np.roll(
            post_collision[..., direction], shift=(cy, cx), axis=(0, 1)
        )
        opposite = int(D2Q9_OPPOSITE[direction])
        if cy == 1:
            streamed[1, :, direction] = post_collision[1, :, opposite]
        elif cy == -1:
            streamed[-2, :, direction] = post_collision[-2, :, opposite]
    # The outer rows are inactive wall nodes. They remain finite and stationary;
    # physical fluid occupies rows 1 through ny-2.
    streamed[0, :, :] = post_collision[0, :, :]
    streamed[-1, :, :] = post_collision[-1, :, :]
    return np.ascontiguousarray(streamed, dtype=np.float32)


def _equilibrium_population(
    direction: int,
    rho: np.float32,
    u: np.float32,
    v: np.float32,
) -> np.float32:
    cx = np.float32(D2Q9_VELOCITIES[direction, 0])
    cy = np.float32(D2Q9_VELOCITIES[direction, 1])
    cu = np.float32(cx * u + cy * v)
    speed_squared = np.float32(u * u + v * v)
    return np.float32(
        D2Q9_WEIGHTS[direction]
        * rho
        * (
            np.float32(1.0)
            + np.float32(3.0) * cu
            + np.float32(4.5) * cu * cu
            - np.float32(1.5) * speed_squared
        )
    )


def _zou_he_provisional(
    populations: Float32Array, u: np.float32
) -> tuple[Float32Array, np.float32]:
    result = populations.copy()
    rho = np.float32(
        (result[0] + result[2] + result[4] + np.float32(2.0) * (result[3] + result[6] + result[7]))
        / (np.float32(1.0) - u)
    )
    result[1] = np.float32(result[3] + np.float32(2.0 / 3.0) * rho * u)
    result[5] = np.float32(
        result[7] + np.float32(0.5) * (result[4] - result[2]) + np.float32(1.0 / 6.0) * rho * u
    )
    result[8] = np.float32(
        result[6] + np.float32(0.5) * (result[2] - result[4]) + np.float32(1.0 / 6.0) * rho * u
    )
    return result, rho


def _regularize(populations: Float32Array, rho: np.float32, u: np.float32) -> Float32Array:
    equilibrium = np.empty(Q, dtype=np.float32)
    for direction in range(Q):
        equilibrium[direction] = _equilibrium_population(direction, rho, u, np.float32(0.0))
    non_equilibrium = populations - equilibrium
    pi_xx = np.float32(0.0)
    pi_xy = np.float32(0.0)
    pi_yy = np.float32(0.0)
    for direction in range(Q):
        cx = np.float32(D2Q9_VELOCITIES[direction, 0])
        cy = np.float32(D2Q9_VELOCITIES[direction, 1])
        pi_xx = np.float32(pi_xx + cx * cx * non_equilibrium[direction])
        pi_xy = np.float32(pi_xy + cx * cy * non_equilibrium[direction])
        pi_yy = np.float32(pi_yy + cy * cy * non_equilibrium[direction])

    result = np.empty(Q, dtype=np.float32)
    cs2 = np.float32(CS2)
    for direction in range(Q):
        cx = np.float32(D2Q9_VELOCITIES[direction, 0])
        cy = np.float32(D2Q9_VELOCITIES[direction, 1])
        contraction = np.float32(
            (cx * cx - cs2) * pi_xx + np.float32(2.0) * cx * cy * pi_xy + (cy * cy - cs2) * pi_yy
        )
        result[direction] = np.float32(
            equilibrium[direction] + np.float32(4.5) * D2Q9_WEIGHTS[direction] * contraction
        )
    return result


def apply_inlet_outlet_numpy(streamed: Float32Array, *, inlet_velocity_lu: float) -> Float32Array:
    """Apply regularized inlet and zero-gradient non-equilibrium outlet reconstruction."""

    validate_populations(streamed)
    ny, _ = streamed.shape[:2]
    _validate_channel_shape((streamed.shape[0], streamed.shape[1]))
    inlet_velocity = _validate_inlet_velocity(inlet_velocity_lu)
    result = streamed.copy()

    for y in range(1, ny - 1):
        provisional, rho = _zou_he_provisional(result[y, 0], inlet_velocity)
        if 1 < y < ny - 2:
            result[y, 0] = _regularize(provisional, rho, inlet_velocity)
        else:
            # Wall links own diagonal corner populations. The inlet owns only
            # the remaining west-facing incoming links at these two cells.
            result[y, 0, 1] = provisional[1]
            if y != 1:
                result[y, 0, 5] = provisional[5]
            if y != ny - 2:
                result[y, 0, 8] = provisional[8]

        neighbor = result[y, -2]
        neighbor_rho = np.float32(neighbor[0])
        for direction in range(1, Q):
            neighbor_rho = np.float32(neighbor_rho + neighbor[direction])
        neighbor_u = np.float32(
            (neighbor[1] - neighbor[3] + neighbor[5] - neighbor[6] - neighbor[7] + neighbor[8])
            / neighbor_rho
        )
        neighbor_v = np.float32(
            (neighbor[2] - neighbor[4] + neighbor[5] + neighbor[6] - neighbor[7] - neighbor[8])
            / neighbor_rho
        )
        for direction in (3, 6, 7):
            if (y == 1 and direction == 6) or (y == ny - 2 and direction == 7):
                continue
            equilibrium = _equilibrium_population(direction, neighbor_rho, neighbor_u, neighbor_v)
            non_equilibrium = np.float32(neighbor[direction] - equilibrium)
            result[y, -1, direction] = np.float32(equilibrium + non_equilibrium)

    if not np.isfinite(result).all():
        raise NumericalStabilityError("BC-1 BOUNDARY: reconstruction produced NaN or infinity")
    return np.ascontiguousarray(result, dtype=np.float32)


def pull_stream_channel_numpy(
    post_collision: Float32Array,
    *,
    inlet_velocity_lu: float,
) -> Float32Array:
    """Pull with exclusive wall/inlet/outlet ownership and fixed corner priority."""

    streamed = pull_stream_halfway_walls_numpy(post_collision)
    return apply_inlet_outlet_numpy(streamed, inlet_velocity_lu=inlet_velocity_lu)


def sponge_columns(config: DerivedLatticeConfig) -> int:
    """Return the RFC-0003 sponge width or reject an impossible channel."""

    if not isinstance(config, DerivedLatticeConfig):
        raise TypeError("config must be a DerivedLatticeConfig instance")
    columns = max(
        MIN_SPONGE_COLUMNS,
        math.floor(SPONGE_LENGTH_DIAMETERS * config.reference_diameter_lu + 0.5),
    )
    if columns >= config.nx:
        raise DomainError("BC-1 SPONGE: sponge occupies the entire streamwise grid")
    return columns


def sponge_strengths(config: DerivedLatticeConfig) -> Float32Array:
    """Return the fixed quadratic streamwise sponge profile."""

    columns = sponge_columns(config)
    result = np.zeros(config.nx, dtype=np.float32)
    start = config.nx - columns
    denominator = np.float32(columns - 1)
    for x in range(start, config.nx):
        fraction = np.float32(x - start) / denominator
        result[x] = np.float32(MAX_SPONGE_STRENGTH) * fraction * fraction
    result.flags.writeable = False
    return result


def validate_sponge_mask(config: DerivedLatticeConfig, mask: BoolArray) -> None:
    """Reject a mask that violates geometry preflight sponge exclusion."""

    columns = sponge_columns(config)
    _validate_mask(mask, config.shape)
    if np.any(mask[1:-1, config.nx - columns :]):
        raise DomainError("BC-1 SPONGE: obstacle mask enters the outlet sponge")


def apply_sponge_numpy(
    post_collision: Float32Array,
    config: DerivedLatticeConfig,
    mask: BoolArray,
    *,
    inlet_velocity_lu: float,
) -> Float32Array:
    """Relax post-collision fluid populations toward ramped inlet equilibrium."""

    validate_populations(post_collision)
    if post_collision.shape[:2] != config.shape:
        raise DomainError("BC-1 SPONGE: populations and config grids must match")
    _validate_channel_shape(config.shape)
    validate_sponge_mask(config, mask)
    inlet_velocity = _validate_inlet_velocity(inlet_velocity_lu)
    strengths = sponge_strengths(config)
    result = post_collision.copy()
    for x in np.flatnonzero(strengths > np.float32(0.0)):
        strength = strengths[x]
        for direction in range(Q):
            target = _equilibrium_population(
                direction, np.float32(1.0), inlet_velocity, np.float32(0.0)
            )
            column = result[1:-1, x, direction]
            column[:] = column + strength * (target - column)
    if not np.isfinite(result).all():
        raise NumericalStabilityError("BC-1 SPONGE: relaxation produced NaN or infinity")
    return np.ascontiguousarray(result, dtype=np.float32)


def numpy_channel_step(
    populations: Float32Array,
    config: DerivedLatticeConfig,
    mask: BoolArray,
    *,
    inlet_velocity_lu: float,
) -> NumpyLatticeState:
    """Run collision, sponge, channel streaming, and moment reduction in RFC order."""

    post_collision = collide_numpy(populations, omega=config.omega)
    damped = apply_sponge_numpy(
        post_collision,
        config,
        mask,
        inlet_velocity_lu=inlet_velocity_lu,
    )
    streamed = pull_stream_channel_numpy(damped, inlet_velocity_lu=inlet_velocity_lu)
    rho, velocity = macroscopic_moments(streamed)
    return NumpyLatticeState(f=streamed, rho=rho, velocity=velocity)


__all__ = [
    "MAX_SPONGE_STRENGTH",
    "MIN_CHANNEL_NX",
    "MIN_CHANNEL_NY",
    "BoundaryOwner",
    "apply_inlet_outlet_numpy",
    "apply_sponge_numpy",
    "channel_boundary_ownership",
    "numpy_channel_step",
    "pull_stream_channel_numpy",
    "pull_stream_halfway_walls_numpy",
    "sponge_columns",
    "sponge_strengths",
    "validate_sponge_mask",
]
