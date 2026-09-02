"""D2Q9 lattice constants, configuration derivation, and fp32 moments."""

from __future__ import annotations

import math
from typing import Final, TypeVar

import numpy as np
import numpy.typing as npt

from soufflerie.errors import NumericalStabilityError
from soufflerie.numerics import (
    CS2,
    DEFAULT_SAMPLE_INTERVAL,
    MAX_INLET_VELOCITY_LU,
    MAX_NOMINAL_MACH,
    MAX_TAU,
    MIN_TAU,
    RHO_REF,
    SPEED_OF_SOUND,
    DerivedLatticeConfig,
    LatticeConfig,
    _require_integer,
    derive_lattice,
    kinematic_viscosity_lu,
    preflight_lattice,
    relaxation_time,
    reynolds_from_relaxation_time,
)

Q: Final = 9


ScalarT = TypeVar("ScalarT", bound=np.generic)


def _readonly(array: npt.NDArray[ScalarT]) -> npt.NDArray[ScalarT]:
    array.flags.writeable = False
    return array


D2Q9_VELOCITIES = _readonly(
    np.array(
        [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 1),
            (-1, 1),
            (-1, -1),
            (1, -1),
        ],
        dtype=np.int8,
    )
)
D2Q9_WEIGHTS = _readonly(
    np.array(
        [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0] + [1.0 / 36.0] * 4,
        dtype=np.float32,
    )
)
D2Q9_OPPOSITE = _readonly(np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int8))


def inlet_ramp(step: int, ramp_steps: int) -> float:
    """Return the deterministic half-cosine inlet multiplier for one step."""

    _require_integer("step", step, minimum=0)
    _require_integer("ramp_steps", ramp_steps, minimum=0)
    if ramp_steps == 0 or step >= ramp_steps:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * step / ramp_steps))


def _validate_density_velocity(
    rho: npt.NDArray[np.float32], velocity: npt.NDArray[np.float32]
) -> None:
    if not isinstance(rho, np.ndarray) or rho.dtype != np.float32 or rho.ndim != 2:
        raise NumericalStabilityError("LBM-6 PRECISION: rho must be a two-dimensional fp32 array")
    if not rho.flags.c_contiguous or rho.size == 0:
        raise NumericalStabilityError("LBM-6 PRECISION: rho must be non-empty and C-contiguous")
    if not np.isfinite(rho).all() or not np.all(rho > np.float32(0.0)):
        raise NumericalStabilityError("LBM-2 FINITE: rho must be finite and strictly positive")
    if (
        not isinstance(velocity, np.ndarray)
        or velocity.dtype != np.float32
        or velocity.shape != (*rho.shape, 2)
    ):
        raise NumericalStabilityError(
            "LBM-6 PRECISION: velocity must be fp32 with shape [ny, nx, 2]"
        )
    if not velocity.flags.c_contiguous or not np.isfinite(velocity).all():
        raise NumericalStabilityError("LBM-2 FINITE: velocity must be finite and C-contiguous")


def validate_populations(populations: npt.NDArray[np.float32]) -> None:
    """Validate the canonical in-memory population layout and precision."""

    if (
        not isinstance(populations, np.ndarray)
        or populations.dtype != np.float32
        or populations.ndim != 3
        or populations.shape[-1] != Q
    ):
        raise NumericalStabilityError(
            "LBM-6 PRECISION: populations must be fp32 with shape [ny, nx, 9]"
        )
    if populations.shape[0] == 0 or populations.shape[1] == 0:
        raise NumericalStabilityError("LBM-6 PRECISION: population dimensions must be positive")
    if not populations.flags.c_contiguous:
        raise NumericalStabilityError("LBM-6 PRECISION: populations must be C-contiguous")
    if not np.isfinite(populations).all():
        raise NumericalStabilityError("LBM-2 FINITE: populations contain NaN or infinity")


def equilibrium(
    rho: npt.NDArray[np.float32], velocity: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Compute the D2Q9 equilibrium in fixed direction order using fp32 state."""

    _validate_density_velocity(rho, velocity)
    u = velocity[..., 0]
    v = velocity[..., 1]
    speed_squared = u * u + v * v
    result = np.empty((*rho.shape, Q), dtype=np.float32)
    three = np.float32(3.0)
    four_point_five = np.float32(4.5)
    one_point_five = np.float32(1.5)
    for direction in range(Q):
        cx = np.float32(D2Q9_VELOCITIES[direction, 0])
        cy = np.float32(D2Q9_VELOCITIES[direction, 1])
        cu = cx * u + cy * v
        result[..., direction] = (
            D2Q9_WEIGHTS[direction]
            * rho
            * (
                np.float32(1.0)
                + three * cu
                + four_point_five * cu * cu
                - one_point_five * speed_squared
            )
        )
    return result


def macroscopic_moments(
    populations: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Recover density and velocity with fixed-order D2Q9 momentum sums."""

    validate_populations(populations)
    rho = populations[..., 0].copy()
    for direction in range(1, Q):
        rho += populations[..., direction]
    if not np.isfinite(rho).all() or not np.all(rho > np.float32(0.0)):
        raise NumericalStabilityError("LBM-2 FINITE: recovered density must be finite and positive")

    momentum_x = (
        populations[..., 1]
        - populations[..., 3]
        + populations[..., 5]
        - populations[..., 6]
        - populations[..., 7]
        + populations[..., 8]
    )
    momentum_y = (
        populations[..., 2]
        - populations[..., 4]
        + populations[..., 5]
        + populations[..., 6]
        - populations[..., 7]
        - populations[..., 8]
    )
    velocity = np.empty((*rho.shape, 2), dtype=np.float32)
    velocity[..., 0] = momentum_x / rho
    velocity[..., 1] = momentum_y / rho
    if not np.isfinite(velocity).all():
        raise NumericalStabilityError("LBM-2 FINITE: recovered velocity must be finite")
    return np.ascontiguousarray(rho), velocity


__all__ = [
    "CS2",
    "D2Q9_OPPOSITE",
    "D2Q9_VELOCITIES",
    "D2Q9_WEIGHTS",
    "DEFAULT_SAMPLE_INTERVAL",
    "MAX_INLET_VELOCITY_LU",
    "MAX_NOMINAL_MACH",
    "MAX_TAU",
    "MIN_TAU",
    "RHO_REF",
    "SPEED_OF_SOUND",
    "DerivedLatticeConfig",
    "LatticeConfig",
    "Q",
    "derive_lattice",
    "equilibrium",
    "inlet_ramp",
    "kinematic_viscosity_lu",
    "macroscopic_moments",
    "preflight_lattice",
    "relaxation_time",
    "reynolds_from_relaxation_time",
    "validate_populations",
]
