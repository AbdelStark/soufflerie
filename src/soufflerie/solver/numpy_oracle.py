"""Pure NumPy D2Q9 BGK oracle for small periodic kernel comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from soufflerie.errors import DomainError, NumericalStabilityError
from soufflerie.solver.lattice import (
    D2Q9_VELOCITIES,
    MAX_TAU,
    MIN_TAU,
    DerivedLatticeConfig,
    Q,
    equilibrium,
    macroscopic_moments,
    validate_populations,
)


@dataclass(frozen=True, slots=True)
class NumpyLatticeState:
    """Validated fp32 oracle state in canonical row-major layouts."""

    f: npt.NDArray[np.float32]
    rho: npt.NDArray[np.float32]
    velocity: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        validate_populations(self.f)
        if (
            not isinstance(self.rho, np.ndarray)
            or self.rho.dtype != np.float32
            or self.rho.shape != self.f.shape[:2]
        ):
            raise NumericalStabilityError("LBM-6 PRECISION: state rho shape or dtype is invalid")
        if (
            not isinstance(self.velocity, np.ndarray)
            or self.velocity.dtype != np.float32
            or self.velocity.shape != (*self.f.shape[:2], 2)
        ):
            raise NumericalStabilityError(
                "LBM-6 PRECISION: state velocity shape or dtype is invalid"
            )
        if not self.rho.flags.c_contiguous or not self.velocity.flags.c_contiguous:
            raise NumericalStabilityError("LBM-6 PRECISION: state arrays must be C-contiguous")
        if not np.isfinite(self.rho).all() or not np.all(self.rho > np.float32(0.0)):
            raise NumericalStabilityError("LBM-2 FINITE: state rho must be finite and positive")
        if not np.isfinite(self.velocity).all():
            raise NumericalStabilityError("LBM-2 FINITE: state velocity must be finite")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.f.shape[0], self.f.shape[1])


def _validate_mask(mask: npt.NDArray[np.bool_], shape: tuple[int, int]) -> None:
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.shape != shape
        or not mask.flags.c_contiguous
    ):
        raise DomainError("LBM-6 PRECISION: mask must be C-contiguous bool with shape [ny, nx]")


def initialize_numpy(
    config: DerivedLatticeConfig,
    mask: npt.NDArray[np.bool_] | None = None,
) -> NumpyLatticeState:
    """Initialize rho=1 and equilibrium inlet flow, with zero obstacle velocity."""

    if mask is not None:
        _validate_mask(mask, config.shape)
    rho = np.ones(config.shape, dtype=np.float32)
    velocity = np.zeros((*config.shape, 2), dtype=np.float32)
    velocity[..., 0] = np.float32(config.inlet_velocity_lu)
    if mask is not None:
        velocity[mask] = np.float32(0.0)
    populations = equilibrium(rho, velocity)
    return NumpyLatticeState(f=populations, rho=rho, velocity=velocity)


def collide_numpy(populations: npt.NDArray[np.float32], *, omega: float) -> npt.NDArray[np.float32]:
    """Apply one fixed-order fp32 BGK collision without mutating input."""

    validate_populations(populations)
    if isinstance(omega, bool) or not isinstance(omega, (int, float)):
        raise DomainError("LBM-3 STABILITY: omega must be a finite number")
    relaxation_rate = float(omega)
    minimum_omega = 1.0 / MAX_TAU
    maximum_omega = 1.0 / MIN_TAU
    if (
        not math.isfinite(relaxation_rate)
        or relaxation_rate < math.nextafter(minimum_omega, -math.inf)
        or relaxation_rate > math.nextafter(maximum_omega, math.inf)
    ):
        raise DomainError(
            f"LBM-3 STABILITY: omega must correspond to tau in [{MIN_TAU}, {MAX_TAU}]"
        )
    rho, velocity = macroscopic_moments(populations)
    target = equilibrium(rho, velocity)
    rate = np.float32(relaxation_rate)
    result = populations - rate * (populations - target)
    if not np.isfinite(result).all():
        raise NumericalStabilityError("LBM-2 FINITE: collision produced NaN or infinity")
    return np.ascontiguousarray(result, dtype=np.float32)


def pull_stream_periodic_numpy(
    post_collision: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Pull every population from its periodic upstream cell exactly once."""

    validate_populations(post_collision)
    streamed = np.empty_like(post_collision)
    for direction in range(Q):
        cx = int(D2Q9_VELOCITIES[direction, 0])
        cy = int(D2Q9_VELOCITIES[direction, 1])
        streamed[..., direction] = np.roll(
            post_collision[..., direction], shift=(cy, cx), axis=(0, 1)
        )
    return streamed


def numpy_periodic_step(
    populations: npt.NDArray[np.float32], config: DerivedLatticeConfig
) -> NumpyLatticeState:
    """Run collision then periodic pull streaming and recover next-step moments."""

    validate_populations(populations)
    if populations.shape[:2] != config.shape:
        raise DomainError(
            f"LBM-6 PRECISION: populations grid {populations.shape[:2]} "
            f"does not match config grid {config.shape}"
        )
    post_collision = collide_numpy(populations, omega=config.omega)
    streamed = pull_stream_periodic_numpy(post_collision)
    rho, velocity = macroscopic_moments(streamed)
    return NumpyLatticeState(f=streamed, rho=rho, velocity=velocity)


__all__ = [
    "NumpyLatticeState",
    "collide_numpy",
    "initialize_numpy",
    "numpy_periodic_step",
    "pull_stream_periodic_numpy",
]
