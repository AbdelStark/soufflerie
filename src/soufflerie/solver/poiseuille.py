"""Deterministic periodic-x D2Q9 Poiseuille acceptance fixture."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from soufflerie.errors import DomainError, InternalInvariantError
from soufflerie.solver.boundaries import pull_stream_halfway_walls_numpy
from soufflerie.solver.lattice import (
    D2Q9_VELOCITIES,
    D2Q9_WEIGHTS,
    MAX_TAU,
    MIN_TAU,
    Q,
)

POISEUILLE_ERROR_THRESHOLD = 0.01


@dataclass(frozen=True, slots=True)
class PoiseuilleFixture:
    """A body-forced periodic channel with half-way no-slip walls."""

    channel_height: int
    width: int
    steps: int
    tau: float = 0.8
    maximum_velocity_lu: float = 0.05
    excluded_wall_cells: int = 1

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("channel_height", self.channel_height, 6),
            ("width", self.width, 1),
            ("steps", self.steps, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise DomainError(f"Poiseuille {name} must be an integer >= {minimum}")
        if not math.isfinite(self.tau) or not MIN_TAU <= self.tau <= MAX_TAU:
            raise DomainError(f"Poiseuille tau must be in [{MIN_TAU}, {MAX_TAU}]")
        if (
            not math.isfinite(self.maximum_velocity_lu)
            or self.maximum_velocity_lu <= 0.0
            or self.maximum_velocity_lu > 0.1
        ):
            raise DomainError("Poiseuille maximum_velocity_lu must be in (0, 0.1]")
        if self.excluded_wall_cells != 1:
            raise DomainError("Poiseuille acceptance excludes exactly one fluid cell per wall")

    @property
    def ny(self) -> int:
        return self.channel_height + 2

    @property
    def kinematic_viscosity_lu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def body_force_lu(self) -> float:
        return 8.0 * self.kinematic_viscosity_lu * self.maximum_velocity_lu / self.channel_height**2


@dataclass(frozen=True, slots=True)
class PoiseuilleResult:
    fixture: PoiseuilleFixture
    measured_profile: npt.NDArray[np.float64]
    analytic_profile: npt.NDArray[np.float64]
    relative_l2_error: float
    max_relative_error: float
    initial_mass: float
    final_mass: float
    mass_drift_ratio: float

    def __post_init__(self) -> None:
        expected_shape = (self.fixture.channel_height,)
        for name, profile in (
            ("measured_profile", self.measured_profile),
            ("analytic_profile", self.analytic_profile),
        ):
            if (
                not isinstance(profile, np.ndarray)
                or profile.dtype != np.float64
                or profile.shape != expected_shape
                or not profile.flags.c_contiguous
                or not np.isfinite(profile).all()
            ):
                raise InternalInvariantError(
                    f"{name} must be finite C-contiguous float64 with shape {expected_shape}"
                )
        scalars = (
            self.relative_l2_error,
            self.max_relative_error,
            self.initial_mass,
            self.final_mass,
            self.mass_drift_ratio,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise InternalInvariantError("Poiseuille result scalars must be finite")
        if self.initial_mass <= 0.0 or self.final_mass <= 0.0:
            raise InternalInvariantError("Poiseuille masses must be positive")

    @property
    def passed(self) -> bool:
        return poiseuille_gate_passes(self.relative_l2_error, self.max_relative_error)


def analytic_poiseuille_profile(fixture: PoiseuilleFixture) -> npt.NDArray[np.float64]:
    """Evaluate the half-way-wall analytic parabola at fluid cell centers."""

    distance_from_lower_wall = np.arange(fixture.channel_height, dtype=np.float64) + 0.5
    height = float(fixture.channel_height)
    result = (
        fixture.body_force_lu
        / (2.0 * fixture.kinematic_viscosity_lu)
        * distance_from_lower_wall
        * (height - distance_from_lower_wall)
    )
    return np.ascontiguousarray(result, dtype=np.float64)


def poiseuille_gate_passes(relative_l2_error: float, max_relative_error: float) -> bool:
    """Apply both inclusive analytic error thresholds without rounding."""

    return (
        math.isfinite(relative_l2_error)
        and math.isfinite(max_relative_error)
        and relative_l2_error <= POISEUILLE_ERROR_THRESHOLD
        and max_relative_error <= POISEUILLE_ERROR_THRESHOLD
    )


def _density(populations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    rho = populations[..., 0].copy()
    for direction in range(1, Q):
        rho += populations[..., direction]
    return rho


def run_poiseuille_fixture(fixture: PoiseuilleFixture) -> PoiseuilleResult:
    """Run the independent NumPy D2Q9 analytic-channel fixture."""

    shape = (fixture.ny, fixture.width)
    populations = np.broadcast_to(D2Q9_WEIGHTS, (*shape, Q)).copy()
    initial_mass = float(np.sum(_density(populations), dtype=np.float64))
    force = np.float32(fixture.body_force_lu)
    omega = np.float32(fixture.omega)
    force_prefactor = np.float32(1.0 - 0.5 * fixture.omega)

    for _ in range(fixture.steps):
        rho = _density(populations)
        u = (
            populations[..., 1]
            - populations[..., 3]
            + populations[..., 5]
            - populations[..., 6]
            - populations[..., 7]
            + populations[..., 8]
            + np.float32(0.5) * force
        ) / rho
        v = (
            populations[..., 2]
            - populations[..., 4]
            + populations[..., 5]
            + populations[..., 6]
            - populations[..., 7]
            - populations[..., 8]
        ) / rho
        speed_squared = u * u + v * v
        post_collision = np.empty_like(populations)
        for direction in range(Q):
            cx = np.float32(D2Q9_VELOCITIES[direction, 0])
            cy = np.float32(D2Q9_VELOCITIES[direction, 1])
            cu = cx * u + cy * v
            equilibrium = (
                D2Q9_WEIGHTS[direction]
                * rho
                * (
                    np.float32(1.0)
                    + np.float32(3.0) * cu
                    + np.float32(4.5) * cu * cu
                    - np.float32(1.5) * speed_squared
                )
            )
            forcing = (
                D2Q9_WEIGHTS[direction]
                * force_prefactor
                * (np.float32(3.0) * (cx - u) + np.float32(9.0) * cu * cx)
                * force
            )
            post_collision[..., direction] = (
                populations[..., direction]
                - omega * (populations[..., direction] - equilibrium)
                + forcing
            )

        streamed = pull_stream_halfway_walls_numpy(post_collision)
        streamed[0] = populations[0]
        streamed[-1] = populations[-1]
        populations = streamed

    final_rho = _density(populations)
    final_u = (
        populations[..., 1]
        - populations[..., 3]
        + populations[..., 5]
        - populations[..., 6]
        - populations[..., 7]
        + populations[..., 8]
        + np.float32(0.5) * force
    ) / final_rho
    measured = np.ascontiguousarray(
        np.mean(final_u[1:-1], axis=1, dtype=np.float64), dtype=np.float64
    )
    analytic = analytic_poiseuille_profile(fixture)
    exclusion = fixture.excluded_wall_cells
    interior = slice(exclusion, -exclusion)
    delta = measured[interior] - analytic[interior]
    relative_l2 = float(np.linalg.norm(delta) / np.linalg.norm(analytic[interior]))
    max_relative = float(np.max(np.abs(delta)) / np.max(np.abs(analytic[interior])))
    final_mass = float(np.sum(final_rho, dtype=np.float64))
    return PoiseuilleResult(
        fixture=fixture,
        measured_profile=measured,
        analytic_profile=analytic,
        relative_l2_error=relative_l2,
        max_relative_error=max_relative,
        initial_mass=initial_mass,
        final_mass=final_mass,
        mass_drift_ratio=abs(final_mass - initial_mass) / initial_mass,
    )


__all__ = [
    "POISEUILLE_ERROR_THRESHOLD",
    "PoiseuilleFixture",
    "PoiseuilleResult",
    "analytic_poiseuille_profile",
    "poiseuille_gate_passes",
    "run_poiseuille_fixture",
]
