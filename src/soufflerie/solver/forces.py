"""Deterministic obstacle-link enumeration and momentum-exchange forces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from soufflerie.errors import DomainError, InternalInvariantError
from soufflerie.solver.lattice import (
    D2Q9_VELOCITIES,
    RHO_REF,
    DerivedLatticeConfig,
    validate_populations,
)

BoolArray = npt.NDArray[np.bool_]
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
Int8Array = npt.NDArray[np.int8]
Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]
ScalarT = TypeVar("ScalarT", bound=np.generic)
FORCE_REDUCTION_TILE_LINKS = 256


def _readonly_copy(array: npt.NDArray[ScalarT]) -> npt.NDArray[ScalarT]:
    result = np.ascontiguousarray(array.copy())
    result.flags.writeable = False
    return result


def _validate_obstacle_mask(mask: BoolArray) -> tuple[int, int]:
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.ndim != 2
        or mask.size == 0
        or not mask.flags.c_contiguous
    ):
        raise DomainError(
            "FORCE-1 MASK: obstacle mask must be non-empty C-contiguous bool [ny, nx]"
        )
    return (mask.shape[0], mask.shape[1])


@dataclass(frozen=True, slots=True)
class ObstacleLinks:
    """Row-major fluid-to-solid links with ascending D2Q9 directions."""

    grid_shape: tuple[int, int]
    fluid_y: Int32Array
    fluid_x: Int32Array
    direction: Int8Array

    def __post_init__(self) -> None:
        ny, nx = self.grid_shape
        if ny <= 0 or nx <= 0:
            raise InternalInvariantError("FORCE-1 LINKS: grid shape must be positive")
        count = self.fluid_y.shape
        expected = (
            ("fluid_y", self.fluid_y, np.dtype(np.int32)),
            ("fluid_x", self.fluid_x, np.dtype(np.int32)),
            ("direction", self.direction, np.dtype(np.int8)),
        )
        for name, array, dtype in expected:
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != dtype
                or array.ndim != 1
                or array.shape != count
                or not array.flags.c_contiguous
                or array.flags.writeable
            ):
                raise InternalInvariantError(
                    f"FORCE-1 LINKS: {name} must be a read-only C-contiguous vector"
                )
        if np.any(self.fluid_y < 0) or np.any(self.fluid_y >= ny):
            raise InternalInvariantError("FORCE-1 LINKS: fluid_y is outside the grid")
        if np.any(self.fluid_x < 0) or np.any(self.fluid_x >= nx):
            raise InternalInvariantError("FORCE-1 LINKS: fluid_x is outside the grid")
        if np.any(self.direction <= 0) or np.any(self.direction >= 9):
            raise InternalInvariantError("FORCE-1 LINKS: directions must be in [1, 8]")
        if self.direction.size > 1:
            keys = (
                self.fluid_y.astype(np.int64) * nx + self.fluid_x.astype(np.int64)
            ) * 9 + self.direction.astype(np.int64)
            if np.any(np.diff(keys) <= 0):
                raise InternalInvariantError(
                    "FORCE-1 LINKS: links must be unique and in fixed row/direction order"
                )

    @property
    def count(self) -> int:
        return int(self.direction.size)


def enumerate_obstacle_links(mask: BoolArray) -> ObstacleLinks:
    """Enumerate each in-domain fluid-to-solid D2Q9 link exactly once."""

    ny, nx = _validate_obstacle_mask(mask)
    fluid_y: list[int] = []
    fluid_x: list[int] = []
    directions: list[int] = []
    for y in range(ny):
        for x in range(nx):
            if mask[y, x]:
                continue
            for direction in range(1, 9):
                neighbor_x = x + int(D2Q9_VELOCITIES[direction, 0])
                neighbor_y = y + int(D2Q9_VELOCITIES[direction, 1])
                if 0 <= neighbor_y < ny and 0 <= neighbor_x < nx and mask[neighbor_y, neighbor_x]:
                    fluid_y.append(y)
                    fluid_x.append(x)
                    directions.append(direction)
    return ObstacleLinks(
        grid_shape=(ny, nx),
        fluid_y=_readonly_copy(np.asarray(fluid_y, dtype=np.int32)),
        fluid_x=_readonly_copy(np.asarray(fluid_x, dtype=np.int32)),
        direction=_readonly_copy(np.asarray(directions, dtype=np.int8)),
    )


@dataclass(frozen=True, slots=True)
class ObstacleForce:
    """One fp64 deterministic momentum-exchange reduction and normalization."""

    link_count: int
    fx_lu: float
    fy_lu: float
    normalization_lu: float
    cd: float
    cl: float

    def __post_init__(self) -> None:
        if self.link_count < 0:
            raise InternalInvariantError("FORCE-1 NORMALIZATION: link_count must be nonnegative")
        values = (self.fx_lu, self.fy_lu, self.normalization_lu, self.cd, self.cl)
        if not all(math.isfinite(value) for value in values):
            raise InternalInvariantError("FORCE-1 NORMALIZATION: force values must be finite")
        if self.normalization_lu <= 0.0:
            raise InternalInvariantError("FORCE-1 NORMALIZATION: denominator must be positive")


def momentum_exchange_force(
    post_collision: Float32Array,
    links: ObstacleLinks,
    config: DerivedLatticeConfig,
) -> ObstacleForce:
    """Reduce obstacle force in the links declared fixed order without atomics."""

    validate_populations(post_collision)
    if not isinstance(links, ObstacleLinks):
        raise TypeError("links must be an ObstacleLinks instance")
    if not isinstance(config, DerivedLatticeConfig):
        raise TypeError("config must be a DerivedLatticeConfig instance")
    if post_collision.shape[:2] != config.shape or links.grid_shape != config.shape:
        raise DomainError("FORCE-1 NORMALIZATION: populations, links, and config grids must match")

    fx = 0.0
    fy = 0.0
    for tile_start in range(0, links.count, FORCE_REDUCTION_TILE_LINKS):
        tile_fx = 0.0
        tile_fy = 0.0
        tile_stop = min(tile_start + FORCE_REDUCTION_TILE_LINKS, links.count)
        for index in range(tile_start, tile_stop):
            y = int(links.fluid_y[index])
            x = int(links.fluid_x[index])
            direction = int(links.direction[index])
            momentum = 2.0 * float(post_collision[y, x, direction])
            tile_fx += momentum * int(D2Q9_VELOCITIES[direction, 0])
            tile_fy += momentum * int(D2Q9_VELOCITIES[direction, 1])
        fx += tile_fx
        fy += tile_fy

    normalization = (
        0.5
        * RHO_REF
        * config.inlet_velocity_lu
        * config.inlet_velocity_lu
        * config.reference_diameter_lu
    )
    return ObstacleForce(
        link_count=links.count,
        fx_lu=fx,
        fy_lu=fy,
        normalization_lu=normalization,
        cd=fx / normalization,
        cl=fy / normalization,
    )


@dataclass(frozen=True, slots=True)
class ObstacleForceHistory:
    """Immutable persisted-boundary force history arrays."""

    steps: Int64Array
    fx_lu: Float64Array
    fy_lu: Float64Array
    cd: Float32Array
    cl: Float32Array

    def __post_init__(self) -> None:
        shape = self.steps.shape
        expected = (
            ("steps", self.steps, np.dtype(np.int64)),
            ("fx_lu", self.fx_lu, np.dtype(np.float64)),
            ("fy_lu", self.fy_lu, np.dtype(np.float64)),
            ("cd", self.cd, np.dtype(np.float32)),
            ("cl", self.cl, np.dtype(np.float32)),
        )
        for name, array, dtype in expected:
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != dtype
                or array.ndim != 1
                or array.shape != shape
                or not array.flags.c_contiguous
                or array.flags.writeable
                or not np.isfinite(array).all()
            ):
                raise InternalInvariantError(
                    f"FORCE-1 HISTORY: {name} must be a finite read-only vector"
                )
        if np.any(self.steps < 0) or (self.steps.size > 1 and np.any(np.diff(self.steps) <= 0)):
            raise InternalInvariantError("FORCE-1 HISTORY: steps must be increasing")

    @property
    def count(self) -> int:
        return int(self.steps.size)


@dataclass(slots=True)
class ForceHistoryRecorder:
    """Append strictly ordered force samples and freeze them for persistence."""

    _steps: list[int] = field(default_factory=list)
    _forces: list[ObstacleForce] = field(default_factory=list)

    def record(self, step: int, force: ObstacleForce) -> None:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise DomainError("FORCE-1 HISTORY: step must be a nonnegative integer")
        if self._steps and step <= self._steps[-1]:
            raise DomainError("FORCE-1 HISTORY: steps must be strictly increasing")
        if not isinstance(force, ObstacleForce):
            raise TypeError("force must be an ObstacleForce instance")
        self._steps.append(step)
        self._forces.append(force)

    def snapshot(self) -> ObstacleForceHistory:
        return ObstacleForceHistory(
            steps=_readonly_copy(np.asarray(self._steps, dtype=np.int64)),
            fx_lu=_readonly_copy(
                np.asarray([force.fx_lu for force in self._forces], dtype=np.float64)
            ),
            fy_lu=_readonly_copy(
                np.asarray([force.fy_lu for force in self._forces], dtype=np.float64)
            ),
            cd=_readonly_copy(np.asarray([force.cd for force in self._forces], dtype=np.float32)),
            cl=_readonly_copy(np.asarray([force.cl for force in self._forces], dtype=np.float32)),
        )


__all__ = [
    "FORCE_REDUCTION_TILE_LINKS",
    "ForceHistoryRecorder",
    "ObstacleForce",
    "ObstacleForceHistory",
    "ObstacleLinks",
    "enumerate_obstacle_links",
    "momentum_exchange_force",
]
