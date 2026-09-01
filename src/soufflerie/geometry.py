"""Deterministic ellipse geometry and fail-closed domain preflight."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from soufflerie.errors import DomainError
from soufflerie.schemas import GridSpec, ShapeParams

Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

REFERENCE_DIAMETER_FRACTION = 0.125
OBSTACLE_CENTER_X_FRACTION = 0.30
OBSTACLE_CENTER_Y_FRACTION = 0.50
MIN_MINOR_DIAMETER_CELLS = 12.0
INLET_CLEARANCE_DIAMETERS = 2.0
OUTLET_CLEARANCE_DIAMETERS = 4.0
WALL_CLEARANCE_DIAMETERS = 1.0
SPONGE_LENGTH_DIAMETERS = 2.0
MIN_SPONGE_COLUMNS = 16


@dataclass(frozen=True, slots=True)
class GeometryDiagnostics:
    """Finite scalar evidence emitted by a successful geometry preflight."""

    grid_shape: tuple[int, int]
    center_x_lu: float
    center_y_lu: float
    reference_diameter_lu: float
    semi_major_lu: float
    semi_minor_lu: float
    scaled_minor_diameter_lu: float
    inlet_clearance_lu: float
    outlet_clearance_lu: float
    lower_wall_clearance_lu: float
    upper_wall_clearance_lu: float
    sponge_columns: int
    sponge_start_x_lu: float
    obstacle_cell_count: int
    fluid_cell_count: int
    inlet_outlet_connected: bool


@dataclass(frozen=True, slots=True)
class _AnalyticGeometry:
    center_x: float
    center_y: float
    reference_diameter: float
    semi_major: float
    semi_minor: float
    cosine: float
    sine: float
    extent_x: float
    extent_y: float
    inlet_clearance: float
    outlet_clearance: float
    lower_wall_clearance: float
    upper_wall_clearance: float
    sponge_columns: int
    sponge_start_x: float


def _require_contract_types(shape: ShapeParams, grid: GridSpec) -> None:
    if not isinstance(shape, ShapeParams):
        raise TypeError("shape must be a ShapeParams instance")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec instance")


def reference_diameter_lu(grid: GridSpec) -> float:
    """Return the single RFC-0003 unscaled reference diameter."""

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec instance")
    return REFERENCE_DIAMETER_FRACTION * float(grid.ny)


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _analytic_geometry(shape: ShapeParams, grid: GridSpec) -> _AnalyticGeometry:
    _require_contract_types(shape, grid)
    diameter = reference_diameter_lu(grid)
    semi_major = 0.5 * diameter * shape.scale
    semi_minor = semi_major * shape.aspect_ratio
    center_x = OBSTACLE_CENTER_X_FRACTION * float(grid.nx - 1)
    center_y = OBSTACLE_CENTER_Y_FRACTION * float(grid.ny - 1)
    angle = math.radians(shape.rotation_deg)
    if shape.aspect_ratio == 1.0:
        # A circle is rotation invariant by contract. Avoiding trigonometric
        # round-off makes that invariant bitwise on the same platform.
        cosine = 1.0
        sine = 0.0
    else:
        cosine = math.cos(angle)
        sine = math.sin(angle)
    extent_x = math.hypot(semi_major * cosine, semi_minor * sine)
    extent_y = math.hypot(semi_major * sine, semi_minor * cosine)
    sponge_columns = max(
        MIN_SPONGE_COLUMNS,
        _round_half_up(SPONGE_LENGTH_DIAMETERS * diameter),
    )
    return _AnalyticGeometry(
        center_x=center_x,
        center_y=center_y,
        reference_diameter=diameter,
        semi_major=semi_major,
        semi_minor=semi_minor,
        cosine=cosine,
        sine=sine,
        extent_x=extent_x,
        extent_y=extent_y,
        inlet_clearance=center_x - extent_x,
        outlet_clearance=float(grid.nx - 1) - center_x - extent_x,
        lower_wall_clearance=center_y - extent_y,
        upper_wall_clearance=float(grid.ny - 1) - center_y - extent_y,
        sponge_columns=sponge_columns,
        sponge_start_x=float(grid.nx - sponge_columns),
    )


def _preflight_analytic(geometry: _AnalyticGeometry, grid: GridSpec) -> None:
    minor_diameter = 2.0 * geometry.semi_minor
    if minor_diameter < MIN_MINOR_DIAMETER_CELLS:
        raise DomainError(
            "GEO-2 RESOLUTION: scaled minor diameter "
            f"{minor_diameter:.17g} is below {MIN_MINOR_DIAMETER_CELLS:g} cells"
        )
    if geometry.sponge_columns >= grid.nx:
        raise DomainError("GEO-2 SPONGE: sponge occupies the entire streamwise grid")

    required = (
        (
            "inlet",
            geometry.inlet_clearance,
            INLET_CLEARANCE_DIAMETERS * geometry.reference_diameter,
        ),
        (
            "outlet",
            geometry.outlet_clearance,
            OUTLET_CLEARANCE_DIAMETERS * geometry.reference_diameter,
        ),
        (
            "lower wall",
            geometry.lower_wall_clearance,
            WALL_CLEARANCE_DIAMETERS * geometry.reference_diameter,
        ),
        (
            "upper wall",
            geometry.upper_wall_clearance,
            WALL_CLEARANCE_DIAMETERS * geometry.reference_diameter,
        ),
    )
    for boundary, observed, minimum in required:
        if observed < minimum:
            raise DomainError(
                f"GEO-2 CLEARANCE: {boundary} clearance {observed:.17g} "
                f"is below {minimum:.17g} lattice units"
            )

    obstacle_right = geometry.center_x + geometry.extent_x
    if obstacle_right >= geometry.sponge_start_x:
        raise DomainError(
            "GEO-2 SPONGE: obstacle enters the outlet sponge "
            f"starting at x={geometry.sponge_start_x:.17g}"
        )


def _ellipse_sdf_unchecked(
    geometry: _AnalyticGeometry,
    grid: GridSpec,
) -> Float32Array:
    x = np.arange(grid.nx, dtype=np.float32)[None, :] - np.float32(geometry.center_x)
    y = np.arange(grid.ny, dtype=np.float32)[:, None] - np.float32(geometry.center_y)
    cosine = np.float32(geometry.cosine)
    sine = np.float32(geometry.sine)
    x_prime = cosine * x + sine * y
    y_prime = -sine * x + cosine * y
    q = np.sqrt(
        np.square(x_prime / np.float32(geometry.semi_major))
        + np.square(y_prime / np.float32(geometry.semi_minor))
    )
    result = np.ascontiguousarray(
        (q - np.float32(1.0)) * np.float32(min(geometry.semi_major, geometry.semi_minor)),
        dtype=np.float32,
    )
    result.flags.writeable = False
    return result


def ellipse_sdf(shape: ShapeParams, grid: GridSpec) -> Float32Array:
    """Rasterize the RFC-0003 algebraic ellipse SDF at lattice-cell centers."""

    return _ellipse_sdf_unchecked(_analytic_geometry(shape, grid), grid)


def _validated_sdf(sdf: Float32Array) -> Float32Array:
    if not isinstance(sdf, np.ndarray) or sdf.dtype != np.float32 or sdf.ndim != 2:
        raise DomainError("GEO-1 SDF: expected a two-dimensional float32 array")
    if sdf.size == 0 or not sdf.flags.c_contiguous:
        raise DomainError("GEO-1 SDF: array must be non-empty and C-contiguous")
    if not np.isfinite(sdf).all():
        raise DomainError("GEO-1 SDF: array must contain only finite values")
    return sdf


def obstacle_mask(sdf: Float32Array) -> BoolArray:
    """Return the canonical read-only solid mask, where the zero contour is solid."""

    checked = _validated_sdf(sdf)
    result = np.ascontiguousarray(checked <= np.float32(0.0), dtype=np.bool_)
    result.flags.writeable = False
    return result


def normalized_sdf_input(shape: ShapeParams, grid: GridSpec) -> Float32Array:
    """Return the clipped dimensionless SDF model channel in ``[-1, 1]``."""

    sdf = ellipse_sdf(shape, grid)
    result = np.ascontiguousarray(
        np.clip(
            sdf / np.float32(reference_diameter_lu(grid)),
            np.float32(-1.0),
            np.float32(1.0),
        ),
        dtype=np.float32,
    )
    result.flags.writeable = False
    return result


def _fluid_connects_inlet_to_outlet(solid: BoolArray) -> bool:
    ny, nx = solid.shape
    if ny < 3:
        return False
    visited = np.zeros(solid.shape, dtype=np.bool_)
    frontier: deque[int] = deque()
    for y in range(1, ny - 1):
        if not solid[y, 0]:
            visited[y, 0] = True
            frontier.append(y * nx)

    while frontier:
        index = frontier.popleft()
        y, x = divmod(index, nx)
        if x == nx - 1:
            return True
        if x > 0 and not solid[y, x - 1] and not visited[y, x - 1]:
            visited[y, x - 1] = True
            frontier.append(index - 1)
        if x + 1 < nx and not solid[y, x + 1] and not visited[y, x + 1]:
            visited[y, x + 1] = True
            frontier.append(index + 1)
        if y > 1 and not solid[y - 1, x] and not visited[y - 1, x]:
            visited[y - 1, x] = True
            frontier.append(index - nx)
        if y + 1 < ny - 1 and not solid[y + 1, x] and not visited[y + 1, x]:
            visited[y + 1, x] = True
            frontier.append(index + nx)
    return False


def validate_geometry(shape: ShapeParams, grid: GridSpec) -> GeometryDiagnostics:
    """Validate resolution, clearance, sponge, and raster connectivity before a solve."""

    geometry = _analytic_geometry(shape, grid)
    # Scalar rejection deliberately precedes dense geometry allocation.
    _preflight_analytic(geometry, grid)
    sdf = _ellipse_sdf_unchecked(geometry, grid)
    solid = obstacle_mask(sdf)
    connected = _fluid_connects_inlet_to_outlet(solid)
    if not connected:
        raise DomainError("GEO-1 CONNECTIVITY: no one-cell fluid path joins inlet and outlet")
    obstacle_cells = int(np.count_nonzero(solid))
    if obstacle_cells == 0:
        raise DomainError("GEO-2 RESOLUTION: rasterized obstacle contains no solid cells")
    return GeometryDiagnostics(
        grid_shape=grid.shape,
        center_x_lu=geometry.center_x,
        center_y_lu=geometry.center_y,
        reference_diameter_lu=geometry.reference_diameter,
        semi_major_lu=geometry.semi_major,
        semi_minor_lu=geometry.semi_minor,
        scaled_minor_diameter_lu=2.0 * geometry.semi_minor,
        inlet_clearance_lu=geometry.inlet_clearance,
        outlet_clearance_lu=geometry.outlet_clearance,
        lower_wall_clearance_lu=geometry.lower_wall_clearance,
        upper_wall_clearance_lu=geometry.upper_wall_clearance,
        sponge_columns=geometry.sponge_columns,
        sponge_start_x_lu=geometry.sponge_start_x,
        obstacle_cell_count=obstacle_cells,
        fluid_cell_count=int(solid.size - obstacle_cells),
        inlet_outlet_connected=True,
    )


__all__ = [
    "INLET_CLEARANCE_DIAMETERS",
    "MIN_MINOR_DIAMETER_CELLS",
    "MIN_SPONGE_COLUMNS",
    "OBSTACLE_CENTER_X_FRACTION",
    "OBSTACLE_CENTER_Y_FRACTION",
    "OUTLET_CLEARANCE_DIAMETERS",
    "REFERENCE_DIAMETER_FRACTION",
    "SPONGE_LENGTH_DIAMETERS",
    "WALL_CLEARANCE_DIAMETERS",
    "GeometryDiagnostics",
    "ellipse_sdf",
    "normalized_sdf_input",
    "obstacle_mask",
    "reference_diameter_lu",
    "validate_geometry",
]
