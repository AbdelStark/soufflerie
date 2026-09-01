"""Deterministic force-history and predicted-field diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from soufflerie.errors import DomainError, InternalInvariantError
from soufflerie.geometry import (
    CONTROL_SURFACE_CLEARANCE_CELLS,
    ControlSurface,
    control_surface_from_sdf,
)
from soufflerie.schemas import FlowFields
from soufflerie.solver.forces import ObstacleForceHistory
from soufflerie.solver.lattice import CS2, RHO_REF, DerivedLatticeConfig

MEAN_FORCE_WINDOW_STEPS = 4_000
MIN_RESOLVED_LIFT_CYCLES = 8.0
MIN_STROUHAL = 0.05
MAX_STROUHAL = 0.4


class StrouhalUnavailableReason(StrEnum):
    """Stable reason codes for a lift spectrum that cannot support Strouhal."""

    INSUFFICIENT_SAMPLES = "insufficient_samples"
    IRREGULAR_SAMPLING = "irregular_sampling"
    DEGENERATE_LIFT = "degenerate_lift"
    NO_RESOLVABLE_BAND = "no_resolvable_band"
    NO_SPECTRAL_PEAK = "no_spectral_peak"
    INSUFFICIENT_CYCLES = "insufficient_cycles"


@dataclass(frozen=True, slots=True)
class StrouhalEstimate:
    """A resolved Strouhal value or typed evidence explaining its absence."""

    strouhal: float | None
    reason: StrouhalUnavailableReason | None
    frequency_lu: float | None
    resolved_cycles: float
    sample_count: int
    sample_interval_steps: int | None

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise InternalInvariantError("DIAG-1 STROUHAL: sample_count must be nonnegative")
        if self.sample_interval_steps is not None and self.sample_interval_steps <= 0:
            raise InternalInvariantError(
                "DIAG-1 STROUHAL: sample interval must be positive when present"
            )
        if not math.isfinite(self.resolved_cycles) or self.resolved_cycles < 0.0:
            raise InternalInvariantError(
                "DIAG-1 STROUHAL: resolved cycle count must be finite and nonnegative"
            )
        if self.frequency_lu is not None and (
            not math.isfinite(self.frequency_lu) or self.frequency_lu <= 0.0
        ):
            raise InternalInvariantError(
                "DIAG-1 STROUHAL: candidate frequency must be finite and positive"
            )
        if self.reason is None:
            if (
                self.strouhal is None
                or not math.isfinite(self.strouhal)
                or not MIN_STROUHAL <= self.strouhal <= MAX_STROUHAL
                or self.frequency_lu is None
                or self.resolved_cycles < MIN_RESOLVED_LIFT_CYCLES
            ):
                raise InternalInvariantError(
                    "DIAG-1 STROUHAL: an available estimate must satisfy the spectral contract"
                )
        elif self.strouhal is not None:
            raise InternalInvariantError(
                "DIAG-1 STROUHAL: unavailable estimates must not contain a scalar value"
            )

    @property
    def available(self) -> bool:
        return self.reason is None


@dataclass(frozen=True, slots=True)
class MeanForceCoefficients:
    """Fp64 means over the right-closed trailing force-history window."""

    cd: float
    cl: float
    sample_count: int
    first_step: int
    last_step: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.cd) or not math.isfinite(self.cl):
            raise InternalInvariantError("DIAG-1 FORCE_MEAN: coefficients must be finite")
        if self.sample_count <= 0 or self.first_step < 0 or self.last_step < self.first_step:
            raise InternalInvariantError("DIAG-1 FORCE_MEAN: sample window is incoherent")


def mean_force_coefficients(
    history: ObstacleForceHistory,
    *,
    window_steps: int = MEAN_FORCE_WINDOW_STEPS,
) -> MeanForceCoefficients:
    """Average Cd and Cl over samples in ``(last_step-window_steps, last_step]``."""

    if not isinstance(history, ObstacleForceHistory):
        raise TypeError("history must be an ObstacleForceHistory instance")
    if isinstance(window_steps, bool) or not isinstance(window_steps, int) or window_steps <= 0:
        raise DomainError("DIAG-1 FORCE_MEAN: window_steps must be a positive integer")
    if history.count == 0:
        raise DomainError("DIAG-1 FORCE_MEAN: force history contains no samples")

    last_step = int(history.steps[-1])
    first_index = int(np.searchsorted(history.steps, last_step - window_steps, side="right"))
    cd_sum = 0.0
    cl_sum = 0.0
    for index in range(first_index, history.count):
        cd_sum += float(history.cd[index])
        cl_sum += float(history.cl[index])
    sample_count = history.count - first_index
    return MeanForceCoefficients(
        cd=cd_sum / sample_count,
        cl=cl_sum / sample_count,
        sample_count=sample_count,
        first_step=int(history.steps[first_index]),
        last_step=last_step,
    )


def _unavailable_strouhal(
    reason: StrouhalUnavailableReason,
    *,
    sample_count: int,
    sample_interval_steps: int | None = None,
    frequency_lu: float | None = None,
    resolved_cycles: float = 0.0,
) -> StrouhalEstimate:
    return StrouhalEstimate(
        strouhal=None,
        reason=reason,
        frequency_lu=frequency_lu,
        resolved_cycles=resolved_cycles,
        sample_count=sample_count,
        sample_interval_steps=sample_interval_steps,
    )


def estimate_strouhal(
    history: ObstacleForceHistory,
    config: DerivedLatticeConfig,
) -> StrouhalEstimate:
    """Estimate lift Strouhal from the RFC-0003 fp64 Hann-windowed spectrum."""

    if not isinstance(history, ObstacleForceHistory):
        raise TypeError("history must be an ObstacleForceHistory instance")
    if not isinstance(config, DerivedLatticeConfig):
        raise TypeError("config must be a DerivedLatticeConfig instance")
    if history.count and (
        int(history.steps[0]) <= config.warmup_steps or int(history.steps[-1]) > config.steps
    ):
        raise DomainError("DIAG-1 STROUHAL: history lies outside the post-warmup run schedule")
    if history.count < 4:
        return _unavailable_strouhal(
            StrouhalUnavailableReason.INSUFFICIENT_SAMPLES,
            sample_count=history.count,
        )

    step_differences = np.diff(history.steps)
    sample_interval = int(step_differences[0])
    if sample_interval != config.sample_interval or np.any(step_differences != sample_interval):
        return _unavailable_strouhal(
            StrouhalUnavailableReason.IRREGULAR_SAMPLING,
            sample_count=history.count,
        )

    lift = history.cl.astype(np.float64)
    centered = lift - float(np.mean(lift, dtype=np.float64))
    degeneracy_scale = max(1.0, float(np.max(np.abs(lift))))
    if float(np.ptp(centered)) <= np.finfo(np.float32).eps * degeneracy_scale:
        return _unavailable_strouhal(
            StrouhalUnavailableReason.DEGENERATE_LIFT,
            sample_count=history.count,
            sample_interval_steps=sample_interval,
        )

    windowed = centered * np.hanning(history.count).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(history.count, d=float(sample_interval))
    minimum_frequency = MIN_STROUHAL * config.inlet_velocity_lu / config.reference_diameter_lu
    maximum_frequency = MAX_STROUHAL * config.inlet_velocity_lu / config.reference_diameter_lu
    band_indices = np.flatnonzero(
        (frequencies > 0.0)
        & (frequencies >= minimum_frequency)
        & (frequencies <= maximum_frequency)
    )
    if band_indices.size == 0:
        return _unavailable_strouhal(
            StrouhalUnavailableReason.NO_RESOLVABLE_BAND,
            sample_count=history.count,
            sample_interval_steps=sample_interval,
        )

    peak_index = int(band_indices[int(np.argmax(spectrum[band_indices]))])
    peak_magnitude = float(spectrum[peak_index])
    if not math.isfinite(peak_magnitude) or peak_magnitude <= np.finfo(np.float64).tiny:
        return _unavailable_strouhal(
            StrouhalUnavailableReason.NO_SPECTRAL_PEAK,
            sample_count=history.count,
            sample_interval_steps=sample_interval,
        )

    peak_offset = 0.0
    if 0 < peak_index < spectrum.size - 1:
        tiny = float(np.finfo(np.float64).tiny)
        left = math.log(max(float(spectrum[peak_index - 1]), tiny))
        center = math.log(max(peak_magnitude, tiny))
        right = math.log(max(float(spectrum[peak_index + 1]), tiny))
        denominator = left - 2.0 * center + right
        if math.isfinite(denominator) and abs(denominator) > tiny:
            peak_offset = min(0.5, max(-0.5, 0.5 * (left - right) / denominator))

    frequency_resolution = 1.0 / (history.count * sample_interval)
    frequency_lu = (peak_index + peak_offset) * frequency_resolution
    frequency_lu = min(maximum_frequency, max(minimum_frequency, frequency_lu))
    resolved_cycles = frequency_lu * float(int(history.steps[-1]) - int(history.steps[0]))
    if resolved_cycles < MIN_RESOLVED_LIFT_CYCLES:
        return _unavailable_strouhal(
            StrouhalUnavailableReason.INSUFFICIENT_CYCLES,
            sample_count=history.count,
            sample_interval_steps=sample_interval,
            frequency_lu=frequency_lu,
            resolved_cycles=resolved_cycles,
        )

    strouhal = frequency_lu * config.reference_diameter_lu / config.inlet_velocity_lu
    return StrouhalEstimate(
        strouhal=strouhal,
        reason=None,
        frequency_lu=frequency_lu,
        resolved_cycles=resolved_cycles,
        sample_count=history.count,
        sample_interval_steps=sample_interval,
    )


def select_control_surface(
    fields: FlowFields,
    *,
    sponge_start_x: int,
    clearance_cells: int = CONTROL_SURFACE_CLEARANCE_CELLS,
) -> ControlSurface:
    """Select the tightest axis-aligned rectangle enclosing the clearance band."""

    if not isinstance(fields, FlowFields):
        raise TypeError("fields must be a FlowFields instance")
    return control_surface_from_sdf(
        fields.sdf,
        sponge_start_x=sponge_start_x,
        clearance_cells=clearance_cells,
    )


@dataclass(frozen=True, slots=True)
class FieldDragEstimate:
    """Control-volume drag evidence from one predicted mean field."""

    cd: float
    force_x_lu: float
    pressure_force_x_lu: float
    convective_force_x_lu: float
    normalization_lu: float
    surface: ControlSurface

    def __post_init__(self) -> None:
        values = (
            self.cd,
            self.force_x_lu,
            self.pressure_force_x_lu,
            self.convective_force_x_lu,
            self.normalization_lu,
        )
        if not all(math.isfinite(value) for value in values) or self.normalization_lu <= 0.0:
            raise InternalInvariantError("DIAG-2 FIELD_DRAG: result scalars are incoherent")
        if not math.isclose(
            self.force_x_lu,
            self.pressure_force_x_lu + self.convective_force_x_lu,
            rel_tol=1e-15,
            abs_tol=1e-15,
        ):
            raise InternalInvariantError("DIAG-2 FIELD_DRAG: force components do not sum")


def _fixed_sum(values: npt.NDArray[np.float64]) -> float:
    total = 0.0
    for value in values:
        total += float(value)
    return total


def field_drag_coefficient(
    fields: FlowFields,
    *,
    sponge_start_x: int,
    inlet_velocity_lu: float,
    reference_diameter_lu: float,
) -> FieldDragEstimate:
    """Integrate pressure and convective flux on a selected rectangular surface."""

    if not isinstance(fields, FlowFields):
        raise TypeError("fields must be a FlowFields instance")
    for name, value in (
        ("inlet_velocity_lu", inlet_velocity_lu),
        ("reference_diameter_lu", reference_diameter_lu),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DomainError(f"DIAG-2 FIELD_DRAG: {name} must be a finite positive number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise DomainError(f"DIAG-2 FIELD_DRAG: {name} must be a finite positive number")

    surface = select_control_surface(fields, sponge_start_x=sponge_start_x)
    y_slice = slice(surface.bottom_y, surface.top_y + 1)
    x_slice = slice(surface.left_x, surface.right_x + 1)
    rho = fields.rho.astype(np.float64, copy=False)
    u = fields.u.astype(np.float64, copy=False)
    v = fields.v.astype(np.float64, copy=False)
    pressure = CS2 * rho
    streamwise_flux = rho * u * u
    transverse_flux = rho * u * v

    pressure_force = _fixed_sum(pressure[y_slice, surface.left_x]) - _fixed_sum(
        pressure[y_slice, surface.right_x]
    )
    convective_force = (
        _fixed_sum(streamwise_flux[y_slice, surface.left_x])
        - _fixed_sum(streamwise_flux[y_slice, surface.right_x])
        + _fixed_sum(transverse_flux[surface.bottom_y, x_slice])
        - _fixed_sum(transverse_flux[surface.top_y, x_slice])
    )
    force_x = pressure_force + convective_force
    normalization = (
        0.5
        * RHO_REF
        * float(inlet_velocity_lu)
        * float(inlet_velocity_lu)
        * float(reference_diameter_lu)
    )
    return FieldDragEstimate(
        cd=force_x / normalization,
        force_x_lu=force_x,
        pressure_force_x_lu=pressure_force,
        convective_force_x_lu=convective_force,
        normalization_lu=normalization,
        surface=surface,
    )


__all__ = [
    "CONTROL_SURFACE_CLEARANCE_CELLS",
    "MAX_STROUHAL",
    "MEAN_FORCE_WINDOW_STEPS",
    "MIN_RESOLVED_LIFT_CYCLES",
    "MIN_STROUHAL",
    "ControlSurface",
    "FieldDragEstimate",
    "MeanForceCoefficients",
    "StrouhalEstimate",
    "StrouhalUnavailableReason",
    "estimate_strouhal",
    "field_drag_coefficient",
    "mean_force_coefficients",
    "select_control_surface",
]
