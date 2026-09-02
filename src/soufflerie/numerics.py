"""Allocation-free lattice configuration and stability preflight."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from soufflerie.errors import DomainError
from soufflerie.geometry import reference_diameter_lu
from soufflerie.schemas import CaseConfig

CS2: Final = 1.0 / 3.0
SPEED_OF_SOUND: Final = math.sqrt(CS2)
RHO_REF: Final = 1.0
MIN_TAU: Final = 0.5005
MAX_TAU: Final = 1.95
MAX_INLET_VELOCITY_LU: Final = 0.1
MAX_NOMINAL_MACH: Final = 0.1733
DEFAULT_SAMPLE_INTERVAL: Final = 10
MAX_RAMP_STEPS: Final = 2_000


def _require_integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DomainError(f"LBM-3 STABILITY: {name} must be an integer >= {minimum}")


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError(f"LBM-3 STABILITY: {name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DomainError(f"LBM-3 STABILITY: {name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class LatticeConfig:
    """Underived numerical inputs from the accepted RFC-0002 interface."""

    nx: int
    ny: int
    steps: int
    warmup_steps: int
    sample_interval: int
    inlet_velocity_lu: float
    reynolds: float
    reference_diameter_lu: float

    def __post_init__(self) -> None:
        _require_integer("nx", self.nx, minimum=3)
        _require_integer("ny", self.ny, minimum=3)
        _require_integer("steps", self.steps, minimum=1)
        _require_integer("warmup_steps", self.warmup_steps, minimum=0)
        _require_integer("sample_interval", self.sample_interval, minimum=1)
        if self.warmup_steps >= self.steps:
            raise DomainError("LBM-3 STABILITY: warmup_steps must be less than steps")
        if self.sample_interval > self.steps - self.warmup_steps:
            raise DomainError(
                "LBM-3 STABILITY: sample_interval must fit within the post-warmup window"
            )

        inlet_velocity = _finite_float("inlet_velocity_lu", self.inlet_velocity_lu)
        reynolds = _finite_float("reynolds", self.reynolds)
        diameter = _finite_float("reference_diameter_lu", self.reference_diameter_lu)
        if inlet_velocity <= 0.0 or inlet_velocity > MAX_INLET_VELOCITY_LU:
            raise DomainError(
                f"LBM-3 STABILITY: inlet_velocity_lu must be in (0, {MAX_INLET_VELOCITY_LU}]"
            )
        if reynolds <= 0.0:
            raise DomainError("LBM-3 STABILITY: reynolds must be positive")
        if diameter <= 0.0:
            raise DomainError("LBM-3 STABILITY: reference_diameter_lu must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)


@dataclass(frozen=True, slots=True)
class DerivedLatticeConfig:
    """A preflighted lattice configuration with no clipped numerical values."""

    nx: int
    ny: int
    steps: int
    warmup_steps: int
    sample_interval: int
    inlet_velocity_lu: float
    reynolds: float
    reference_diameter_lu: float
    kinematic_viscosity_lu: float
    tau: float
    omega: float
    nominal_mach: float
    rho_ref: float
    ramp_steps: int

    def __post_init__(self) -> None:
        base = self.base
        expected_viscosity = kinematic_viscosity_lu(
            inlet_velocity_lu=base.inlet_velocity_lu,
            reference_diameter_lu=base.reference_diameter_lu,
            reynolds=base.reynolds,
        )
        expected_tau = 3.0 * expected_viscosity + 0.5
        expected = (
            ("kinematic_viscosity_lu", self.kinematic_viscosity_lu, expected_viscosity),
            ("tau", self.tau, expected_tau),
            ("omega", self.omega, 1.0 / expected_tau),
            ("nominal_mach", self.nominal_mach, base.inlet_velocity_lu / SPEED_OF_SOUND),
        )
        for name, actual, target in expected:
            if not math.isfinite(actual) or not math.isclose(
                actual, target, rel_tol=1e-15, abs_tol=1e-15
            ):
                raise DomainError(f"LBM-3 STABILITY: derived {name} is not coherent")
        if expected_tau < math.nextafter(MIN_TAU, -math.inf) or expected_tau > math.nextafter(
            MAX_TAU, math.inf
        ):
            raise DomainError("LBM-3 STABILITY: derived tau is outside the accepted interval")
        if base.inlet_velocity_lu / SPEED_OF_SOUND > MAX_NOMINAL_MACH:
            raise DomainError("LBM-3 STABILITY: derived nominal Mach exceeds the accepted maximum")
        if self.rho_ref != RHO_REF:
            raise DomainError(f"LBM-3 STABILITY: rho_ref must equal {RHO_REF}")
        if self.ramp_steps != min(MAX_RAMP_STEPS, base.warmup_steps):
            raise DomainError("LBM-3 STABILITY: ramp_steps does not match warmup policy")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def base(self) -> LatticeConfig:
        return LatticeConfig(
            nx=self.nx,
            ny=self.ny,
            steps=self.steps,
            warmup_steps=self.warmup_steps,
            sample_interval=self.sample_interval,
            inlet_velocity_lu=self.inlet_velocity_lu,
            reynolds=self.reynolds,
            reference_diameter_lu=self.reference_diameter_lu,
        )


def kinematic_viscosity_lu(
    *, inlet_velocity_lu: float, reference_diameter_lu: float, reynolds: float
) -> float:
    """Return ``nu = U_ref * D_lu / Re`` without clipping inputs or output."""

    inlet = _finite_float("inlet_velocity_lu", inlet_velocity_lu)
    diameter = _finite_float("reference_diameter_lu", reference_diameter_lu)
    re = _finite_float("reynolds", reynolds)
    if inlet <= 0.0 or diameter <= 0.0 or re <= 0.0:
        raise DomainError("LBM-3 STABILITY: velocity, diameter, and Reynolds must be positive")
    viscosity = inlet * diameter / re
    if not math.isfinite(viscosity) or viscosity <= 0.0:
        raise DomainError("LBM-3 STABILITY: derived viscosity must be finite and positive")
    return viscosity


def relaxation_time(
    *, inlet_velocity_lu: float, reference_diameter_lu: float, reynolds: float
) -> float:
    """Derive BGK relaxation time from the Reynolds mapping."""

    return (
        3.0
        * kinematic_viscosity_lu(
            inlet_velocity_lu=inlet_velocity_lu,
            reference_diameter_lu=reference_diameter_lu,
            reynolds=reynolds,
        )
        + 0.5
    )


def reynolds_from_relaxation_time(
    *, tau: float, inlet_velocity_lu: float, reference_diameter_lu: float
) -> float:
    """Invert the RFC-0002 Reynolds mapping without modifying ``tau``."""

    relaxation = _finite_float("tau", tau)
    inlet = _finite_float("inlet_velocity_lu", inlet_velocity_lu)
    diameter = _finite_float("reference_diameter_lu", reference_diameter_lu)
    if relaxation <= 0.5:
        raise DomainError("LBM-3 STABILITY: tau must be greater than 0.5")
    if inlet <= 0.0 or diameter <= 0.0:
        raise DomainError("LBM-3 STABILITY: velocity and diameter must be positive")
    result = inlet * diameter / ((relaxation - 0.5) / 3.0)
    if not math.isfinite(result) or result <= 0.0:
        raise DomainError("LBM-3 STABILITY: derived Reynolds must be finite and positive")
    return result


def preflight_lattice(config: LatticeConfig) -> DerivedLatticeConfig:
    """Derive and validate all RFC-0002 lattice stability quantities."""

    viscosity = kinematic_viscosity_lu(
        inlet_velocity_lu=config.inlet_velocity_lu,
        reference_diameter_lu=config.reference_diameter_lu,
        reynolds=config.reynolds,
    )
    tau = 3.0 * viscosity + 0.5
    minimum_tau = math.nextafter(MIN_TAU, -math.inf)
    maximum_tau = math.nextafter(MAX_TAU, math.inf)
    if tau < minimum_tau or tau > maximum_tau:
        raise DomainError(
            f"LBM-3 STABILITY: tau={tau:.17g} is outside [{MIN_TAU}, {MAX_TAU}]; "
            "increase resolution or change the requested case without clipping"
        )
    mach = config.inlet_velocity_lu / SPEED_OF_SOUND
    if mach > MAX_NOMINAL_MACH:
        raise DomainError(f"LBM-3 STABILITY: nominal Mach={mach:.17g} exceeds {MAX_NOMINAL_MACH}")
    omega = 1.0 / tau
    return DerivedLatticeConfig(
        nx=config.nx,
        ny=config.ny,
        steps=config.steps,
        warmup_steps=config.warmup_steps,
        sample_interval=config.sample_interval,
        inlet_velocity_lu=config.inlet_velocity_lu,
        reynolds=config.reynolds,
        reference_diameter_lu=config.reference_diameter_lu,
        kinematic_viscosity_lu=viscosity,
        tau=tau,
        omega=omega,
        nominal_mach=mach,
        rho_ref=RHO_REF,
        ramp_steps=min(MAX_RAMP_STEPS, config.warmup_steps),
    )


def derive_lattice(
    case: CaseConfig,
    *,
    sample_interval: int = DEFAULT_SAMPLE_INTERVAL,
) -> DerivedLatticeConfig:
    """Map a canonical case to its unscaled diameter and preflighted lattice."""

    return preflight_lattice(
        LatticeConfig(
            nx=case.nx,
            ny=case.ny,
            steps=case.steps,
            warmup_steps=case.warmup_steps,
            sample_interval=sample_interval,
            inlet_velocity_lu=case.inlet_velocity_lu,
            reynolds=case.reynolds,
            reference_diameter_lu=reference_diameter_lu(case.grid),
        )
    )


__all__ = [
    "CS2",
    "DEFAULT_SAMPLE_INTERVAL",
    "MAX_INLET_VELOCITY_LU",
    "MAX_NOMINAL_MACH",
    "MAX_RAMP_STEPS",
    "MAX_TAU",
    "MIN_TAU",
    "RHO_REF",
    "SPEED_OF_SOUND",
    "DerivedLatticeConfig",
    "LatticeConfig",
    "derive_lattice",
    "kinematic_viscosity_lu",
    "preflight_lattice",
    "relaxation_time",
    "reynolds_from_relaxation_time",
]
