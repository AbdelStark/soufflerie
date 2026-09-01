from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from soufflerie.errors import DomainError, NumericalStabilityError
from soufflerie.schemas import CaseConfig, ShapeParams
from soufflerie.solver.lattice import (
    CS2,
    D2Q9_OPPOSITE,
    D2Q9_VELOCITIES,
    D2Q9_WEIGHTS,
    MAX_TAU,
    MIN_TAU,
    DerivedLatticeConfig,
    LatticeConfig,
    derive_lattice,
    equilibrium,
    inlet_ramp,
    macroscopic_moments,
    preflight_lattice,
    relaxation_time,
    reynolds_from_relaxation_time,
)


def _lattice(**overrides: object) -> LatticeConfig:
    values: dict[str, object] = {
        "nx": 8,
        "ny": 6,
        "steps": 100,
        "warmup_steps": 20,
        "sample_interval": 10,
        "inlet_velocity_lu": 0.05,
        "reynolds": 100.0,
        "reference_diameter_lu": 32.0,
    }
    values.update(overrides)
    return LatticeConfig(**values)  # type: ignore[arg-type]


def _for_tau(tau: float) -> LatticeConfig:
    inlet = 0.05
    diameter = 32.0
    reynolds = reynolds_from_relaxation_time(
        tau=tau,
        inlet_velocity_lu=inlet,
        reference_diameter_lu=diameter,
    )
    return _lattice(
        inlet_velocity_lu=inlet,
        reference_diameter_lu=diameter,
        reynolds=reynolds,
    )


def test_d2q9_constants_match_the_rfc_and_are_immutable() -> None:
    np.testing.assert_array_equal(
        D2Q9_VELOCITIES,
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
        ),
    )
    np.testing.assert_allclose(
        D2Q9_WEIGHTS,
        np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9] + [1 / 36] * 4),
        rtol=0.0,
        atol=2e-8,
    )
    np.testing.assert_array_equal(D2Q9_OPPOSITE, [0, 3, 4, 1, 2, 7, 8, 5, 6])
    np.testing.assert_array_equal(D2Q9_OPPOSITE[D2Q9_OPPOSITE], np.arange(9))
    assert math.isclose(CS2, 1 / 3)
    assert not D2Q9_VELOCITIES.flags.writeable
    assert not D2Q9_WEIGHTS.flags.writeable
    assert not D2Q9_OPPOSITE.flags.writeable
    with pytest.raises(ValueError):
        D2Q9_WEIGHTS[0] = np.float32(0.0)


def test_rest_equilibrium_recovers_density_and_zero_momentum() -> None:
    rho = np.array([[0.75, 1.0, 1.25], [1.1, 0.9, 1.0]], dtype=np.float32)
    velocity = np.zeros((*rho.shape, 2), dtype=np.float32)
    populations = equilibrium(rho, velocity)

    np.testing.assert_allclose(populations, rho[..., None] * D2Q9_WEIGHTS, rtol=1e-7)
    recovered_rho, recovered_velocity = macroscopic_moments(populations)
    np.testing.assert_allclose(recovered_rho, rho, rtol=3e-7, atol=3e-7)
    np.testing.assert_allclose(recovered_velocity, 0.0, rtol=0.0, atol=1e-8)
    assert populations.dtype == np.float32
    assert populations.flags.c_contiguous


@pytest.mark.parametrize(
    ("u", "v"),
    [(0.05, 0.0), (-0.03, 0.02), (0.001, -0.002), (0.08, 0.04)],
)
def test_low_velocity_equilibrium_recovers_declared_moments(u: float, v: float) -> None:
    rho = np.full((3, 4), 1.07, dtype=np.float32)
    velocity = np.empty((3, 4, 2), dtype=np.float32)
    velocity[..., 0] = np.float32(u)
    velocity[..., 1] = np.float32(v)

    populations = equilibrium(rho, velocity)
    recovered_rho, recovered_velocity = macroscopic_moments(populations)

    assert np.isfinite(populations).all()
    np.testing.assert_allclose(recovered_rho, rho, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(recovered_velocity, velocity, rtol=2e-6, atol=2e-7)


def test_tau_reynolds_mapping_round_trips_without_clipping() -> None:
    tau = relaxation_time(
        inlet_velocity_lu=0.05,
        reference_diameter_lu=32.0,
        reynolds=100.0,
    )
    assert tau == pytest.approx(0.548)
    assert reynolds_from_relaxation_time(
        tau=tau,
        inlet_velocity_lu=0.05,
        reference_diameter_lu=32.0,
    ) == pytest.approx(100.0)
    with pytest.raises(DomainError, match="tau must be greater"):
        reynolds_from_relaxation_time(
            tau=0.5,
            inlet_velocity_lu=0.05,
            reference_diameter_lu=32.0,
        )


@pytest.mark.parametrize("tau", [MIN_TAU, MIN_TAU + 1e-7, MAX_TAU - 1e-7, MAX_TAU])
def test_tau_preflight_accepts_each_boundary_and_just_inside(tau: float) -> None:
    derived = preflight_lattice(_for_tau(tau))
    assert derived.tau == pytest.approx(tau, rel=0.0, abs=3e-16)
    assert derived.omega == pytest.approx(1.0 / tau)


@pytest.mark.parametrize("tau", [MIN_TAU - 1e-7, MAX_TAU + 1e-7])
def test_tau_preflight_rejects_just_outside_without_clipping(tau: float) -> None:
    with pytest.raises(DomainError, match="outside"):
        preflight_lattice(_for_tau(tau))


def test_velocity_grid_and_schedule_boundaries_are_strict() -> None:
    assert preflight_lattice(_lattice(inlet_velocity_lu=0.1)).nominal_mach <= 0.1733
    assert _lattice(nx=3, ny=3).shape == (3, 3)
    assert _lattice(steps=2, warmup_steps=1, sample_interval=1).steps == 2

    invalid_cases: list[tuple[dict[str, object], str]] = [
        ({"inlet_velocity_lu": math.nextafter(0.1, math.inf)}, "inlet_velocity_lu"),
        ({"inlet_velocity_lu": math.nextafter(0.0, math.inf)}, "viscosity"),
        ({"nx": 2}, "nx"),
        ({"ny": 2}, "ny"),
        ({"steps": 0}, "steps"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"warmup_steps": 100}, "warmup_steps"),
        ({"sample_interval": 0}, "sample_interval"),
        ({"sample_interval": 81}, "sample_interval"),
        ({"reynolds": 0.0}, "reynolds"),
        ({"reference_diameter_lu": 0.0}, "reference_diameter_lu"),
    ]
    for overrides, message in invalid_cases:
        with pytest.raises(DomainError, match=message):
            preflight_lattice(_lattice(**overrides))

    with pytest.raises(DomainError, match="nx"):
        _lattice(nx=True)
    with pytest.raises(DomainError, match="finite"):
        _lattice(reynolds=math.inf)


def test_case_derivation_uses_canonical_reference_length_and_schedule() -> None:
    case = CaseConfig(
        shape=ShapeParams(aspect_ratio=1.0, rotation_deg=0.0, scale=1.0),
        reynolds=100.0,
        nx=512,
        ny=256,
        steps=20_000,
        warmup_steps=8_000,
        inlet_velocity_lu=0.05,
        seed=0,
    )
    derived = derive_lattice(case)

    assert isinstance(derived, DerivedLatticeConfig)
    assert derived.reference_diameter_lu == 32.0
    assert derived.kinematic_viscosity_lu == pytest.approx(0.016)
    assert derived.tau == pytest.approx(0.548)
    assert derived.rho_ref == 1.0
    assert derived.sample_interval == 10
    assert derived.ramp_steps == 2_000
    assert derived.base == _lattice(nx=512, ny=256, steps=20_000, warmup_steps=8_000)
    with pytest.raises(DomainError, match="derived tau"):
        replace(derived, tau=0.6)
    with pytest.raises(DomainError, match="rho_ref"):
        replace(derived, rho_ref=0.9)


def test_inlet_ramp_has_exact_endpoints_and_monotonic_interior() -> None:
    values = [inlet_ramp(step, 10) for step in range(11)]
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert values == sorted(values)
    assert inlet_ramp(4, 0) == 1.0
    assert inlet_ramp(11, 10) == 1.0
    with pytest.raises(DomainError, match="step"):
        inlet_ramp(-1, 10)


def test_moment_boundaries_reject_wrong_precision_shape_and_nonfinite_state() -> None:
    rho = np.ones((2, 2), dtype=np.float32)
    velocity = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(NumericalStabilityError, match="fp32"):
        equilibrium(rho.astype(np.float64), velocity)
    with pytest.raises(NumericalStabilityError, match="strictly positive"):
        equilibrium(np.zeros((2, 2), dtype=np.float32), velocity)
    with pytest.raises(NumericalStabilityError, match="shape"):
        macroscopic_moments(np.ones((2, 2, 8), dtype=np.float32))
    invalid = equilibrium(rho, velocity)
    invalid[0, 0, 0] = np.nan
    with pytest.raises(NumericalStabilityError, match="NaN"):
        macroscopic_moments(invalid)
