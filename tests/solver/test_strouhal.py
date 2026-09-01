from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pytest

import soufflerie.solver as solver_public
from soufflerie.errors import DomainError
from soufflerie.solver.diagnostics import (
    StrouhalUnavailableReason,
    estimate_strouhal,
    mean_force_coefficients,
)
from soufflerie.solver.forces import ObstacleForceHistory
from soufflerie.solver.lattice import DerivedLatticeConfig, LatticeConfig, preflight_lattice

ScalarT = TypeVar("ScalarT", bound=np.generic)


def _readonly(array: npt.NDArray[ScalarT]) -> npt.NDArray[ScalarT]:
    result = np.ascontiguousarray(array)
    result.flags.writeable = False
    return result


def _history(steps: npt.NDArray[np.int64], cl: npt.NDArray[np.float64]) -> ObstacleForceHistory:
    count = steps.size
    return ObstacleForceHistory(
        steps=_readonly(np.asarray(steps, dtype=np.int64)),
        fx_lu=_readonly(np.zeros(count, dtype=np.float64)),
        fy_lu=_readonly(np.zeros(count, dtype=np.float64)),
        cd=_readonly(np.linspace(0.0, 1.0, count, dtype=np.float32)),
        cl=_readonly(np.asarray(cl, dtype=np.float32)),
    )


def _config() -> DerivedLatticeConfig:
    return preflight_lattice(
        LatticeConfig(
            nx=80,
            ny=32,
            steps=20_000,
            warmup_steps=8_000,
            sample_interval=10,
            inlet_velocity_lu=0.05,
            reynolds=100.0,
            reference_diameter_lu=4.0,
        )
    )


def test_off_bin_noisy_lift_recovers_strouhal_with_parabolic_interpolation() -> None:
    config = _config()
    target_strouhal = 0.173
    target_frequency = target_strouhal * config.inlet_velocity_lu / config.reference_diameter_lu
    steps = np.arange(8_010, 17_010, 10, dtype=np.int64)
    rng = np.random.default_rng(20260901)
    lift = np.sin(2.0 * np.pi * target_frequency * steps.astype(np.float64))
    lift += 0.02 * rng.standard_normal(steps.size)

    estimate = estimate_strouhal(_history(steps, lift), config)

    assert estimate.available
    assert estimate.reason is None
    assert estimate.strouhal == pytest.approx(target_strouhal, rel=0.005)
    assert estimate.frequency_lu == pytest.approx(target_frequency, rel=0.005)
    assert estimate.resolved_cycles >= 8.0
    assert estimate.sample_interval_steps == 10


def test_zero_lift_returns_none_with_typed_reason() -> None:
    config = _config()
    steps = np.arange(8_010, 17_010, 10, dtype=np.int64)

    estimate = estimate_strouhal(_history(steps, np.zeros(steps.size)), config)

    assert estimate.strouhal is None
    assert estimate.reason is StrouhalUnavailableReason.DEGENERATE_LIFT
    assert not estimate.available


def test_resolved_peak_with_fewer_than_eight_cycles_is_unavailable() -> None:
    config = _config()
    target_frequency = 0.17 * config.inlet_velocity_lu / config.reference_diameter_lu
    steps = np.arange(8_010, 10_010, 10, dtype=np.int64)
    lift = np.sin(2.0 * np.pi * target_frequency * steps.astype(np.float64))

    estimate = estimate_strouhal(_history(steps, lift), config)

    assert estimate.strouhal is None
    assert estimate.reason is StrouhalUnavailableReason.INSUFFICIENT_CYCLES
    assert estimate.frequency_lu is not None
    assert estimate.resolved_cycles < 8.0


def test_short_and_irregular_histories_return_typed_unavailable_results() -> None:
    config = _config()
    short_steps = np.array([8_010, 8_020, 8_030], dtype=np.int64)
    short = estimate_strouhal(_history(short_steps, np.array([0.0, 1.0, 0.0])), config)
    assert short.reason is StrouhalUnavailableReason.INSUFFICIENT_SAMPLES

    irregular_steps = np.array([8_010, 8_020, 8_040, 8_050], dtype=np.int64)
    irregular = estimate_strouhal(
        _history(irregular_steps, np.array([0.0, 1.0, 0.0, -1.0])), config
    )
    assert irregular.reason is StrouhalUnavailableReason.IRREGULAR_SAMPLING


def test_unresolvable_band_and_zero_windowed_energy_have_distinct_reasons() -> None:
    config = _config()
    four_steps = np.arange(8_010, 8_050, 10, dtype=np.int64)
    unresolved = estimate_strouhal(_history(four_steps, np.array([0.0, 1.0, 0.0, -1.0])), config)
    assert unresolved.reason is StrouhalUnavailableReason.NO_RESOLVABLE_BAND

    twenty_steps = np.arange(8_010, 8_210, 10, dtype=np.int64)
    endpoint_only_lift = np.ones(twenty_steps.size, dtype=np.float64)
    endpoint_only_lift[0] = 0.0
    endpoint_only_lift[-1] = 2.0
    no_peak = estimate_strouhal(_history(twenty_steps, endpoint_only_lift), config)
    assert no_peak.reason is StrouhalUnavailableReason.NO_SPECTRAL_PEAK


def test_history_outside_run_schedule_is_rejected() -> None:
    config = _config()
    steps = np.array([7_990, 8_000, 8_010, 8_020], dtype=np.int64)
    with pytest.raises(DomainError, match="post-warmup run schedule"):
        estimate_strouhal(_history(steps, np.ones(steps.size)), config)


def test_force_means_use_only_the_right_closed_last_four_thousand_steps() -> None:
    steps = np.arange(1_000, 7_000, 1_000, dtype=np.int64)
    history = _history(steps, np.arange(1.0, 7.0, dtype=np.float64))
    cd = np.arange(1.0, 7.0, dtype=np.float32)
    cd.flags.writeable = False
    history = ObstacleForceHistory(
        steps=history.steps,
        fx_lu=history.fx_lu,
        fy_lu=history.fy_lu,
        cd=cd,
        cl=history.cl,
    )

    summary = mean_force_coefficients(history)

    assert summary.sample_count == 4
    assert summary.first_step == 3_000
    assert summary.last_step == 6_000
    assert summary.cd == pytest.approx(4.5)
    assert summary.cl == pytest.approx(4.5)


def test_force_mean_rejects_empty_history_and_invalid_window() -> None:
    empty = _history(np.array([], dtype=np.int64), np.array([], dtype=np.float64))
    with pytest.raises(DomainError, match="contains no samples"):
        mean_force_coefficients(empty)
    with pytest.raises(DomainError, match="positive integer"):
        mean_force_coefficients(empty, window_steps=0)


def test_diagnostic_contract_is_exported_from_the_solver_package() -> None:
    assert solver_public.estimate_strouhal is estimate_strouhal
    assert solver_public.mean_force_coefficients is mean_force_coefficients
    assert solver_public.StrouhalUnavailableReason is StrouhalUnavailableReason
