from __future__ import annotations

import math

import numpy as np
import pytest

from soufflerie.errors import DomainError
from soufflerie.solver.poiseuille import (
    POISEUILLE_ERROR_THRESHOLD,
    PoiseuilleFixture,
    PoiseuilleResult,
    analytic_poiseuille_profile,
    poiseuille_gate_passes,
    run_poiseuille_fixture,
)


@pytest.fixture(scope="module")
def channel_results() -> tuple[PoiseuilleResult, PoiseuilleResult]:
    return (
        run_poiseuille_fixture(PoiseuilleFixture(channel_height=16, width=2, steps=4_000)),
        run_poiseuille_fixture(PoiseuilleFixture(channel_height=32, width=2, steps=12_000)),
    )


def test_analytic_profile_uses_half_way_walls_and_is_symmetric() -> None:
    fixture = PoiseuilleFixture(channel_height=16, width=2, steps=1)
    profile = analytic_poiseuille_profile(fixture)

    assert profile.dtype == np.float64
    np.testing.assert_allclose(profile, profile[::-1], rtol=0.0, atol=1e-17)
    assert np.argmax(profile) in {7, 8}
    assert profile[0] > 0.0
    assert profile[0] < profile[1]


def test_two_grid_sizes_pass_both_one_percent_gates_and_converge(
    channel_results: tuple[PoiseuilleResult, PoiseuilleResult],
) -> None:
    coarse, fine = channel_results
    for result in channel_results:
        assert result.passed
        assert result.relative_l2_error <= POISEUILLE_ERROR_THRESHOLD
        assert result.max_relative_error <= POISEUILLE_ERROR_THRESHOLD
        assert result.mass_drift_ratio < 0.001
        assert result.measured_profile.shape == (result.fixture.channel_height,)
        np.testing.assert_allclose(
            result.measured_profile[1:-1],
            result.analytic_profile[1:-1],
            rtol=POISEUILLE_ERROR_THRESHOLD,
            atol=2e-6,
        )
    assert fine.fixture.channel_height == 2 * coarse.fixture.channel_height
    assert fine.relative_l2_error < coarse.relative_l2_error
    assert fine.max_relative_error < coarse.max_relative_error


def test_poiseuille_gate_threshold_is_inclusive_and_nonfinite_is_red() -> None:
    threshold = POISEUILLE_ERROR_THRESHOLD
    assert poiseuille_gate_passes(threshold, threshold)
    assert not poiseuille_gate_passes(math.nextafter(threshold, math.inf), threshold)
    assert not poiseuille_gate_passes(threshold, math.nextafter(threshold, math.inf))
    assert not poiseuille_gate_passes(math.nan, 0.0)
    assert not poiseuille_gate_passes(0.0, math.inf)


def test_fixture_preflight_rejects_unresolved_or_unstable_channels() -> None:
    with pytest.raises(DomainError, match="channel_height"):
        PoiseuilleFixture(channel_height=5, width=2, steps=100)
    with pytest.raises(DomainError, match="tau"):
        PoiseuilleFixture(channel_height=16, width=2, steps=100, tau=0.5)
    with pytest.raises(DomainError, match="excludes exactly one"):
        PoiseuilleFixture(
            channel_height=16,
            width=2,
            steps=100,
            excluded_wall_cells=0,
        )
