from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from soufflerie.errors import NonConvergenceError, NumericalStabilityError
from soufflerie.solver.lattice import DerivedLatticeConfig, LatticeConfig, preflight_lattice
from soufflerie.solver.lifecycle import (
    FailedSolverRun,
    NumpyPeriodicState,
    NumpyPeriodicStepper,
    SolverConvergenceFailure,
    SolverStabilityFailure,
    run_lifecycle,
)


def _config() -> DerivedLatticeConfig:
    return preflight_lattice(
        LatticeConfig(
            nx=3,
            ny=3,
            steps=4_001,
            warmup_steps=1,
            sample_interval=20,
            inlet_velocity_lu=0.05,
            reynolds=100.0,
            reference_diameter_lu=32.0,
        )
    )


Mutation = Callable[[NumpyPeriodicState], None]


class InjectingStepper(NumpyPeriodicStepper):
    def __init__(self, *, inject_at: int, mutation: Mutation) -> None:
        self.inject_at = inject_at
        self.mutation = mutation
        self.steps_advanced = 0

    def advance(
        self,
        state: object,
        config: DerivedLatticeConfig,
        mask: npt.NDArray[np.bool_],
        *,
        completed_step: int,
        inlet_velocity_lu: float,
    ) -> None:
        super().advance(
            state,
            config,
            mask,
            completed_step=completed_step,
            inlet_velocity_lu=inlet_velocity_lu,
        )
        self.steps_advanced = completed_step
        if completed_step == self.inject_at:
            assert isinstance(state, NumpyPeriodicState)
            self.mutation(state)


def _set_nonfinite(state: NumpyPeriodicState) -> None:
    state.f[0, 0, 0] = np.float32(np.nan)


def _set_low_density(state: NumpyPeriodicState) -> None:
    state.rho[0, 0] = np.nextafter(np.float32(0.5), np.float32(0.0))


def _set_high_density(state: NumpyPeriodicState) -> None:
    state.rho[0, 0] = np.nextafter(np.float32(1.5), np.float32(np.inf))


def _set_high_speed(state: NumpyPeriodicState) -> None:
    state.velocity[0, 0, 0] = np.nextafter(np.float32(0.2), np.float32(np.inf))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_set_nonfinite, "NaN or infinity"),
        (_set_low_density, "rho is outside"),
        (_set_high_density, "rho is outside"),
        (_set_high_speed, "maximum speed exceeds"),
    ],
)
def test_runtime_instability_stops_at_first_diagnostic_and_returns_no_result(
    mutation: Mutation, message: str
) -> None:
    config = _config()
    stepper = InjectingStepper(inject_at=20, mutation=mutation)

    with pytest.raises(SolverStabilityFailure, match=message) as captured:
        run_lifecycle(config, np.zeros(config.shape, dtype=np.bool_), stepper=stepper)

    error = captured.value
    assert isinstance(error, NumericalStabilityError)
    assert not error.retryable
    assert stepper.steps_advanced == 20
    assert isinstance(error.failed_run, FailedSolverRun)
    assert error.failed_run.steps_completed == 20
    assert error.failed_run.error_code == "SOLVER_UNSTABLE"
    assert error.failed_run.diagnostics is None
    assert not hasattr(error.failed_run, "mean_fields")


def test_mass_drift_fails_only_after_run_with_invalid_diagnostics() -> None:
    config = _config()

    def drift(state: NumpyPeriodicState) -> None:
        state.rho *= np.float32(1.002)

    stepper = InjectingStepper(inject_at=config.steps, mutation=drift)
    with pytest.raises(SolverConvergenceFailure, match="mass drift") as captured:
        run_lifecycle(config, np.zeros(config.shape, dtype=np.bool_), stepper=stepper)

    error = captured.value
    assert isinstance(error, NonConvergenceError)
    assert not error.retryable
    assert stepper.steps_advanced == config.steps
    assert error.failed_run.steps_completed == config.steps
    assert error.failed_run.sample_count == 200
    assert error.failed_run.error_code == "SOLVER_NOT_CONVERGED"
    assert error.failed_run.diagnostics is not None
    assert not error.failed_run.diagnostics.valid
    assert not error.failed_run.diagnostics.converged
    assert error.failed_run.diagnostics.mass_drift_ratio >= 0.001
    assert not hasattr(error.failed_run, "mean_fields")
