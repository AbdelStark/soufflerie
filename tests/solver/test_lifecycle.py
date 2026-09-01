from __future__ import annotations

import importlib.util

import numpy as np
import numpy.typing as npt
import pytest

from soufflerie.errors import DomainError
from soufflerie.solver.lattice import (
    DerivedLatticeConfig,
    LatticeConfig,
    inlet_ramp,
    preflight_lattice,
)
from soufflerie.solver.lifecycle import (
    MIN_AVERAGING_SAMPLES,
    MIN_AVERAGING_STEPS,
    CompletedLatticeRun,
    NumpyPeriodicState,
    NumpyPeriodicStepper,
    StepDiagnostics,
    WarpPeriodicStepper,
    averaging_sample_steps,
    run_lifecycle,
)


def _config(
    *,
    steps: int = 4_001,
    warmup_steps: int = 1,
    sample_interval: int = 20,
) -> DerivedLatticeConfig:
    return preflight_lattice(
        LatticeConfig(
            nx=3,
            ny=3,
            steps=steps,
            warmup_steps=warmup_steps,
            sample_interval=sample_interval,
            inlet_velocity_lu=0.05,
            reynolds=100.0,
            reference_diameter_lu=32.0,
        )
    )


def _mask(config: DerivedLatticeConfig) -> npt.NDArray[np.bool_]:
    return np.zeros(config.shape, dtype=np.bool_)


class RecordingStepper(NumpyPeriodicStepper):
    def __init__(self) -> None:
        self.inlet_targets: list[float] = []

    def advance(
        self,
        state: object,
        config: DerivedLatticeConfig,
        mask: npt.NDArray[np.bool_],
        *,
        completed_step: int,
        inlet_velocity_lu: float,
    ) -> None:
        self.inlet_targets.append(inlet_velocity_lu)
        super().advance(
            state,
            config,
            mask,
            completed_step=completed_step,
            inlet_velocity_lu=inlet_velocity_lu,
        )


def test_averaging_schedule_accepts_exact_minimums_and_rejects_just_outside() -> None:
    config = _config()
    steps = averaging_sample_steps(config)
    assert config.steps - config.warmup_steps == MIN_AVERAGING_STEPS
    assert len(steps) == MIN_AVERAGING_SAMPLES
    assert steps[0] == 21
    assert steps[-1] == 4_001

    with pytest.raises(DomainError, match="4000 post-warmup steps"):
        averaging_sample_steps(_config(steps=4_000))
    with pytest.raises(DomainError, match="200 post-warmup samples"):
        averaging_sample_steps(_config(sample_interval=21))


def test_lifecycle_returns_validated_fp32_means_and_fp64_diagnostics() -> None:
    config = _config()
    progress: list[StepDiagnostics] = []
    result = run_lifecycle(config, _mask(config), progress=progress.append)

    assert isinstance(result, CompletedLatticeRun)
    assert result.config == config
    assert result.device_class == "cpu-numpy"
    assert result.mean_fields.shape == config.shape
    for field in (result.mean_fields.u, result.mean_fields.v, result.mean_fields.rho):
        assert field.dtype == np.float32
        assert field.flags.c_contiguous
        assert np.isfinite(field).all()
    np.testing.assert_allclose(result.mean_fields.u, 0.05, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(result.mean_fields.v, 0.0, rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(result.mean_fields.rho, 1.0, rtol=2e-6, atol=2e-7)
    assert result.diagnostics.valid
    assert result.diagnostics.converged
    assert result.diagnostics.steps_completed == config.steps
    assert result.diagnostics.sample_count == MIN_AVERAGING_SAMPLES
    assert result.diagnostics.mass_drift_ratio < 0.001
    assert isinstance(result.diagnostics.initial_mass, float)
    assert [item.step for item in progress] == [*range(20, 4_001, 20), 4_001]


def test_lifecycle_forwards_the_declared_half_cosine_ramp_to_every_step() -> None:
    config = _config(steps=6_000, warmup_steps=2_000)
    stepper = RecordingStepper()
    run_lifecycle(config, _mask(config), stepper=stepper)

    assert len(stepper.inlet_targets) == config.steps
    assert stepper.inlet_targets[0] == pytest.approx(
        config.inlet_velocity_lu * inlet_ramp(1, config.ramp_steps)
    )
    assert stepper.inlet_targets[1_999] == config.inlet_velocity_lu
    assert stepper.inlet_targets[-1] == config.inlet_velocity_lu
    assert stepper.inlet_targets == sorted(stepper.inlet_targets)


def test_lifecycle_rejects_mask_shape_and_dtype_before_execution() -> None:
    config = _config()
    with pytest.raises(DomainError, match="solver mask"):
        run_lifecycle(config, np.zeros((2, 2), dtype=np.bool_))
    with pytest.raises(DomainError, match="solver mask"):
        run_lifecycle(config, np.zeros(config.shape, dtype=np.uint8))


def test_warp_periodic_lifecycle_matches_numpy_means_on_cpu() -> None:
    if importlib.util.find_spec("warp") is None:
        pytest.skip("install the Soufflerie 'solver' extra to run Warp lifecycle integration")
    config = _config()
    mask = _mask(config)
    expected = run_lifecycle(config, mask)
    actual = run_lifecycle(config, mask, stepper=WarpPeriodicStepper("cpu"))

    np.testing.assert_allclose(actual.mean_fields.u, expected.mean_fields.u, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(actual.mean_fields.v, expected.mean_fields.v, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        actual.mean_fields.rho,
        expected.mean_fields.rho,
        rtol=2e-6,
        atol=2e-7,
    )
    assert actual.diagnostics.mass_drift_ratio < 0.001


def test_density_and_speed_endpoints_are_inclusive_in_fp32_state() -> None:
    config = _config()

    class EndpointStepper(NumpyPeriodicStepper):
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
            if completed_step == 20:
                assert isinstance(state, NumpyPeriodicState)
                state.rho[0, 0] = np.float32(0.5)
                state.rho[0, 1] = np.float32(1.5)
                state.velocity[0, 2, 0] = np.float32(0.2)

    result = run_lifecycle(config, _mask(config), stepper=EndpointStepper())
    assert result.diagnostics.min_rho == 0.5
    assert result.diagnostics.max_rho == 1.5
    assert result.diagnostics.max_speed_lu == float(np.float32(0.2))
