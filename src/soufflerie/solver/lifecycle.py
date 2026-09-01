"""Solver lifecycle, runtime diagnostics, and deterministic time averaging."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from soufflerie.errors import (
    DomainError,
    InternalInvariantError,
    NonConvergenceError,
    NumericalStabilityError,
)
from soufflerie.schemas import SolverDiagnostics, validate_array
from soufflerie.solver.boundaries import numpy_channel_step
from soufflerie.solver.forces import (
    ForceHistoryRecorder,
    ObstacleForceHistory,
    ObstacleLinks,
    enumerate_obstacle_links,
)
from soufflerie.solver.kernels import LatticeState, WarpKernelAdapter
from soufflerie.solver.lattice import (
    DerivedLatticeConfig,
    inlet_ramp,
)
from soufflerie.solver.numpy_oracle import initialize_numpy, numpy_periodic_step
from soufflerie.solver.obstacle import numpy_obstacle_step

MIN_AVERAGING_STEPS = 4_000
MIN_AVERAGING_SAMPLES = 200
MIN_RUNTIME_RHO = 0.5
MAX_RUNTIME_RHO = 1.5
MAX_RUNTIME_SPEED_LU = 0.2
MAX_MASS_DRIFT_RATIO = 0.001


@dataclass(frozen=True, slots=True)
class RawLatticeSnapshot:
    """Detached host arrays that may contain an invalid numerical state."""

    f: npt.NDArray[np.float32]
    rho: npt.NDArray[np.float32]
    velocity: npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """One finite fp64-derived diagnostic boundary."""

    step: int
    mass: float
    mass_drift_ratio: float
    min_rho: float
    max_rho: float
    max_speed_lu: float
    inlet_ramp: float
    inlet_velocity_lu: float

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise InternalInvariantError("diagnostic step must be a nonnegative integer")
        scalars = (
            self.mass,
            self.mass_drift_ratio,
            self.min_rho,
            self.max_rho,
            self.max_speed_lu,
            self.inlet_ramp,
            self.inlet_velocity_lu,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise InternalInvariantError("diagnostic scalars must be finite")
        if self.mass <= 0.0 or self.mass_drift_ratio < 0.0:
            raise InternalInvariantError("diagnostic mass values are incoherent")
        if self.min_rho <= 0.0 or self.min_rho > self.max_rho:
            raise InternalInvariantError("diagnostic density bounds are incoherent")
        if self.max_speed_lu < 0.0:
            raise InternalInvariantError("diagnostic speed must be nonnegative")
        if not 0.0 <= self.inlet_ramp <= 1.0 or self.inlet_velocity_lu < 0.0:
            raise InternalInvariantError("diagnostic inlet ramp values are incoherent")


@dataclass(frozen=True, slots=True)
class MeanFlowFields:
    """Validated fp32 time means before geometry/result assembly."""

    u: npt.NDArray[np.float32]
    v: npt.NDArray[np.float32]
    rho: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        validate_array(self.u, name="u_mean", dtype=np.dtype(np.float32), ndim=2)
        validate_array(self.v, name="v_mean", dtype=np.dtype(np.float32), shape=self.u.shape)
        validate_array(
            self.rho,
            name="rho_mean",
            dtype=np.dtype(np.float32),
            shape=self.u.shape,
        )
        if not np.all(self.rho > np.float32(0.0)):
            raise InternalInvariantError("LBM-2 FINITE: mean rho must be strictly positive")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.u.shape[0], self.u.shape[1])


@dataclass(frozen=True, slots=True)
class CompletedLatticeRun:
    """Successful lifecycle output; geometry and force assembly occur later."""

    config: DerivedLatticeConfig
    mean_fields: MeanFlowFields
    diagnostics: SolverDiagnostics
    sample_steps: tuple[int, ...]
    history: tuple[StepDiagnostics, ...]
    device_class: str

    def __post_init__(self) -> None:
        if not self.diagnostics.valid or not self.diagnostics.converged:
            raise InternalInvariantError(
                "completed lattice run requires valid converged diagnostics"
            )
        if self.mean_fields.shape != self.config.shape:
            raise InternalInvariantError("mean-field grid must match the completed run config")
        if self.diagnostics.steps_completed != self.config.steps:
            raise InternalInvariantError("completed diagnostics must cover the full run schedule")
        if self.diagnostics.sample_count != len(self.sample_steps):
            raise InternalInvariantError(
                "diagnostic sample_count must match averaging sample_steps"
            )
        if self.sample_steps != averaging_sample_steps(self.config):
            raise InternalInvariantError("completed sample_steps must match the declared schedule")
        if any(step <= 0 for step in self.sample_steps) or any(
            right <= left
            for left, right in zip(self.sample_steps, self.sample_steps[1:], strict=False)
        ):
            raise InternalInvariantError("averaging sample_steps must be positive and increasing")
        if (
            not self.history
            or self.history[0].step != 0
            or self.history[-1].step != self.config.steps
        ):
            raise InternalInvariantError(
                "diagnostic history must span initialization through completion"
            )
        if not self.device_class:
            raise InternalInvariantError("device_class must not be empty")


@dataclass(frozen=True, slots=True)
class FailedSolverRun:
    """Scalar-only failure evidence that cannot masquerade as a completed run."""

    config: DerivedLatticeConfig
    steps_completed: int
    sample_count: int
    error_code: str
    reason: str
    last_valid: StepDiagnostics | None
    diagnostics: SolverDiagnostics | None = None

    def __post_init__(self) -> None:
        if self.steps_completed < 0 or self.steps_completed > self.config.steps:
            raise InternalInvariantError("failed-run steps_completed is outside the run schedule")
        if self.sample_count < 0:
            raise InternalInvariantError("failed-run sample_count must be nonnegative")
        if not self.error_code or not self.reason:
            raise InternalInvariantError("failed-run error code and reason are required")
        if self.diagnostics is not None and self.diagnostics.valid:
            raise InternalInvariantError("failed-run diagnostics must never be valid")


class SolverStabilityFailure(NumericalStabilityError):
    """Immediate runtime stability failure with safe scalar evidence."""

    def __init__(self, message: str, *, failed_run: FailedSolverRun) -> None:
        self.failed_run = failed_run
        super().__init__(message)


class SolverConvergenceFailure(NonConvergenceError):
    """Post-run mass/convergence failure with safe scalar evidence."""

    def __init__(self, message: str, *, failed_run: FailedSolverRun) -> None:
        self.failed_run = failed_run
        super().__init__(message)


class LifecycleStepper(Protocol):
    """Execution boundary consumed by the framework-independent lifecycle."""

    device_class: str

    def initialize(self, config: DerivedLatticeConfig, mask: npt.NDArray[np.bool_]) -> object: ...

    def advance(
        self,
        state: object,
        config: DerivedLatticeConfig,
        mask: npt.NDArray[np.bool_],
        *,
        completed_step: int,
        inlet_velocity_lu: float,
    ) -> None: ...

    def snapshot(self, state: object) -> RawLatticeSnapshot: ...


@dataclass(slots=True)
class NumpyPeriodicState:
    f: npt.NDArray[np.float32]
    rho: npt.NDArray[np.float32]
    velocity: npt.NDArray[np.float32]


class NumpyPeriodicStepper:
    """Lightweight periodic lifecycle driver used for CPU contract evidence."""

    device_class = "cpu-numpy"

    def initialize(self, config: DerivedLatticeConfig, mask: npt.NDArray[np.bool_]) -> object:
        initial = initialize_numpy(config, mask)
        return NumpyPeriodicState(initial.f.copy(), initial.rho.copy(), initial.velocity.copy())

    def advance(
        self,
        state: object,
        config: DerivedLatticeConfig,
        mask: npt.NDArray[np.bool_],
        *,
        completed_step: int,
        inlet_velocity_lu: float,
    ) -> None:
        del mask, completed_step, inlet_velocity_lu
        if not isinstance(state, NumpyPeriodicState):
            raise InternalInvariantError("NumPy lifecycle received an incompatible state")
        next_state = numpy_periodic_step(state.f, config)
        state.f = next_state.f
        state.rho = next_state.rho
        state.velocity = next_state.velocity

    def snapshot(self, state: object) -> RawLatticeSnapshot:
        if not isinstance(state, NumpyPeriodicState):
            raise InternalInvariantError("NumPy lifecycle received an incompatible state")
        return RawLatticeSnapshot(
            f=np.array(state.f, dtype=np.float32, order="C", copy=True),
            rho=np.array(state.rho, dtype=np.float32, order="C", copy=True),
            velocity=np.array(state.velocity, dtype=np.float32, order="C", copy=True),
        )


class NumpyChannelStepper(NumpyPeriodicStepper):
    """NumPy channel driver with ramped inlet, walls, outlet, and sponge."""

    def initialize(self, config: DerivedLatticeConfig, mask: npt.NDArray[np.bool_]) -> object:
        _validate_mask(mask, config.shape)
        channel_mask = mask.copy()
        channel_mask[0, :] = True
        channel_mask[-1, :] = True
        initial = initialize_numpy(config, channel_mask)
        return NumpyPeriodicState(initial.f.copy(), initial.rho.copy(), initial.velocity.copy())

    def advance(
        self,
        state: object,
        config: DerivedLatticeConfig,
        mask: npt.NDArray[np.bool_],
        *,
        completed_step: int,
        inlet_velocity_lu: float,
    ) -> None:
        del completed_step
        if not isinstance(state, NumpyPeriodicState):
            raise InternalInvariantError("NumPy channel lifecycle received an incompatible state")
        next_state = numpy_channel_step(
            state.f,
            config,
            mask,
            inlet_velocity_lu=inlet_velocity_lu,
        )
        state.f = next_state.f
        state.rho = next_state.rho
        state.velocity = next_state.velocity


class NumpyObstacleStepper(NumpyChannelStepper):
    """NumPy channel driver with obstacle bounce-back and sampled force history."""

    def __init__(self) -> None:
        self._links: ObstacleLinks | None = None
        self._history = ForceHistoryRecorder()

    def initialize(self, config: DerivedLatticeConfig, mask: npt.NDArray[np.bool_]) -> object:
        self._links = enumerate_obstacle_links(mask)
        self._history = ForceHistoryRecorder()
        return super().initialize(config, mask)

    def advance(
        self,
        state: object,
        config: DerivedLatticeConfig,
        mask: npt.NDArray[np.bool_],
        *,
        completed_step: int,
        inlet_velocity_lu: float,
    ) -> None:
        if not isinstance(state, NumpyPeriodicState) or self._links is None:
            raise InternalInvariantError("NumPy obstacle lifecycle is not initialized")
        result = numpy_obstacle_step(
            state.f,
            config,
            mask,
            self._links,
            inlet_velocity_lu=inlet_velocity_lu,
        )
        state.f = result.state.f
        state.rho = result.state.rho
        state.velocity = result.state.velocity
        if (
            completed_step > config.warmup_steps
            and (completed_step - config.warmup_steps) % config.sample_interval == 0
        ):
            self._history.record(completed_step, result.force)

    def force_history(self) -> ObstacleForceHistory:
        return self._history.snapshot()


class WarpPeriodicStepper:
    """Optional Warp driver for the periodic kernel stage sequence."""

    def __init__(self, device: str = "cpu") -> None:
        self._adapter = WarpKernelAdapter(device)
        self.device_class = self._adapter.device

    def initialize(self, config: DerivedLatticeConfig, mask: npt.NDArray[np.bool_]) -> object:
        return self._adapter.initialize(config, mask)

    def advance(
        self,
        state: object,
        config: DerivedLatticeConfig,
        mask: npt.NDArray[np.bool_],
        *,
        completed_step: int,
        inlet_velocity_lu: float,
    ) -> None:
        del mask, completed_step, inlet_velocity_lu
        if not isinstance(state, LatticeState):
            raise InternalInvariantError("Warp lifecycle received an incompatible state")
        self._adapter.step(state, config)

    def snapshot(self, state: object) -> RawLatticeSnapshot:
        if not isinstance(state, LatticeState):
            raise InternalInvariantError("Warp lifecycle received an incompatible state")
        self._adapter.synchronize()
        return RawLatticeSnapshot(
            f=np.array(state.f.numpy(), dtype=np.float32, order="C", copy=True),
            rho=np.array(state.rho.numpy(), dtype=np.float32, order="C", copy=True),
            velocity=np.array(state.velocity.numpy(), dtype=np.float32, order="C", copy=True),
        )


class WarpChannelStepper(WarpPeriodicStepper):
    """Warp channel driver matching the NumPy boundary-stage oracle."""

    def initialize(self, config: DerivedLatticeConfig, mask: npt.NDArray[np.bool_]) -> object:
        _validate_mask(mask, config.shape)
        channel_mask = mask.copy()
        channel_mask[0, :] = True
        channel_mask[-1, :] = True
        return self._adapter.initialize(config, channel_mask)

    def advance(
        self,
        state: object,
        config: DerivedLatticeConfig,
        mask: npt.NDArray[np.bool_],
        *,
        completed_step: int,
        inlet_velocity_lu: float,
    ) -> None:
        del completed_step
        if not isinstance(state, LatticeState):
            raise InternalInvariantError("Warp channel lifecycle received an incompatible state")
        self._adapter.step_channel(
            state,
            config,
            mask,
            inlet_velocity_lu=inlet_velocity_lu,
        )


class WarpObstacleStepper(WarpChannelStepper):
    """Warp obstacle driver with deterministic host-ordered force reduction."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)
        self._links: ObstacleLinks | None = None
        self._history = ForceHistoryRecorder()

    def initialize(self, config: DerivedLatticeConfig, mask: npt.NDArray[np.bool_]) -> object:
        self._links = enumerate_obstacle_links(mask)
        self._history = ForceHistoryRecorder()
        return super().initialize(config, mask)

    def advance(
        self,
        state: object,
        config: DerivedLatticeConfig,
        mask: npt.NDArray[np.bool_],
        *,
        completed_step: int,
        inlet_velocity_lu: float,
    ) -> None:
        if not isinstance(state, LatticeState) or self._links is None:
            raise InternalInvariantError("Warp obstacle lifecycle is not initialized")
        sample_due = (
            completed_step > config.warmup_steps
            and (completed_step - config.warmup_steps) % config.sample_interval == 0
        )
        force = self._adapter.step_obstacle(
            state,
            config,
            mask,
            self._links,
            inlet_velocity_lu=inlet_velocity_lu,
            measure_force=sample_due,
        )
        if sample_due:
            if force is None:
                raise InternalInvariantError("Warp force sample was not reduced")
            self._history.record(completed_step, force)

    def force_history(self) -> ObstacleForceHistory:
        return self._history.snapshot()


def averaging_sample_steps(config: DerivedLatticeConfig) -> tuple[int, ...]:
    """Validate and return the fixed post-warmup averaging schedule."""

    window_steps = config.steps - config.warmup_steps
    if window_steps < MIN_AVERAGING_STEPS:
        raise DomainError(
            f"LBM averaging requires at least {MIN_AVERAGING_STEPS} post-warmup steps"
        )
    sample_count = window_steps // config.sample_interval
    if sample_count < MIN_AVERAGING_SAMPLES:
        raise DomainError(
            f"LBM averaging requires at least {MIN_AVERAGING_SAMPLES} post-warmup samples"
        )
    return tuple(
        config.warmup_steps + index * config.sample_interval for index in range(1, sample_count + 1)
    )


def _validate_mask(mask: npt.NDArray[np.bool_], shape: tuple[int, int]) -> None:
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.shape != shape
        or not mask.flags.c_contiguous
    ):
        raise DomainError("solver mask must be C-contiguous bool with config grid shape")


def _validate_snapshot_layout(snapshot: RawLatticeSnapshot, shape: tuple[int, int]) -> None:
    expected = (
        ("populations", snapshot.f, (*shape, 9)),
        ("rho", snapshot.rho, shape),
        ("velocity", snapshot.velocity, (*shape, 2)),
    )
    for name, array, target in expected:
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.float32
            or array.shape != target
            or not array.flags.c_contiguous
        ):
            raise InternalInvariantError(
                f"LBM-6 PRECISION: {name} must be C-contiguous fp32 with shape {target}"
            )


def _inspect_snapshot(
    snapshot: RawLatticeSnapshot,
    config: DerivedLatticeConfig,
    *,
    step: int,
    initial_mass: float | None,
) -> StepDiagnostics:
    _validate_snapshot_layout(snapshot, config.shape)
    for name, array in (
        ("populations", snapshot.f),
        ("rho", snapshot.rho),
        ("velocity", snapshot.velocity),
    ):
        if not np.isfinite(array).all():
            raise NumericalStabilityError(
                f"LBM-2 FINITE: {name} contains NaN or infinity at step {step}"
            )
    min_rho = float(np.min(snapshot.rho))
    max_rho = float(np.max(snapshot.rho))
    velocity64 = snapshot.velocity.astype(np.float64)
    max_speed = float(np.max(np.hypot(velocity64[..., 0], velocity64[..., 1])))
    mass = float(np.sum(snapshot.rho, dtype=np.float64))
    if mass <= 0.0:
        raise NumericalStabilityError(f"LBM-2 FINITE: total mass is not positive at step {step}")
    if min_rho < MIN_RUNTIME_RHO or max_rho > MAX_RUNTIME_RHO:
        raise NumericalStabilityError(
            f"LBM-3 STABILITY: rho is outside [{MIN_RUNTIME_RHO}, {MAX_RUNTIME_RHO}] at step {step}"
        )
    # The persisted state is fp32; accept its nearest representation of the
    # inclusive decimal endpoint instead of rejecting 0.2 as 0.20000000298.
    speed_limit = float(np.float32(MAX_RUNTIME_SPEED_LU))
    if max_speed > speed_limit:
        raise NumericalStabilityError(
            f"LBM-3 STABILITY: maximum speed exceeds {MAX_RUNTIME_SPEED_LU} at step {step}"
        )
    reference_mass = mass if initial_mass is None else initial_mass
    ramp = inlet_ramp(step, config.ramp_steps)
    return StepDiagnostics(
        step=step,
        mass=mass,
        mass_drift_ratio=abs(mass - reference_mass) / reference_mass,
        min_rho=min_rho,
        max_rho=max_rho,
        max_speed_lu=max_speed,
        inlet_ramp=ramp,
        inlet_velocity_lu=config.inlet_velocity_lu * ramp,
    )


def _failed_run(
    config: DerivedLatticeConfig,
    *,
    steps_completed: int,
    sample_count: int,
    error_code: str,
    reason: str,
    inspections: list[StepDiagnostics],
    diagnostics: SolverDiagnostics | None = None,
) -> FailedSolverRun:
    return FailedSolverRun(
        config=config,
        steps_completed=steps_completed,
        sample_count=sample_count,
        error_code=error_code,
        reason=reason,
        last_valid=inspections[-1] if inspections else None,
        diagnostics=diagnostics,
    )


def run_lifecycle(
    config: DerivedLatticeConfig,
    mask: npt.NDArray[np.bool_],
    *,
    stepper: LifecycleStepper | None = None,
    progress: Callable[[StepDiagnostics], None] | None = None,
) -> CompletedLatticeRun:
    """Run a preflighted lifecycle or raise a typed failure with scalar evidence."""

    sample_steps = averaging_sample_steps(config)
    sample_step_set = frozenset(sample_steps)
    _validate_mask(mask, config.shape)
    driver = NumpyPeriodicStepper() if stepper is None else stepper
    state = driver.initialize(config, mask)
    inspections: list[StepDiagnostics] = []
    try:
        initial = _inspect_snapshot(driver.snapshot(state), config, step=0, initial_mass=None)
    except NumericalStabilityError as exc:
        failure = _failed_run(
            config,
            steps_completed=0,
            sample_count=0,
            error_code=exc.code,
            reason=str(exc),
            inspections=inspections,
        )
        raise SolverStabilityFailure(str(exc), failed_run=failure) from exc
    inspections.append(initial)
    initial_mass = initial.mass

    u_sum = np.zeros(config.shape, dtype=np.float64)
    v_sum = np.zeros(config.shape, dtype=np.float64)
    rho_sum = np.zeros(config.shape, dtype=np.float64)
    samples_seen = 0

    for completed_step in range(1, config.steps + 1):
        ramp = inlet_ramp(completed_step, config.ramp_steps)
        try:
            driver.advance(
                state,
                config,
                mask,
                completed_step=completed_step,
                inlet_velocity_lu=config.inlet_velocity_lu * ramp,
            )
        except NumericalStabilityError as exc:
            failure = _failed_run(
                config,
                steps_completed=completed_step,
                sample_count=samples_seen,
                error_code=exc.code,
                reason=str(exc),
                inspections=inspections,
            )
            raise SolverStabilityFailure(str(exc), failed_run=failure) from exc

        is_average_sample = completed_step in sample_step_set
        is_diagnostic = (
            completed_step % config.sample_interval == 0 or completed_step == config.steps
        )
        if not is_average_sample and not is_diagnostic:
            continue
        try:
            snapshot = driver.snapshot(state)
            observed = _inspect_snapshot(
                snapshot,
                config,
                step=completed_step,
                initial_mass=initial_mass,
            )
        except NumericalStabilityError as exc:
            failure = _failed_run(
                config,
                steps_completed=completed_step,
                sample_count=samples_seen,
                error_code=exc.code,
                reason=str(exc),
                inspections=inspections,
            )
            raise SolverStabilityFailure(str(exc), failed_run=failure) from exc
        inspections.append(observed)
        if is_diagnostic and progress is not None:
            progress(observed)
        if is_average_sample:
            u_sum += snapshot.velocity[..., 0].astype(np.float64)
            v_sum += snapshot.velocity[..., 1].astype(np.float64)
            rho_sum += snapshot.rho.astype(np.float64)
            samples_seen += 1

    if samples_seen != len(sample_steps):
        raise InternalInvariantError("averaging loop did not consume its declared sample schedule")
    final = inspections[-1]
    mass_drift = abs(final.mass - initial_mass) / initial_mass
    global_min_rho = min(item.min_rho for item in inspections)
    global_max_rho = max(item.max_rho for item in inspections)
    global_max_speed = max(item.max_speed_lu for item in inspections)
    if mass_drift >= MAX_MASS_DRIFT_RATIO:
        reason = (
            f"LBM-1 CONSERVATION: mass drift {mass_drift:.17g} is not below {MAX_MASS_DRIFT_RATIO}"
        )
        diagnostics = SolverDiagnostics(
            steps_completed=config.steps,
            sample_count=samples_seen,
            initial_mass=initial_mass,
            final_mass=final.mass,
            mass_drift_ratio=mass_drift,
            min_rho=global_min_rho,
            max_rho=global_max_rho,
            max_speed_lu=global_max_speed,
            converged=False,
            valid=False,
            messages=(reason,),
        )
        failure = _failed_run(
            config,
            steps_completed=config.steps,
            sample_count=samples_seen,
            error_code=NonConvergenceError.code,
            reason=reason,
            inspections=inspections,
            diagnostics=diagnostics,
        )
        raise SolverConvergenceFailure(reason, failed_run=failure)

    inverse_count = 1.0 / samples_seen
    means = MeanFlowFields(
        u=np.ascontiguousarray((u_sum * inverse_count).astype(np.float32)),
        v=np.ascontiguousarray((v_sum * inverse_count).astype(np.float32)),
        rho=np.ascontiguousarray((rho_sum * inverse_count).astype(np.float32)),
    )
    diagnostics = SolverDiagnostics(
        steps_completed=config.steps,
        sample_count=samples_seen,
        initial_mass=initial_mass,
        final_mass=final.mass,
        mass_drift_ratio=mass_drift,
        min_rho=global_min_rho,
        max_rho=global_max_rho,
        max_speed_lu=global_max_speed,
        converged=True,
        valid=True,
    )
    return CompletedLatticeRun(
        config=config,
        mean_fields=means,
        diagnostics=diagnostics,
        sample_steps=sample_steps,
        history=tuple(inspections),
        device_class=driver.device_class,
    )


__all__ = [
    "MAX_MASS_DRIFT_RATIO",
    "MAX_RUNTIME_RHO",
    "MAX_RUNTIME_SPEED_LU",
    "MIN_AVERAGING_SAMPLES",
    "MIN_AVERAGING_STEPS",
    "MIN_RUNTIME_RHO",
    "CompletedLatticeRun",
    "FailedSolverRun",
    "LifecycleStepper",
    "MeanFlowFields",
    "NumpyChannelStepper",
    "NumpyObstacleStepper",
    "NumpyPeriodicState",
    "NumpyPeriodicStepper",
    "RawLatticeSnapshot",
    "SolverConvergenceFailure",
    "SolverStabilityFailure",
    "StepDiagnostics",
    "WarpChannelStepper",
    "WarpObstacleStepper",
    "WarpPeriodicStepper",
    "averaging_sample_steps",
    "run_lifecycle",
]
