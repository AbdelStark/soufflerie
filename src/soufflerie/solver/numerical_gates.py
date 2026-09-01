"""Schema-versioned CPU numerical gate generation and validation."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator, model_validator

from soufflerie.errors import InternalInvariantError
from soufflerie.schemas import StrictFrozenModel, VersionedModel
from soufflerie.solver.kernels import WarpKernelAdapter
from soufflerie.solver.lattice import LatticeConfig, macroscopic_moments, preflight_lattice
from soufflerie.solver.numpy_oracle import initialize_numpy
from soufflerie.solver.poiseuille import PoiseuilleFixture, run_poiseuille_fixture

CPU_GATE_GENERATION_REVISION: Final[Literal["cpu-gates-v1"]] = "cpu-gates-v1"
MASS_DRIFT_THRESHOLD = 0.001


class PoiseuilleGateSummary(StrictFrozenModel):
    backend: Literal["numpy-d2q9-periodic-x-body-force"]
    channel_height: int = Field(ge=6)
    width: int = Field(ge=1)
    steps: int = Field(ge=1)
    tau: float = Field(gt=0.5, allow_inf_nan=False)
    body_force_lu: float = Field(gt=0.0, allow_inf_nan=False)
    excluded_wall_cells: Literal[1]
    relative_l2_error: float = Field(ge=0.0, allow_inf_nan=False)
    max_relative_error: float = Field(ge=0.0, allow_inf_nan=False)
    threshold: float = Field(ge=0.01, le=0.01, allow_inf_nan=False)
    passed: bool

    @model_validator(mode="after")
    def _gate_matches_metrics(self) -> Self:
        expected = (
            self.relative_l2_error <= self.threshold and self.max_relative_error <= self.threshold
        )
        if self.passed != expected:
            raise ValueError("Poiseuille passed flag does not match metrics")
        return self


class MassGateSummary(StrictFrozenModel):
    backend: Literal["warp-periodic-cpu"]
    steps: Literal[20_000]
    grid_nx: int = Field(ge=3)
    grid_ny: int = Field(ge=3)
    initial_mass: float = Field(gt=0.0, allow_inf_nan=False)
    final_mass: float = Field(gt=0.0, allow_inf_nan=False)
    mass_drift_ratio: float = Field(ge=0.0, allow_inf_nan=False)
    threshold: float = Field(ge=0.001, le=0.001, allow_inf_nan=False)
    passed: bool

    @model_validator(mode="after")
    def _gate_matches_metrics(self) -> Self:
        expected_drift = abs(self.final_mass - self.initial_mass) / self.initial_mass
        if not np.isclose(self.mass_drift_ratio, expected_drift, rtol=1e-15, atol=1e-15):
            raise ValueError("mass drift ratio does not match masses")
        if self.passed != (self.mass_drift_ratio < self.threshold):
            raise ValueError("mass passed flag does not match strict threshold")
        return self


class DeterminismGateSummary(StrictFrozenModel):
    backend: Literal["warp-periodic-cpu"]
    repetitions: Literal[2]
    steps_per_repetition: Literal[20_000]
    final_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_arrays_bitwise_equal: bool
    histories_bitwise_equal: bool
    passed: bool

    @model_validator(mode="after")
    def _gate_matches_evidence(self) -> Self:
        expected = self.final_arrays_bitwise_equal and self.histories_bitwise_equal
        if self.passed != expected:
            raise ValueError("determinism passed flag does not match equality evidence")
        return self


class CpuSolverGateSummary(VersionedModel):
    generation_revision: Literal["cpu-gates-v1"]
    platform_class: str = Field(min_length=1)
    python_version: str = Field(pattern=r"^3\.11\.")
    numpy_version: Literal["2.2.6"]
    warp_version: Literal["1.17.0"]
    poiseuille: tuple[PoiseuilleGateSummary, PoiseuilleGateSummary]
    mass: MassGateSummary
    determinism: DeterminismGateSummary
    overall_passed: bool

    @field_validator("poiseuille", mode="before")
    @classmethod
    def _json_array_to_fixed_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _summary_is_coherent(self) -> Self:
        if self.poiseuille[0].channel_height == self.poiseuille[1].channel_height:
            raise ValueError("Poiseuille summaries must use two grid sizes")
        expected = (
            all(item.passed for item in self.poiseuille)
            and self.mass.passed
            and self.determinism.passed
        )
        if self.overall_passed != expected:
            raise ValueError("overall_passed does not match required CPU gates")
        return self


@dataclass(frozen=True, slots=True)
class PeriodicRegressionResult:
    initial_mass: float
    final_mass: float
    mass_drift_ratio: float
    final_state_sha256: str
    history_sha256: str
    final_arrays_bitwise_equal: bool
    histories_bitwise_equal: bool


def mass_gate_passes(mass_drift_ratio: float) -> bool:
    return np.isfinite(mass_drift_ratio) and mass_drift_ratio < MASS_DRIFT_THRESHOLD


def _state_digest(
    populations: npt.NDArray[np.float32],
    rho: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
) -> str:
    digest = hashlib.sha256()
    for array in (populations, rho, velocity):
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _history_digest(history: npt.NDArray[np.float64]) -> str:
    digest = hashlib.sha256()
    digest.update(history.dtype.str.encode("ascii"))
    digest.update(history.tobytes(order="C"))
    return digest.hexdigest()


def _periodic_run() -> tuple[
    float,
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float64],
]:
    config = preflight_lattice(
        LatticeConfig(
            nx=7,
            ny=5,
            steps=20_000,
            warmup_steps=16_000,
            sample_interval=20,
            inlet_velocity_lu=0.05,
            reynolds=100.0,
            reference_diameter_lu=32.0,
        )
    )
    populations = initialize_numpy(config).f.copy()
    populations[1, 2, 1] += np.float32(0.002)
    populations[3, 4, 6] -= np.float32(0.001)
    initial_rho, _ = macroscopic_moments(populations)
    initial_mass = float(np.sum(initial_rho, dtype=np.float64))
    adapter = WarpKernelAdapter("cpu")
    state = adapter.from_numpy(populations, config)
    mass_history = np.empty(20, dtype=np.float64)
    for step in range(1, config.steps + 1):
        adapter.step(state, config)
        if step % 1_000 == 0:
            snapshot = adapter.snapshot(state)
            mass_history[step // 1_000 - 1] = np.sum(snapshot.rho, dtype=np.float64)
    final = adapter.snapshot(state)
    return initial_mass, final.f, final.rho, final.velocity, mass_history


def run_periodic_regression() -> PeriodicRegressionResult:
    """Run two independent 20,000-step CPU Warp regressions."""

    first = _periodic_run()
    second = _periodic_run()
    initial_mass = first[0]
    final_mass = float(np.sum(first[2], dtype=np.float64))
    final_equal = all(
        np.array_equal(left, right) for left, right in zip(first[1:4], second[1:4], strict=True)
    )
    histories_equal = np.array_equal(first[4], second[4])
    first_digest = _state_digest(first[1], first[2], first[3])
    second_digest = _state_digest(second[1], second[2], second[3])
    if final_equal and first_digest != second_digest:
        raise InternalInvariantError("bitwise-equal states produced different digests")
    return PeriodicRegressionResult(
        initial_mass=initial_mass,
        final_mass=final_mass,
        mass_drift_ratio=abs(final_mass - initial_mass) / initial_mass,
        final_state_sha256=first_digest,
        history_sha256=_history_digest(first[4]),
        final_arrays_bitwise_equal=final_equal,
        histories_bitwise_equal=histories_equal,
    )


def generate_cpu_gate_summary() -> CpuSolverGateSummary:
    fixtures = (
        PoiseuilleFixture(channel_height=16, width=2, steps=4_000),
        PoiseuilleFixture(channel_height=32, width=2, steps=12_000),
    )
    poiseuille_results = tuple(run_poiseuille_fixture(item) for item in fixtures)
    poiseuille = tuple(
        PoiseuilleGateSummary(
            backend="numpy-d2q9-periodic-x-body-force",
            channel_height=result.fixture.channel_height,
            width=result.fixture.width,
            steps=result.fixture.steps,
            tau=result.fixture.tau,
            body_force_lu=result.fixture.body_force_lu,
            excluded_wall_cells=1,
            relative_l2_error=result.relative_l2_error,
            max_relative_error=result.max_relative_error,
            threshold=0.01,
            passed=result.passed,
        )
        for result in poiseuille_results
    )
    if len(poiseuille) != 2:
        raise InternalInvariantError("CPU gate generation requires exactly two Poiseuille cases")
    periodic = run_periodic_regression()
    mass_passed = mass_gate_passes(periodic.mass_drift_ratio)
    determinism_passed = periodic.final_arrays_bitwise_equal and periodic.histories_bitwise_equal
    if np.__version__ != "2.2.6" or version("warp-lang") != "1.17.0":
        raise InternalInvariantError(
            "CPU golden generation requires the locked NumPy/Warp versions"
        )
    return CpuSolverGateSummary(
        generation_revision=CPU_GATE_GENERATION_REVISION,
        platform_class=f"{platform.system().lower()}-{platform.machine().lower()}-cpu",
        python_version=platform.python_version(),
        numpy_version="2.2.6",
        warp_version="1.17.0",
        poiseuille=(poiseuille[0], poiseuille[1]),
        mass=MassGateSummary(
            backend="warp-periodic-cpu",
            steps=20_000,
            grid_nx=7,
            grid_ny=5,
            initial_mass=periodic.initial_mass,
            final_mass=periodic.final_mass,
            mass_drift_ratio=periodic.mass_drift_ratio,
            threshold=0.001,
            passed=mass_passed,
        ),
        determinism=DeterminismGateSummary(
            backend="warp-periodic-cpu",
            repetitions=2,
            steps_per_repetition=20_000,
            final_state_sha256=periodic.final_state_sha256,
            history_sha256=periodic.history_sha256,
            final_arrays_bitwise_equal=periodic.final_arrays_bitwise_equal,
            histories_bitwise_equal=periodic.histories_bitwise_equal,
            passed=determinism_passed,
        ),
        overall_passed=all(result.passed for result in poiseuille_results)
        and mass_passed
        and determinism_passed,
    )


def render_cpu_gate_summary(summary: CpuSolverGateSummary) -> str:
    return json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def load_cpu_gate_summary(path: Path) -> CpuSolverGateSummary:
    return CpuSolverGateSummary.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "CPU_GATE_GENERATION_REVISION",
    "MASS_DRIFT_THRESHOLD",
    "CpuSolverGateSummary",
    "DeterminismGateSummary",
    "MassGateSummary",
    "PeriodicRegressionResult",
    "PoiseuilleGateSummary",
    "generate_cpu_gate_summary",
    "load_cpu_gate_summary",
    "mass_gate_passes",
    "render_cpu_gate_summary",
    "run_periodic_regression",
]
