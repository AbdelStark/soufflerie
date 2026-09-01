from __future__ import annotations

import importlib.util

import numpy as np
import numpy.typing as npt
import pytest

from soufflerie.solver.kernels import WarpKernelAdapter
from soufflerie.solver.lattice import DerivedLatticeConfig, LatticeConfig, preflight_lattice
from soufflerie.solver.numpy_oracle import NumpyLatticeState, initialize_numpy


def _require_warp() -> None:
    if importlib.util.find_spec("warp") is None:
        pytest.skip("install the Soufflerie 'solver' extra to run Warp determinism tests")


def _config() -> DerivedLatticeConfig:
    return preflight_lattice(
        LatticeConfig(
            nx=7,
            ny=5,
            steps=100,
            warmup_steps=20,
            sample_interval=10,
            inlet_velocity_lu=0.05,
            reynolds=100.0,
            reference_diameter_lu=32.0,
        )
    )


def _run(steps: int) -> tuple[NumpyLatticeState, npt.NDArray[np.float32]]:
    config = _config()
    initial = initialize_numpy(config).f.copy()
    initial[1, 2, 1] += np.float32(0.002)
    initial[3, 4, 6] -= np.float32(0.001)
    adapter = WarpKernelAdapter("cpu")
    state = adapter.from_numpy(initial, config)
    for _ in range(steps):
        adapter.step(state, config)
    return adapter.snapshot(state), initial


def test_repeated_cpu_runs_are_bitwise_equal_at_persisted_boundaries() -> None:
    _require_warp()
    first, first_input = _run(12)
    second, second_input = _run(12)

    np.testing.assert_array_equal(first_input, second_input)
    np.testing.assert_array_equal(first.f, second.f)
    np.testing.assert_array_equal(first.rho, second.rho)
    np.testing.assert_array_equal(first.velocity, second.velocity)


def test_snapshot_is_a_detached_stable_host_copy() -> None:
    _require_warp()
    config = _config()
    adapter = WarpKernelAdapter("cpu")
    state = adapter.initialize(config)
    before = adapter.snapshot(state)

    adapter.step(state, config)
    after = adapter.snapshot(state)

    assert not np.shares_memory(before.f, after.f)
    np.testing.assert_array_equal(before.f, initialize_numpy(config).f)
    assert before.f.dtype == after.f.dtype == np.float32
