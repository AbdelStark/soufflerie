from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from soufflerie.errors import (
    DependencyUnavailableError,
    DeviceUnavailableError,
    DomainError,
    InternalInvariantError,
)
from soufflerie.solver.kernels import WarpKernelAdapter, allocate, initialize
from soufflerie.solver.lattice import DerivedLatticeConfig, LatticeConfig, preflight_lattice
from soufflerie.solver.numpy_oracle import (
    collide_numpy,
    initialize_numpy,
    numpy_periodic_step,
    pull_stream_periodic_numpy,
)


def _config(*, nx: int = 6, ny: int = 5) -> DerivedLatticeConfig:
    return preflight_lattice(
        LatticeConfig(
            nx=nx,
            ny=ny,
            steps=100,
            warmup_steps=20,
            sample_interval=10,
            inlet_velocity_lu=0.05,
            reynolds=100.0,
            reference_diameter_lu=32.0,
        )
    )


def _require_warp() -> None:
    if importlib.util.find_spec("warp") is None:
        pytest.skip("install the Soufflerie 'solver' extra to run Warp kernel tests")


def _populations() -> tuple[npt.NDArray[np.float32], DerivedLatticeConfig]:
    config = _config()
    populations = initialize_numpy(config).f.copy()
    perturbation = np.linspace(-0.001, 0.001, populations.size, dtype=np.float32).reshape(
        populations.shape
    )
    populations += perturbation
    return populations, config


def test_public_solver_import_does_not_load_warp() -> None:
    code = """
import sys
import soufflerie
from soufflerie.solver import WarpKernelAdapter
assert WarpKernelAdapter.__name__ == 'WarpKernelAdapter'
assert 'warp' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True)


def test_missing_optional_runtime_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    kernels = importlib.import_module("soufflerie.solver.kernels")
    real_import = importlib.import_module

    def missing_backend(name: str, package: str | None = None) -> object:
        if name == "soufflerie.solver._warp_backend":
            raise ModuleNotFoundError("No module named 'warp'", name="warp")
        return real_import(name, package)

    monkeypatch.setattr(kernels.importlib, "import_module", missing_backend)
    with pytest.raises(DependencyUnavailableError, match=r"solver.*extra"):
        WarpKernelAdapter()


def test_invalid_device_raises_typed_error() -> None:
    _require_warp()
    with pytest.raises(DeviceUnavailableError, match="unavailable"):
        WarpKernelAdapter("definitely-not-a-device")


def test_allocate_owns_exact_canonical_fp32_buffers() -> None:
    _require_warp()
    config = _config()
    adapter = WarpKernelAdapter("cpu")
    state = adapter.allocate(config)

    assert state.shape == config.shape
    assert tuple(state.f.shape) == (*config.shape, 9)
    assert tuple(state.f_next.shape) == (*config.shape, 9)
    assert tuple(state.rho.shape) == config.shape
    assert tuple(state.velocity.shape) == (*config.shape, 2)
    for array in (state.f, state.f_next, state.rho, state.velocity):
        assert array.numpy().dtype == np.float32
        assert str(array.device) == "cpu"

    public_state = allocate(config, device="cpu")
    assert public_state.shape == config.shape
    mask = np.zeros(config.shape, dtype=np.bool_)
    initialized = initialize(config, mask, device="cpu")
    assert initialized.shape == config.shape


def test_cpu_collision_and_reduction_match_numpy_oracle() -> None:
    _require_warp()
    populations, config = _populations()
    adapter = WarpKernelAdapter("cpu")
    state = adapter.from_numpy(populations, config)

    adapter.collide(state, config)
    actual_post = adapter.post_collision_numpy(state)
    expected_post = collide_numpy(populations, omega=config.omega)
    np.testing.assert_allclose(actual_post, expected_post, rtol=2e-6, atol=2e-7)

    adapter.pull_stream_periodic(state)
    adapter.reduce_macroscopic(state)
    actual = adapter.snapshot(state)
    expected = numpy_periodic_step(populations, config)
    np.testing.assert_allclose(actual.f, expected.f, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(actual.rho, expected.rho, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(actual.velocity, expected.velocity, rtol=2e-6, atol=2e-7)


def test_pull_stream_writes_every_destination_once_with_exact_mapping() -> None:
    _require_warp()
    populations, config = _populations()
    adapter = WarpKernelAdapter("cpu")
    state = adapter.from_numpy(populations, config)
    adapter.collide(state, config)
    expected = pull_stream_periodic_numpy(adapter.post_collision_numpy(state))

    adapter.pull_stream_periodic(state)
    actual = adapter.snapshot(state).f

    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()


def test_state_and_config_grid_mismatch_is_rejected() -> None:
    _require_warp()
    adapter = WarpKernelAdapter("cpu")
    state = adapter.initialize(_config())
    with pytest.raises(DomainError, match="does not match"):
        adapter.collide(state, _config(nx=7))

    state.f_next = state.rho
    with pytest.raises(InternalInvariantError, match="f_next shape"):
        adapter.collide(state, _config())


def test_streaming_source_has_single_non_atomic_destination_assignment() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "soufflerie" / "solver" / "_warp_backend.py"
    ).read_text(encoding="utf-8")
    assert "wp.atomic" not in source
    assert source.count("streamed[y, x, direction] =") == 1
    assert "dim=(ny, nx, 9)" in source
