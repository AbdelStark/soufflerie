"""Lazy Warp adapter for deterministic D2Q9 collision and streaming kernels."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from soufflerie.errors import (
    DependencyUnavailableError,
    DeviceUnavailableError,
    DomainError,
    InternalInvariantError,
)
from soufflerie.solver.lattice import DerivedLatticeConfig, validate_populations
from soufflerie.solver.numpy_oracle import NumpyLatticeState, initialize_numpy


class WarpArray(Protocol):
    """The small Warp array surface exposed by a lattice state."""

    shape: tuple[int, ...]
    device: object

    def numpy(self) -> npt.NDArray[np.float32]: ...


class _WarpBackend(Protocol):
    def resolve_device(self, requested: str) -> str: ...

    def allocate_state(
        self, shape: tuple[int, int], device: str
    ) -> tuple[WarpArray, WarpArray, WarpArray, WarpArray]: ...

    def upload_state(
        self,
        populations: npt.NDArray[np.float32],
        rho: npt.NDArray[np.float32],
        velocity: npt.NDArray[np.float32],
        device: str,
    ) -> tuple[WarpArray, WarpArray, WarpArray, WarpArray]: ...

    def launch_collision(
        self,
        populations: WarpArray,
        post_collision: WarpArray,
        omega: float,
        shape: tuple[int, int],
        device: str,
    ) -> None: ...

    def launch_pull_stream(
        self,
        post_collision: WarpArray,
        streamed: WarpArray,
        shape: tuple[int, int],
        device: str,
    ) -> None: ...

    def launch_macroscopic(
        self,
        populations: WarpArray,
        rho: WarpArray,
        velocity: WarpArray,
        shape: tuple[int, int],
        device: str,
    ) -> None: ...

    def synchronize(self, device: str) -> None: ...

    def array_is_float32(self, array: WarpArray) -> bool: ...


@dataclass(slots=True)
class LatticeState:
    """Mutable Warp buffers with the exact RFC-0002 memory ownership layout."""

    f: WarpArray
    f_next: WarpArray
    rho: WarpArray
    velocity: WarpArray

    @property
    def shape(self) -> tuple[int, int]:
        raw_shape = tuple(self.f.shape)
        if len(raw_shape) != 3:
            raise InternalInvariantError("LBM-6 PRECISION: state f must have three dimensions")
        return (raw_shape[0], raw_shape[1])


def _load_backend() -> _WarpBackend:
    try:
        module: ModuleType = importlib.import_module("soufflerie.solver._warp_backend")
    except ModuleNotFoundError as exc:
        if exc.name == "warp":
            raise DependencyUnavailableError(
                "Warp is unavailable; install Soufflerie with the 'solver' extra"
            ) from exc
        raise
    return cast(_WarpBackend, module)


class WarpKernelAdapter:
    """Own a single resolved device and synchronized unfused kernel stages."""

    def __init__(self, device: str = "cpu") -> None:
        if not isinstance(device, str) or not device:
            raise DeviceUnavailableError("solver device must be a non-empty string")
        self._backend = _load_backend()
        try:
            self.device = self._backend.resolve_device(device)
        except (RuntimeError, ValueError) as exc:
            raise DeviceUnavailableError(f"solver device {device!r} is unavailable") from exc

    def _validate_state(
        self, state: LatticeState, config: DerivedLatticeConfig | None = None
    ) -> tuple[int, int]:
        shape = state.shape
        expected = {
            "f": (*shape, 9),
            "f_next": (*shape, 9),
            "rho": shape,
            "velocity": (*shape, 2),
        }
        for name, target in expected.items():
            array = getattr(state, name)
            actual = tuple(array.shape)
            if actual != target:
                raise InternalInvariantError(
                    f"LBM-6 PRECISION: state {name} shape {actual} does not match {target}"
                )
            if not self._backend.array_is_float32(array):
                raise InternalInvariantError(f"LBM-6 PRECISION: state {name} must use Warp float32")
        if shape[0] <= 0 or shape[1] <= 0:
            raise InternalInvariantError("LBM-6 PRECISION: state grid dimensions must be positive")
        devices = {str(getattr(state, name).device) for name in expected}
        if devices != {self.device}:
            raise DeviceUnavailableError(
                f"state buffers must all belong to adapter device {self.device!r}"
            )
        if config is not None and shape != config.shape:
            raise DomainError(
                f"LBM-6 PRECISION: state grid {shape} does not match config grid {config.shape}"
            )
        return shape

    def allocate(self, config: DerivedLatticeConfig) -> LatticeState:
        """Allocate zeroed canonical buffers on the adapter's device."""

        arrays = self._backend.allocate_state(config.shape, self.device)
        state = LatticeState(*arrays)
        self._validate_state(state, config)
        return state

    def from_numpy(
        self, populations: npt.NDArray[np.float32], config: DerivedLatticeConfig
    ) -> LatticeState:
        """Upload validated populations and their recovered moments."""

        validate_populations(populations)
        if populations.shape[:2] != config.shape:
            raise DomainError(
                f"LBM-6 PRECISION: populations grid {populations.shape[:2]} "
                f"does not match config grid {config.shape}"
            )
        from soufflerie.solver.lattice import macroscopic_moments

        rho, velocity = macroscopic_moments(populations)
        arrays = self._backend.upload_state(populations, rho, velocity, self.device)
        state = LatticeState(*arrays)
        self._validate_state(state, config)
        return state

    def initialize(
        self,
        config: DerivedLatticeConfig,
        mask: npt.NDArray[np.bool_] | None = None,
    ) -> LatticeState:
        """Upload deterministic equilibrium initialization from the NumPy contract."""

        initial = initialize_numpy(config, mask)
        arrays = self._backend.upload_state(initial.f, initial.rho, initial.velocity, self.device)
        state = LatticeState(*arrays)
        self._validate_state(state, config)
        return state

    def collide(self, state: LatticeState, config: DerivedLatticeConfig) -> None:
        """Write post-collision populations to ``state.f_next``."""

        shape = self._validate_state(state, config)
        self._backend.launch_collision(
            state.f,
            state.f_next,
            config.omega,
            shape,
            self.device,
        )

    def pull_stream_periodic(self, state: LatticeState) -> None:
        """Pull ``f_next`` into ``f`` with one writer per periodic destination."""

        shape = self._validate_state(state)
        self._backend.launch_pull_stream(state.f_next, state.f, shape, self.device)

    def reduce_macroscopic(self, state: LatticeState) -> None:
        """Recover density and velocity from the streamed ``state.f``."""

        shape = self._validate_state(state)
        self._backend.launch_macroscopic(
            state.f,
            state.rho,
            state.velocity,
            shape,
            self.device,
        )

    def step(self, state: LatticeState, config: DerivedLatticeConfig) -> None:
        """Run synchronized collision, periodic pull, and reduction without fusion."""

        self.collide(state, config)
        self.pull_stream_periodic(state)
        self.reduce_macroscopic(state)

    def synchronize(self) -> None:
        self._backend.synchronize(self.device)

    def snapshot(self, state: LatticeState) -> NumpyLatticeState:
        """Synchronize and copy canonical persisted-boundary arrays to host."""

        self._validate_state(state)
        self.synchronize()
        return NumpyLatticeState(
            f=np.array(state.f.numpy(), dtype=np.float32, order="C", copy=True),
            rho=np.array(state.rho.numpy(), dtype=np.float32, order="C", copy=True),
            velocity=np.array(state.velocity.numpy(), dtype=np.float32, order="C", copy=True),
        )

    def post_collision_numpy(self, state: LatticeState) -> npt.NDArray[np.float32]:
        """Synchronize and copy the post-collision comparison buffer."""

        self._validate_state(state)
        self.synchronize()
        result = np.array(state.f_next.numpy(), dtype=np.float32, order="C", copy=True)
        validate_populations(result)
        return result


def allocate(config: DerivedLatticeConfig, *, device: str = "cpu") -> LatticeState:
    """Allocate an RFC-shaped state through a lazily constructed adapter."""

    return WarpKernelAdapter(device).allocate(config)


def initialize(
    config: DerivedLatticeConfig,
    mask: npt.NDArray[np.bool_],
    device: str = "cpu",
) -> LatticeState:
    """Implement the RFC initialization signature for the Warp adapter."""

    return WarpKernelAdapter(device).initialize(config, mask)


__all__ = [
    "LatticeState",
    "WarpArray",
    "WarpKernelAdapter",
    "allocate",
    "initialize",
]
