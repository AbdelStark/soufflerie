"""Numerical solver contracts and lightweight reference implementations."""

from soufflerie.solver.kernels import LatticeState, WarpKernelAdapter, allocate, initialize
from soufflerie.solver.lattice import (
    CS2,
    D2Q9_OPPOSITE,
    D2Q9_VELOCITIES,
    D2Q9_WEIGHTS,
    DerivedLatticeConfig,
    LatticeConfig,
    derive_lattice,
    equilibrium,
    inlet_ramp,
    kinematic_viscosity_lu,
    macroscopic_moments,
    preflight_lattice,
    relaxation_time,
    reynolds_from_relaxation_time,
)
from soufflerie.solver.numpy_oracle import (
    NumpyLatticeState,
    collide_numpy,
    initialize_numpy,
    numpy_periodic_step,
    pull_stream_periodic_numpy,
)

__all__ = [
    "CS2",
    "D2Q9_OPPOSITE",
    "D2Q9_VELOCITIES",
    "D2Q9_WEIGHTS",
    "DerivedLatticeConfig",
    "LatticeConfig",
    "LatticeState",
    "NumpyLatticeState",
    "WarpKernelAdapter",
    "allocate",
    "collide_numpy",
    "derive_lattice",
    "equilibrium",
    "initialize",
    "initialize_numpy",
    "inlet_ramp",
    "kinematic_viscosity_lu",
    "macroscopic_moments",
    "numpy_periodic_step",
    "preflight_lattice",
    "pull_stream_periodic_numpy",
    "relaxation_time",
    "reynolds_from_relaxation_time",
]
