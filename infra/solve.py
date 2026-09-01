"""Authenticated remote kernel smoke; domain solve orchestration lands in issue #42."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Literal

import numpy as np
import numpy.typing as npt

from infra.app import app, image, settings, volume
from infra.policy import (
    REMOTE_RETRIES,
    SMOKE_MAX_CONTAINERS,
    SOLVE_TIMEOUT_SECONDS,
    VOLUME_MOUNT,
)
from infra.runtime_manifest import (
    KernelSmokeEvidence,
    KernelSmokeResult,
    load_build_manifest,
)


def _state_sha256(arrays: tuple[npt.NDArray[np.float32], ...]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@app.function(
    image=image,
    gpu=settings.remote_gpu,
    volumes={VOLUME_MOUNT: volume},
    timeout=SOLVE_TIMEOUT_SECONDS,
    max_containers=SMOKE_MAX_CONTAINERS,
    retries=REMOTE_RETRIES,
)
def kernel_smoke_remote(requested_device_class: Literal["L40S", "A10G"]) -> str:
    """Execute two synchronized periodic D2Q9 steps on the requested GPU."""

    import warp as wp

    from soufflerie.solver.kernels import WarpKernelAdapter
    from soufflerie.solver.lattice import LatticeConfig, preflight_lattice

    build = load_build_manifest()
    if build.source_dirty:
        raise RuntimeError("authenticated smoke requires an image built from clean source")

    started = time.perf_counter()
    config = preflight_lattice(
        LatticeConfig(
            nx=8,
            ny=8,
            steps=2,
            warmup_steps=0,
            sample_interval=1,
            inlet_velocity_lu=0.05,
            reynolds=100.0,
            reference_diameter_lu=32.0,
        )
    )
    adapter = WarpKernelAdapter("cuda:0")
    state = adapter.initialize(config)
    initial = adapter.snapshot(state)
    for _ in range(2):
        adapter.step(state, config)
    final = adapter.snapshot(state)
    adapter.synchronize()
    elapsed = time.perf_counter() - started

    device = wp.get_device(adapter.device)
    if not device.is_cuda:
        raise RuntimeError("remote kernel smoke resolved a non-CUDA device")
    initial_mass = float(np.sum(initial.rho, dtype=np.float64))
    final_mass = float(np.sum(final.rho, dtype=np.float64))
    result = KernelSmokeResult.create(
        build=build.model_dump(mode="json"),
        requested_device_class=requested_device_class,
        resolved_device=adapter.device,
        device_name=device.name,
        cuda_arch=int(device.arch),
        volume_name="soufflerie-data",
        volume_mount=VOLUME_MOUNT,
        kernel_steps=2,
        state_sha256=_state_sha256((final.f, final.rho, final.velocity)),
        initial_mass=initial_mass,
        final_mass=final_mass,
        wall_seconds=elapsed,
        gpu_seconds=elapsed,
        passed=True,
    )
    return result.model_dump_json()


@app.local_entrypoint()
def main(smoke: bool = False) -> None:
    """Run the authenticated two-repetition kernel acceptance smoke."""

    if not smoke:
        raise RuntimeError("issue #41 exposes only --smoke; domain solve lands in issue #42")
    first = KernelSmokeResult.model_validate_json(kernel_smoke_remote.remote(settings.remote_gpu))
    second = KernelSmokeResult.model_validate_json(kernel_smoke_remote.remote(settings.remote_gpu))
    evidence = KernelSmokeEvidence.create(first, second)
    print(json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True))


__all__ = ["kernel_smoke_remote", "main"]
