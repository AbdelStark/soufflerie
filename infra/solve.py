"""Authenticated kernel smoke and idempotent remote solver entrypoint."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from infra.app import app, checkout, image, settings, volume
from infra.policy import (
    REMOTE_RETRIES,
    SMOKE_MAX_CONTAINERS,
    SOLVE_TIMEOUT_SECONDS,
    VOLUME_MOUNT,
)
from infra.remote_execution import (
    REMOTE_ARTIFACT_ROOT,
    RemoteSolveRequest,
    SolveSummary,
    encode_remote_model,
)
from infra.runtime_manifest import (
    KernelSmokeEvidence,
    KernelSmokeResult,
    load_build_manifest,
)
from infra.solve_worker import run_solver_case, solve_remote
from soufflerie.config import load_config
from soufflerie.datagen.run_artifact import LocalRunArtifactStore
from soufflerie.schemas import ArtifactRef, CaseConfig, canonical_sha256


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


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=SOLVE_TIMEOUT_SECONDS,
    max_containers=SMOKE_MAX_CONTAINERS,
    retries=REMOTE_RETRIES,
)
def summarize_run_remote(reference: ArtifactRef) -> SolveSummary:
    """Reload and verify a committed run before returning its small receipt."""

    parsed = ArtifactRef.model_validate_json(reference.model_dump_json())
    volume.reload()
    store = LocalRunArtifactStore(Path(VOLUME_MOUNT) / REMOTE_ARTIFACT_ROOT)
    run = store.open_run(parsed)
    provenance = run.metadata.provenance
    return SolveSummary.create(
        artifact=run.reference,
        case_id=run.metadata.case_id,
        source_revision=provenance.source_revision,
        device_class=provenance.device_class,
        wall_seconds=(provenance.completed_at - provenance.started_at).total_seconds(),
        gpu_seconds=provenance.gpu_seconds,
        final_state="succeeded",
    )


@app.local_entrypoint()
def main(smoke: bool = False, config: str = "") -> None:
    """Run the kernel smoke or one non-release standalone domain solve."""

    if smoke and config:
        raise RuntimeError("choose either --smoke or --config")
    if smoke:
        first = KernelSmokeResult.model_validate_json(
            kernel_smoke_remote.remote(settings.remote_gpu)
        )
        second = KernelSmokeResult.model_validate_json(
            kernel_smoke_remote.remote(settings.remote_gpu)
        )
        evidence = KernelSmokeEvidence.create(first, second)
        print(json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    if not config:
        raise RuntimeError("one of --smoke or --config is required")
    if checkout.source_dirty:
        raise RuntimeError("remote solve submission requires a clean source revision")
    case = load_config(Path(config), CaseConfig)
    design_id = canonical_sha256(
        {
            "schema_version": 1,
            "design_kind": "standalone-solve-v1",
            "shape": case.shape.model_dump(mode="json"),
            "reynolds": case.reynolds,
        }
    )[:20]
    sweep_digest = canonical_sha256(
        {
            "schema_version": 1,
            "operation_kind": "single",
            "case_id": case.case_id,
            "design_id": design_id,
            "source_revision": checkout.source_revision,
            "lock_sha256": checkout.lock_sha256,
            "requested_device_class": settings.remote_gpu,
        }
    )
    request = RemoteSolveRequest.create(
        operation_kind="single",
        sweep_digest=sweep_digest,
        design_id=design_id,
        split="test",
        case=case,
        requested_device_class=settings.remote_gpu,
        source_revision=checkout.source_revision,
        lock_sha256=checkout.lock_sha256,
        attempt_id=f"single-{uuid.uuid4().hex}",
    )
    reference = solve_remote.remote(
        encode_remote_model(request),
        f"single.{request.case.case_id}.{request.attempt_id}",
    )
    summary = summarize_run_remote.remote(reference)
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))


__all__ = [
    "kernel_smoke_remote",
    "main",
    "run_solver_case",
    "solve_remote",
    "summarize_run_remote",
]
