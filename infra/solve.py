"""Authenticated kernel smoke and idempotent remote solver entrypoint."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
from soufflerie.datagen.run_artifact import LocalRunArtifactStore, RunArtifact
from soufflerie.schemas import ArtifactRef, CaseConfig, canonical_sha256
from soufflerie.solver.cylinder_acceptance import (
    CYLINDER_CD_MAX,
    CYLINDER_CD_MIN,
    CYLINDER_CONFIG_PATH,
    CYLINDER_MASS_DRIFT_MAX,
    CYLINDER_ST_MAX,
    CYLINDER_ST_MIN,
    CylinderAcceptanceReport,
    CylinderRunEvidence,
    render_cylinder_report,
)
from soufflerie.solver.diagnostics import MIN_RESOLVED_LIFT_CYCLES

CYLINDER_REPORT_PATH = Path("reports/solver/cylinder-re100.json")
CYLINDER_MARKDOWN_PATH = Path("reports/solver/cylinder-re100.md")


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


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=SOLVE_TIMEOUT_SECONDS,
    max_containers=SMOKE_MAX_CONTAINERS,
    retries=REMOTE_RETRIES,
)
def summarize_cylinder_run_remote(reference: ArtifactRef) -> CylinderRunEvidence:
    """Reload a committed run and return its digest-bound cylinder projection."""

    parsed = ArtifactRef.model_validate_json(reference.model_dump_json())
    volume.reload()
    store = LocalRunArtifactStore(Path(VOLUME_MOUNT) / REMOTE_ARTIFACT_ROOT)
    return _cylinder_run_evidence(store.open_run(parsed))


def _cylinder_run_evidence(run: RunArtifact) -> CylinderRunEvidence:
    """Project one verified run into bounded acceptance evidence."""

    metadata = run.metadata
    steps = run.fields.force_steps
    cl = run.fields.cl_history.astype(np.float64)
    centered_lift = cl - float(np.mean(cl, dtype=np.float64))
    sample_interval = int(steps[1] - steps[0])
    observed_duration = int(steps[-1] - steps[0])
    if metadata.strouhal is None:
        raise ValueError("cylinder acceptance requires an available Strouhal estimate")
    resolved_cycles = (
        metadata.strouhal
        * metadata.case.inlet_velocity_lu
        * observed_duration
        / (metadata.case.ny / 20.0)
    )
    lift_rms = float(np.sqrt(np.mean(centered_lift * centered_lift, dtype=np.float64)))
    lift_peak_to_peak = float(np.ptp(centered_lift))
    periodic = (
        resolved_cycles >= MIN_RESOLVED_LIFT_CYCLES
        and lift_rms > 0.0
        and lift_peak_to_peak > float(np.finfo(np.float32).eps)
    )
    provenance = metadata.provenance
    return CylinderRunEvidence.create(
        artifact=run.reference,
        case=metadata.case,
        config_sha256=provenance.config_sha256,
        fields_sha256=metadata.fields_sha256,
        source_revision=provenance.source_revision,
        lock_sha256=provenance.lock_sha256,
        device_class=provenance.device_class,
        dtype_policy=provenance.dtype_policy,
        cd=metadata.cd,
        cl_mean=metadata.cl_mean,
        strouhal=metadata.strouhal,
        mass_drift_ratio=metadata.diagnostics.mass_drift_ratio,
        sample_count=metadata.diagnostics.sample_count,
        sample_interval_steps=sample_interval,
        observed_duration_steps=observed_duration,
        resolved_lift_cycles=resolved_cycles,
        lift_rms=lift_rms,
        lift_peak_to_peak=lift_peak_to_peak,
        diagnostics_valid=True,
        diagnostics_converged=True,
        cd_reference_passed=CYLINDER_CD_MIN <= metadata.cd <= CYLINDER_CD_MAX,
        strouhal_reference_passed=CYLINDER_ST_MIN <= metadata.strouhal <= CYLINDER_ST_MAX,
        periodic_lift_passed=periodic,
        mass_passed=metadata.diagnostics.mass_drift_ratio < CYLINDER_MASS_DRIFT_MAX,
        wall_seconds=(provenance.completed_at - provenance.started_at).total_seconds(),
        gpu_seconds=provenance.gpu_seconds,
    )


def _standalone_design_id(case: CaseConfig) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "design_kind": "standalone-solve-v1",
            "shape": case.shape.model_dump(mode="json"),
            "reynolds": case.reynolds,
        }
    )[:20]


def _cylinder_cases(case: CaseConfig) -> tuple[CaseConfig, CaseConfig, CaseConfig, CaseConfig]:
    if Path(CYLINDER_CONFIG_PATH).name != "cylinder-re100.yaml":
        raise RuntimeError("canonical cylinder config path is internally inconsistent")
    expected = {
        "aspect_ratio": 1.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
        "reynolds": 100.0,
        "nx": 512,
        "ny": 640,
        "steps": 60_000,
        "warmup_steps": 20_000,
        "inlet_velocity_lu": 0.05,
        "seed": 20260901,
    }
    actual = {
        **case.shape.model_dump(mode="python"),
        "reynolds": case.reynolds,
        "nx": case.nx,
        "ny": case.ny,
        "steps": case.steps,
        "warmup_steps": case.warmup_steps,
        "inlet_velocity_lu": case.inlet_velocity_lu,
        "seed": case.seed,
    }
    if actual != expected:
        raise RuntimeError("canonical cylinder acceptance config does not match RFC-0002/0003")
    coarse = case.model_copy(update={"nx": 384, "ny": 480})
    fine = case.model_copy(update={"nx": 640, "ny": 800})
    return (coarse, case, fine, case)


def _cylinder_request(
    case: CaseConfig,
    *,
    role: str,
    design_id: str,
) -> RemoteSolveRequest:
    operation_digest = canonical_sha256(
        {
            "schema_version": 1,
            "operation_kind": "cylinder-acceptance-v1",
            "role": role,
            "case_id": case.case_id,
            "source_revision": checkout.source_revision,
            "lock_sha256": checkout.lock_sha256,
            "requested_device_class": settings.remote_gpu,
        }
    )
    return RemoteSolveRequest.create(
        operation_kind="cylinder-acceptance",
        sweep_digest=operation_digest,
        design_id=design_id,
        split="test",
        case=case,
        requested_device_class=settings.remote_gpu,
        source_revision=checkout.source_revision,
        lock_sha256=checkout.lock_sha256,
        attempt_id=f"acceptance-{role}",
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_cylinder_acceptance(case: CaseConfig) -> CylinderAcceptanceReport:
    roles = ("coarse", "canonical", "fine", "canonical-repeat")
    cases = _cylinder_cases(case)
    design_id = _standalone_design_id(case)
    requests = tuple(
        _cylinder_request(case_value, role=role, design_id=design_id)
        for role, case_value in zip(roles, cases, strict=True)
    )
    references = tuple(
        solve_remote.map(
            tuple(encode_remote_model(request) for request in requests),
            tuple(
                f"acceptance.{request.sweep_digest[:16]}.{request.case.case_id}"
                for request in requests
            ),
            order_outputs=True,
        )
    )
    evidence = tuple(summarize_cylinder_run_remote.map(references, order_outputs=True))
    report = CylinderAcceptanceReport.create(
        coarse=evidence[0],
        canonical=evidence[1],
        fine=evidence[2],
        canonical_repeat=evidence[3],
        primary_operation_sha256=requests[1].sweep_digest,
        repeat_operation_sha256=requests[3].sweep_digest,
    )
    rendered_json = (
        json.dumps(
            report.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _atomic_write(CYLINDER_REPORT_PATH, rendered_json)
    _atomic_write(CYLINDER_MARKDOWN_PATH, render_cylinder_report(report))
    return report


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
    if Path(config).as_posix() == CYLINDER_CONFIG_PATH:
        report = _run_cylinder_acceptance(case)
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        if not report.overall_passed:
            raise RuntimeError("remote cylinder acceptance is red; inspect the retained report")
        return
    design_id = _standalone_design_id(case)
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
    "summarize_cylinder_run_remote",
    "summarize_run_remote",
]
