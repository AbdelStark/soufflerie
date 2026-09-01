"""Modal adapter for the provider-neutral idempotent solve transaction."""

from __future__ import annotations

from pathlib import Path

from infra.app import app, image, settings, volume
from infra.policy import (
    REMOTE_RETRIES,
    SOLVE_MAX_CONTAINERS,
    SOLVE_TIMEOUT_SECONDS,
    VOLUME_MOUNT,
)
from infra.remote_execution import REMOTE_ARTIFACT_ROOT
from infra.runtime_manifest import load_build_manifest
from infra.solve_execution import execute_solve_request, run_solver_case
from soufflerie.schemas import ArtifactRef


def _artifact_root() -> Path:
    return Path(VOLUME_MOUNT) / REMOTE_ARTIFACT_ROOT


@app.function(
    image=image,
    gpu=settings.remote_gpu,
    volumes={VOLUME_MOUNT: volume},
    timeout=SOLVE_TIMEOUT_SECONDS,
    max_containers=SOLVE_MAX_CONTAINERS,
    retries=REMOTE_RETRIES,
)
def solve_remote(case_json: bytes, correlation_id: str) -> ArtifactRef:
    """Reparse, claim, solve, publish, and commit one idempotent case."""

    build = load_build_manifest()
    return execute_solve_request(
        case_json,
        correlation_id,
        root=_artifact_root(),
        build=build,
        reload_volume=volume.reload,
        commit_volume=volume.commit,
        solve_case=run_solver_case,
    )


__all__ = ["execute_solve_request", "run_solver_case", "solve_remote"]
