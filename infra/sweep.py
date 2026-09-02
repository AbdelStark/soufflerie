"""Resumable remote sweep orchestration over idempotent solve workers."""

from __future__ import annotations

import json
from pathlib import Path

from infra.app import app, checkout, image, settings, volume
from infra.policy import (
    REMOTE_RETRIES,
    SOLVE_TIMEOUT_SECONDS,
    SWEEP_MAX_CONTAINERS,
    SWEEP_TIMEOUT_SECONDS,
    VOLUME_MOUNT,
)
from infra.remote_execution import (
    REMOTE_ARTIFACT_ROOT,
    RemoteSweepRequest,
    SweepSummary,
    encode_remote_model,
    parse_remote_model,
    publish_remote_request,
)
from infra.runtime_manifest import load_build_manifest
from infra.solve_worker import solve_remote
from infra.sweep_execution import (
    assert_sweep_request_matches_build,
    orchestrate_sweep,
)
from soufflerie.config import SweepConfig, load_config
from soufflerie.schemas import ArtifactRef


def _artifact_root() -> Path:
    return Path(VOLUME_MOUNT) / REMOTE_ARTIFACT_ROOT


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=SOLVE_TIMEOUT_SECONDS,
    max_containers=SWEEP_MAX_CONTAINERS,
    retries=REMOTE_RETRIES,
)
def stage_sweep_request_remote(request_json: bytes) -> ArtifactRef:
    """Reparse and atomically stage one immutable bounded sweep request."""

    request = parse_remote_model(request_json, RemoteSweepRequest)
    build = load_build_manifest()
    assert_sweep_request_matches_build(request, build)
    volume.reload()
    root = _artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    reference = publish_remote_request(root, request_json)
    volume.commit()
    return reference


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=SWEEP_TIMEOUT_SECONDS,
    max_containers=SWEEP_MAX_CONTAINERS,
    retries=REMOTE_RETRIES,
)
def sweep_remote(config_ref: ArtifactRef) -> SweepSummary:
    """Verify a staged request, fan out bounded work, and resume missing cases."""

    build = load_build_manifest()

    def submit(payloads: tuple[bytes, ...], owners: tuple[str, ...]) -> None:
        # State, rather than provider exception serialization, is authoritative.
        list(
            solve_remote.map(
                payloads,
                owners,
                order_outputs=True,
                return_exceptions=True,
            )
        )

    return orchestrate_sweep(
        config_ref,
        root=_artifact_root(),
        build=build,
        reload_volume=volume.reload,
        commit_volume=volume.commit,
        submit=submit,
    )


@app.local_entrypoint()
def main(
    config: str = "",
    n: int = 0,
    force_failure_once: bool = True,
    output: str = "",
) -> None:
    """Run the canonical sweep, or the explicitly selected eight-case smoke."""

    if not config:
        raise RuntimeError("--config is required")
    if n not in {0, 8}:
        raise RuntimeError("--n accepts only 8 for smoke; omit it for the 1,000-case design")
    if checkout.source_dirty:
        raise RuntimeError("remote sweep submission requires a clean source revision")
    sweep_config = load_config(Path(config), SweepConfig)
    if n == 8:
        request = RemoteSweepRequest.create(
            config=sweep_config,
            requested_device_class=settings.remote_gpu,
            source_revision=checkout.source_revision,
            lock_sha256=checkout.lock_sha256,
            force_failure_once=force_failure_once,
        )
    else:
        request = RemoteSweepRequest.create_canonical(
            config=sweep_config,
            requested_device_class=settings.remote_gpu,
            source_revision=checkout.source_revision,
            lock_sha256=checkout.lock_sha256,
        )
    request_reference = stage_sweep_request_remote.remote(encode_remote_model(request))
    summary = sweep_remote.remote(request_reference)
    rendered = json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if output:
        output_path = Path(output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    if summary.final_state != "succeeded":
        raise RuntimeError("remote sweep is incomplete; rerun to resume eligible cases")


__all__ = [
    "main",
    "orchestrate_sweep",
    "stage_sweep_request_remote",
    "sweep_remote",
]
