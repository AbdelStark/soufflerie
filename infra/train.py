"""Remote three-seed training entrypoint over the provider-neutral pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from infra.app import app, checkout, image, settings, volume
from infra.policy import (
    REMOTE_RETRIES,
    TRAIN_MAX_CONTAINERS,
    TRAIN_TIMEOUT_SECONDS,
    VOLUME_MOUNT,
)
from infra.remote_execution import REMOTE_ARTIFACT_ROOT, encode_remote_model, parse_remote_model
from infra.runtime_manifest import load_build_manifest
from infra.train_validate_execution import (
    RemoteTrainingRequest,
    TrainingRunIndex,
    TrainingSeedReceipt,
    assert_request_matches_build,
    load_workflow_request,
    publish_workflow_request,
)
from infra.training_pipeline import execute_training_seed
from soufflerie.config import TrainingConfig, load_config
from soufflerie.schemas import ArtifactRef


def _artifact_root() -> Path:
    return Path(VOLUME_MOUNT) / REMOTE_ARTIFACT_ROOT


def _checked_dataset_reference(path: Path, *, dataset_id: str) -> ArtifactRef:
    """Load the canonical dataset reference from a checked sweep summary."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = value["dataset_reference"]
        reference = ArtifactRef.model_validate(payload)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError("dataset summary does not contain a valid ArtifactRef") from error
    if reference.artifact_type != "dataset" or reference.artifact_id != dataset_id:
        raise RuntimeError("dataset summary does not match the training config")
    return reference


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=TRAIN_TIMEOUT_SECONDS,
    max_containers=TRAIN_MAX_CONTAINERS,
    retries=REMOTE_RETRIES,
)
def stage_training_request_remote(request_json: bytes) -> ArtifactRef:
    """Reparse and atomically stage one immutable training request."""

    request = parse_remote_model(request_json, RemoteTrainingRequest)
    build = load_build_manifest()
    assert_request_matches_build(request, build)
    volume.reload()
    root = _artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    reference = publish_workflow_request(
        root,
        request_json,
        kind="training",
        model=RemoteTrainingRequest,
    )
    volume.commit()
    return reference


@app.function(
    image=image,
    gpu=settings.remote_gpu,
    volumes={VOLUME_MOUNT: volume},
    timeout=TRAIN_TIMEOUT_SECONDS,
    max_containers=TRAIN_MAX_CONTAINERS,
    retries=REMOTE_RETRIES,
)
def train_remote(config_ref: ArtifactRef, seed: int) -> TrainingSeedReceipt:
    """Verify one seed request, execute it, and return a small artifact receipt."""

    build = load_build_manifest()
    volume.reload()
    request = load_workflow_request(
        _artifact_root(),
        config_ref,
        kind="training",
        model=RemoteTrainingRequest,
    )
    return execute_training_seed(
        request,
        seed=seed,
        root=_artifact_root(),
        build=build,
        reload_volume=volume.reload,
        commit_volume=volume.commit,
    )


@app.local_entrypoint()
def main(
    config: str = "",
    dataset_summary: str = "reports/dataset/sweep-summary.json",
    output: str = "",
) -> None:
    """Run all three declared seeds concurrently and print their receipts."""

    if not config:
        raise RuntimeError("--config is required")
    if checkout.source_dirty:
        raise RuntimeError("remote training submission requires a clean source revision")
    training_config = load_config(Path(config), TrainingConfig)
    dataset = _checked_dataset_reference(
        Path(dataset_summary),
        dataset_id=training_config.dataset_id,
    )
    request = RemoteTrainingRequest.create(
        config=training_config,
        dataset=dataset,
        requested_device_class=settings.remote_gpu,
        source_revision=checkout.source_revision,
        lock_sha256=checkout.lock_sha256,
    )
    request_reference = stage_training_request_remote.remote(encode_remote_model(request))
    receipts = tuple(
        train_remote.map(
            (request_reference,) * len(training_config.seeds),
            training_config.seeds,
            order_outputs=True,
        )
    )
    typed = cast(tuple[TrainingSeedReceipt, TrainingSeedReceipt, TrainingSeedReceipt], receipts)
    index = TrainingRunIndex.create(request=request, receipts=typed)
    rendered = json.dumps(index.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if output:
        target = Path(output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")


__all__ = ["main", "stage_training_request_remote", "train_remote"]
