"""Remote validation entrypoint over the provider-neutral report pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from infra.app import app, checkout, image, settings, volume
from infra.policy import (
    REMOTE_RETRIES,
    VALIDATE_MAX_CONTAINERS,
    VALIDATE_TIMEOUT_SECONDS,
    VOLUME_MOUNT,
)
from infra.remote_execution import REMOTE_ARTIFACT_ROOT, encode_remote_model, parse_remote_model
from infra.runtime_manifest import load_build_manifest
from infra.train_validate_execution import (
    RemoteValidationRequest,
    TrainingRunIndex,
    ValidationReceipt,
    assert_request_matches_build,
    load_workflow_request,
    publish_workflow_request,
)
from infra.validation_pipeline import execute_validation
from soufflerie.config import ValidationConfig, load_config
from soufflerie.schemas import ArtifactRef


def _artifact_root() -> Path:
    return Path(VOLUME_MOUNT) / REMOTE_ARTIFACT_ROOT


def _load_training_index(path: Path) -> TrainingRunIndex:
    """Load the complete, digest-bound output of the three-seed training run."""

    try:
        content = path.read_bytes()
    except OSError as error:
        raise RuntimeError("training index cannot be read") from error
    try:
        return TrainingRunIndex.model_validate_json(content)
    except (TypeError, ValueError) as error:
        raise RuntimeError("training index is invalid") from error


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=VALIDATE_TIMEOUT_SECONDS,
    max_containers=VALIDATE_MAX_CONTAINERS,
    retries=REMOTE_RETRIES,
)
def stage_validation_request_remote(request_json: bytes) -> ArtifactRef:
    """Reparse and atomically stage one immutable validation request."""

    request = parse_remote_model(request_json, RemoteValidationRequest)
    build = load_build_manifest()
    assert_request_matches_build(request, build)
    volume.reload()
    root = _artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    reference = publish_workflow_request(
        root,
        request_json,
        kind="validation",
        model=RemoteValidationRequest,
    )
    volume.commit()
    return reference


@app.function(
    image=image,
    gpu=settings.remote_gpu,
    volumes={VOLUME_MOUNT: volume},
    timeout=VALIDATE_TIMEOUT_SECONDS,
    max_containers=VALIDATE_MAX_CONTAINERS,
    retries=REMOTE_RETRIES,
)
def validate_remote(config_ref: ArtifactRef) -> ValidationReceipt:
    """Verify all immutable parents, run validation, and publish the report."""

    build = load_build_manifest()
    volume.reload()
    request = load_workflow_request(
        _artifact_root(),
        config_ref,
        kind="validation",
        model=RemoteValidationRequest,
    )
    return execute_validation(
        request,
        root=_artifact_root(),
        build=build,
        reload_volume=volume.reload,
        commit_volume=volume.commit,
    )


@app.local_entrypoint()
def main(
    config: str = "",
    training_index: str = "reports/training/index.json",
    output: str = "",
) -> None:
    """Assemble exact training parents, run validation, and print its receipt."""

    if not config:
        raise RuntimeError("--config is required")
    if checkout.source_dirty:
        raise RuntimeError("remote validation submission requires a clean source revision")
    validation_config = load_config(Path(config), ValidationConfig)
    index = _load_training_index(Path(training_index))
    if validation_config.dataset_id != index.request.dataset.artifact_id:
        raise RuntimeError("validation config and training dataset differ")
    receipts_by_model = {item.model.artifact_id: item for item in index.receipts}
    try:
        ordered_receipts = tuple(
            receipts_by_model[model_id] for model_id in validation_config.ensemble_model_ids
        )
    except KeyError as error:
        raise RuntimeError("validation config and training models differ") from error
    baseline_sha256s = index.receipts[0].baseline_sha256s
    if tuple(value[:20] for value in baseline_sha256s) != validation_config.baseline_ids:
        raise RuntimeError("validation config and fitted baseline identities differ")
    if index.request.lock_sha256 != checkout.lock_sha256:
        raise RuntimeError("training and validation lock digests differ")
    request = RemoteValidationRequest.create(
        config=validation_config,
        dataset=index.request.dataset,
        models=(
            ordered_receipts[0].model,
            ordered_receipts[1].model,
            ordered_receipts[2].model,
        ),
        baseline_sha256s=baseline_sha256s,
        selected_model_id=index.selected_model_id,
        solver_sha256=index.receipts[0].solver_sha256,
        requested_device_class=settings.remote_gpu,
        source_revision=checkout.source_revision,
        lock_sha256=checkout.lock_sha256,
        precision=index.request.config.precision,
        model_source_revision=index.request.source_revision,
    )
    if (
        request.source_revision != checkout.source_revision
        or request.lock_sha256 != checkout.lock_sha256
        or request.requested_device_class != settings.remote_gpu
    ):
        raise RuntimeError("validation request does not match the local submission identity")
    request_reference = stage_validation_request_remote.remote(encode_remote_model(request))
    receipt = validate_remote.remote(request_reference)
    rendered = json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if output:
        target = Path(output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")


__all__ = ["main", "stage_validation_request_remote", "validate_remote"]
