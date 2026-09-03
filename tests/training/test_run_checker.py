from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from infra.train_validate_execution import (
    ExecutionAccounting,
    RemoteTrainingRequest,
    TrainingRunIndex,
    TrainingSeedReceipt,
)
from scripts.check_training_run import check_training_run, load_training_index
from soufflerie.config import TrainingConfig
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ArtifactRef


def _dataset() -> ArtifactRef:
    return ArtifactRef(
        artifact_type="dataset",
        artifact_id="a" * 20,
        sha256="a" * 64,
        size_bytes=123,
        uri=f"datasets/{'a' * 20}",
    )


def _index(*, wall_seconds: float = 120.0) -> TrainingRunIndex:
    dataset = _dataset()
    config = TrainingConfig(
        dataset_id=dataset.artifact_id,
        seeds=(17, 23, 31),
        epochs=100,
        precision="bf16",
    )
    request = RemoteTrainingRequest.create(
        config=config,
        dataset=dataset,
        requested_device_class="L40S",
        source_revision="1" * 40,
        lock_sha256="2" * 64,
    )
    completed = datetime(2026, 9, 3, tzinfo=UTC)
    receipts = []
    for seed, character, score in zip(config.seeds, "bcd", (0.1, 0.2, 0.3), strict=True):
        model = ArtifactRef(
            artifact_type="model",
            artifact_id=character * 20,
            sha256=character * 64,
            size_bytes=456,
            uri=f"models/{character * 20}",
        )
        checkpoint_sha256 = f"{seed:064x}"
        receipts.append(
            TrainingSeedReceipt.create(
                experiment_id=request.experiment_id,
                seed=seed,
                completed_epochs=100,
                model=model,
                best_checkpoint_id=checkpoint_sha256[:20],
                validation_score=score,
                selection_id="6" * 20,
                selection_sha256="6" * 64,
                deployable_seed=17,
                baseline_sha256s=("e" * 64, "f" * 64),
                solver_sha256="9" * 64,
                accounting=ExecutionAccounting(
                    started_at=completed - timedelta(seconds=wall_seconds),
                    completed_at=completed,
                    wall_seconds=wall_seconds,
                    gpu_seconds=min(100.0, wall_seconds),
                    peak_allocated_bytes=10,
                    peak_reserved_bytes=20,
                    device_class="L40S",
                    device_name="NVIDIA L40S",
                    precision="bf16",
                    source_revision=request.source_revision,
                    lock_sha256=request.lock_sha256,
                ),
                parent_sha256={
                    "architecture": request.architecture_sha256,
                    "best_checkpoint": checkpoint_sha256,
                    "config": config.config_digest,
                    "dataset": dataset.sha256,
                    "model": model.sha256,
                },
            )
        )
    return TrainingRunIndex.create(
        request=request,
        receipts=cast(
            tuple[TrainingSeedReceipt, TrainingSeedReceipt, TrainingSeedReceipt],
            tuple(receipts),
        ),
    )


def test_checker_accepts_exact_canonical_run_and_encoding(tmp_path: Path) -> None:
    index = _index()
    check_training_run(index, config=index.request.config, dataset=index.request.dataset)
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps(index.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert load_training_index(path) == index


def test_checker_rejects_budget_config_and_noncanonical_encoding(tmp_path: Path) -> None:
    slow = _index(wall_seconds=3600.0)
    with pytest.raises(ArtifactIntegrityError, match="TRAIN-RUN-7 BUDGET"):
        check_training_run(slow, config=slow.request.config, dataset=slow.request.dataset)

    index = _index()
    changed = index.request.config.model_copy(update={"learning_rate": 0.002})
    with pytest.raises(ArtifactIntegrityError, match="TRAIN-RUN-3 CONFIG"):
        check_training_run(index, config=changed, dataset=index.request.dataset)

    path = tmp_path / "index.json"
    path.write_text(index.model_dump_json(), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="TRAIN-RUN-1 ENCODING"):
        load_training_index(path)
