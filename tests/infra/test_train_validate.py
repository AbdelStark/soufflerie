from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from infra.remote_execution import encode_remote_model
from infra.runtime_manifest import RuntimeBuildManifest
from infra.train_validate_execution import (
    ExecutionAccounting,
    RemoteTrainingRequest,
    RemoteValidationRequest,
    TrainingRunIndex,
    TrainingSeedReceipt,
    ValidationReceipt,
    assert_report_matches_request,
    assert_request_matches_build,
    assert_training_seed,
    load_workflow_request,
    publish_workflow_request,
)
from soufflerie.config import TrainingConfig, ValidationConfig, load_config
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ArtifactRef, Provenance
from soufflerie.validation.gates import ValidationReport

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _reference(kind: str, digest: str) -> ArtifactRef:
    plural = "models" if kind == "model" else "datasets"
    return ArtifactRef(
        artifact_type=kind,
        artifact_id=digest[:20],
        sha256=digest,
        size_bytes=123,
        uri=f"{plural}/{digest[:20]}",
    )


def _training_request() -> RemoteTrainingRequest:
    dataset = _reference("dataset", "a" * 64)
    return RemoteTrainingRequest.create(
        config=TrainingConfig(dataset_id=dataset.artifact_id, seeds=(17, 23, 31)),
        dataset=dataset,
        requested_device_class="L40S",
        source_revision="1" * 40,
        lock_sha256="2" * 64,
    )


def _validation_request(
    *,
    source_revision: str = "1" * 40,
    model_source_revision: str | None = None,
) -> RemoteValidationRequest:
    dataset = _reference("dataset", "a" * 64)
    models = cast(
        tuple[ArtifactRef, ArtifactRef, ArtifactRef],
        tuple(_reference("model", character * 64) for character in "bcd"),
    )
    config = ValidationConfig(
        dataset_id=dataset.artifact_id,
        ensemble_model_ids=cast(
            tuple[str, str, str], tuple(reference.artifact_id for reference in models)
        ),
        baseline_ids=("e" * 20, "f" * 20),
        report_seed=7,
        bootstrap_resamples=100,
    )
    return RemoteValidationRequest.create(
        config=config,
        dataset=dataset,
        models=models,
        baseline_sha256s=("e" * 64, "f" * 64),
        selected_model_id=models[1].artifact_id,
        solver_sha256="9" * 64,
        requested_device_class="L40S",
        source_revision=source_revision,
        lock_sha256="2" * 64,
        model_source_revision=model_source_revision,
    )


def _build() -> RuntimeBuildManifest:
    return RuntimeBuildManifest.create(
        lock_sha256="2" * 64,
        source_revision="1" * 40,
        source_dirty=False,
        packages={"soufflerie": "0.1.0"},
    )


def _accounting() -> ExecutionAccounting:
    started = datetime(2026, 9, 3, tzinfo=UTC)
    return ExecutionAccounting(
        started_at=started,
        completed_at=started + timedelta(seconds=3),
        wall_seconds=3.0,
        gpu_seconds=2.0,
        peak_allocated_bytes=10,
        peak_reserved_bytes=20,
        device_class="L40S",
        device_name="NVIDIA L40S",
        precision="bf16",
        source_revision="1" * 40,
        lock_sha256="2" * 64,
    )


def test_canonical_validation_configs_bind_checked_training_index() -> None:
    index = TrainingRunIndex.model_validate_json(
        (PROJECT_ROOT / "reports" / "training" / "index.json").read_bytes()
    )
    expected_models = tuple(receipt.model.artifact_id for receipt in index.receipts)
    expected_baselines = tuple(value[:20] for value in index.receipts[0].baseline_sha256s)

    for name in ("release-v1.yaml", "smoke.yaml"):
        config = load_config(PROJECT_ROOT / "configs" / "validation" / name, ValidationConfig)
        assert config.dataset_id == index.request.dataset.artifact_id
        assert config.ensemble_model_ids == expected_models
        assert config.baseline_ids == expected_baselines


def test_training_request_round_trip_stage_and_load_is_content_addressed(
    tmp_path: Path,
) -> None:
    request = _training_request()
    content = encode_remote_model(request)
    reference = publish_workflow_request(
        tmp_path,
        content,
        kind="training",
        model=RemoteTrainingRequest,
    )
    assert (
        load_workflow_request(
            tmp_path,
            reference,
            kind="training",
            model=RemoteTrainingRequest,
        )
        == request
    )
    assert (
        publish_workflow_request(
            tmp_path,
            content,
            kind="training",
            model=RemoteTrainingRequest,
        )
        == reference
    )
    assert_request_matches_build(request, _build())
    for seed in request.config.seeds:
        assert_training_seed(request, seed)


def test_training_rejects_seed_request_build_and_storage_identity_drift(
    tmp_path: Path,
) -> None:
    request = _training_request()
    with pytest.raises(ArtifactIntegrityError, match="SEED_MISMATCH"):
        assert_training_seed(request, 99)
    with pytest.raises(ArtifactIntegrityError, match="SEED_MISMATCH"):
        assert_training_seed(request, cast(int, True))
    with pytest.raises(RuntimeError, match="source revision"):
        assert_request_matches_build(
            request,
            _build().model_copy(update={"source_revision": "3" * 40}),
        )

    content = encode_remote_model(request)
    reference = publish_workflow_request(
        tmp_path,
        content,
        kind="training",
        model=RemoteTrainingRequest,
    )
    with pytest.raises(ArtifactIntegrityError, match="request reference is incoherent"):
        load_workflow_request(
            tmp_path,
            reference.model_copy(update={"uri": f"requests/training/{'0' * 64}.json"}),
            kind="training",
            model=RemoteTrainingRequest,
        )

    tampered = request.model_dump(mode="python")
    tampered["experiment_id"] = "0" * 20
    with pytest.raises(ValidationError, match="complete training identity"):
        RemoteTrainingRequest.model_validate(tampered)

    with pytest.raises(ValidationError, match="at most 100"):
        RemoteTrainingRequest.create(
            config=request.config.model_copy(update={"epochs": 101}),
            dataset=request.dataset,
            requested_device_class=request.requested_device_class,
            source_revision=request.source_revision,
            lock_sha256=request.lock_sha256,
        )


def test_validation_request_binds_every_parent_and_refuses_report_mismatch(
    tmp_path: Path,
) -> None:
    request = _validation_request()
    assert_request_matches_build(request, _build())
    content = encode_remote_model(request)
    reference = publish_workflow_request(
        tmp_path,
        content,
        kind="validation",
        model=RemoteValidationRequest,
    )
    assert (
        load_workflow_request(
            tmp_path,
            reference,
            kind="validation",
            model=RemoteValidationRequest,
        )
        == request
    )
    provenance = Provenance.model_construct(
        source_revision=request.source_revision,
        source_dirty=False,
        python_version="3.11.14",
        lock_sha256=request.lock_sha256,
        packages={"soufflerie": "0.1.0"},
        os="linux",
        architecture="x86_64",
        device_class="L40S",
        dtype_policy="bf16",
        config_sha256=request.config.config_digest,
        parent_sha256=request.expected_parent_sha256,
        seeds=(),
        deterministic=True,
        started_at=datetime(2026, 9, 3, tzinfo=UTC),
        completed_at=datetime(2026, 9, 3, tzinfo=UTC),
        gpu_seconds=0.0,
    )
    report = ValidationReport.model_construct(
        report_id="0" * 20,
        report_sha256="0" * 64,
        dataset_id=request.dataset.artifact_id,
        selected_model_id=request.selected_model_id,
        ensemble_model_ids=request.config.ensemble_model_ids,
        baseline_ids=request.config.baseline_ids,
        metrics={},
        gates=(),
        overall_status="red",
        provenance=provenance,
    )
    assert_report_matches_request(report, request)

    wrong = report.model_copy(update={"selected_model_id": request.models[0].artifact_id})
    with pytest.raises(ArtifactIntegrityError, match="MODEL_MISMATCH"):
        assert_report_matches_request(wrong, request)
    wrong_provenance = provenance.model_copy(
        update={"parent_sha256": {**request.expected_parent_sha256, "solver": "8" * 64}}
    )
    wrong = report.model_copy(update={"provenance": wrong_provenance})
    with pytest.raises(ArtifactIntegrityError, match="PARENT_MISMATCH"):
        assert_report_matches_request(wrong, request)


def test_validation_request_separates_model_and_report_source_revisions() -> None:
    request = _validation_request(
        source_revision="3" * 40,
        model_source_revision="1" * 40,
    )
    assert request.model_source_revision == "1" * 40
    assert request.source_revision == "3" * 40
    assert_request_matches_build(
        request,
        _build().model_copy(update={"source_revision": "3" * 40}),
    )

    tampered = request.model_dump(mode="python")
    tampered["model_source_revision"] = "4" * 40
    with pytest.raises(ValidationError, match="request_digest"):
        RemoteValidationRequest.model_validate(tampered)


def test_receipts_bind_terminal_artifacts_resources_and_full_parent_digests() -> None:
    model = _reference("model", "b" * 64)
    training = TrainingSeedReceipt.create(
        experiment_id="c" * 20,
        seed=17,
        completed_epochs=100,
        model=model,
        best_checkpoint_id="d" * 20,
        validation_score=0.25,
        selection_id="6" * 20,
        selection_sha256="6" * 64,
        deployable_seed=17,
        baseline_sha256s=("e" * 64, "f" * 64),
        solver_sha256="9" * 64,
        accounting=_accounting(),
        parent_sha256={
            "architecture": "4" * 64,
            "best_checkpoint": "d" * 64,
            "config": "5" * 64,
            "dataset": "a" * 64,
            "model": model.sha256,
        },
    )
    assert training.final_state == "succeeded"

    request = _validation_request()
    report = ArtifactRef(
        artifact_type="validation_report",
        artifact_id="7" * 20,
        sha256="7" * 64,
        size_bytes=456,
        uri=f"validation/{'7' * 20}",
    )
    validation = ValidationReceipt.create(
        report=report,
        overall_status="red",
        accounting=_accounting(),
        parent_sha256=request.expected_parent_sha256,
    )
    assert validation.final_state == "succeeded"

    bad = training.model_dump(mode="python")
    bad["parent_sha256"]["model"] = "8" * 64
    with pytest.raises(ValidationError, match="model parent differs"):
        TrainingSeedReceipt.model_validate(bad)


def test_training_index_is_a_complete_identity_checked_validation_handoff() -> None:
    request = _training_request()
    receipts: list[TrainingSeedReceipt] = []
    for seed, character, score in zip(request.config.seeds, "bcd", (0.1, 0.2, 0.3), strict=True):
        model = _reference("model", character * 64)
        checkpoint_digest = f"{seed:064x}"
        receipts.append(
            TrainingSeedReceipt.create(
                experiment_id=request.experiment_id,
                seed=seed,
                completed_epochs=request.config.epochs,
                model=model,
                best_checkpoint_id=checkpoint_digest[:20],
                validation_score=score,
                selection_id="6" * 20,
                selection_sha256="6" * 64,
                deployable_seed=17,
                baseline_sha256s=("e" * 64, "f" * 64),
                solver_sha256="9" * 64,
                accounting=_accounting(),
                parent_sha256={
                    "architecture": request.architecture_sha256,
                    "best_checkpoint": checkpoint_digest,
                    "config": request.config.config_digest,
                    "dataset": request.dataset.sha256,
                    "model": model.sha256,
                },
            )
        )
    typed = cast(
        tuple[TrainingSeedReceipt, TrainingSeedReceipt, TrainingSeedReceipt],
        tuple(receipts),
    )
    index = TrainingRunIndex.create(request=request, receipts=typed)
    assert index.selected_model_id == receipts[0].model.artifact_id
    assert TrainingRunIndex.model_validate_json(index.model_dump_json()) == index

    tampered = index.model_dump(mode="python")
    tampered_receipts = list(tampered["receipts"])
    original = receipts[1]
    tampered_receipts[1] = TrainingSeedReceipt.create(
        experiment_id=original.experiment_id,
        seed=original.seed,
        completed_epochs=original.completed_epochs,
        model=original.model,
        best_checkpoint_id=original.best_checkpoint_id,
        validation_score=original.validation_score,
        selection_id=original.selection_id,
        selection_sha256=original.selection_sha256,
        deployable_seed=23,
        baseline_sha256s=original.baseline_sha256s,
        solver_sha256=original.solver_sha256,
        accounting=original.accounting,
        parent_sha256=original.parent_sha256,
    )
    tampered["receipts"] = tuple(tampered_receipts)
    with pytest.raises(ValidationError, match="deployable seed"):
        TrainingRunIndex.model_validate(tampered)
