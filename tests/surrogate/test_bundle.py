from __future__ import annotations

import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from soufflerie.errors import (
    ArtifactIntegrityError,
    DependencyUnavailableError,
    DeviceUnavailableError,
)
from soufflerie.schemas import ArrayDescriptor, canonical_json_bytes, sha256_bytes
from soufflerie.surrogate.architecture import FnoArchitecture
from soufflerie.surrogate.bundle import (
    EXPECTED_MODEL_TENSORS,
    MODEL_TENSOR_BYTES,
    MODEL_WEIGHTS_FILE_CAP_BYTES,
    CompatibilityRange,
    LocalModelBundleStore,
    ModelBundleMetadata,
    ModelCardGate,
    ModelTensorDescriptor,
    build_model_bundle,
    instantiate_bundle_predictor,
    render_model_card,
    snapshot_fno_weights,
)
from soufflerie.surrogate.fno import FnoPredictor
from soufflerie.surrogate.preprocessing import (
    MODEL_SPATIAL_SHAPE,
    PredictionBatch,
)
from tests.model_bundle_helpers import (
    CODE_REVISION,
    DATASET_ID,
    DATASET_SHA256,
    EXPERIMENT_ID,
    LOCK_DIGEST,
    make_test_bundle,
    model_card,
    preprocessing_statistics,
    zero_weights,
)


def test_tensor_allowlist_matches_the_fixed_fno_and_supports_five_dimensions() -> None:
    assert len(EXPECTED_MODEL_TENSORS) == 28
    assert MODEL_TENSOR_BYTES == 151_123_216
    assert MODEL_TENSOR_BYTES < MODEL_WEIGHTS_FILE_CAP_BYTES
    assert tuple(item.name for item in EXPECTED_MODEL_TENSORS) == tuple(
        sorted(item.name for item in EXPECTED_MODEL_TENSORS)
    )
    spectral = next(item for item in EXPECTED_MODEL_TENSORS if item.name.endswith("weights1"))
    assert spectral.shape == (64, 64, 24, 24, 2)
    assert spectral.nbytes == 18_874_368
    assert (
        ArrayDescriptor(
            dtype="float32",
            shape=spectral.shape,
            unit="dimensionless",
        ).shape
        == spectral.shape
    )
    with pytest.raises(ValidationError, match="byte count"):
        ModelTensorDescriptor(name="bad.weight", shape=(2, 2), nbytes=4)
    with pytest.raises(ValidationError, match="dimensions must be positive"):
        ModelTensorDescriptor(name="bad.weight", shape=(0,), nbytes=4)


def test_runtime_ranges_and_model_card_metadata_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError, match="Soufflerie compatibility range"):
        CompatibilityRange(
            minimum_soufflerie="0.2.0",
            maximum_soufflerie_exclusive="0.1.0",
        )
    with pytest.raises(ValidationError, match="nonnegative version integers"):
        CompatibilityRange(minimum_python=(-1, 11))
    with pytest.raises(ValidationError, match="Python compatibility range"):
        CompatibilityRange(minimum_python=(3, 11), maximum_python_exclusive=(3, 11))
    with pytest.raises(ArtifactIntegrityError, match="Python runtime"):
        CompatibilityRange(
            minimum_python=(99, 0),
            maximum_python_exclusive=(99, 1),
        ).validate_core_runtime()
    with pytest.raises(ValidationError, match="gate names must be unique"):
        card = model_card()
        type(card).model_validate(
            {
                **card.model_dump(mode="python"),
                "gates": (card.gates[0], card.gates[0]),
            }
        )

    def missing_version(_package: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing_version)
    CompatibilityRange().validate_core_runtime()


def test_metadata_identity_binds_weights_preprocessing_card_and_runtime_contract() -> None:
    architecture_sha256 = sha256_bytes(canonical_json_bytes(FnoArchitecture()))
    preprocessing_sha256 = sha256_bytes(canonical_json_bytes(preprocessing_statistics()))

    def metadata(
        *,
        weights: str = "1" * 64,
        preprocessing: str = preprocessing_sha256,
        dataset: str = DATASET_SHA256,
    ) -> ModelBundleMetadata:
        return ModelBundleMetadata.create(
            dataset_id=DATASET_ID,
            dataset_sha256=dataset,
            experiment_id=EXPERIMENT_ID,
            seed=7,
            selected_epoch=12,
            weights_sha256=weights,
            weights_file_bytes=151_126_144,
            preprocessing_sha256=preprocessing,
            architecture_sha256=architecture_sha256,
            code_revision=CODE_REVISION,
            lock_digest=LOCK_DIGEST,
            model_card=model_card(),
        )

    baseline = metadata()
    assert ModelBundleMetadata.model_validate_json(baseline.model_dump_json()) == baseline
    assert baseline.model_id == baseline.model_sha256[:20]
    assert metadata(weights="2" * 64).model_id != baseline.model_id
    assert metadata(preprocessing="3" * 64).model_id != baseline.model_id
    assert metadata(dataset=DATASET_ID + "f" * 44).model_id != baseline.model_id
    changed_card = baseline.model_card.model_copy(
        update={"summary": "A different factual model summary."}
    )
    changed = ModelBundleMetadata.create(
        dataset_id=DATASET_ID,
        dataset_sha256=DATASET_SHA256,
        experiment_id=EXPERIMENT_ID,
        seed=7,
        selected_epoch=12,
        weights_sha256="1" * 64,
        weights_file_bytes=151_126_144,
        preprocessing_sha256=preprocessing_sha256,
        architecture_sha256=architecture_sha256,
        code_revision=CODE_REVISION,
        lock_digest=LOCK_DIGEST,
        model_card=changed_card,
    )
    assert changed.model_id != baseline.model_id
    assert "## Validation gates" in render_model_card(baseline)
    assert "not evaluated" in render_model_card(baseline)

    invalid_updates = (
        ({"architecture_sha256": "f" * 64}, "architecture digest"),
        ({"model_sha256": "f" * 64}, "bundle logical identity"),
        ({"model_id": "f" * 20}, "prefix of model_sha256"),
        ({"model_card_sha256": "f" * 64}, "generated model card"),
    )
    baseline_payload = baseline.model_dump(mode="python")
    for update, message in invalid_updates:
        with pytest.raises(ValidationError, match=message):
            ModelBundleMetadata.model_validate({**baseline_payload, **update})

    rebound_tensors = [item.model_dump(mode="python") for item in baseline.tensors]
    rebound_tensors[0]["name"] = "attacker.weight"
    with pytest.raises(ValidationError, match="fno2d-v1 allowlist"):
        ModelBundleMetadata.model_validate({**baseline_payload, "tensors": rebound_tensors})
    with pytest.raises(ValidationError, match="full parent dataset digest"):
        ModelBundleMetadata.model_validate({**baseline_payload, "dataset_sha256": "f" * 64})


def test_full_bundle_round_trip_is_atomic_read_only_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = make_test_bundle()
    store = LocalModelBundleStore(tmp_path)

    reference = store.publish(bundle)
    loaded = store.open(reference)
    repeated = store.publish(bundle)

    assert reference == bundle.reference == repeated
    assert loaded.reference == reference
    assert loaded.metadata == bundle.metadata
    assert loaded.preprocessing == bundle.preprocessing
    assert loaded.architecture == bundle.architecture
    assert loaded.model_card_markdown == bundle.model_card_markdown
    assert set(loaded.weights) == {item.name for item in EXPECTED_MODEL_TENSORS}
    np.testing.assert_array_equal(
        loaded.weights["core.decoder_net.final_layer.linear.bias"],
        [0.25, -0.5, 0.75],
    )
    assert all(not array.flags.writeable for array in loaded.weights.values())
    assert {path.name for path in (tmp_path / reference.uri).iterdir()} == store.expected_files
    with pytest.raises(TypeError, match="PublishedModelBundle"):
        instantiate_bundle_predictor(cast(Any, object()), device="cpu")
    with pytest.raises(DeviceUnavailableError, match="cpu or cuda"):
        instantiate_bundle_predictor(loaded, device="mps")

    original_import = importlib.import_module

    def unavailable_torch(name: str) -> Any:
        if name == "torch":
            raise ImportError("injected")
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", unavailable_torch)
    with pytest.raises(DependencyUnavailableError, match="locked 'ml' extra"):
        instantiate_bundle_predictor(loaded, device="cpu")


@pytest.mark.parametrize("stage", ("members_written", "verified", "committed"))
def test_faults_never_publish_a_visible_partial_bundle(tmp_path: Path, stage: str) -> None:
    bundle = make_test_bundle()

    def fail(current: str) -> None:
        if current == stage:
            raise RuntimeError("injected")

    store = LocalModelBundleStore(tmp_path, fault_injector=fail)
    with pytest.raises(RuntimeError, match="injected"):
        store.publish(bundle)
    assert not (tmp_path / bundle.reference.uri).exists()
    assert list((tmp_path / ".staging" / "models").iterdir()) == []


def test_export_rejects_missing_extra_malformed_and_nonfinite_weight_arrays() -> None:
    valid = zero_weights()
    missing = dict(valid)
    missing.pop(next(iter(missing)))
    with pytest.raises(ArtifactIntegrityError, match="closed allowlist"):
        make_test_bundle(weights=missing)
    extra = {**valid, "extra.weight": np.zeros((1,), dtype=np.float32)}
    with pytest.raises(ArtifactIntegrityError, match="closed allowlist"):
        make_test_bundle(weights=extra)

    name = "cd_head.4.weight"
    for replacement in (
        np.zeros((1, 32), dtype=np.float64),
        np.zeros((32, 1), dtype=np.float32),
        np.asfortranarray(np.zeros((32, 64), dtype=np.float32)),
        np.full((1, 32), np.nan, dtype=np.float32),
    ):
        malformed: dict[str, npt.NDArray[Any]] = dict(valid)
        target = "cd_head.2.weight" if replacement.shape == (32, 64) else name
        malformed[target] = replacement
        with pytest.raises(ArtifactIntegrityError):
            make_test_bundle(weights=cast(dict[str, Any], malformed))


def test_model_card_and_bundle_reject_incoherent_evidence_or_dataset() -> None:
    with pytest.raises(ValidationError, match="must be coherent"):
        ModelCardGate(name="Gate", status="green", threshold="below 1")
    with pytest.raises(ValidationError, match="must be coherent"):
        ModelCardGate(
            name="Gate",
            status="not_evaluated",
            threshold="below 1",
            measured="not applicable",
        )
    bundle = make_test_bundle()
    other_preprocessing = preprocessing_statistics(dataset_id="f" * 20)
    with pytest.raises(ArtifactIntegrityError, match="preprocessing dataset"):
        replace(bundle, preprocessing=other_preprocessing)
    with pytest.raises(ArtifactIntegrityError, match="weights byte count"):
        replace(
            bundle,
            metadata=bundle.metadata.model_copy(
                update={"weights_file_bytes": len(bundle.weights_bytes) + 1}
            ),
        )
    with pytest.raises(ArtifactIntegrityError, match="member digest"):
        replace(
            bundle,
            metadata=bundle.metadata.model_copy(update={"weights_sha256": "f" * 64}),
        )
    noncanonical_card = "# Attacker-rebound card\n"
    with pytest.raises(ArtifactIntegrityError, match="model card is not canonical"):
        replace(
            bundle,
            metadata=bundle.metadata.model_copy(
                update={"model_card_sha256": sha256_bytes(noncanonical_card.encode("utf-8"))}
            ),
            model_card_markdown=noncanonical_card,
        )
    with pytest.raises(TypeError, match="preprocessing must"):
        build_model_bundle(
            weights=zero_weights(),
            preprocessing=cast(Any, object()),
            dataset_sha256=DATASET_SHA256,
            experiment_id=EXPERIMENT_ID,
            seed=7,
            selected_epoch=12,
            code_revision=CODE_REVISION,
            lock_digest=LOCK_DIGEST,
            model_card=model_card(),
        )
    with pytest.raises(ArtifactIntegrityError, match="only fno2d-v1"):
        build_model_bundle(
            weights=zero_weights(),
            preprocessing=preprocessing_statistics(),
            dataset_sha256=DATASET_SHA256,
            experiment_id=EXPERIMENT_ID,
            seed=7,
            selected_epoch=12,
            code_revision=CODE_REVISION,
            lock_digest=LOCK_DIGEST,
            model_card=model_card(),
            architecture=cast(Any, object()),
        )
    with pytest.raises(TypeError, match="predictor must"):
        snapshot_fno_weights(cast(Any, object()))


def test_incompatible_but_well_formed_bundle_is_not_publishable(tmp_path: Path) -> None:
    compatibility = CompatibilityRange(
        minimum_soufflerie="0.2.0",
        maximum_soufflerie_exclusive="0.3.0",
    )
    bundle = make_test_bundle(compatibility=compatibility)
    with pytest.raises(ArtifactIntegrityError, match="COMPATIBILITY"):
        LocalModelBundleStore(tmp_path).publish(bundle)
    assert not (tmp_path / bundle.reference.uri).exists()


@pytest.mark.slow
def test_real_runtime_export_load_is_bitwise_identical_on_cpu(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("physicsnemo") is None:
        pytest.skip("the optional ml runtime is not installed")
    torch = cast(Any, importlib.import_module("torch"))
    torch.manual_seed(23)
    original = FnoPredictor()
    weights = snapshot_fno_weights(original)
    bundle = make_test_bundle(weights=weights)
    store = LocalModelBundleStore(tmp_path)
    loaded = store.open(store.publish(bundle))
    restored = instantiate_bundle_predictor(loaded, device="cpu")
    batch = PredictionBatch(
        inputs=torch.randn((1, 2, *MODEL_SPATIAL_SHAPE), dtype=torch.float32),
        fluid_mask=torch.ones((1, 1, *MODEL_SPATIAL_SHAPE), dtype=torch.bool),
        design_params=torch.zeros((1, 4), dtype=torch.float32),
    )

    expected = original.predict(batch)
    actual = restored.predict(batch)

    assert torch.equal(expected.fields_normalized, actual.fields_normalized)
    assert torch.equal(expected.cd_head, actual.cd_head)
