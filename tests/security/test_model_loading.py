from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from safetensors.numpy import save as save_tensors

from soufflerie.artifacts import ReaderLimits
from soufflerie.errors import ArtifactIntegrityError, SchemaVersionError
from soufflerie.schemas import ArtifactRef, canonical_json_bytes, sha256_bytes
from soufflerie.surrogate.architecture import FnoArchitecture
from soufflerie.surrogate.bundle import (
    MODEL_ARCHITECTURE_NAME,
    MODEL_CARD_NAME,
    MODEL_COMMIT_NAME,
    MODEL_METADATA_NAME,
    MODEL_PREPROCESSING_NAME,
    MODEL_ROOT_PREFIX,
    MODEL_WEIGHTS_NAME,
    LocalModelBundleStore,
    ModelBundleMetadata,
    render_model_card,
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
)


@dataclass(frozen=True)
class CommittedFixture:
    root: Path
    store: LocalModelBundleStore
    reference: ArtifactRef
    original_files: dict[str, bytes]

    @property
    def directory(self) -> Path:
        return self.root / self.reference.uri


@pytest.fixture(scope="module")
def committed(tmp_path_factory: pytest.TempPathFactory) -> CommittedFixture:
    root = tmp_path_factory.mktemp("model-loading")
    store = LocalModelBundleStore(root)
    reference = store.publish(make_test_bundle())
    directory = root / reference.uri
    originals = {path.name: path.read_bytes() for path in directory.iterdir()}
    return CommittedFixture(root=root, store=store, reference=reference, original_files=originals)


def _restore(fixture: CommittedFixture) -> None:
    fixture.directory.mkdir(parents=True, exist_ok=True)
    for path in fixture.directory.iterdir():
        if path.is_dir() and not path.is_symlink():
            os.rmdir(path)
        else:
            path.unlink()
    for name, content in fixture.original_files.items():
        (fixture.directory / name).write_bytes(content)


def test_missing_extra_and_symlinked_members_fail_closed(committed: CommittedFixture) -> None:
    try:
        (committed.directory / MODEL_CARD_NAME).unlink()
        with pytest.raises(ArtifactIntegrityError, match="closed layout"):
            committed.store.open(committed.reference)
        _restore(committed)

        (committed.directory / "extra.bin").write_bytes(b"extra")
        with pytest.raises(ArtifactIntegrityError, match="closed layout"):
            committed.store.open(committed.reference)
        _restore(committed)

        card = committed.directory / MODEL_CARD_NAME
        card.unlink()
        card.symlink_to(committed.directory / MODEL_METADATA_NAME)
        with pytest.raises(ArtifactIntegrityError, match="regular file"):
            committed.store.open(committed.reference)
    finally:
        _restore(committed)


def test_modified_weights_and_every_bound_sidecar_fail_digest_checks(
    committed: CommittedFixture,
) -> None:
    try:
        for name in (
            MODEL_WEIGHTS_NAME,
            MODEL_PREPROCESSING_NAME,
            MODEL_ARCHITECTURE_NAME,
            MODEL_CARD_NAME,
        ):
            path = committed.directory / name
            content = bytearray(path.read_bytes())
            content[-1] ^= 1
            path.write_bytes(content)
            with pytest.raises(ArtifactIntegrityError, match=r"SHA-256|JSON|card"):
                committed.store.open(committed.reference)
            _restore(committed)
    finally:
        _restore(committed)


def test_missing_invalid_or_rebound_commit_marker_never_opens(
    committed: CommittedFixture,
) -> None:
    marker = committed.directory / MODEL_COMMIT_NAME
    try:
        marker.unlink()
        with pytest.raises(ArtifactIntegrityError):
            committed.store.open(committed.reference)
        _restore(committed)
        marker.write_bytes(b"not-a-digest\n")
        with pytest.raises(ArtifactIntegrityError, match="COMMIT"):
            committed.store.open(committed.reference)
        _restore(committed)
        marker.write_bytes(b"\xff" * 65)
        with pytest.raises(ArtifactIntegrityError, match="not ASCII"):
            committed.store.open(committed.reference)
        _restore(committed)
        marker.write_text("0" * 64 + "\n", encoding="ascii")
        with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
            committed.store.open(committed.reference)
    finally:
        _restore(committed)


def test_wrong_schema_is_rejected_before_any_weight_decode(committed: CommittedFixture) -> None:
    metadata_path = committed.directory / MODEL_METADATA_NAME
    marker = committed.directory / MODEL_COMMIT_NAME
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["schema_version"] = 2
        content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        metadata_path.write_bytes(content)
        marker.write_text(sha256_bytes(content) + "\n", encoding="ascii")
        with pytest.raises(SchemaVersionError):
            committed.store.open(committed.reference)
    finally:
        _restore(committed)


def _write_attacker_bundle(
    root: Path, weights_bytes: bytes
) -> tuple[LocalModelBundleStore, ArtifactRef]:
    preprocessing = preprocessing_statistics()
    architecture = FnoArchitecture()
    metadata = ModelBundleMetadata.create(
        dataset_id=DATASET_ID,
        dataset_sha256=DATASET_SHA256,
        experiment_id=EXPERIMENT_ID,
        seed=7,
        selected_epoch=12,
        weights_sha256=sha256_bytes(weights_bytes),
        weights_file_bytes=len(weights_bytes),
        preprocessing_sha256=sha256_bytes(canonical_json_bytes(preprocessing)),
        architecture_sha256=sha256_bytes(canonical_json_bytes(architecture)),
        code_revision=CODE_REVISION,
        lock_digest=LOCK_DIGEST,
        model_card=model_card(),
    )
    files = {
        MODEL_METADATA_NAME: canonical_json_bytes(metadata),
        MODEL_WEIGHTS_NAME: weights_bytes,
        MODEL_PREPROCESSING_NAME: canonical_json_bytes(preprocessing),
        MODEL_ARCHITECTURE_NAME: canonical_json_bytes(architecture),
        MODEL_CARD_NAME: render_model_card(metadata).encode(),
    }
    directory = root / MODEL_ROOT_PREFIX / metadata.model_id
    directory.mkdir(parents=True)
    for name, content in files.items():
        (directory / name).write_bytes(content)
    metadata_digest = sha256_bytes(files[MODEL_METADATA_NAME])
    (directory / MODEL_COMMIT_NAME).write_text(metadata_digest + "\n", encoding="ascii")
    size = sum(len(content) for content in files.values()) + 65
    reference = ArtifactRef(
        artifact_type="model",
        artifact_id=metadata.model_id,
        sha256=metadata.model_sha256,
        size_bytes=size,
        uri=f"{MODEL_ROOT_PREFIX}/{metadata.model_id}",
    )
    return LocalModelBundleStore(root), reference


@pytest.mark.parametrize(
    "weights",
    (
        {"extra.weight": np.zeros((1,), dtype=np.float32)},
        {"cd_head.0.bias": np.zeros((64,), dtype=np.float32)},
    ),
)
def test_attacker_rebound_extra_or_missing_tensor_file_fails_the_allowlist(
    tmp_path: Path,
    weights: dict[str, npt.NDArray[Any]],
) -> None:
    content = save_tensors(weights)
    store, reference = _write_attacker_bundle(tmp_path, content)
    with pytest.raises(ArtifactIntegrityError, match="names do not exactly match"):
        store.open(reference)


def test_tensor_allocation_cap_is_enforced_before_safetensor_loading(
    committed: CommittedFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from safetensors import numpy as safetensors_numpy

    def forbidden(_content: bytes) -> object:
        raise AssertionError("safetensors decoder ran before allocation preflight")

    monkeypatch.setattr(safetensors_numpy, "load", forbidden)
    strict_store = LocalModelBundleStore(
        committed.root,
        limits=ReaderLimits(max_member_bytes=8),
    )
    with pytest.raises(ArtifactIntegrityError, match="member byte cap"):
        strict_store.open(committed.reference)


def test_reference_identity_and_size_are_authoritative(committed: CommittedFixture) -> None:
    with pytest.raises(ArtifactIntegrityError, match="expected a model"):
        committed.store.open(cast(Any, object()))
    with pytest.raises(ArtifactIntegrityError, match="reference byte count"):
        committed.store.open(
            committed.reference.model_copy(
                update={"size_bytes": committed.reference.size_bytes + 1}
            )
        )
    with pytest.raises(ArtifactIntegrityError, match="reference is incoherent"):
        committed.store.open(committed.reference.model_copy(update={"sha256": "f" * 64}))
    metadata_mismatch = committed.reference.model_copy(
        update={
            "sha256": committed.reference.artifact_id + "f" * 44,
        }
    )
    with pytest.raises(ArtifactIntegrityError, match="reference and metadata disagree"):
        committed.store.open(metadata_mismatch)
    absent = committed.reference.model_copy(
        update={
            "artifact_id": "f" * 20,
            "sha256": "f" * 64,
            "uri": f"models/{'f' * 20}",
        }
    )
    with pytest.raises(ArtifactIntegrityError, match="unable to inspect bundle"):
        committed.store.open(absent)
