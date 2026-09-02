"""Checksum-bound packaged CPU smoke model and installed-wheel acceptance path."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import importlib.resources
import io
import math
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
from pydantic import Field, model_validator

from soufflerie.artifacts import ReaderLimits, safe_read_bytes, safe_read_json
from soufflerie.errors import ArtifactIntegrityError, DependencyUnavailableError
from soufflerie.schemas import (
    ArtifactRef,
    ContentId,
    Sha256,
    StrictFrozenModel,
    VersionedModel,
    canonical_json_bytes,
    canonical_sha256,
)
from soufflerie.surrogate.architecture import FnoArchitecture
from soufflerie.surrogate.bundle import (
    MODEL_ARCHITECTURE_NAME,
    MODEL_CARD_NAME,
    MODEL_JSON_CAP_BYTES,
    MODEL_METADATA_NAME,
    MODEL_PREPROCESSING_NAME,
    MODEL_WEIGHTS_FILE_CAP_BYTES,
    LocalModelBundleStore,
    ModelBundle,
    ModelBundleMetadata,
    instantiate_bundle_predictor,
)
from soufflerie.surrogate.preprocessing import (
    MODEL_SPATIAL_SHAPE,
    PredictionBatch,
    PreprocessingStatistics,
)

BUNDLED_RESOURCE_PREFIX = "resources/model"
BUNDLED_RESOURCE_NAME = "resource.json"
BUNDLED_COMPRESSED_WEIGHTS_NAME: Literal["model.safetensors.gz"] = "model.safetensors.gz"
BUNDLED_RESOURCE_KIND: Literal["synthetic-cpu-smoke-v1"] = "synthetic-cpu-smoke-v1"
BUNDLED_COMPRESSED_CAP_BYTES = 4 * 1024 * 1024

_RESOURCE_FILES = frozenset(
    {
        BUNDLED_RESOURCE_NAME,
        MODEL_METADATA_NAME,
        MODEL_PREPROCESSING_NAME,
        MODEL_ARCHITECTURE_NAME,
        MODEL_CARD_NAME,
        BUNDLED_COMPRESSED_WEIGHTS_NAME,
    }
)
_RESOURCE_LIMITS = ReaderLimits(
    max_file_bytes=max(MODEL_JSON_CAP_BYTES, BUNDLED_COMPRESSED_CAP_BYTES),
    max_json_bytes=MODEL_JSON_CAP_BYTES,
)


class SmokeDatasetParent(StrictFrozenModel):
    """Transparent synthetic parent identity for the untrained smoke fixture."""

    kind: Literal["synthetic-normalization-fixture-v1"] = "synthetic-normalization-fixture-v1"
    sample_count: Literal[1] = 1
    output_means: tuple[float, float, float] = (0.0, 0.0, 0.0)
    output_standard_deviations: tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    )
    scientific_evidence: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("output_means", "output_standard_deviations"):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized

    @model_validator(mode="after")
    def _normalization_is_the_fixed_fixture(self) -> Self:
        if self.output_means != (0.0, 0.0, 0.0) or self.output_standard_deviations != (
            1.0,
            1.0,
            1.0,
        ):
            raise ValueError("smoke fixture normalization must remain zero-mean and unit-scale")
        return self

    @property
    def sha256(self) -> str:
        return canonical_sha256(self)


class BundledModelResource(VersionedModel):
    """Outer package record binding compressed bytes to one safe model bundle."""

    resource_kind: Literal["synthetic-cpu-smoke-v1"] = BUNDLED_RESOURCE_KIND
    model: ArtifactRef
    bundle_metadata_sha256: Sha256
    fixture_parent: SmokeDatasetParent
    fixture_parent_sha256: Sha256
    weights_encoding: Literal["gzip"] = "gzip"
    compressed_weights_name: Literal["model.safetensors.gz"] = BUNDLED_COMPRESSED_WEIGHTS_NAME
    compressed_weights_sha256: Sha256
    compressed_weights_bytes: int = Field(ge=1, le=BUNDLED_COMPRESSED_CAP_BYTES)
    uncompressed_weights_bytes: int = Field(ge=1, le=MODEL_WEIGHTS_FILE_CAP_BYTES)
    expected_fields_sha256: Sha256
    expected_cd_sha256: Sha256
    representative_of_trained_quality: Literal[False] = False

    @model_validator(mode="after")
    def _identity_is_coherent(self) -> Self:
        if self.model.artifact_type != "model":
            raise ValueError("bundled resource must reference a model artifact")
        if self.fixture_parent_sha256 != self.fixture_parent.sha256:
            raise ValueError("fixture parent digest does not match its logical record")
        return self


class BundledCpuSmokeResult(VersionedModel):
    """Finite schema-v1 evidence returned by the real packaged CPU prediction path."""

    resource_kind: Literal["synthetic-cpu-smoke-v1"] = BUNDLED_RESOURCE_KIND
    model_id: ContentId
    model_sha256: Sha256
    dataset_id: ContentId
    dataset_sha256: Sha256
    device: Literal["cpu"] = "cpu"
    fields_shape: tuple[Literal[1], Literal[3], Literal[320], Literal[256]]
    cd_shape: tuple[Literal[1]]
    fields_minimum: float = Field(allow_inf_nan=False)
    fields_maximum: float = Field(allow_inf_nan=False)
    cd: float = Field(allow_inf_nan=False)
    fields_sha256: Sha256
    cd_sha256: Sha256
    representative_of_trained_quality: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("fields_shape", "cd_shape"):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized

    @model_validator(mode="after")
    def _range_is_increasing(self) -> Self:
        if self.fields_minimum > self.fields_maximum:
            raise ValueError("field bounds must be increasing")
        return self


def _packaged_resource_root() -> Path:
    resource = importlib.resources.files("soufflerie").joinpath(*BUNDLED_RESOURCE_PREFIX.split("/"))
    if not isinstance(resource, Path):
        raise ArtifactIntegrityError(
            "BUNDLED-1 PACKAGE: model resources require an installed filesystem wheel"
        )
    return resource


def _assert_resource_layout(root: Path) -> None:
    try:
        details = root.lstat()
        names = {entry.name for entry in root.iterdir()}
    except OSError as error:
        raise ArtifactIntegrityError(
            "BUNDLED-1 PACKAGE: unable to inspect packaged model resources"
        ) from error
    if not stat.S_ISDIR(details.st_mode) or names != _RESOURCE_FILES:
        raise ArtifactIntegrityError(
            "BUNDLED-1 PACKAGE: packaged model resources do not match the closed layout"
        )
    for name in names:
        try:
            member = (root / name).lstat()
        except OSError as error:
            raise ArtifactIntegrityError(
                f"BUNDLED-1 PACKAGE: unable to inspect resource {name!r}"
            ) from error
        if not stat.S_ISREG(member.st_mode):
            raise ArtifactIntegrityError(
                f"BUNDLED-1 PACKAGE: resource {name!r} is not a regular file"
            )


def _decompress_weights(content: bytes) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(content), mode="rb") as stream:
            decoded = stream.read(MODEL_WEIGHTS_FILE_CAP_BYTES + 1)
    except (EOFError, OSError) as error:
        raise ArtifactIntegrityError(
            "BUNDLED-2 COMPRESSION: packaged model weights are not valid gzip"
        ) from error
    if len(decoded) > MODEL_WEIGHTS_FILE_CAP_BYTES:
        raise ArtifactIntegrityError(
            "BUNDLED-2 COMPRESSION: decoded model weights exceed the byte cap"
        )
    return decoded


def _load_resource(root: Path) -> tuple[BundledModelResource, ModelBundle]:
    _assert_resource_layout(root)
    descriptor = safe_read_json(
        root,
        BUNDLED_RESOURCE_NAME,
        model=BundledModelResource,
        limits=_RESOURCE_LIMITS,
    )
    metadata = safe_read_json(
        root,
        MODEL_METADATA_NAME,
        model=ModelBundleMetadata,
        limits=_RESOURCE_LIMITS,
        expected_sha256=descriptor.bundle_metadata_sha256,
    )
    preprocessing = safe_read_json(
        root,
        MODEL_PREPROCESSING_NAME,
        model=PreprocessingStatistics,
        limits=_RESOURCE_LIMITS,
        expected_sha256=metadata.preprocessing_sha256,
    )
    architecture = safe_read_json(
        root,
        MODEL_ARCHITECTURE_NAME,
        model=FnoArchitecture,
        limits=_RESOURCE_LIMITS,
        expected_sha256=metadata.architecture_sha256,
    )
    card_bytes = safe_read_bytes(
        root,
        MODEL_CARD_NAME,
        max_bytes=MODEL_JSON_CAP_BYTES,
        expected_sha256=metadata.model_card_sha256,
    )
    try:
        card = card_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactIntegrityError(
            "BUNDLED-1 PACKAGE: packaged model card is not UTF-8"
        ) from error
    compressed = safe_read_bytes(
        root,
        BUNDLED_COMPRESSED_WEIGHTS_NAME,
        max_bytes=BUNDLED_COMPRESSED_CAP_BYTES,
        expected_sha256=descriptor.compressed_weights_sha256,
    )
    if len(compressed) != descriptor.compressed_weights_bytes:
        raise ArtifactIntegrityError("BUNDLED-2 COMPRESSION: compressed weights byte count changed")
    weights = _decompress_weights(compressed)
    if len(weights) != descriptor.uncompressed_weights_bytes:
        raise ArtifactIntegrityError(
            "BUNDLED-2 COMPRESSION: uncompressed weights byte count changed"
        )
    if metadata.dataset_sha256 != descriptor.fixture_parent_sha256:
        raise ArtifactIntegrityError(
            "BUNDLED-3 LINEAGE: model does not reference the packaged fixture parent"
        )
    bundle = ModelBundle(
        metadata=metadata,
        weights_bytes=weights,
        preprocessing=preprocessing,
        architecture=architecture,
        model_card_markdown=card,
    )
    if bundle.reference != descriptor.model:
        raise ArtifactIntegrityError(
            "BUNDLED-3 IDENTITY: packaged resource and model bundle disagree"
        )
    return descriptor, bundle


def _materialize_bundled_cpu_model_from(resource_root: Path, store_root: Path) -> ArtifactRef:
    descriptor, bundle = _load_resource(resource_root.resolve())
    reference = LocalModelBundleStore(store_root).publish(bundle)
    if reference != descriptor.model:
        raise ArtifactIntegrityError("BUNDLED-3 IDENTITY: published smoke model reference changed")
    return reference


def materialize_bundled_cpu_model(store_root: Path) -> ArtifactRef:
    """Verify and atomically expand the packaged smoke model into ``store_root``."""

    if not isinstance(store_root, Path):
        raise TypeError("store_root must be a Path")
    return _materialize_bundled_cpu_model_from(_packaged_resource_root(), store_root)


def bundled_model_resource() -> BundledModelResource:
    """Read the checksum and immutable model reference shipped in this installation."""

    root = _packaged_resource_root()
    _assert_resource_layout(root)
    return safe_read_json(
        root,
        BUNDLED_RESOURCE_NAME,
        model=BundledModelResource,
        limits=_RESOURCE_LIMITS,
    )


def _array_sha256(value: Any) -> str:
    array = np.asarray(value.detach().cpu().contiguous().numpy(), dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _import_torch() -> Any:
    return importlib.import_module("torch")


def run_bundled_cpu_smoke(store_root: Path) -> BundledCpuSmokeResult:
    """Materialize the synthetic bundle and run one real fixed-FNO CPU prediction."""

    reference = materialize_bundled_cpu_model(store_root)
    store = LocalModelBundleStore(store_root)
    bundle = store.open(reference)
    try:
        torch = _import_torch()
    except ImportError as error:
        raise DependencyUnavailableError(
            "bundled CPU prediction requires the locked 'ml' extra"
        ) from error
    predictor = instantiate_bundle_predictor(bundle, device="cpu")
    batch = PredictionBatch(
        inputs=torch.zeros((1, 2, *MODEL_SPATIAL_SHAPE), dtype=torch.float32),
        fluid_mask=torch.ones((1, 1, *MODEL_SPATIAL_SHAPE), dtype=torch.bool),
        design_params=torch.zeros((1, 4), dtype=torch.float32),
    )
    prediction = predictor.predict(batch)
    fields = cast(Any, prediction.fields_normalized)
    cd_head = cast(Any, prediction.cd_head)
    fields_sha256 = _array_sha256(fields)
    cd_sha256 = _array_sha256(cd_head)
    descriptor = bundled_model_resource()
    if (
        fields_sha256 != descriptor.expected_fields_sha256
        or cd_sha256 != descriptor.expected_cd_sha256
    ):
        raise ArtifactIntegrityError("BUNDLED-4 PREDICTION: packaged CPU smoke output changed")
    fields_minimum = float(fields.min().item())
    fields_maximum = float(fields.max().item())
    cd = float(cd_head[0].item())
    if not all(math.isfinite(value) for value in (fields_minimum, fields_maximum, cd)):
        raise ArtifactIntegrityError("BUNDLED-4 PREDICTION: smoke output is not finite")
    fields_shape: tuple[Literal[1], Literal[3], Literal[320], Literal[256]] = (
        1,
        3,
        320,
        256,
    )
    cd_shape: tuple[Literal[1]] = (1,)
    return BundledCpuSmokeResult(
        model_id=bundle.metadata.model_id,
        model_sha256=bundle.metadata.model_sha256,
        dataset_id=bundle.metadata.dataset_id,
        dataset_sha256=bundle.metadata.dataset_sha256,
        fields_shape=fields_shape,
        cd_shape=cd_shape,
        fields_minimum=fields_minimum,
        fields_maximum=fields_maximum,
        cd=cd,
        fields_sha256=fields_sha256,
        cd_sha256=cd_sha256,
    )


def rendered_bundled_resource_bytes(descriptor: BundledModelResource) -> bytes:
    """Canonical resource bytes shared by the generator and regeneration test."""

    return canonical_json_bytes(descriptor)


__all__ = [
    "BUNDLED_COMPRESSED_WEIGHTS_NAME",
    "BUNDLED_RESOURCE_KIND",
    "BUNDLED_RESOURCE_NAME",
    "BUNDLED_RESOURCE_PREFIX",
    "BundledCpuSmokeResult",
    "BundledModelResource",
    "SmokeDatasetParent",
    "bundled_model_resource",
    "materialize_bundled_cpu_model",
    "rendered_bundled_resource_bytes",
    "run_bundled_cpu_smoke",
]
