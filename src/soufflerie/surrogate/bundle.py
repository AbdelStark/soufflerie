"""Safe immutable model bundles with a closed FNO tensor allowlist."""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from errno import EEXIST, ENOTEMPTY
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import Field, StringConstraints, model_validator

from soufflerie.artifacts import (
    DEFAULT_READER_LIMITS,
    ReaderLimits,
    safe_read_bytes,
    safe_read_json,
    safe_read_tensors,
)
from soufflerie.errors import (
    ArtifactIntegrityError,
    DependencyUnavailableError,
    DeviceUnavailableError,
)
from soufflerie.schemas import (
    ArrayDescriptor,
    ArtifactRef,
    ContentId,
    Sha256,
    StrictFrozenModel,
    VersionedModel,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)
from soufflerie.surrogate.architecture import FnoArchitecture
from soufflerie.surrogate.fno import FnoPredictor
from soufflerie.surrogate.preprocessing import PreprocessingStatistics

MODEL_ROOT_PREFIX = "models"
MODEL_METADATA_NAME = "bundle.json"
MODEL_WEIGHTS_NAME = "model.safetensors"
MODEL_PREPROCESSING_NAME = "preprocessing.json"
MODEL_ARCHITECTURE_NAME = "architecture.json"
MODEL_CARD_NAME = "model-card.md"
MODEL_COMMIT_NAME = "COMMITTED"
MODEL_WEIGHTS_FILE_CAP_BYTES = 192 * 1024 * 1024
MODEL_TENSOR_MEMBER_CAP_BYTES = 24 * 1024 * 1024
MODEL_JSON_CAP_BYTES = 1024 * 1024
MODEL_CARD_CAP_BYTES = 128 * 1024

_DEVICE_PATTERN = re.compile(r"^(?:cpu|cuda(?::[0-9]+)?)$")
_SOURCE_REVISION_PATTERN = r"^[0-9a-f]{40}$"
_TENSOR_NAME_PATTERN = r"^[a-z][a-z0-9_.]{0,127}$"
_SINGLE_LINE_PATTERN = r"^[^\r\n|]{1,240}$"
_UINT64_MAX = 2**64 - 1
_SEMVER_PATTERN = r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"

SingleLine = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=_SINGLE_LINE_PATTERN),
]
TensorName = Annotated[str, StringConstraints(pattern=_TENSOR_NAME_PATTERN)]
SemanticVersion = Annotated[str, StringConstraints(pattern=_SEMVER_PATTERN)]


class ModelTensorDescriptor(StrictFrozenModel):
    """One exact float32 state tensor and its pre-allocation byte count."""

    name: TensorName
    dtype: Literal["float32"] = "float32"
    shape: tuple[int, ...] = Field(min_length=1, max_length=5)
    nbytes: int = Field(ge=1, le=MODEL_TENSOR_MEMBER_CAP_BYTES)

    @model_validator(mode="before")
    @classmethod
    def _normalize_shape(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        shape = normalized.get("shape")
        if isinstance(shape, list):
            normalized["shape"] = tuple(shape)
        return normalized

    @model_validator(mode="after")
    def _bytes_match_shape(self) -> Self:
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("model tensor dimensions must be positive")
        expected = math.prod(self.shape) * np.dtype(np.float32).itemsize
        if self.nbytes != expected:
            raise ValueError("model tensor byte count must match its float32 shape")
        return self

    def array_descriptor(self) -> ArrayDescriptor:
        return ArrayDescriptor(dtype="float32", shape=self.shape, unit="dimensionless")


def _tensor(name: str, shape: tuple[int, ...]) -> ModelTensorDescriptor:
    return ModelTensorDescriptor(
        name=name,
        shape=shape,
        nbytes=math.prod(shape) * np.dtype(np.float32).itemsize,
    )


def _expected_tensor_descriptors() -> tuple[ModelTensorDescriptor, ...]:
    values = [
        _tensor("cd_head.0.bias", (64,)),
        _tensor("cd_head.0.weight", (64, 68)),
        _tensor("cd_head.2.bias", (32,)),
        _tensor("cd_head.2.weight", (32, 64)),
        _tensor("cd_head.4.bias", (1,)),
        _tensor("cd_head.4.weight", (1, 32)),
        _tensor("core.decoder_net.final_layer.linear.bias", (3,)),
        _tensor("core.decoder_net.final_layer.linear.weight", (3, 128)),
        _tensor("core.decoder_net.layers.0.linear.bias", (128,)),
        _tensor("core.decoder_net.layers.0.linear.weight", (128, 64)),
        _tensor("core.spec_encoder.lift_network.bias", (64,)),
        _tensor("core.spec_encoder.lift_network.weight", (64, 2, 1, 1)),
    ]
    for index in range(4):
        values.extend(
            (
                _tensor(f"core.spec_encoder.conv_layers.{index}.bias", (64,)),
                _tensor(
                    f"core.spec_encoder.conv_layers.{index}.weight",
                    (64, 64, 1, 1),
                ),
                _tensor(
                    f"core.spec_encoder.spconv_layers.{index}.weights1",
                    (64, 64, 24, 24, 2),
                ),
                _tensor(
                    f"core.spec_encoder.spconv_layers.{index}.weights2",
                    (64, 64, 24, 24, 2),
                ),
            )
        )
    return tuple(sorted(values, key=lambda item: item.name))


EXPECTED_MODEL_TENSORS = _expected_tensor_descriptors()
EXPECTED_MODEL_TENSOR_MAP: Mapping[str, ArrayDescriptor] = MappingProxyType(
    {descriptor.name: descriptor.array_descriptor() for descriptor in EXPECTED_MODEL_TENSORS}
)
MODEL_TENSOR_BYTES = sum(descriptor.nbytes for descriptor in EXPECTED_MODEL_TENSORS)


def _semantic_version(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(_SEMVER_PATTERN, value)
    if match is None:
        raise ValueError(f"invalid semantic version {value!r}")
    return cast(tuple[int, int, int], tuple(int(item) for item in value.split(".")))


class CompatibilityRange(VersionedModel):
    """Closed runtime range for the v0.1 FNO bundle format."""

    minimum_soufflerie: SemanticVersion = "0.1.0"
    maximum_soufflerie_exclusive: SemanticVersion = "0.2.0"
    minimum_python: tuple[int, int] = (3, 11)
    maximum_python_exclusive: tuple[int, int] = (3, 12)
    torch: Literal["2.10.0"] = "2.10.0"
    physicsnemo: Literal["2.2.1"] = "2.2.1"
    minimum_bundle_schema: Literal[1] = 1
    maximum_bundle_schema: Literal[1] = 1

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("minimum_python", "maximum_python_exclusive"):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized

    @model_validator(mode="after")
    def _ranges_are_increasing_and_support_this_schema(self) -> Self:
        if _semantic_version(self.minimum_soufflerie) >= _semantic_version(
            self.maximum_soufflerie_exclusive
        ):
            raise ValueError("Soufflerie compatibility range must be increasing")
        for name, version in (
            ("minimum_python", self.minimum_python),
            ("maximum_python_exclusive", self.maximum_python_exclusive),
        ):
            if any(isinstance(item, bool) or item < 0 for item in version):
                raise ValueError(f"{name} must contain nonnegative version integers")
        if self.minimum_python >= self.maximum_python_exclusive:
            raise ValueError("Python compatibility range must be increasing")
        if not self.minimum_bundle_schema <= 1 <= self.maximum_bundle_schema:
            raise ValueError("bundle compatibility range must include schema version 1")
        return self

    def validate_core_runtime(self) -> None:
        current_python = (sys.version_info.major, sys.version_info.minor)
        if not self.minimum_python <= current_python < self.maximum_python_exclusive:
            raise ArtifactIntegrityError(
                "BUNDLE-5 COMPATIBILITY: Python runtime is outside the declared range"
            )
        try:
            package_version = importlib.metadata.version("soufflerie")
        except importlib.metadata.PackageNotFoundError:
            package_version = "0.1.0"
        current_package = _semantic_version(package_version)
        if not (
            _semantic_version(self.minimum_soufflerie)
            <= current_package
            < _semantic_version(self.maximum_soufflerie_exclusive)
        ):
            raise ArtifactIntegrityError(
                "BUNDLE-5 COMPATIBILITY: Soufflerie runtime is outside the declared range"
            )


class ModelCardGate(StrictFrozenModel):
    """One required validation gate rendered without recomputation."""

    name: SingleLine
    status: Literal["not_evaluated", "green", "red"]
    threshold: SingleLine
    measured: SingleLine | None = None
    evidence_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def _status_has_coherent_evidence(self) -> Self:
        evaluated = self.status in {"green", "red"}
        if evaluated != (self.measured is not None) or evaluated != (
            self.evidence_sha256 is not None
        ):
            raise ValueError(
                "model-card gate status, measured value, and evidence must be coherent"
            )
        return self


class ModelCardMetadata(StrictFrozenModel):
    """Bounded factual content rendered into the immutable model card."""

    display_name: SingleLine
    summary: SingleLine
    intended_uses: tuple[SingleLine, ...] = Field(min_length=1, max_length=8)
    limitations: tuple[SingleLine, ...] = Field(min_length=1, max_length=12)
    gates: tuple[ModelCardGate, ...] = Field(min_length=1, max_length=32)
    license: Literal["Apache-2.0"] = "Apache-2.0"

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("intended_uses", "limitations", "gates"):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized

    @model_validator(mode="after")
    def _gate_names_are_unique(self) -> Self:
        names = [gate.name for gate in self.gates]
        if len(names) != len(set(names)):
            raise ValueError("model-card gate names must be unique")
        return self


class ModelBundleMetadata(VersionedModel):
    """Commit-marker-bound identity and member contract for one deployable FNO."""

    model_id: ContentId
    model_sha256: Sha256
    architecture: Literal["fno2d-v1"] = "fno2d-v1"
    dataset_id: ContentId
    dataset_sha256: Sha256
    experiment_id: ContentId
    seed: int = Field(ge=0, le=_UINT64_MAX)
    selected_epoch: int = Field(ge=1, le=100)
    weights_sha256: Sha256
    weights_file_bytes: int = Field(ge=1, le=MODEL_WEIGHTS_FILE_CAP_BYTES)
    weights_tensor_bytes: int = Field(
        default=MODEL_TENSOR_BYTES,
        ge=MODEL_TENSOR_BYTES,
        le=MODEL_TENSOR_BYTES,
        json_schema_extra={"const": MODEL_TENSOR_BYTES},
    )
    preprocessing_sha256: Sha256
    architecture_sha256: Sha256
    model_card_sha256: Sha256
    code_revision: Annotated[str, StringConstraints(pattern=_SOURCE_REVISION_PATTERN)]
    source_dirty: Literal[False] = False
    lock_digest: Sha256
    compatibility: CompatibilityRange = Field(default_factory=CompatibilityRange)
    max_weights_file_bytes: int = Field(
        default=MODEL_WEIGHTS_FILE_CAP_BYTES,
        ge=MODEL_WEIGHTS_FILE_CAP_BYTES,
        le=MODEL_WEIGHTS_FILE_CAP_BYTES,
        json_schema_extra={"const": MODEL_WEIGHTS_FILE_CAP_BYTES},
    )
    tensors: tuple[ModelTensorDescriptor, ...] = Field(
        default=EXPECTED_MODEL_TENSORS,
        min_length=len(EXPECTED_MODEL_TENSORS),
        max_length=len(EXPECTED_MODEL_TENSORS),
        json_schema_extra={
            "const": [item.model_dump(mode="json") for item in EXPECTED_MODEL_TENSORS]
        },
    )
    model_card: ModelCardMetadata

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        tensors = normalized.get("tensors")
        if isinstance(tensors, list):
            normalized["tensors"] = tuple(tensors)
        return normalized

    def logical_identity(self) -> dict[str, object]:
        return self.model_dump(
            mode="python",
            exclude={"model_id", "model_sha256", "model_card_sha256"},
        )

    @model_validator(mode="after")
    def _identity_and_allowlist_are_coherent(self) -> Self:
        if self.dataset_id != self.dataset_sha256[:20]:
            raise ValueError("dataset_id must prefix the full parent dataset digest")
        if self.tensors != EXPECTED_MODEL_TENSORS:
            raise ValueError("bundle tensor descriptors must match the fno2d-v1 allowlist")
        if self.weights_tensor_bytes != sum(item.nbytes for item in self.tensors):
            raise ValueError("weights_tensor_bytes does not match tensor descriptors")
        expected_architecture_sha256 = sha256_bytes(canonical_json_bytes(FnoArchitecture()))
        if self.architecture_sha256 != expected_architecture_sha256:
            raise ValueError("architecture digest does not match fno2d-v1")
        expected_model_sha256 = canonical_sha256(self.logical_identity())
        if self.model_sha256 != expected_model_sha256:
            raise ValueError("model_sha256 does not match bundle logical identity")
        if self.model_id != self.model_sha256[:20]:
            raise ValueError("model_id must be the prefix of model_sha256")
        expected_card_sha256 = sha256_bytes(render_model_card(self).encode("utf-8"))
        if self.model_card_sha256 != expected_card_sha256:
            raise ValueError("model_card_sha256 does not match generated model card")
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        dataset_sha256: str,
        experiment_id: str,
        seed: int,
        selected_epoch: int,
        weights_sha256: str,
        weights_file_bytes: int,
        preprocessing_sha256: str,
        architecture_sha256: str,
        code_revision: str,
        lock_digest: str,
        model_card: ModelCardMetadata,
        compatibility: CompatibilityRange | None = None,
    ) -> ModelBundleMetadata:
        values: dict[str, object] = {
            "dataset_id": dataset_id,
            "dataset_sha256": dataset_sha256,
            "experiment_id": experiment_id,
            "seed": seed,
            "selected_epoch": selected_epoch,
            "weights_sha256": weights_sha256,
            "weights_file_bytes": weights_file_bytes,
            "preprocessing_sha256": preprocessing_sha256,
            "architecture_sha256": architecture_sha256,
            "code_revision": code_revision,
            "lock_digest": lock_digest,
            "compatibility": compatibility or CompatibilityRange(),
            "model_card": model_card,
        }
        identity = {
            "schema_version": 1,
            "architecture": "fno2d-v1",
            "source_dirty": False,
            "weights_tensor_bytes": MODEL_TENSOR_BYTES,
            "max_weights_file_bytes": MODEL_WEIGHTS_FILE_CAP_BYTES,
            "tensors": [item.model_dump(mode="python") for item in EXPECTED_MODEL_TENSORS],
            **values,
        }
        model_sha256 = canonical_sha256(identity)
        payload = {
            "schema_version": 1,
            "model_id": model_sha256[:20],
            "model_sha256": model_sha256,
            "model_card_sha256": "0" * 64,
            **values,
        }
        temporary = cls.model_construct(**cast(Any, payload))
        card_sha256 = sha256_bytes(render_model_card(temporary).encode("utf-8"))
        payload["model_card_sha256"] = card_sha256
        return cls.model_validate(payload)


def render_model_card(metadata: ModelBundleMetadata) -> str:
    """Render one deterministic evidence-bounded Markdown model card."""

    gates = [
        "| Gate | Status | Threshold | Measured | Evidence SHA-256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    gates.extend(
        "| "
        + " | ".join(
            (
                gate.name,
                gate.status,
                gate.threshold,
                gate.measured or "not evaluated",
                gate.evidence_sha256 or "not available",
            )
        )
        + " |"
        for gate in metadata.model_card.gates
    )
    lines = [
        f"# {metadata.model_card.display_name}",
        "",
        metadata.model_card.summary,
        "",
        "## Identity",
        "",
        f"- Model ID: `{metadata.model_id}`",
        f"- Dataset ID: `{metadata.dataset_id}`",
        f"- Dataset SHA-256: `{metadata.dataset_sha256}`",
        f"- Experiment ID: `{metadata.experiment_id}`",
        f"- Architecture: `{metadata.architecture}`",
        f"- Selected epoch: `{metadata.selected_epoch}`",
        f"- Training seed: `{metadata.seed}`",
        f"- Weights SHA-256: `{metadata.weights_sha256}`",
        f"- Source revision: `{metadata.code_revision}`",
        f"- License: `{metadata.model_card.license}`",
        "",
        "## Intended use",
        "",
        *(f"- {item}" for item in metadata.model_card.intended_uses),
        "",
        "## Validation gates",
        "",
        *gates,
        "",
        "## Limitations",
        "",
        *(f"- {item}" for item in metadata.model_card.limitations),
        "",
    ]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ModelBundle:
    metadata: ModelBundleMetadata
    weights_bytes: bytes
    preprocessing: PreprocessingStatistics
    architecture: FnoArchitecture
    model_card_markdown: str

    def __post_init__(self) -> None:
        if self.metadata.dataset_id != self.preprocessing.dataset_id:
            raise ArtifactIntegrityError(
                "BUNDLE-2 IDENTITY: preprocessing dataset does not match bundle"
            )
        if self.architecture != FnoArchitecture():
            raise ArtifactIntegrityError("BUNDLE-3 ARCHITECTURE: architecture is not allowed")
        if len(self.weights_bytes) != self.metadata.weights_file_bytes:
            raise ArtifactIntegrityError("BUNDLE-2 IDENTITY: weights byte count mismatch")
        digests = {
            "weights": sha256_bytes(self.weights_bytes),
            "preprocessing": sha256_bytes(canonical_json_bytes(self.preprocessing)),
            "architecture": sha256_bytes(canonical_json_bytes(self.architecture)),
            "model_card": sha256_bytes(self.model_card_markdown.encode("utf-8")),
        }
        expected = {
            "weights": self.metadata.weights_sha256,
            "preprocessing": self.metadata.preprocessing_sha256,
            "architecture": self.metadata.architecture_sha256,
            "model_card": self.metadata.model_card_sha256,
        }
        if digests != expected:
            raise ArtifactIntegrityError("BUNDLE-2 IDENTITY: member digest mismatch")
        if self.model_card_markdown != render_model_card(self.metadata):
            raise ArtifactIntegrityError("BUNDLE-4 CARD: model card is not canonical")

    @property
    def reference(self) -> ArtifactRef:
        size = (
            len(self.weights_bytes)
            + len(canonical_json_bytes(self.metadata))
            + len(canonical_json_bytes(self.preprocessing))
            + len(canonical_json_bytes(self.architecture))
            + len(self.model_card_markdown.encode("utf-8"))
            + 65
        )
        return ArtifactRef(
            artifact_type="model",
            artifact_id=self.metadata.model_id,
            sha256=self.metadata.model_sha256,
            size_bytes=size,
            uri=f"{MODEL_ROOT_PREFIX}/{self.metadata.model_id}",
        )


@dataclass(frozen=True, slots=True)
class PublishedModelBundle:
    reference: ArtifactRef
    metadata: ModelBundleMetadata
    weights: Mapping[str, npt.NDArray[np.float32]]
    preprocessing: PreprocessingStatistics
    architecture: FnoArchitecture
    model_card_markdown: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))


def _validate_weight_arrays(
    weights: Mapping[str, npt.NDArray[Any]],
) -> dict[str, npt.NDArray[np.float32]]:
    if set(weights) != set(EXPECTED_MODEL_TENSOR_MAP):
        raise ArtifactIntegrityError("BUNDLE-1 TENSORS: weights do not match the closed allowlist")
    validated: dict[str, npt.NDArray[np.float32]] = {}
    for name, descriptor in EXPECTED_MODEL_TENSOR_MAP.items():
        value = weights[name]
        descriptor.validate_array(value, name=name)
        if not value.flags.c_contiguous:
            raise ArtifactIntegrityError(f"BUNDLE-1 TENSORS: {name!r} must be C-contiguous")
        validated[name] = cast(npt.NDArray[np.float32], value)
    return validated


def _encode_safetensors(
    weights: Mapping[str, npt.NDArray[np.float32]],
) -> bytes:
    """Encode a canonical safetensors header and sorted contiguous tensor region."""

    header: dict[str, object] = {
        "__metadata__": {"architecture": "fno2d-v1", "format": "soufflerie"}
    }
    offset = 0
    for name in sorted(weights):
        value = weights[name]
        end = offset + value.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    output = io.BytesIO()
    output.write(struct.pack("<Q", len(header_bytes)))
    output.write(header_bytes)
    for name in sorted(weights):
        output.write(weights[name].tobytes(order="C"))
    return output.getvalue()


def build_model_bundle(
    *,
    weights: Mapping[str, npt.NDArray[Any]],
    preprocessing: PreprocessingStatistics,
    dataset_sha256: str,
    experiment_id: str,
    seed: int,
    selected_epoch: int,
    code_revision: str,
    lock_digest: str,
    model_card: ModelCardMetadata,
    architecture: FnoArchitecture | None = None,
    compatibility: CompatibilityRange | None = None,
) -> ModelBundle:
    """Encode validated CPU float32 weights and all logical sidecars."""

    if not isinstance(preprocessing, PreprocessingStatistics):
        raise TypeError("preprocessing must be PreprocessingStatistics")
    resolved_architecture = architecture or FnoArchitecture()
    if resolved_architecture != FnoArchitecture():
        raise ArtifactIntegrityError("BUNDLE-3 ARCHITECTURE: only fno2d-v1 is allowed")
    validated = _validate_weight_arrays(weights)
    try:
        weights_bytes = _encode_safetensors(validated)
    except Exception as error:
        raise ArtifactIntegrityError("BUNDLE-1 TENSORS: safe weight encoding failed") from error
    if len(weights_bytes) > MODEL_WEIGHTS_FILE_CAP_BYTES:
        raise ArtifactIntegrityError("BUNDLE-1 TENSORS: weights exceed the file byte cap")
    preprocessing_sha256 = sha256_bytes(canonical_json_bytes(preprocessing))
    architecture_sha256 = sha256_bytes(canonical_json_bytes(resolved_architecture))
    metadata = ModelBundleMetadata.create(
        dataset_id=preprocessing.dataset_id,
        dataset_sha256=dataset_sha256,
        experiment_id=experiment_id,
        seed=seed,
        selected_epoch=selected_epoch,
        weights_sha256=sha256_bytes(weights_bytes),
        weights_file_bytes=len(weights_bytes),
        preprocessing_sha256=preprocessing_sha256,
        architecture_sha256=architecture_sha256,
        code_revision=code_revision,
        lock_digest=lock_digest,
        model_card=model_card,
        compatibility=compatibility,
    )
    return ModelBundle(
        metadata=metadata,
        weights_bytes=weights_bytes,
        preprocessing=preprocessing,
        architecture=resolved_architecture,
        model_card_markdown=render_model_card(metadata),
    )


def snapshot_fno_weights(predictor: FnoPredictor) -> dict[str, npt.NDArray[np.float32]]:
    """Explicitly copy one fp32 predictor state to immutable CPU NumPy arrays."""

    if not isinstance(predictor, FnoPredictor):
        raise TypeError("predictor must be an FnoPredictor")
    state = predictor.state_dict()
    if set(state) != set(EXPECTED_MODEL_TENSOR_MAP):
        raise ArtifactIntegrityError("BUNDLE-1 TENSORS: predictor state names changed")
    result: dict[str, npt.NDArray[np.float32]] = {}
    for name, descriptor in EXPECTED_MODEL_TENSOR_MAP.items():
        tensor = state[name]
        if str(tensor.dtype) != "torch.float32" or tuple(tensor.shape) != descriptor.shape:
            raise ArtifactIntegrityError(f"BUNDLE-1 TENSORS: predictor tensor {name!r} drifted")
        if not bool(tensor.is_contiguous()) or not bool(tensor.isfinite().all().item()):
            raise ArtifactIntegrityError(f"BUNDLE-1 TENSORS: predictor tensor {name!r} is invalid")
        array = np.array(tensor.detach().cpu().numpy(), dtype=np.float32, order="C", copy=True)
        array.flags.writeable = False
        result[name] = array
    return result


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_real_directory(root: Path, *parts: str) -> Path:
    current = root
    for part in parts:
        candidate = current / part
        created = False
        try:
            candidate.mkdir()
            created = True
        except FileExistsError:
            pass
        try:
            details = candidate.lstat()
        except OSError as error:
            raise ArtifactIntegrityError(
                f"BUNDLE-6 STORE: unable to inspect component {part!r}"
            ) from error
        if not stat.S_ISDIR(details.st_mode):
            raise ArtifactIntegrityError(
                f"BUNDLE-6 STORE: component {part!r} is not a real directory"
            )
        if created:
            _fsync_directory(candidate)
            _fsync_directory(current)
        current = candidate
    return current


class LocalModelBundleStore:
    """Stage, fully verify, commit, and atomically publish immutable model bundles."""

    expected_files = frozenset(
        {
            MODEL_METADATA_NAME,
            MODEL_WEIGHTS_NAME,
            MODEL_PREPROCESSING_NAME,
            MODEL_ARCHITECTURE_NAME,
            MODEL_CARD_NAME,
            MODEL_COMMIT_NAME,
        }
    )

    def __init__(
        self,
        root: Path,
        *,
        limits: ReaderLimits = DEFAULT_READER_LIMITS,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self._weight_limits = limits.model_copy(
            update={
                "max_file_bytes": min(limits.max_file_bytes, MODEL_WEIGHTS_FILE_CAP_BYTES),
                "max_member_bytes": min(limits.max_member_bytes, MODEL_TENSOR_MEMBER_CAP_BYTES),
                "max_total_uncompressed_bytes": min(
                    limits.max_total_uncompressed_bytes, MODEL_WEIGHTS_FILE_CAP_BYTES
                ),
                "max_tensor_count": len(EXPECTED_MODEL_TENSORS),
            }
        )
        self._fault_injector = fault_injector

    def _inject(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)

    @staticmethod
    def _key(prefix: str, name: str) -> str:
        return f"{prefix}/{name}" if prefix else name

    def _assert_exact_files(self, directory: Path) -> None:
        try:
            details = directory.lstat()
        except OSError as error:
            raise ArtifactIntegrityError("BUNDLE-6 STORE: unable to inspect bundle") from error
        if not stat.S_ISDIR(details.st_mode):
            raise ArtifactIntegrityError("BUNDLE-6 STORE: bundle is not a real directory")
        try:
            names = {entry.name for entry in directory.iterdir()}
        except OSError as error:
            raise ArtifactIntegrityError("BUNDLE-6 STORE: unable to list bundle") from error
        if names != self.expected_files:
            raise ArtifactIntegrityError(
                "BUNDLE-6 STORE: bundle members must exactly match the closed layout"
            )
        for name in names:
            member = directory / name
            member_details = member.lstat()
            if not stat.S_ISREG(member_details.st_mode):
                raise ArtifactIntegrityError(
                    f"BUNDLE-6 STORE: member {name!r} is not a regular file"
                )

    def _load_members(
        self,
        *,
        root: Path,
        prefix: str,
        metadata_digest: str,
    ) -> tuple[
        ModelBundleMetadata,
        dict[str, npt.NDArray[Any]],
        PreprocessingStatistics,
        FnoArchitecture,
        str,
    ]:
        metadata = safe_read_json(
            root,
            self._key(prefix, MODEL_METADATA_NAME),
            model=ModelBundleMetadata,
            limits=self.limits.model_copy(
                update={
                    "max_file_bytes": MODEL_JSON_CAP_BYTES,
                    "max_json_bytes": MODEL_JSON_CAP_BYTES,
                }
            ),
            expected_sha256=metadata_digest,
        )
        metadata.compatibility.validate_core_runtime()
        architecture = safe_read_json(
            root,
            self._key(prefix, MODEL_ARCHITECTURE_NAME),
            model=FnoArchitecture,
            limits=self.limits,
            expected_sha256=metadata.architecture_sha256,
        )
        if architecture != FnoArchitecture():
            raise ArtifactIntegrityError("BUNDLE-3 ARCHITECTURE: architecture is not allowed")
        preprocessing = safe_read_json(
            root,
            self._key(prefix, MODEL_PREPROCESSING_NAME),
            model=PreprocessingStatistics,
            limits=self.limits,
            expected_sha256=metadata.preprocessing_sha256,
        )
        if preprocessing.dataset_id != metadata.dataset_id:
            raise ArtifactIntegrityError(
                "BUNDLE-2 IDENTITY: preprocessing dataset does not match metadata"
            )
        card_bytes = safe_read_bytes(
            root,
            self._key(prefix, MODEL_CARD_NAME),
            max_bytes=MODEL_CARD_CAP_BYTES,
            expected_sha256=metadata.model_card_sha256,
        )
        try:
            card = card_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArtifactIntegrityError("BUNDLE-4 CARD: model card is not UTF-8") from error
        if card != render_model_card(metadata):
            raise ArtifactIntegrityError("BUNDLE-4 CARD: model card is not canonical")
        weights = safe_read_tensors(
            root,
            self._key(prefix, MODEL_WEIGHTS_NAME),
            expected=EXPECTED_MODEL_TENSOR_MAP,
            limits=self._weight_limits,
            expected_sha256=metadata.weights_sha256,
        )
        weights_file = safe_read_bytes(
            root,
            self._key(prefix, MODEL_WEIGHTS_NAME),
            max_bytes=MODEL_WEIGHTS_FILE_CAP_BYTES,
            expected_sha256=metadata.weights_sha256,
        )
        if len(weights_file) != metadata.weights_file_bytes:
            raise ArtifactIntegrityError("BUNDLE-2 IDENTITY: weights file size mismatch")
        return metadata, weights, preprocessing, architecture, card

    def publish(self, bundle: ModelBundle) -> ArtifactRef:
        if not isinstance(bundle, ModelBundle):
            raise TypeError("bundle must be a ModelBundle")
        reference = bundle.reference
        target = self.root / reference.uri
        if target.exists() or target.is_symlink():
            return self.open(
                reference.model_copy(update={"size_bytes": self._committed_size(reference.uri)})
            ).reference

        staging_parent = _ensure_real_directory(self.root, ".staging", MODEL_ROOT_PREFIX)
        staging = Path(tempfile.mkdtemp(prefix=f"{bundle.metadata.model_id}-", dir=staging_parent))
        try:
            files = {
                MODEL_METADATA_NAME: canonical_json_bytes(bundle.metadata),
                MODEL_WEIGHTS_NAME: bundle.weights_bytes,
                MODEL_PREPROCESSING_NAME: canonical_json_bytes(bundle.preprocessing),
                MODEL_ARCHITECTURE_NAME: canonical_json_bytes(bundle.architecture),
                MODEL_CARD_NAME: bundle.model_card_markdown.encode("utf-8"),
            }
            for name, content in files.items():
                path = staging / name
                path.write_bytes(content)
                _fsync_file(path)
            self._inject("members_written")

            metadata_digest = sha256_bytes(files[MODEL_METADATA_NAME])
            self._load_members(root=staging, prefix="", metadata_digest=metadata_digest)
            self._inject("verified")

            marker = staging / MODEL_COMMIT_NAME
            marker.write_text(metadata_digest + "\n", encoding="ascii")
            _fsync_file(marker)
            _fsync_directory(staging)
            self._inject("committed")

            parent = _ensure_real_directory(self.root, MODEL_ROOT_PREFIX)
            try:
                os.rename(staging, target)
            except OSError as error:
                if error.errno not in {EEXIST, ENOTEMPTY}:
                    raise
                reference = self.open(
                    reference.model_copy(update={"size_bytes": self._committed_size(reference.uri)})
                ).reference
            _fsync_directory(parent)
            self._inject("published")
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return reference

    def _committed_size(self, prefix: str) -> int:
        caps = {
            MODEL_METADATA_NAME: MODEL_JSON_CAP_BYTES,
            MODEL_WEIGHTS_NAME: MODEL_WEIGHTS_FILE_CAP_BYTES,
            MODEL_PREPROCESSING_NAME: MODEL_JSON_CAP_BYTES,
            MODEL_ARCHITECTURE_NAME: MODEL_JSON_CAP_BYTES,
            MODEL_CARD_NAME: MODEL_CARD_CAP_BYTES,
            MODEL_COMMIT_NAME: 65,
        }
        return sum(
            len(safe_read_bytes(self.root, self._key(prefix, name), max_bytes=cap))
            for name, cap in caps.items()
        )

    def open(self, reference: ArtifactRef) -> PublishedModelBundle:
        if not isinstance(reference, ArtifactRef) or reference.artifact_type != "model":
            raise ArtifactIntegrityError("BUNDLE-2 IDENTITY: expected a model ArtifactRef")
        expected_uri = f"{MODEL_ROOT_PREFIX}/{reference.artifact_id}"
        if reference.uri != expected_uri or reference.sha256[:20] != reference.artifact_id:
            raise ArtifactIntegrityError("BUNDLE-2 IDENTITY: model reference is incoherent")
        directory = self.root / reference.uri
        self._assert_exact_files(directory)
        marker = safe_read_bytes(
            self.root,
            self._key(reference.uri, MODEL_COMMIT_NAME),
            max_bytes=65,
        )
        try:
            metadata_digest = marker.decode("ascii").removesuffix("\n")
        except UnicodeDecodeError as error:
            raise ArtifactIntegrityError("BUNDLE-6 COMMIT: marker is not ASCII") from error
        if len(marker) != 65 or re.fullmatch(r"[0-9a-f]{64}", metadata_digest) is None:
            raise ArtifactIntegrityError("BUNDLE-6 COMMIT: marker must contain one metadata digest")
        metadata, weights, preprocessing, architecture, card = self._load_members(
            root=self.root,
            prefix=reference.uri,
            metadata_digest=metadata_digest,
        )
        if metadata.model_id != reference.artifact_id or metadata.model_sha256 != reference.sha256:
            raise ArtifactIntegrityError("BUNDLE-2 IDENTITY: reference and metadata disagree")
        actual_size = self._committed_size(reference.uri)
        if actual_size != reference.size_bytes:
            raise ArtifactIntegrityError("BUNDLE-2 IDENTITY: reference byte count mismatch")
        return PublishedModelBundle(
            reference=reference,
            metadata=metadata,
            weights=cast(dict[str, npt.NDArray[np.float32]], weights),
            preprocessing=preprocessing,
            architecture=architecture,
            model_card_markdown=card,
        )


def instantiate_bundle_predictor(
    bundle: PublishedModelBundle,
    *,
    device: str,
) -> FnoPredictor:
    """Construct the fixed runtime and explicitly copy verified weights to one device."""

    if not isinstance(bundle, PublishedModelBundle):
        raise TypeError("bundle must be a PublishedModelBundle")
    if _DEVICE_PATTERN.fullmatch(device) is None:
        raise DeviceUnavailableError("device must be cpu or cuda[:index]")
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise DependencyUnavailableError(
            "bundle inference requires the locked 'ml' extra"
        ) from error
    runtime_torch = str(getattr(torch, "__version__", "")).split("+")[0]
    if runtime_torch != bundle.metadata.compatibility.torch:
        raise DependencyUnavailableError(
            f"bundle requires Torch {bundle.metadata.compatibility.torch}, got {runtime_torch}"
        )
    predictor = FnoPredictor(bundle.architecture)
    if device.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise DeviceUnavailableError(f"requested device {device} is unavailable")
    state: dict[str, Any] = {}
    for name, array in bundle.weights.items():
        owned = np.array(array, dtype=np.float32, order="C", copy=True)
        tensor = torch.from_numpy(owned)
        if device != "cpu":
            try:
                tensor = tensor.to(device=device)
            except (RuntimeError, ValueError) as error:
                raise DeviceUnavailableError(f"requested device {device} is unavailable") from error
        state[name] = tensor
    predictor.to(device)
    result = predictor.load_state_dict(state, strict=True)
    missing = tuple(getattr(result, "missing_keys", ()))
    unexpected = tuple(getattr(result, "unexpected_keys", ()))
    if missing or unexpected:
        raise ArtifactIntegrityError("BUNDLE-1 TENSORS: state loading changed the allowlist")
    predictor.eval()
    return predictor


__all__ = [
    "EXPECTED_MODEL_TENSORS",
    "MODEL_TENSOR_BYTES",
    "MODEL_WEIGHTS_FILE_CAP_BYTES",
    "CompatibilityRange",
    "LocalModelBundleStore",
    "ModelBundle",
    "ModelBundleMetadata",
    "ModelCardGate",
    "ModelCardMetadata",
    "ModelTensorDescriptor",
    "PublishedModelBundle",
    "build_model_bundle",
    "instantiate_bundle_predictor",
    "render_model_card",
    "snapshot_fno_weights",
]
