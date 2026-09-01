"""Strict provider-neutral contracts for remote solve and smoke-sweep execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self, TypeVar

from pydantic import Field, StringConstraints, ValidationError, field_validator, model_validator

from soufflerie.artifacts import safe_read_bytes
from soufflerie.config import SweepConfig
from soufflerie.datagen._local_files import ensure_real_directory, fsync_directory, fsync_file
from soufflerie.errors import ArtifactIntegrityError, ConfigurationError
from soufflerie.geometry import validate_geometry
from soufflerie.schemas import (
    ArtifactRef,
    CaseConfig,
    ContentId,
    Sha256,
    ShapeParams,
    Split,
    VersionedModel,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)

MAX_REMOTE_INPUT_BYTES = 16 * 1024
REMOTE_ARTIFACT_ROOT = "soufflerie/v1"
SMOKE_SAMPLE_COUNT: Literal[8] = 8
SMOKE_DESIGN_KIND: Literal["remote-smoke-v1"] = "remote-smoke-v1"

CorrelationId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$"),
]
AttemptId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
DeviceClass = Literal["L40S", "A10G"]

_CORRELATION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_ModelT = TypeVar("_ModelT", bound=VersionedModel)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_remote_model(data: bytes, model: type[_ModelT]) -> _ModelT:
    """Parse one exact canonical JSON model after enforcing the RPC byte cap."""

    if not isinstance(data, bytes):
        raise ConfigurationError("remote input must be bytes")
    if not data or len(data) > MAX_REMOTE_INPUT_BYTES:
        raise ConfigurationError(
            f"remote input must contain 1..{MAX_REMOTE_INPUT_BYTES} canonical JSON bytes"
        )
    try:
        json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
        value = model.model_validate_json(data)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
        ValidationError,
    ) as error:
        raise ConfigurationError(f"remote input does not match {model.__name__}") from error
    if canonical_json_bytes(value) != data:
        raise ConfigurationError("remote input must use the canonical JSON encoding")
    return value


def encode_remote_model(value: VersionedModel) -> bytes:
    """Encode a versioned model for a bounded RPC and fail before submission."""

    if not isinstance(value, VersionedModel):
        raise TypeError("remote values must be versioned models")
    content = canonical_json_bytes(value)
    if len(content) > MAX_REMOTE_INPUT_BYTES:
        raise ConfigurationError(f"remote input exceeds {MAX_REMOTE_INPUT_BYTES} bytes")
    return content


def validate_correlation_id(value: str) -> str:
    if not isinstance(value, str) or _CORRELATION_PATTERN.fullmatch(value) is None:
        raise ConfigurationError("correlation_id is not a bounded safe token")
    return value


class RemoteSweepRequest(VersionedModel):
    """Immutable request for the deliberately non-release eight-case smoke design."""

    design_kind: Literal["remote-smoke-v1"] = SMOKE_DESIGN_KIND
    config: SweepConfig
    sample_count: Literal[8] = SMOKE_SAMPLE_COUNT
    requested_device_class: DeviceClass
    source_revision: Revision
    lock_sha256: Sha256
    force_failure_once: bool = True
    request_digest: Sha256

    @model_validator(mode="before")
    @classmethod
    def _normalize_config_json_tuple(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        config = normalized.get("config")
        if isinstance(config, Mapping):
            normalized_config = dict(config)
            split_counts = normalized_config.get("split_counts")
            if isinstance(split_counts, list):
                normalized_config["split_counts"] = tuple(split_counts)
            normalized["config"] = normalized_config
        return normalized

    def logical_identity(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "design_kind": self.design_kind,
            "config": self.config.model_dump(mode="json"),
            "sample_count": self.sample_count,
            "requested_device_class": self.requested_device_class,
            "source_revision": self.source_revision,
            "lock_sha256": self.lock_sha256,
        }

    @model_validator(mode="after")
    def _digest_is_coherent(self) -> Self:
        if self.request_digest != canonical_sha256(self.logical_identity()):
            raise ValueError("request_digest does not match the smoke sweep identity")
        return self

    @classmethod
    def create(
        cls,
        *,
        config: SweepConfig,
        requested_device_class: DeviceClass,
        source_revision: str,
        lock_sha256: str,
        force_failure_once: bool = True,
    ) -> Self:
        values: dict[str, object] = {
            "schema_version": 1,
            "design_kind": SMOKE_DESIGN_KIND,
            "config": config,
            "sample_count": SMOKE_SAMPLE_COUNT,
            "requested_device_class": requested_device_class,
            "source_revision": source_revision,
            "lock_sha256": lock_sha256,
            "force_failure_once": force_failure_once,
        }
        identity = {
            "schema_version": 1,
            "design_kind": SMOKE_DESIGN_KIND,
            "config": config.model_dump(mode="json"),
            "sample_count": SMOKE_SAMPLE_COUNT,
            "requested_device_class": requested_device_class,
            "source_revision": source_revision,
            "lock_sha256": lock_sha256,
        }
        return cls.model_validate({**values, "request_digest": canonical_sha256(identity)})


class SmokeDesignPoint(VersionedModel):
    """One deterministic point in the smoke-only stratified design."""

    index: int = Field(ge=0, lt=SMOKE_SAMPLE_COUNT)
    design_id: ContentId
    split: Split
    case: CaseConfig


_SMOKE_PERMUTATIONS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (3, 0, 6, 2, 7, 4, 1, 5),
    (6, 2, 5, 1, 4, 0, 7, 3),
    (1, 5, 3, 7, 0, 4, 2, 6),
)


def _stratum_value(minimum: float, maximum: float, stratum: int) -> float:
    unit = (float(stratum) + 0.5) / float(SMOKE_SAMPLE_COUNT)
    return minimum + (maximum - minimum) * unit


def smoke_design(request: RemoteSweepRequest) -> tuple[SmokeDesignPoint, ...]:
    """Build eight fixed stratified points without claiming release LHS semantics."""

    if not isinstance(request, RemoteSweepRequest):
        raise TypeError("request must be a RemoteSweepRequest")
    config = request.config
    split_by_index: tuple[Split, ...] = (
        "train",
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
    )
    points: list[SmokeDesignPoint] = []
    for index in range(SMOKE_SAMPLE_COUNT):
        physical = {
            "aspect_ratio": _stratum_value(
                config.aspect_ratio.minimum,
                config.aspect_ratio.maximum,
                _SMOKE_PERMUTATIONS[0][index],
            ),
            "rotation_deg": _stratum_value(
                config.rotation_deg.minimum,
                config.rotation_deg.maximum,
                _SMOKE_PERMUTATIONS[1][index],
            ),
            "scale": _stratum_value(
                config.scale.minimum,
                config.scale.maximum,
                _SMOKE_PERMUTATIONS[2][index],
            ),
            "reynolds": _stratum_value(
                config.reynolds.minimum,
                config.reynolds.maximum,
                _SMOKE_PERMUTATIONS[3][index],
            ),
        }
        design_id = canonical_sha256(
            {
                "schema_version": 1,
                "design_kind": SMOKE_DESIGN_KIND,
                "physical": physical,
            }
        )[:20]
        seed_material = hashlib.sha256(
            f"smoke-seed-v1:{config.seed}:{index}".encode("ascii")
        ).digest()
        case = CaseConfig(
            shape=ShapeParams(
                aspect_ratio=physical["aspect_ratio"],
                rotation_deg=physical["rotation_deg"],
                scale=physical["scale"],
            ),
            reynolds=physical["reynolds"],
            nx=config.grid.nx,
            ny=config.grid.ny,
            steps=config.run.steps,
            warmup_steps=config.run.warmup_steps,
            inlet_velocity_lu=config.run.inlet_velocity_lu,
            seed=int.from_bytes(seed_material[:8], "big"),
        )
        validate_geometry(case.shape, case.grid)
        points.append(
            SmokeDesignPoint(
                index=index,
                design_id=design_id,
                split=split_by_index[index],
                case=case,
            )
        )
    if len({point.case.case_id for point in points}) != SMOKE_SAMPLE_COUNT:
        raise ArtifactIntegrityError("remote smoke design produced duplicate case identities")
    return tuple(points)


class RemoteSolveRequest(VersionedModel):
    """Canonical solve envelope; attempt mechanics are outside logical run identity."""

    operation_kind: Literal["single", "smoke-sweep"]
    sweep_digest: Sha256
    design_id: ContentId
    split: Split
    case: CaseConfig
    requested_device_class: DeviceClass
    source_revision: Revision
    lock_sha256: Sha256
    attempt_id: AttemptId
    force_retryable_failure: bool = False
    request_digest: Sha256

    def logical_identity(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operation_kind": self.operation_kind,
            "sweep_digest": self.sweep_digest,
            "design_id": self.design_id,
            "split": self.split,
            "case": self.case.model_dump(mode="json"),
            "requested_device_class": self.requested_device_class,
            "source_revision": self.source_revision,
            "lock_sha256": self.lock_sha256,
        }

    @model_validator(mode="after")
    def _request_is_coherent(self) -> Self:
        if self.force_retryable_failure and self.operation_kind != "smoke-sweep":
            raise ValueError("forced retryable failure is restricted to smoke sweeps")
        if self.request_digest != canonical_sha256(self.logical_identity()):
            raise ValueError("request_digest does not match the solve identity")
        return self

    @classmethod
    def create(
        cls,
        *,
        operation_kind: Literal["single", "smoke-sweep"],
        sweep_digest: str,
        design_id: str,
        split: Split,
        case: CaseConfig,
        requested_device_class: DeviceClass,
        source_revision: str,
        lock_sha256: str,
        attempt_id: str,
        force_retryable_failure: bool = False,
    ) -> Self:
        values: dict[str, object] = {
            "schema_version": 1,
            "operation_kind": operation_kind,
            "sweep_digest": sweep_digest,
            "design_id": design_id,
            "split": split,
            "case": case,
            "requested_device_class": requested_device_class,
            "source_revision": source_revision,
            "lock_sha256": lock_sha256,
            "attempt_id": attempt_id,
            "force_retryable_failure": force_retryable_failure,
        }
        identity = {
            "schema_version": 1,
            "operation_kind": operation_kind,
            "sweep_digest": sweep_digest,
            "design_id": design_id,
            "split": split,
            "case": case.model_dump(mode="json"),
            "requested_device_class": requested_device_class,
            "source_revision": source_revision,
            "lock_sha256": lock_sha256,
        }
        return cls.model_validate({**values, "request_digest": canonical_sha256(identity)})


class SweepSummary(VersionedModel):
    """Small terminal or resumable outcome returned by the remote orchestrator."""

    sweep_digest: Sha256
    config_digest: Sha256
    design_kind: Literal["remote-smoke-v1"] = SMOKE_DESIGN_KIND
    requested_device_class: DeviceClass
    source_revision: Revision
    case_count: Literal[8] = SMOKE_SAMPLE_COUNT
    pending_count: int = Field(ge=0, le=SMOKE_SAMPLE_COUNT)
    running_count: int = Field(ge=0, le=SMOKE_SAMPLE_COUNT)
    succeeded_count: int = Field(ge=0, le=SMOKE_SAMPLE_COUNT)
    failed_count: int = Field(ge=0, le=SMOKE_SAMPLE_COUNT)
    initial_submitted_case_ids: tuple[ContentId, ...]
    resumed_case_ids: tuple[ContentId, ...]
    skipped_case_ids: tuple[ContentId, ...]
    run_references: tuple[ArtifactRef, ...]
    attempt_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    estimated_bytes: int = Field(ge=0)
    wall_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    final_state: Literal["succeeded", "incomplete"]
    evidence_sha256: Sha256

    @field_validator(
        "initial_submitted_case_ids",
        "resumed_case_ids",
        "skipped_case_ids",
        "run_references",
        mode="before",
    )
    @classmethod
    def _json_arrays_to_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _summary_is_coherent(self) -> Self:
        total = self.pending_count + self.running_count + self.succeeded_count + self.failed_count
        if total != self.case_count:
            raise ValueError("sweep state counts must sum to case_count")
        case_id_groups = (
            self.initial_submitted_case_ids,
            self.resumed_case_ids,
            self.skipped_case_ids,
        )
        if any(len(set(group)) != len(group) for group in case_id_groups):
            raise ValueError("sweep case ID lists must not contain duplicates")
        if len({reference.sha256 for reference in self.run_references}) != len(self.run_references):
            raise ValueError("sweep run references must have unique full digests")
        if len(self.run_references) != self.succeeded_count:
            raise ValueError("each successful case must have one run reference")
        expected_state = "succeeded" if self.succeeded_count == self.case_count else "incomplete"
        if self.final_state != expected_state:
            raise ValueError("final_state does not match the state counts")
        payload = self.model_dump(mode="json", exclude={"evidence_sha256"})
        if self.evidence_sha256 != canonical_sha256(payload):
            raise ValueError("evidence_sha256 does not match the sweep summary")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        payload: dict[str, object] = {"schema_version": 1, **values}
        return cls.model_validate({**payload, "evidence_sha256": canonical_sha256(payload)})


class SolveSummary(VersionedModel):
    """Small verified execution receipt for a committed run reference."""

    artifact: ArtifactRef
    case_id: ContentId
    source_revision: Revision
    device_class: str = Field(min_length=1)
    wall_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    final_state: Literal["succeeded"] = "succeeded"
    evidence_sha256: Sha256

    @model_validator(mode="after")
    def _receipt_is_coherent(self) -> Self:
        if self.artifact.artifact_type != "run":
            raise ValueError("solve receipt requires a run ArtifactRef")
        parts = self.artifact.uri.split("/")
        if len(parts) != 3 or parts[0] != "runs" or parts[1] != self.case_id:
            raise ValueError("solve receipt case ID does not match its artifact path")
        payload = self.model_dump(mode="json", exclude={"evidence_sha256"})
        if self.evidence_sha256 != canonical_sha256(payload):
            raise ValueError("evidence_sha256 does not match the solve receipt")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        payload: dict[str, object] = {"schema_version": 1, **values}
        return cls.model_validate({**payload, "evidence_sha256": canonical_sha256(payload)})


def remote_request_reference(content: bytes) -> ArtifactRef:
    """Return the content-addressed volume key for one bounded sweep request."""

    parse_remote_model(content, RemoteSweepRequest)
    digest = sha256_bytes(content)
    return ArtifactRef(
        artifact_type="sweep_request",
        artifact_id=digest[:20],
        sha256=digest,
        size_bytes=len(content),
        uri=f"requests/sweeps/{digest}.json",
    )


def publish_remote_request(root: Path, content: bytes) -> ArtifactRef:
    """Atomically publish or verify one immutable bounded request document."""

    reference = remote_request_reference(content)
    parent = ensure_real_directory(root.resolve(), "requests", "sweeps")
    target = parent / f"{reference.sha256}.json"
    if target.exists() or target.is_symlink():
        existing = safe_read_bytes(
            root,
            reference.uri,
            max_bytes=MAX_REMOTE_INPUT_BYTES,
            expected_sha256=reference.sha256,
        )
        if existing != content:
            raise ArtifactIntegrityError("content-addressed remote request diverged")
        return reference

    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".request-", suffix=".tmp", dir=parent)
        temporary = Path(name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_file(temporary)
        os.replace(temporary, target)
        temporary = None
        fsync_directory(parent)
    except OSError as error:
        raise ArtifactIntegrityError("atomic remote request publication failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    stored = safe_read_bytes(
        root,
        reference.uri,
        max_bytes=MAX_REMOTE_INPUT_BYTES,
        expected_sha256=reference.sha256,
    )
    if stored != content:
        raise ArtifactIntegrityError("published remote request bytes diverged")
    return reference


def load_remote_request(root: Path, reference: ArtifactRef) -> RemoteSweepRequest:
    """Reload and fully verify a sweep request through its typed reference."""

    if reference.artifact_type != "sweep_request":
        raise ArtifactIntegrityError("remote sweep requires a sweep_request ArtifactRef")
    expected = f"requests/sweeps/{reference.sha256}.json"
    if reference.uri != expected or reference.artifact_id != reference.sha256[:20]:
        raise ArtifactIntegrityError("remote sweep request reference is incoherent")
    content = safe_read_bytes(
        root,
        reference.uri,
        max_bytes=MAX_REMOTE_INPUT_BYTES,
        expected_sha256=reference.sha256,
    )
    if len(content) != reference.size_bytes:
        raise ArtifactIntegrityError("remote sweep request size does not match its reference")
    return parse_remote_model(content, RemoteSweepRequest)


def references_digest(references: Sequence[ArtifactRef]) -> str:
    """Stable rollup for checking digest preservation across resume."""

    return canonical_sha256(
        [reference.model_dump(mode="json") for reference in sorted(references, key=lambda x: x.uri)]
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "MAX_REMOTE_INPUT_BYTES",
    "REMOTE_ARTIFACT_ROOT",
    "SMOKE_DESIGN_KIND",
    "SMOKE_SAMPLE_COUNT",
    "CorrelationId",
    "DeviceClass",
    "RemoteSolveRequest",
    "RemoteSweepRequest",
    "SmokeDesignPoint",
    "SolveSummary",
    "SweepSummary",
    "encode_remote_model",
    "load_remote_request",
    "parse_remote_model",
    "publish_remote_request",
    "references_digest",
    "remote_request_reference",
    "smoke_design",
    "utc_now",
    "validate_correlation_id",
]
