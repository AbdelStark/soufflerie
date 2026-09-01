"""Strict build and kernel-smoke provenance records."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from infra.policy import (
    BUILD_IDENTITY_ENV_NAMES,
    BUILD_LOCK_SHA256_ENV,
    BUILD_SOURCE_DIRTY_ENV,
    BUILD_SOURCE_REVISION_ENV,
    PYTHON_BASE_IMAGE,
    PYTHON_VERSION,
    UV_VERSION,
)
from soufflerie.schemas import VersionedModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"
BUILD_MANIFEST_PATH = Path("/opt/soufflerie/runtime-build.json")


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RuntimeBuildManifest(VersionedModel):
    """Immutable identity of one remotely built runtime image."""

    base_image: Literal[
        "python:3.11.14-slim-bookworm@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"
    ]
    python_version: Literal["3.11.14"]
    uv_version: Literal["0.12.8"]
    lock_sha256: str = Field(pattern=SHA256_PATTERN)
    source_revision: str = Field(pattern=REVISION_PATTERN)
    source_dirty: bool
    packages: dict[str, str] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _digest_is_coherent(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if not isinstance(payload, dict) or self.manifest_sha256 != canonical_sha256(payload):
            raise ValueError("runtime build manifest digest does not match its content")
        return self

    @classmethod
    def create(
        cls,
        *,
        lock_sha256: str,
        source_revision: str,
        source_dirty: bool,
        packages: dict[str, str],
    ) -> Self:
        payload: dict[str, object] = {
            "schema_version": 1,
            "base_image": PYTHON_BASE_IMAGE,
            "python_version": PYTHON_VERSION,
            "uv_version": UV_VERSION,
            "lock_sha256": lock_sha256,
            "source_revision": source_revision,
            "source_dirty": source_dirty,
            "packages": packages,
        }
        return cls.model_validate({**payload, "manifest_sha256": canonical_sha256(payload)})


class KernelSmokeResult(VersionedModel):
    """One real GPU kernel execution with content-linked provenance."""

    build: RuntimeBuildManifest
    requested_device_class: Literal["L40S", "A10G"]
    resolved_device: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    cuda_arch: int = Field(gt=0)
    volume_name: Literal["soufflerie-data"]
    volume_mount: Literal["/data"]
    kernel_steps: Literal[2]
    state_sha256: str = Field(pattern=SHA256_PATTERN)
    initial_mass: float = Field(gt=0.0, allow_inf_nan=False)
    final_mass: float = Field(gt=0.0, allow_inf_nan=False)
    wall_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    passed: Literal[True]
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _result_is_coherent(self) -> Self:
        if not math.isclose(self.initial_mass, self.final_mass, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("kernel smoke must conserve mass")
        payload = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if not isinstance(payload, dict) or self.artifact_sha256 != canonical_sha256(payload):
            raise ValueError("kernel smoke artifact digest does not match its content")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        payload: dict[str, object] = {"schema_version": 1, **values}
        return cls.model_validate({**payload, "artifact_sha256": canonical_sha256(payload)})


class KernelSmokeEvidence(VersionedModel):
    """Two-run evidence that the same locked GPU smoke is deterministic."""

    repetitions: Literal[2]
    runs: tuple[KernelSmokeResult, KernelSmokeResult]
    state_digests_equal: Literal[True]
    build_digests_equal: Literal[True]
    total_gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("runs", mode="before")
    @classmethod
    def _json_array_to_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _evidence_is_coherent(self) -> Self:
        first, second = self.runs
        if first.state_sha256 != second.state_sha256 or not self.state_digests_equal:
            raise ValueError("kernel smoke state digests must match")
        if (
            first.build.manifest_sha256 != second.build.manifest_sha256
            or not self.build_digests_equal
        ):
            raise ValueError("kernel smoke build digests must match")
        expected_seconds = first.gpu_seconds + second.gpu_seconds
        if not math.isclose(self.total_gpu_seconds, expected_seconds, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("total GPU seconds do not match the two runs")
        payload = self.model_dump(mode="json", exclude={"evidence_sha256"})
        if not isinstance(payload, dict) or self.evidence_sha256 != canonical_sha256(payload):
            raise ValueError("kernel smoke evidence digest does not match its content")
        return self

    @classmethod
    def create(cls, first: KernelSmokeResult, second: KernelSmokeResult) -> Self:
        payload: dict[str, object] = {
            "schema_version": 1,
            "repetitions": 2,
            "runs": [first.model_dump(mode="json"), second.model_dump(mode="json")],
            "state_digests_equal": first.state_sha256 == second.state_sha256,
            "build_digests_equal": first.build.manifest_sha256 == second.build.manifest_sha256,
            "total_gpu_seconds": first.gpu_seconds + second.gpu_seconds,
        }
        return cls.model_validate({**payload, "evidence_sha256": canonical_sha256(payload)})


def installed_packages() -> dict[str, str]:
    """Return every installed distribution with normalized, deterministic keys."""

    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata["Name"]
        if not raw_name:
            continue
        name = raw_name.casefold().replace("_", "-")
        version = distribution.version
        previous = packages.setdefault(name, version)
        if previous != version:
            raise RuntimeError(f"conflicting installed versions for {name}")
    return dict(sorted(packages.items()))


def build_manifest_from_environment() -> RuntimeBuildManifest:
    """Collect only the allowlisted build environment fields."""

    missing = [name for name in BUILD_IDENTITY_ENV_NAMES if name not in os.environ]
    if missing:
        raise RuntimeError(f"missing runtime build identities: {', '.join(missing)}")
    if platform.python_version() != PYTHON_VERSION:
        raise RuntimeError(
            f"runtime Python {platform.python_version()} does not match {PYTHON_VERSION}"
        )
    dirty = os.environ[BUILD_SOURCE_DIRTY_ENV]
    if dirty not in {"true", "false"}:
        raise RuntimeError("SOUFFLERIE_BUILD_SOURCE_DIRTY must be true or false")
    return RuntimeBuildManifest.create(
        lock_sha256=os.environ[BUILD_LOCK_SHA256_ENV],
        source_revision=os.environ[BUILD_SOURCE_REVISION_ENV],
        source_dirty=dirty == "true",
        packages=installed_packages(),
    )


def write_build_manifest(path: Path = BUILD_MANIFEST_PATH) -> RuntimeBuildManifest:
    manifest = build_manifest_from_environment()
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_build_manifest(path: Path = BUILD_MANIFEST_PATH) -> RuntimeBuildManifest:
    return RuntimeBuildManifest.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest = write_build_manifest()
    print(f"runtime_build_manifest={manifest.manifest_sha256}")


if __name__ == "__main__":
    main()


__all__ = [
    "BUILD_MANIFEST_PATH",
    "KernelSmokeEvidence",
    "KernelSmokeResult",
    "RuntimeBuildManifest",
    "build_manifest_from_environment",
    "canonical_sha256",
    "installed_packages",
    "load_build_manifest",
    "write_build_manifest",
]
