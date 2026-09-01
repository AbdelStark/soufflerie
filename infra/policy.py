"""Provider-neutral constants and strict settings for the Modal runtime."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_NAME: Final = "soufflerie"
VOLUME_NAME: Final = "soufflerie-data"
VOLUME_MOUNT: Final = "/data"
RUNTIME_SECRET_NAME: Final = "soufflerie-runtime"

PRIMARY_GPU: Final = "L40S"
FALLBACK_GPUS: Final = ("A10G",)
REMOTE_GPU_CHOICES: Final = (PRIMARY_GPU, *FALLBACK_GPUS)

PYTHON_BASE_IMAGE: Final = (
    "python:3.11.14-slim-bookworm@"
    "sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"
)
PYTHON_BASE_IMAGE_SHA256: Final = PYTHON_BASE_IMAGE.rpartition("@")[2]
PYTHON_VERSION: Final = "3.11.14"
UV_VERSION: Final = "0.12.8"
FULL_RUNTIME_EXTRAS: Final = ("solver", "ml", "remote", "serve", "viz")

SOLVE_TIMEOUT_SECONDS: Final = 180
SWEEP_TIMEOUT_SECONDS: Final = 2 * 60 * 60
TRAIN_TIMEOUT_SECONDS: Final = 75 * 60
VALIDATE_TIMEOUT_SECONDS: Final = 30 * 60
SOLVE_MAX_CONTAINERS: Final = 100
SWEEP_MAX_CONTAINERS: Final = 1
TRAIN_MAX_CONTAINERS: Final = 3
VALIDATE_MAX_CONTAINERS: Final = 1
SERVICE_MAX_CONTAINERS: Final = 1
SMOKE_MAX_CONTAINERS: Final = 1
REMOTE_RETRIES: Final = 0

BUILD_LOCK_SHA256_ENV: Final = "SOUFFLERIE_BUILD_LOCK_SHA256"
BUILD_SOURCE_DIRTY_ENV: Final = "SOUFFLERIE_BUILD_SOURCE_DIRTY"
BUILD_SOURCE_REVISION_ENV: Final = "SOUFFLERIE_BUILD_SOURCE_REVISION"
BUILD_IDENTITY_ENV_NAMES: Final = (
    BUILD_LOCK_SHA256_ENV,
    BUILD_SOURCE_DIRTY_ENV,
    BUILD_SOURCE_REVISION_ENV,
)


class RemoteRuntimeSettings(BaseSettings):
    """Explicit operator choices; credentials remain in the Modal profile."""

    model_config = SettingsConfigDict(
        env_prefix="SOUFFLERIE_",
        extra="forbid",
        frozen=True,
        strict=True,
    )

    remote_gpu: Literal["L40S", "A10G"] = PRIMARY_GPU


@dataclass(frozen=True, slots=True)
class CheckoutState:
    source_revision: str
    source_dirty: bool
    lock_sha256: str


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def checkout_state(repository_root: Path) -> CheckoutState:
    """Read the exact source and lock identities used to construct an image."""

    revision = _git_output(repository_root, "rev-parse", "HEAD")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("source revision must be a full lowercase Git SHA")
    status = _git_output(repository_root, "status", "--porcelain", "--untracked-files=all")
    lock_path = repository_root / "uv.lock"
    if not lock_path.is_file():
        raise RuntimeError("uv.lock is required for the remote runtime")
    return CheckoutState(
        source_revision=revision,
        source_dirty=bool(status),
        lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    )


def checkout_state_for_runtime(repository_root: Path) -> CheckoutState:
    """Use baked worker identities, falling back to the local Git checkout."""

    present = {name: os.environ.get(name) for name in BUILD_IDENTITY_ENV_NAMES}
    if all(value is None for value in present.values()):
        return checkout_state(repository_root)
    missing = [name for name, value in present.items() if value is None]
    if missing:
        raise RuntimeError(f"incomplete baked build identities: {', '.join(missing)}")

    revision = present[BUILD_SOURCE_REVISION_ENV]
    lock_sha256 = present[BUILD_LOCK_SHA256_ENV]
    dirty = present[BUILD_SOURCE_DIRTY_ENV]
    if (
        revision is None
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError("baked source revision must be a full lowercase Git SHA")
    if (
        lock_sha256 is None
        or len(lock_sha256) != 64
        or any(character not in "0123456789abcdef" for character in lock_sha256)
    ):
        raise RuntimeError("baked lock identity must be a lowercase SHA-256 digest")
    if dirty not in {"true", "false"}:
        raise RuntimeError("baked source dirty identity must be true or false")
    return CheckoutState(
        source_revision=revision,
        source_dirty=dirty == "true",
        lock_sha256=lock_sha256,
    )


__all__ = [
    "APP_NAME",
    "BUILD_IDENTITY_ENV_NAMES",
    "BUILD_LOCK_SHA256_ENV",
    "BUILD_SOURCE_DIRTY_ENV",
    "BUILD_SOURCE_REVISION_ENV",
    "FALLBACK_GPUS",
    "FULL_RUNTIME_EXTRAS",
    "PRIMARY_GPU",
    "PYTHON_BASE_IMAGE",
    "PYTHON_BASE_IMAGE_SHA256",
    "PYTHON_VERSION",
    "REMOTE_GPU_CHOICES",
    "REMOTE_RETRIES",
    "RUNTIME_SECRET_NAME",
    "SERVICE_MAX_CONTAINERS",
    "SMOKE_MAX_CONTAINERS",
    "SOLVE_MAX_CONTAINERS",
    "SOLVE_TIMEOUT_SECONDS",
    "SWEEP_MAX_CONTAINERS",
    "SWEEP_TIMEOUT_SECONDS",
    "TRAIN_MAX_CONTAINERS",
    "TRAIN_TIMEOUT_SECONDS",
    "UV_VERSION",
    "VALIDATE_MAX_CONTAINERS",
    "VALIDATE_TIMEOUT_SECONDS",
    "VOLUME_MOUNT",
    "VOLUME_NAME",
    "CheckoutState",
    "RemoteRuntimeSettings",
    "checkout_state",
    "checkout_state_for_runtime",
]
