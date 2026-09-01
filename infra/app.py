"""The single Modal application, image, volume, and resource policy."""

from __future__ import annotations

from pathlib import Path

import modal

from infra.policy import (
    APP_NAME,
    BUILD_LOCK_SHA256_ENV,
    BUILD_SOURCE_DIRTY_ENV,
    BUILD_SOURCE_REVISION_ENV,
    FULL_RUNTIME_EXTRAS,
    PYTHON_BASE_IMAGE,
    REMOTE_RETRIES,
    UV_VERSION,
    VOLUME_NAME,
    RemoteRuntimeSettings,
    checkout_state_for_runtime,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/opt/soufflerie"
BUILD_MANIFEST_PATH = f"{REMOTE_ROOT}/runtime-build.json"

settings = RemoteRuntimeSettings()
checkout = checkout_state_for_runtime(REPOSITORY_ROOT)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(PYTHON_BASE_IMAGE)
    .entrypoint([])
    .uv_sync(
        uv_project_dir=str(REPOSITORY_ROOT),
        extras=list(FULL_RUNTIME_EXTRAS),
        groups=[],
        frozen=True,
        uv_version=UV_VERSION,
    )
    .add_local_file(REPOSITORY_ROOT / "pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml", copy=True)
    .add_local_file(REPOSITORY_ROOT / "uv.lock", f"{REMOTE_ROOT}/uv.lock", copy=True)
    .add_local_file(REPOSITORY_ROOT / "README.md", f"{REMOTE_ROOT}/README.md", copy=True)
    .add_local_file(REPOSITORY_ROOT / "LICENSE", f"{REMOTE_ROOT}/LICENSE", copy=True)
    .add_local_file(REPOSITORY_ROOT / "NOTICE", f"{REMOTE_ROOT}/NOTICE", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "src", f"{REMOTE_ROOT}/src", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "configs", f"{REMOTE_ROOT}/configs", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "schemas", f"{REMOTE_ROOT}/schemas", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "infra", f"{REMOTE_ROOT}/infra", copy=True)
    .workdir(REMOTE_ROOT)
    .run_commands(
        f"/.uv/uv pip install --python /.uv/.venv/bin/python --no-deps {REMOTE_ROOT}",
    )
    .env(
        {
            BUILD_LOCK_SHA256_ENV: checkout.lock_sha256,
            BUILD_SOURCE_DIRTY_ENV: str(checkout.source_dirty).lower(),
            BUILD_SOURCE_REVISION_ENV: checkout.source_revision,
        }
    )
    .run_commands(f"python -m infra.runtime_manifest {BUILD_MANIFEST_PATH}")
)

if REMOTE_RETRIES != 0:
    raise RuntimeError("the shared remote policy must not enable implicit retries")


__all__ = ["app", "checkout", "image", "settings", "volume"]
