from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from infra.policy import (
    APP_NAME,
    BUILD_LOCK_SHA256_ENV,
    BUILD_SOURCE_DIRTY_ENV,
    BUILD_SOURCE_REVISION_ENV,
    FULL_RUNTIME_EXTRAS,
    PRIMARY_GPU,
    PYTHON_BASE_IMAGE,
    REMOTE_RETRIES,
    RUNTIME_SECRET_NAME,
    SMOKE_MAX_CONTAINERS,
    SOLVE_TIMEOUT_SECONDS,
    UV_VERSION,
    VOLUME_MOUNT,
    VOLUME_NAME,
    RemoteRuntimeSettings,
    checkout_state,
    checkout_state_for_runtime,
)
from infra.runtime_manifest import (
    KernelSmokeEvidence,
    KernelSmokeResult,
    RuntimeBuildManifest,
)

ROOT = Path(__file__).parents[2]
INFRA = ROOT / "infra"


class _FakeImage:
    calls: ClassVar[list[tuple[str, tuple[object, ...], dict[str, object]]]] = []

    @classmethod
    def from_registry(cls, *args: object, **kwargs: object) -> _FakeImage:
        cls.calls.append(("from_registry", args, kwargs))
        return cls()

    def __getattr__(self, name: str) -> Any:
        def record(*args: object, **kwargs: object) -> _FakeImage:
            self.calls.append((name, args, kwargs))
            return self

        return record


class _FakeVolume:
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []

    @classmethod
    def from_name(cls, name: str, **kwargs: object) -> _FakeVolume:
        cls.calls.append((name, kwargs))
        return cls()


class _FakeRemoteFunction:
    def __init__(self, function: object) -> None:
        self.function = function

    def remote(self, *_args: object, **_kwargs: object) -> str:
        raise AssertionError("stubbed policy imports must never call Modal")


class _FakeApp:
    instances: ClassVar[list[_FakeApp]] = []

    def __init__(self, name: str) -> None:
        self.name = name
        self.function_policies: list[dict[str, object]] = []
        self.instances.append(self)

    def function(self, **kwargs: object) -> Any:
        self.function_policies.append(kwargs)

        def decorate(function: object) -> _FakeRemoteFunction:
            return _FakeRemoteFunction(function)

        return decorate

    def local_entrypoint(self) -> Any:
        def decorate(function: object) -> object:
            return function

        return decorate


def _fake_modal() -> ModuleType:
    module = ModuleType("modal")
    module.App = _FakeApp  # type: ignore[attr-defined]
    module.Image = _FakeImage  # type: ignore[attr-defined]
    module.Volume = _FakeVolume  # type: ignore[attr-defined]
    return module


def _build_manifest() -> RuntimeBuildManifest:
    return RuntimeBuildManifest.create(
        lock_sha256="1" * 64,
        source_revision="2" * 40,
        source_dirty=False,
        packages={"soufflerie": "0.1.0", "warp-lang": "1.17.0"},
    )


def _smoke_result(*, seconds: float = 0.25) -> KernelSmokeResult:
    return KernelSmokeResult.create(
        build=_build_manifest().model_dump(mode="json"),
        requested_device_class="L40S",
        resolved_device="cuda:0",
        device_name="NVIDIA L40S",
        cuda_arch=89,
        volume_name=VOLUME_NAME,
        volume_mount=VOLUME_MOUNT,
        kernel_steps=2,
        state_sha256="3" * 64,
        initial_mass=64.0,
        final_mass=64.0,
        wall_seconds=seconds,
        gpu_seconds=seconds,
        passed=True,
    )


def test_remote_policy_is_fixed_strict_and_cleanly_identified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOUFFLERIE_REMOTE_GPU", raising=False)
    assert RemoteRuntimeSettings().remote_gpu == PRIMARY_GPU
    assert RemoteRuntimeSettings(remote_gpu="A10G").remote_gpu == "A10G"
    with pytest.raises(ValidationError, match="remote_gpu"):
        RemoteRuntimeSettings.model_validate({"remote_gpu": "T4"})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RemoteRuntimeSettings.model_validate({"token": "must-not-be-read"})

    checkout = checkout_state(ROOT)
    assert len(checkout.source_revision) == 40
    assert len(checkout.lock_sha256) == 64
    assert RUNTIME_SECRET_NAME == "soufflerie-runtime"
    assert REMOTE_RETRIES == 0


def test_worker_reimport_uses_only_complete_baked_build_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BUILD_LOCK_SHA256_ENV, "1" * 64)
    monkeypatch.setenv(BUILD_SOURCE_REVISION_ENV, "2" * 40)
    monkeypatch.setenv(BUILD_SOURCE_DIRTY_ENV, "false")
    monkeypatch.setattr(
        "infra.policy._git_output",
        lambda *_args, **_kwargs: pytest.fail("worker reimport must not execute Git"),
    )
    first = checkout_state_for_runtime(Path("/checkout-not-present"))
    second = checkout_state_for_runtime(Path("/another-missing-checkout"))
    assert first == second
    assert first.source_dirty is False

    monkeypatch.delenv(BUILD_LOCK_SHA256_ENV)
    with pytest.raises(RuntimeError, match="incomplete baked build identities"):
        checkout_state_for_runtime(Path("/checkout-not-present"))


def test_build_and_smoke_records_are_digest_bound_and_coherent() -> None:
    build = _build_manifest()
    loaded = RuntimeBuildManifest.model_validate_json(build.model_dump_json())
    assert loaded == build

    first = _smoke_result()
    second = _smoke_result(seconds=0.5)
    evidence = KernelSmokeEvidence.create(first, second)
    assert evidence.state_digests_equal
    assert evidence.build_digests_equal
    assert evidence.total_gpu_seconds == 0.75

    tampered = first.model_dump(mode="python")
    tampered["final_mass"] = 63.0
    with pytest.raises(ValidationError, match="conserve mass"):
        KernelSmokeResult.model_validate(tampered)

    tampered = first.model_dump(mode="python")
    tampered["device_name"] = "different"
    with pytest.raises(ValidationError, match="artifact digest"):
        KernelSmokeResult.model_validate(tampered)


def test_stubbed_entrypoint_imports_reuse_the_single_shared_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeImage.calls.clear()
    _FakeVolume.calls.clear()
    _FakeApp.instances.clear()
    monkeypatch.setitem(sys.modules, "modal", _fake_modal())
    monkeypatch.delenv("SOUFFLERIE_REMOTE_GPU", raising=False)
    for name in ("infra.solve", "infra.app"):
        sys.modules.pop(name, None)

    app_module = importlib.import_module("infra.app")
    solve_module = importlib.import_module("infra.solve")

    assert len(_FakeApp.instances) == 1
    fake_app = _FakeApp.instances[0]
    assert fake_app.name == APP_NAME
    assert app_module.app is fake_app
    assert solve_module.app is fake_app
    assert list(inspect.signature(solve_module.kernel_smoke_remote.function).parameters) == [
        "requested_device_class"
    ]
    assert _FakeVolume.calls == [(VOLUME_NAME, {"create_if_missing": True})]

    registry_calls = [call for call in _FakeImage.calls if call[0] == "from_registry"]
    assert registry_calls == [("from_registry", (PYTHON_BASE_IMAGE,), {})]
    uv_calls = [call for call in _FakeImage.calls if call[0] == "uv_sync"]
    assert len(uv_calls) == 1
    assert uv_calls[0][2]["extras"] == list(FULL_RUNTIME_EXTRAS)
    assert uv_calls[0][2]["groups"] == []
    assert uv_calls[0][2]["frozen"] is True
    assert uv_calls[0][2]["uv_version"] == UV_VERSION
    copy_calls = [call for call in _FakeImage.calls if call[0].startswith("add_local_")]
    assert copy_calls
    assert all(call[2]["copy"] is True for call in copy_calls)

    assert fake_app.function_policies == [
        {
            "image": app_module.image,
            "gpu": PRIMARY_GPU,
            "volumes": {VOLUME_MOUNT: app_module.volume},
            "timeout": SOLVE_TIMEOUT_SECONDS,
            "max_containers": SMOKE_MAX_CONTAINERS,
            "retries": REMOTE_RETRIES,
        }
    ]


def test_only_app_module_constructs_provider_resources_and_no_secret_value_is_committed() -> None:
    constructors: dict[str, list[str]] = {"App": [], "Image": [], "Volume": []}
    for path in INFRA.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "App":
                constructors["App"].append(path.name)
            elif node.func.attr == "from_registry":
                constructors["Image"].append(path.name)
            elif node.func.attr == "from_name":
                constructors["Volume"].append(path.name)
    assert constructors == {"App": ["app.py"], "Image": ["app.py"], "Volume": ["app.py"]}

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "TOKEN=" not in env_example
    assert "SECRET=" not in env_example
    assert "PASSWORD=" not in env_example
