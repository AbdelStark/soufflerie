from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from scripts.generate_bundled_model import check_resources, generate_resource_documents
from soufflerie.errors import ArtifactIntegrityError, DependencyUnavailableError
from soufflerie.surrogate import bundled as bundled_module
from soufflerie.surrogate.bundle import LocalModelBundleStore
from soufflerie.surrogate.bundled import (
    BUNDLED_COMPRESSED_WEIGHTS_NAME,
    BUNDLED_RESOURCE_PREFIX,
    BundledCpuSmokeResult,
    BundledModelResource,
    _materialize_bundled_cpu_model_from,
    bundled_model_resource,
    materialize_bundled_cpu_model,
)

PROJECT_ROOT = Path(__file__).parents[2]
RESOURCE_ROOT = PROJECT_ROOT / "src" / "soufflerie" / "resources" / "model"
pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class InstalledWheel:
    wheel: Path
    root: Path
    python: Path


class _FakeTensor:
    def __init__(self, value: np.ndarray[Any, Any]) -> None:
        self._value = value

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def contiguous(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self._value

    def min(self) -> _FakeTensor:
        return _FakeTensor(np.asarray(self._value.min(), dtype=np.float32))

    def max(self) -> _FakeTensor:
        return _FakeTensor(np.asarray(self._value.max(), dtype=np.float32))

    def __getitem__(self, index: int) -> _FakeTensor:
        return _FakeTensor(np.asarray(self._value[index], dtype=np.float32))

    def item(self) -> float:
        return float(self._value.item())


def _install_fake_cpu_runtime(monkeypatch: pytest.MonkeyPatch) -> BundledModelResource:
    descriptor = bundled_model_resource()
    metadata = SimpleNamespace(
        model_id=descriptor.model.artifact_id,
        model_sha256=descriptor.model.sha256,
        dataset_id=descriptor.fixture_parent_sha256[:20],
        dataset_sha256=descriptor.fixture_parent_sha256,
    )
    bundle = SimpleNamespace(metadata=metadata)
    fields = np.empty((1, 3, 320, 256), dtype=np.float32)
    fields[:, 0] = 0.25
    fields[:, 1] = -0.5
    fields[:, 2] = 0.75
    prediction = SimpleNamespace(
        fields_normalized=_FakeTensor(fields),
        cd_head=_FakeTensor(np.asarray([0.125], dtype=np.float32)),
    )
    predictor = SimpleNamespace(predict=lambda _batch: prediction)
    torch = SimpleNamespace(
        float32=np.float32,
        bool=np.bool_,
        zeros=lambda shape, dtype: _FakeTensor(np.zeros(shape, dtype=dtype)),
        ones=lambda shape, dtype: _FakeTensor(np.ones(shape, dtype=dtype)),
    )
    monkeypatch.setattr(
        bundled_module,
        "materialize_bundled_cpu_model",
        lambda _root: descriptor.model,
    )
    monkeypatch.setattr(
        bundled_module,
        "LocalModelBundleStore",
        lambda _root: SimpleNamespace(open=lambda _reference: bundle),
    )
    monkeypatch.setattr(
        bundled_module,
        "instantiate_bundle_predictor",
        lambda _bundle, *, device: predictor,
    )
    monkeypatch.setattr(bundled_module, "PredictionBatch", lambda **_values: object())
    monkeypatch.setattr(bundled_module, "_import_torch", lambda: torch)
    monkeypatch.setattr(bundled_module, "bundled_model_resource", lambda: descriptor)
    return descriptor


def _run_checked(arguments: list[str], *, cwd: Path | None = None) -> None:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, (
        f"command failed: {arguments!r}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _install_wheel_environment(
    *,
    root: Path,
    wheel: Path,
    extra: str | None,
) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("uv is required for the installed-wheel model contract")
    environment = root / (f"venv-{extra}" if extra is not None else "venv-base")
    lock = root / f"pylock.{extra or 'base'}.toml"
    _run_checked([uv, "venv", str(environment)])
    export = [
        uv,
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--format",
        "pylock.toml",
        "--output-file",
        str(lock),
    ]
    if extra is not None:
        export.extend(("--extra", extra))
    _run_checked(export, cwd=PROJECT_ROOT)
    _run_checked(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(environment),
            "--requirements",
            str(lock),
        ]
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run_checked([uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)])
    return python


@pytest.fixture(scope="module")
def installed_base_wheel(tmp_path_factory: pytest.TempPathFactory) -> InstalledWheel:
    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("uv is required for the installed-wheel model contract")
    root = tmp_path_factory.mktemp("bundled-model-wheel")
    distributions = root / "dist"
    _run_checked([uv, "build", "--wheel", "--out-dir", str(distributions)], cwd=PROJECT_ROOT)
    wheel = next(distributions.glob("soufflerie-*.whl"))
    python = _install_wheel_environment(root=root, wheel=wheel, extra=None)
    return InstalledWheel(wheel=wheel, root=root, python=python)


def test_bundled_resource_is_byte_reproducible_bounded_and_materializable(
    tmp_path: Path,
) -> None:
    documents = generate_resource_documents()
    assert check_resources(RESOURCE_ROOT, documents) == ()
    descriptor = bundled_model_resource()
    assert descriptor.compressed_weights_bytes < 256 * 1024
    assert descriptor.uncompressed_weights_bytes == 151_126_144
    assert descriptor.representative_of_trained_quality is False

    reference = materialize_bundled_cpu_model(tmp_path)
    loaded = LocalModelBundleStore(tmp_path).open(reference)
    assert reference == descriptor.model
    assert loaded.metadata.dataset_sha256 == descriptor.fixture_parent_sha256
    assert loaded.metadata.model_card.limitations[0].startswith("Synthetic zero-weight fixture")
    assert "torch" not in sys.modules
    assert "physicsnemo" not in sys.modules


@pytest.mark.parametrize("mutation", ("missing", "extra", "compressed", "metadata"))
def test_bundled_resource_tampering_fails_closed(tmp_path: Path, mutation: str) -> None:
    resource = tmp_path / "resource"
    shutil.copytree(RESOURCE_ROOT, resource)
    if mutation == "missing":
        (resource / "model-card.md").unlink()
    elif mutation == "extra":
        (resource / "extra.bin").write_bytes(b"extra")
    elif mutation == "compressed":
        path = resource / BUNDLED_COMPRESSED_WEIGHTS_NAME
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
    else:
        path = resource / "bundle.json"
        content = bytearray(path.read_bytes())
        content[-2] ^= 1
        path.write_bytes(content)
    with pytest.raises(ArtifactIntegrityError):
        _materialize_bundled_cpu_model_from(resource, tmp_path / "store")


def test_bundled_resource_and_smoke_result_models_reject_rebinding() -> None:
    descriptor = bundled_model_resource()
    with pytest.raises(ValidationError, match="fixture parent digest"):
        BundledModelResource.model_validate(
            {**descriptor.model_dump(mode="python"), "fixture_parent_sha256": "f" * 64}
        )
    valid_result: dict[str, Any] = {
        "model_id": descriptor.model.artifact_id,
        "model_sha256": descriptor.model.sha256,
        "dataset_id": descriptor.fixture_parent_sha256[:20],
        "dataset_sha256": descriptor.fixture_parent_sha256,
        "fields_shape": (1, 3, 320, 256),
        "cd_shape": (1,),
        "fields_minimum": -0.5,
        "fields_maximum": 0.75,
        "cd": 0.125,
        "fields_sha256": descriptor.expected_fields_sha256,
        "cd_sha256": descriptor.expected_cd_sha256,
    }
    with pytest.raises(ValidationError, match="field bounds"):
        BundledCpuSmokeResult.model_validate(
            {
                **valid_result,
                "fields_shape": [1, 3, 320, 256],
                "cd_shape": [1],
                "fields_minimum": 1.0,
                "fields_maximum": -1.0,
            }
        )


def test_bundled_cpu_smoke_returns_round_trippable_exact_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _install_fake_cpu_runtime(monkeypatch)

    result = bundled_module.run_bundled_cpu_smoke(tmp_path)

    assert result.model_sha256 == descriptor.model.sha256
    assert result.fields_shape == (1, 3, 320, 256)
    assert result.cd_shape == (1,)
    assert (result.fields_minimum, result.fields_maximum, result.cd) == (-0.5, 0.75, 0.125)
    assert result.fields_sha256 == descriptor.expected_fields_sha256
    assert result.cd_sha256 == descriptor.expected_cd_sha256
    assert BundledCpuSmokeResult.model_validate_json(result.model_dump_json()) == result


def test_bundled_cpu_smoke_rejects_changed_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_cpu_runtime(monkeypatch)
    monkeypatch.setattr(bundled_module, "_array_sha256", lambda _value: "f" * 64)

    with pytest.raises(ArtifactIntegrityError, match="packaged CPU smoke output changed"):
        bundled_module.run_bundled_cpu_smoke(tmp_path)


def test_bundled_cpu_smoke_reports_missing_ml_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = bundled_model_resource()
    bundle = SimpleNamespace(metadata=SimpleNamespace())
    monkeypatch.setattr(
        bundled_module,
        "materialize_bundled_cpu_model",
        lambda _root: descriptor.model,
    )
    monkeypatch.setattr(
        bundled_module,
        "LocalModelBundleStore",
        lambda _root: SimpleNamespace(open=lambda _reference: bundle),
    )

    def missing_torch() -> None:
        raise ImportError

    monkeypatch.setattr(bundled_module, "_import_torch", missing_torch)
    with pytest.raises(DependencyUnavailableError, match="locked 'ml' extra"):
        bundled_module.run_bundled_cpu_smoke(tmp_path)


def test_wheel_ships_checksum_metadata_and_compressed_weights(
    installed_base_wheel: InstalledWheel,
) -> None:
    prefix = f"soufflerie/{BUNDLED_RESOURCE_PREFIX}/"
    expected = {
        prefix + name
        for name in {
            "resource.json",
            "bundle.json",
            "preprocessing.json",
            "architecture.json",
            "model-card.md",
            BUNDLED_COMPRESSED_WEIGHTS_NAME,
        }
    }
    with zipfile.ZipFile(installed_base_wheel.wheel) as archive:
        names = set(archive.namelist())
        assert expected <= names
        assert prefix + "model.safetensors" not in names
        packaged_descriptor = BundledModelResource.model_validate_json(
            archive.read(prefix + "resource.json")
        )
        assert packaged_descriptor == bundled_model_resource()
        assert archive.getinfo(prefix + BUNDLED_COMPRESSED_WEIGHTS_NAME).file_size < 256 * 1024
        assert "soufflerie/schemas/v1/bundled-model-resource.json" in names
        assert "soufflerie/schemas/v1/bundled-model-smoke-result.json" in names


def test_clean_base_wheel_materializes_without_optional_or_application_imports(
    installed_base_wheel: InstalledWheel,
) -> None:
    cache = installed_base_wheel.root / "base-cache"
    code = """
import json
import sys
from pathlib import Path
from soufflerie.surrogate import materialize_bundled_cpu_model
reference = materialize_bundled_cpu_model(Path(sys.argv[1]))
loaded = set(sys.modules)
forbidden = sorted(name for name in loaded if name.split('.')[0] in {
    'torch', 'physicsnemo', 'modal', 'fastapi', 'gradio', 'warp'
} or name.startswith(('soufflerie.service', 'soufflerie.demo', 'soufflerie.training')))
print(json.dumps({'reference': reference.model_dump(mode='json'), 'forbidden': forbidden}))
"""
    result = subprocess.run(
        [str(installed_base_wheel.python), "-I", "-c", code, str(cache)],
        cwd=installed_base_wheel.root,
        capture_output=True,
        check=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["forbidden"] == []
    assert payload["reference"] == bundled_model_resource().model.model_dump(mode="json")


@pytest.mark.remote
def test_fresh_ml_wheel_returns_schema_valid_finite_cpu_prediction(
    installed_base_wheel: InstalledWheel,
) -> None:
    python = _install_wheel_environment(
        root=installed_base_wheel.root,
        wheel=installed_base_wheel.wheel,
        extra="ml",
    )
    cache = installed_base_wheel.root / "ml-cache"
    code = """
import json
import sys
from pathlib import Path
from soufflerie.surrogate import run_bundled_cpu_smoke
result = run_bundled_cpu_smoke(Path(sys.argv[1]))
loaded = set(sys.modules)
forbidden = sorted(name for name in loaded if name.split('.')[0] in {
    'modal', 'fastapi', 'gradio'
} or name.startswith(('soufflerie.service', 'soufflerie.demo', 'soufflerie.training')))
print(json.dumps({'result': result.model_dump(mode='json'), 'forbidden': forbidden}))
"""
    process = subprocess.run(
        [str(python), "-I", "-c", code, str(cache)],
        cwd=installed_base_wheel.root,
        capture_output=True,
        check=True,
        text=True,
    )
    payload = json.loads(process.stdout)
    assert payload["forbidden"] == []
    result = BundledCpuSmokeResult.model_validate(payload["result"])
    descriptor = bundled_model_resource()
    assert result.model_sha256 == descriptor.model.sha256
    assert result.fields_shape == (1, 3, 320, 256)
    assert result.cd_shape == (1,)
    assert result.fields_minimum == -0.5
    assert result.fields_maximum == 0.75
    assert result.cd == 0.125
    assert result.fields_sha256 == descriptor.expected_fields_sha256
    assert result.cd_sha256 == descriptor.expected_cd_sha256
    assert result.representative_of_trained_quality is False
