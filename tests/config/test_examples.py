from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel

from soufflerie.config import (
    ServiceConfig,
    SweepConfig,
    TrainingConfig,
    ValidationConfig,
    config_digest,
    load_config,
    rendered_config_schema_documents,
)
from soufflerie.observability import rendered_observability_schema_documents
from soufflerie.schemas import CaseConfig, rendered_schema_documents

PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLES: Mapping[Path, type[BaseModel]] = {
    Path("configs/cases/cylinder-re100-v1.yaml"): CaseConfig,
    Path("configs/service/demo-v1.yaml"): ServiceConfig,
    Path("configs/sweeps/mvp-v1.yaml"): SweepConfig,
    Path("configs/training/fno-v1.yaml"): TrainingConfig,
    Path("configs/validation/release-v1.yaml"): ValidationConfig,
}


def test_all_checked_in_examples_are_strict_and_canonical() -> None:
    for relative_path, model in EXAMPLES.items():
        config = load_config(PROJECT_ROOT / relative_path, model)
        assert len(config_digest(config)) == 64


def test_config_identity_does_not_read_environment_or_storage_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_HOST", "0.0.0.0")
    source = PROJECT_ROOT / "configs/service/demo-v1.yaml"
    copy = tmp_path / "renamed.yaml"
    copy.write_bytes(source.read_bytes())
    assert (
        load_config(source, ServiceConfig).config_digest
        == load_config(copy, ServiceConfig).config_digest
    )


def test_all_generated_schema_documents_are_checked_in() -> None:
    expected = {
        **rendered_schema_documents(),
        **rendered_config_schema_documents(),
        **rendered_observability_schema_documents(),
    }
    root = PROJECT_ROOT / "schemas" / "v1"
    assert {path.name for path in root.glob("*.json")} == set(expected)
    for name, content in expected.items():
        assert (root / name).read_text(encoding="utf-8") == content
