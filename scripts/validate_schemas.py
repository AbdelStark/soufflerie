"""Validate checked-in schema exports and canonical YAML examples."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from export_schemas import export
from pydantic import BaseModel

from soufflerie.config import (
    ServiceConfig,
    SweepConfig,
    TrainingConfig,
    ValidationConfig,
    load_config,
)
from soufflerie.schemas import CaseConfig

PROJECT_ROOT = Path(__file__).parents[1]
EXAMPLES: Mapping[Path, type[BaseModel]] = {
    Path("configs/cases/cylinder-re100.yaml"): CaseConfig,
    Path("configs/service/demo-v1.yaml"): ServiceConfig,
    Path("configs/sweeps/mvp-v1.yaml"): SweepConfig,
    Path("configs/training/fno-v1.yaml"): TrainingConfig,
    Path("configs/validation/release-v1.yaml"): ValidationConfig,
}


def main() -> int:
    stale = export(check=True)
    if stale:
        rendered = ", ".join(stale)
        raise SystemExit(f"schema exports are stale: {rendered}")

    for relative_path, model in EXAMPLES.items():
        config: BaseModel = load_config(PROJECT_ROOT / relative_path, model)
        if len(config.model_dump_json()) == 0:
            raise AssertionError(f"empty model JSON for {relative_path}")

    for path in (PROJECT_ROOT / "schemas" / "v1").glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "openapi.json":
            if document.get("openapi") != "3.1.0":
                raise AssertionError(f"unexpected OpenAPI version in {path.name}")
            continue
        if document.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise AssertionError(f"unexpected JSON Schema dialect in {path.name}")

    schema_count = len(list((PROJECT_ROOT / "schemas/v1").glob("*.json")))
    print(f"schema_validation=PASS schemas={schema_count} examples={len(EXAMPLES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
