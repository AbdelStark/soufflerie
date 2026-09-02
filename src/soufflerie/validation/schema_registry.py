"""Generated schema registry for validation artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from pydantic import BaseModel

from soufflerie.validation.gates import GateResult, ValidationReport
from soufflerie.validation.metrics import CaseMetrics, MetricSummary
from soufflerie.validation.ood import OodEvaluation
from soufflerie.validation.sensitivity import SensitivityEvaluation

VALIDATION_SCHEMA_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "case-metrics": CaseMetrics,
        "gate-result": GateResult,
        "metric-summary": MetricSummary,
        "ood-evaluation": OodEvaluation,
        "sensitivity-evaluation": SensitivityEvaluation,
        "validation-report": ValidationReport,
    }
)


def validation_schema_documents() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, model in VALIDATION_SCHEMA_MODELS.items():
        document = cast(dict[str, object], model.model_json_schema(mode="validation"))
        document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        document["$id"] = f"https://github.com/AbdelStark/soufflerie/schemas/v1/{name}.json"
        result[name] = document
    return result


def rendered_validation_schema_documents() -> dict[str, str]:
    return {
        f"{name}.json": json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        for name, document in validation_schema_documents().items()
    }


__all__ = [
    "VALIDATION_SCHEMA_MODELS",
    "rendered_validation_schema_documents",
    "validation_schema_documents",
]
