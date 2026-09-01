"""Generated schema registry for the complete datagen contract family."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from pydantic import BaseModel

from soufflerie.datagen.run_artifact import QuantizationStatistic, RunMetadata
from soufflerie.datagen.sweep_state import CaseState

DATAGEN_SCHEMA_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "case-state": CaseState,
        "quantization-statistic": QuantizationStatistic,
        "run-metadata": RunMetadata,
    }
)


def datagen_schema_documents() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, model in DATAGEN_SCHEMA_MODELS.items():
        document = cast(dict[str, object], model.model_json_schema())
        document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        document["$id"] = f"https://github.com/AbdelStark/soufflerie/schemas/v1/{name}.json"
        result[name] = document
    return result


def rendered_datagen_schema_documents() -> dict[str, str]:
    return {
        f"{name}.json": json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        for name, document in datagen_schema_documents().items()
    }


__all__ = [
    "DATAGEN_SCHEMA_MODELS",
    "datagen_schema_documents",
    "rendered_datagen_schema_documents",
]
