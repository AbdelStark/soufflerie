"""Generated schema registry for training artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from pydantic import BaseModel

from soufflerie.training.baselines import BaselineMetadata
from soufflerie.training.checkpoint import (
    FrozenTrainingSelection,
    TrainingCheckpointMetadata,
)
from soufflerie.training.metrics import TrainingEpochRecord

TRAINING_SCHEMA_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "baseline-metadata": BaselineMetadata,
        "training-checkpoint": TrainingCheckpointMetadata,
        "training-epoch": TrainingEpochRecord,
        "training-selection": FrozenTrainingSelection,
    }
)


def training_schema_documents() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, model in TRAINING_SCHEMA_MODELS.items():
        document = cast(dict[str, object], model.model_json_schema())
        document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        document["$id"] = f"https://github.com/AbdelStark/soufflerie/schemas/v1/{name}.json"
        result[name] = document
    return result


def rendered_training_schema_documents() -> dict[str, str]:
    return {
        f"{name}.json": json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        for name, document in training_schema_documents().items()
    }


__all__ = [
    "TRAINING_SCHEMA_MODELS",
    "rendered_training_schema_documents",
    "training_schema_documents",
]
