"""Manifest-backed training data and deterministic baseline predictors."""

from soufflerie.training.baselines import (
    BaselineKind,
    BaselineMetadata,
    MeanFieldBaseline,
    NearestDesignBaseline,
    fit_baselines,
    fit_mean_field_baseline,
    fit_nearest_design_baseline,
)
from soufflerie.training.data import (
    MAX_TRAINING_BATCH_SIZE,
    ManifestBatch,
    ManifestDataset,
    ManifestRowBatch,
    open_manifest_dataset,
)
from soufflerie.training.schema_registry import rendered_training_schema_documents

__all__ = [
    "MAX_TRAINING_BATCH_SIZE",
    "BaselineKind",
    "BaselineMetadata",
    "ManifestBatch",
    "ManifestDataset",
    "ManifestRowBatch",
    "MeanFieldBaseline",
    "NearestDesignBaseline",
    "fit_baselines",
    "fit_mean_field_baseline",
    "fit_nearest_design_baseline",
    "open_manifest_dataset",
    "rendered_training_schema_documents",
]
