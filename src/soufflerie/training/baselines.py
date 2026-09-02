"""Deterministic mean-field and nearest-design FlowPredictor baselines."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from soufflerie.datagen.manifest import ManifestRow
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ContentId, Sha256, VersionedModel, canonical_sha256, validate_array
from soufflerie.surrogate.preprocessing import (
    MODEL_SPATIAL_SHAPE,
    FlowPredictor,
    PredictionBatch,
    PredictionBatchResult,
    PreprocessingStatistics,
    preprocess_sample,
)
from soufflerie.training.data import ManifestDataset

BaselineKind = Literal["mean-field-v1", "nearest-design-v1"]
DistanceContract = Literal["not-applicable", "unit-interval-euclidean-v1"]
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]


def _array_sha256(array: npt.NDArray[np.generic]) -> str:
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array, dtype=canonical_dtype)
    descriptor = (
        f"{canonical_dtype.str}\0{','.join(str(value) for value in canonical.shape)}\0"
    ).encode("ascii")
    digest = hashlib.sha256(descriptor)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


class BaselineMetadata(VersionedModel):
    """Content identity and train-only lineage for one fitted baseline."""

    artifact_type: Literal["baseline"] = "baseline"
    baseline_kind: BaselineKind
    baseline_id: ContentId
    baseline_sha256: Sha256
    dataset_id: ContentId
    dataset_sha256: Sha256
    preprocessing_sha256: Sha256
    training_split: Literal["train"] = "train"
    training_case_count: Literal[600] = 600
    training_membership_sha256: Sha256
    state_sha256: Sha256
    output_contract: Literal["flow-predictor-normalized-v1"] = "flow-predictor-normalized-v1"
    design_distance: DistanceContract
    fit_dtype: Literal["float64"] = "float64"
    prediction_dtype: Literal["float32"] = "float32"

    def logical_identity(self) -> dict[str, object]:
        return self.model_dump(
            mode="json",
            exclude={"schema_version", "baseline_id", "baseline_sha256"},
        )

    @model_validator(mode="after")
    def _identity_is_coherent(self) -> Self:
        expected = canonical_sha256(self.logical_identity())
        if self.baseline_sha256 != expected or self.baseline_id != expected[:20]:
            raise ValueError("baseline ID and digest must bind the complete logical state")
        expected_distance = (
            "not-applicable"
            if self.baseline_kind == "mean-field-v1"
            else "unit-interval-euclidean-v1"
        )
        if self.design_distance != expected_distance:
            raise ValueError("design distance does not match the baseline kind")
        return self


def _training_membership_sha256(rows: tuple[ManifestRow, ...]) -> str:
    return canonical_sha256(
        [
            {
                "case_id": row.case_id,
                "design_id": row.design_id,
                "run_digest": row.run_digest,
            }
            for row in rows
        ]
    )


def _metadata(
    *,
    kind: BaselineKind,
    dataset: ManifestDataset,
    statistics: PreprocessingStatistics,
    rows: tuple[ManifestRow, ...],
    state_sha256: str,
) -> BaselineMetadata:
    values: dict[str, object] = {
        "artifact_type": "baseline",
        "baseline_kind": kind,
        "dataset_id": dataset.reference.artifact_id,
        "dataset_sha256": dataset.dataset_sha256,
        "preprocessing_sha256": canonical_sha256(statistics),
        "training_split": "train",
        "training_case_count": 600,
        "training_membership_sha256": _training_membership_sha256(rows),
        "state_sha256": state_sha256,
        "output_contract": "flow-predictor-normalized-v1",
        "design_distance": (
            "not-applicable" if kind == "mean-field-v1" else "unit-interval-euclidean-v1"
        ),
        "fit_dtype": "float64",
        "prediction_dtype": "float32",
    }
    digest = canonical_sha256(values)
    return BaselineMetadata(
        **values,  # type: ignore[arg-type]
        baseline_id=digest[:20],
        baseline_sha256=digest,
    )


def _owned_readonly(
    value: npt.NDArray[Any],
    *,
    dtype: npt.DTypeLike,
) -> npt.NDArray[Any]:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _new_tensor(reference: Any, value: npt.NDArray[np.generic], *, name: str) -> Any:
    try:
        return reference.new_tensor(np.array(value, copy=True, order="C"))
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"BASELINE-3 TENSOR: {name} cannot be created on the prediction device"
        ) from error


def _tensor_numpy(value: Any, *, name: str) -> npt.NDArray[np.generic]:
    try:
        result = value.detach().cpu().contiguous().numpy()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"BASELINE-3 TENSOR: {name} cannot be copied for deterministic lookup"
        ) from error
    if not isinstance(result, np.ndarray):
        raise ArtifactIntegrityError(f"BASELINE-3 TENSOR: {name} did not produce a NumPy array")
    return result


def _unit_design(row: ManifestRow) -> tuple[float, float, float, float]:
    return (
        (row.aspect_ratio - 0.5) / 0.5,
        row.rotation_deg / 30.0,
        (row.scale - 0.75) / 0.5,
        (row.reynolds - 40.0) / 260.0,
    )


def _nearest_indices(
    query_model_design: npt.NDArray[np.generic],
    training_unit_design: Float64Array,
    training_design_ids: tuple[str, ...],
) -> tuple[int, ...]:
    query = np.asarray(query_model_design, dtype=np.float64)
    if query.ndim != 2 or query.shape[1:] != (4,) or not np.isfinite(query).all():
        raise ArtifactIntegrityError(
            "BASELINE-3 DISTANCE: query design must be finite with shape [batch,4]"
        )
    if training_unit_design.shape != (len(training_design_ids), 4):
        raise ArtifactIntegrityError("BASELINE-2 FIT: training design index is incoherent")
    query_unit = (query + np.float64(1.0)) * np.float64(0.5)
    indices: list[int] = []
    for candidate in query_unit:
        delta = training_unit_design - candidate[None, :]
        distances = np.sum(delta * delta, axis=1, dtype=np.float64)
        indices.append(int(np.argmin(distances)))
    return tuple(indices)


@dataclass(frozen=True, slots=True)
class MeanFieldBaseline(FlowPredictor):
    """Per-pixel/channel training mean and scalar mean drag baseline."""

    metadata: BaselineMetadata
    fields_normalized: Float32Array
    cd: np.float32

    def __post_init__(self) -> None:
        if self.metadata.baseline_kind != "mean-field-v1":
            raise ArtifactIntegrityError("BASELINE-2 FIT: mean baseline metadata kind changed")
        validate_array(
            self.fields_normalized,
            name="mean_fields_normalized",
            dtype=np.dtype(np.float32),
            shape=(3, *MODEL_SPATIAL_SHAPE),
        )
        if self.fields_normalized.flags.writeable:
            raise ArtifactIntegrityError("BASELINE-2 FIT: mean fields must be read-only")
        if not isinstance(self.cd, np.float32) or not math.isfinite(float(self.cd)):
            raise ArtifactIntegrityError("BASELINE-2 FIT: mean drag must be finite float32")

    def predict(self, batch: PredictionBatch) -> PredictionBatchResult:
        if not isinstance(batch, PredictionBatch):
            raise TypeError("batch must be a PredictionBatch")
        batch_size = int(cast(tuple[int, ...], batch.inputs.shape)[0])
        fields = np.repeat(self.fields_normalized[None, ...], batch_size, axis=0)
        cd = np.full((batch_size,), self.cd, dtype=np.float32)
        return PredictionBatchResult(
            fields_normalized=_new_tensor(batch.inputs, fields, name="mean fields"),
            cd_head=_new_tensor(batch.design_params, cd, name="mean drag"),
        )


@dataclass(frozen=True, slots=True)
class NearestDesignBaseline(FlowPredictor):
    """Lazy verified nearest-training-row baseline with design-ID tie breaking."""

    metadata: BaselineMetadata
    dataset: ManifestDataset
    statistics: PreprocessingStatistics
    training_rows: tuple[ManifestRow, ...]
    training_unit_design: Float64Array
    training_design_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.metadata.baseline_kind != "nearest-design-v1":
            raise ArtifactIntegrityError("BASELINE-2 FIT: nearest metadata kind changed")
        if len(self.training_rows) != 600 or self.training_design_ids != tuple(
            row.design_id for row in self.training_rows
        ):
            raise ArtifactIntegrityError("BASELINE-2 FIT: nearest membership is incoherent")
        if self.training_design_ids != tuple(sorted(self.training_design_ids)):
            raise ArtifactIntegrityError("BASELINE-2 FIT: nearest tie order must be design_id")
        validate_array(
            self.training_unit_design,
            name="training_unit_design",
            dtype=np.dtype(np.float64),
            shape=(600, 4),
        )
        if self.training_unit_design.flags.writeable:
            raise ArtifactIntegrityError("BASELINE-2 FIT: training design index must be read-only")
        if np.any(self.training_unit_design < 0.0) or np.any(self.training_unit_design > 1.0):
            raise ArtifactIntegrityError("BASELINE-2 FIT: training design index must lie in [0,1]")

    def nearest_design_ids(self, batch: PredictionBatch) -> tuple[str, ...]:
        """Return deterministic selected IDs without opening field artifacts."""

        if not isinstance(batch, PredictionBatch):
            raise TypeError("batch must be a PredictionBatch")
        query = _tensor_numpy(batch.design_params, name="query design")
        indices = _nearest_indices(query, self.training_unit_design, self.training_design_ids)
        return tuple(self.training_design_ids[index] for index in indices)

    def predict(self, batch: PredictionBatch) -> PredictionBatchResult:
        if not isinstance(batch, PredictionBatch):
            raise TypeError("batch must be a PredictionBatch")
        query = _tensor_numpy(batch.design_params, name="query design")
        indices = _nearest_indices(query, self.training_unit_design, self.training_design_ids)
        selected = tuple(
            preprocess_sample(self.dataset.load_sample(self.training_rows[index]), self.statistics)
            for index in indices
        )
        fields = np.stack([sample.fields_normalized for sample in selected], axis=0)
        cd = np.asarray([sample.cd for sample in selected], dtype=np.float32)
        return PredictionBatchResult(
            fields_normalized=_new_tensor(batch.inputs, fields, name="nearest fields"),
            cd_head=_new_tensor(batch.design_params, cd, name="nearest drag"),
        )


def _validate_fit_inputs(
    dataset: ManifestDataset,
    statistics: PreprocessingStatistics,
) -> tuple[ManifestRow, ...]:
    if not isinstance(dataset, ManifestDataset):
        raise TypeError("dataset must be a ManifestDataset")
    if not isinstance(statistics, PreprocessingStatistics):
        raise TypeError("statistics must be PreprocessingStatistics")
    if statistics.dataset_id != dataset.reference.artifact_id:
        raise ArtifactIntegrityError("BASELINE-1 LINEAGE: statistics and dataset IDs differ")
    if statistics.training_case_count != 600:
        raise ArtifactIntegrityError("BASELINE-1 LINEAGE: canonical fit requires 600 train rows")
    return dataset.split_rows("train")


def fit_mean_field_baseline(
    dataset: ManifestDataset,
    statistics: PreprocessingStatistics,
) -> MeanFieldBaseline:
    """Stream train membership in design-ID order and fit fp64 mean outputs."""

    rows = _validate_fit_inputs(dataset, statistics)
    field_sum = np.zeros((3, *MODEL_SPATIAL_SHAPE), dtype=np.float64)
    cd_sum = np.float64(0.0)
    for row in rows:
        sample = preprocess_sample(dataset.load_sample(row), statistics)
        field_sum += sample.fields_normalized.astype(np.float64)
        cd_sum += np.float64(sample.cd)
    fields = cast(
        Float32Array,
        _owned_readonly(field_sum / np.float64(len(rows)), dtype=np.float32),
    )
    cd = np.float32(cd_sum / np.float64(len(rows)))
    state_sha256 = canonical_sha256(
        {
            "fields_sha256": _array_sha256(fields),
            "cd_sha256": hashlib.sha256(np.asarray([cd], dtype="<f4").tobytes()).hexdigest(),
        }
    )
    return MeanFieldBaseline(
        metadata=_metadata(
            kind="mean-field-v1",
            dataset=dataset,
            statistics=statistics,
            rows=rows,
            state_sha256=state_sha256,
        ),
        fields_normalized=fields,
        cd=cd,
    )


def fit_nearest_design_baseline(
    dataset: ManifestDataset,
    statistics: PreprocessingStatistics,
) -> NearestDesignBaseline:
    """Fit the fixed unit-interval design index from training rows only."""

    rows = _validate_fit_inputs(dataset, statistics)
    training_design = cast(
        Float64Array,
        _owned_readonly(np.asarray([_unit_design(row) for row in rows]), dtype=np.float64),
    )
    design_ids = tuple(row.design_id for row in rows)
    state_sha256 = canonical_sha256(
        {
            "design_ids": list(design_ids),
            "unit_design_sha256": _array_sha256(training_design),
        }
    )
    return NearestDesignBaseline(
        metadata=_metadata(
            kind="nearest-design-v1",
            dataset=dataset,
            statistics=statistics,
            rows=rows,
            state_sha256=state_sha256,
        ),
        dataset=dataset,
        statistics=statistics,
        training_rows=rows,
        training_unit_design=training_design,
        training_design_ids=design_ids,
    )


def fit_baselines(
    dataset: ManifestDataset,
    statistics: PreprocessingStatistics,
) -> tuple[MeanFieldBaseline, NearestDesignBaseline]:
    """Fit both RFC-0007 baselines against identical frozen train membership."""

    return (
        fit_mean_field_baseline(dataset, statistics),
        fit_nearest_design_baseline(dataset, statistics),
    )


__all__ = [
    "BaselineKind",
    "BaselineMetadata",
    "MeanFieldBaseline",
    "NearestDesignBaseline",
    "fit_baselines",
    "fit_mean_field_baseline",
    "fit_nearest_design_baseline",
]
