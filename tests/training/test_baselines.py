from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

import soufflerie.training.baselines as baseline_module
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import canonical_sha256
from soufflerie.surrogate import FlowPredictor
from soufflerie.surrogate.preprocessing import (
    MODEL_CELL_COUNT,
    PredictionBatch,
    PredictionBatchResult,
)
from soufflerie.training import (
    BaselineMetadata,
    ManifestDataset,
    MeanFieldBaseline,
    NearestDesignBaseline,
    fit_baselines,
    fit_mean_field_baseline,
    fit_nearest_design_baseline,
)
from tests.training.helpers import (
    ArrayTensor,
    TrainingHarness,
    build_harness,
    prediction_batch,
)


@dataclass(frozen=True, slots=True)
class FittedBaselines:
    harness: TrainingHarness
    dataset: ManifestDataset
    mean: MeanFieldBaseline
    nearest: NearestDesignBaseline
    fit_opened: tuple[str, ...]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> FittedBaselines:
    harness = build_harness()
    monkeypatch = pytest.MonkeyPatch()
    dataset = harness.open(tmp_path_factory.mktemp("baseline-artifacts"), monkeypatch)
    harness.opened.clear()
    mean, nearest = fit_baselines(dataset, harness.statistics)
    result = FittedBaselines(
        harness=harness,
        dataset=dataset,
        mean=mean,
        nearest=nearest,
        fit_opened=tuple(harness.opened),
    )
    monkeypatch.undo()
    return result


def _result_arrays(
    result: PredictionBatchResult,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    fields = cast(ArrayTensor, result.fields_normalized).value
    cd = cast(ArrayTensor, result.cd_head).value
    return fields, cd


def _shared_predictor_contract(
    predictor: FlowPredictor,
    batch: PredictionBatch,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The same consumer path used for learned and both baseline predictors."""

    return _result_arrays(predictor.predict(batch))


def test_fit_uses_exactly_train_membership_and_publishes_stable_metadata(
    fitted: FittedBaselines,
) -> None:
    train = fitted.dataset.split_rows("train")
    assert fitted.fit_opened == tuple(row.run_digest for row in train)
    assert not set(fitted.fit_opened) & {
        row.run_digest for row in fitted.dataset.split_rows("validation")
    }
    assert not set(fitted.fit_opened) & {
        row.run_digest for row in fitted.dataset.split_rows("test")
    }

    for metadata in (fitted.mean.metadata, fitted.nearest.metadata):
        assert metadata.dataset_id == fitted.dataset.reference.artifact_id
        assert metadata.dataset_sha256 == fitted.dataset.dataset_sha256
        assert metadata.training_case_count == 600
        assert BaselineMetadata.model_validate_json(metadata.model_dump_json()) == metadata
    assert fitted.mean.metadata.baseline_id != fitted.nearest.metadata.baseline_id


def test_mean_field_is_hand_computed_fp64_then_immutable_float32(
    fitted: FittedBaselines,
) -> None:
    mean = fitted.mean
    np.testing.assert_array_equal(mean.fields_normalized[0], np.float32(0.25))
    np.testing.assert_array_equal(mean.fields_normalized[1], np.float32(-0.5))
    np.testing.assert_array_equal(mean.fields_normalized[2], np.float32(0.125))
    cd_sum = np.float64(0.0)
    for row in fitted.dataset.split_rows("train"):
        cd_sum += np.float64(np.float32(row.cd))
    assert mean.cd == np.float32(cd_sum / np.float64(600))
    assert mean.fields_normalized.dtype == np.dtype(np.float32)
    assert not mean.fields_normalized.flags.writeable

    fitted.harness.opened.clear()
    repeated = fit_mean_field_baseline(fitted.dataset, fitted.harness.statistics)
    assert repeated.metadata == mean.metadata
    assert repeated.cd == mean.cd
    np.testing.assert_array_equal(repeated.fields_normalized, mean.fields_normalized)
    assert fitted.harness.opened == [row.run_digest for row in fitted.dataset.split_rows("train")]


def test_both_baselines_use_one_flow_predictor_result_contract_deterministically(
    fitted: FittedBaselines,
) -> None:
    row = fitted.nearest.training_rows[37]
    unit = np.asarray(baseline_module._unit_design(row), dtype=np.float32)
    query = np.stack((2.0 * unit - 1.0, 2.0 * unit - 1.0))
    batch = prediction_batch(query)

    mean_first = _shared_predictor_contract(fitted.mean, batch)
    mean_second = _shared_predictor_contract(fitted.mean, batch)
    nearest_first = _shared_predictor_contract(fitted.nearest, batch)
    nearest_second = _shared_predictor_contract(fitted.nearest, batch)

    assert fitted.nearest.nearest_design_ids(batch) == (row.design_id, row.design_id)
    for first, second in zip(mean_first, mean_second, strict=True):
        np.testing.assert_array_equal(first, second)
    for first, second in zip(nearest_first, nearest_second, strict=True):
        np.testing.assert_array_equal(first, second)
    assert nearest_first[0].shape == (2, 3, 320, 256)
    np.testing.assert_array_equal(nearest_first[1], np.float32(row.cd))


def test_nearest_distance_is_unit_interval_euclidean_with_design_id_tie_break() -> None:
    training = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    training.flags.writeable = False
    query_model_space = np.asarray([[0.0, -1.0, -1.0, -1.0]], dtype=np.float32)

    selected = baseline_module._nearest_indices(
        query_model_space,
        training,
        ("a" * 20, "b" * 20),
    )

    assert selected == (0,)
    assert baseline_module._nearest_indices(
        np.asarray([[1.0, -1.0, -1.0, -1.0]], dtype=np.float32),
        training,
        ("a" * 20, "b" * 20),
    ) == (1,)


def test_nearest_metadata_and_selection_repeat_without_loading_training_fields(
    fitted: FittedBaselines,
) -> None:
    fitted.harness.opened.clear()
    repeated = fit_nearest_design_baseline(fitted.dataset, fitted.harness.statistics)

    assert repeated.metadata == fitted.nearest.metadata
    assert fitted.harness.opened == []
    np.testing.assert_array_equal(
        repeated.training_unit_design,
        fitted.nearest.training_unit_design,
    )


def test_baseline_metadata_rejects_state_rebinding_and_wrong_distance(
    fitted: FittedBaselines,
) -> None:
    payload = fitted.mean.metadata.model_dump(mode="python")
    with pytest.raises(ValidationError, match="complete logical state"):
        BaselineMetadata.model_validate({**payload, "state_sha256": "f" * 64})
    wrong_identity = {
        **fitted.mean.metadata.logical_identity(),
        "design_distance": "unit-interval-euclidean-v1",
    }
    wrong_digest = canonical_sha256(wrong_identity)
    with pytest.raises(ValidationError, match="design distance"):
        BaselineMetadata.model_validate(
            {
                **payload,
                "design_distance": "unit-interval-euclidean-v1",
                "baseline_id": wrong_digest[:20],
                "baseline_sha256": wrong_digest,
            }
        )


def test_fit_rejects_wrong_dataset_or_noncanonical_training_membership(
    fitted: FittedBaselines,
) -> None:
    wrong_dataset = fitted.harness.statistics.model_copy(update={"dataset_id": "f" * 20})
    with pytest.raises(ArtifactIntegrityError, match="statistics and dataset IDs"):
        fit_mean_field_baseline(fitted.dataset, wrong_dataset)

    wrong_count = fitted.harness.statistics.model_copy(
        update={
            "training_case_count": 599,
            "training_cell_count": 599 * MODEL_CELL_COUNT,
        }
    )
    with pytest.raises(ArtifactIntegrityError, match="requires 600"):
        fit_nearest_design_baseline(fitted.dataset, wrong_count)


def test_predictor_fails_closed_when_tensor_backend_cannot_create_output(
    fitted: FittedBaselines,
) -> None:
    class NoFactoryTensor(ArrayTensor):
        def new_tensor(self, value: np.ndarray[Any, Any]) -> ArrayTensor:
            del value
            raise RuntimeError("unavailable")

    batch = prediction_batch(np.zeros((1, 4), dtype=np.float32))
    broken = PredictionBatch(
        inputs=NoFactoryTensor(cast(ArrayTensor, batch.inputs).value),
        fluid_mask=batch.fluid_mask,
        design_params=batch.design_params,
    )
    with pytest.raises(ArtifactIntegrityError, match="cannot be created"):
        fitted.mean.predict(broken)


def test_nearest_distance_rejects_malformed_or_nonfinite_query() -> None:
    design = np.zeros((2, 4), dtype=np.float64)
    with pytest.raises(ArtifactIntegrityError, match=r"shape \[batch,4\]"):
        baseline_module._nearest_indices(
            np.zeros((4,), dtype=np.float32), design, ("a" * 20, "b" * 20)
        )
    with pytest.raises(ArtifactIntegrityError, match="finite"):
        baseline_module._nearest_indices(
            np.full((1, 4), np.nan, dtype=np.float32),
            design,
            ("a" * 20, "b" * 20),
        )
