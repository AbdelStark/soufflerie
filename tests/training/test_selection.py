from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from soufflerie.errors import ArtifactIntegrityError
from soufflerie.training import (
    FrozenTrainingSelection,
    ValidationCheckpointMetric,
    freeze_validation_selection,
)


def _metric(
    seed: int,
    epoch: int,
    velocity: float,
    cd: float,
    *,
    experiment_id: str = "e" * 20,
) -> ValidationCheckpointMetric:
    return ValidationCheckpointMetric(
        experiment_id=experiment_id,
        dataset_id="a" * 20,
        config_digest="b" * 64,
        checkpoint_id=f"{seed:02x}{epoch:02x}".ljust(20, "0"),
        seed=seed,
        epoch=epoch,
        median_velocity_relative_l2=velocity,
        median_cd_head_relative_error=cd,
        score=velocity + cd,
    )


def test_selection_uses_validation_only_with_earlier_epoch_and_lower_seed_ties() -> None:
    metrics = (
        _metric(17, 1, 0.3, 0.2),
        _metric(17, 2, 0.2, 0.3),
        _metric(17, 3, 0.1, 0.1),
        _metric(23, 4, 0.1, 0.1),
        _metric(23, 5, 0.1, 0.1),
        _metric(31, 6, 0.2, 0.2),
    )
    selection = freeze_validation_selection(metrics, expected_seeds=(17, 23, 31))

    assert [(item.seed, item.epoch) for item in selection.selected] == [
        (17, 3),
        (23, 4),
        (31, 6),
    ]
    assert selection.deployable_seed == 17
    assert selection.test_metrics_read is False
    assert FrozenTrainingSelection.model_validate_json(selection.model_dump_json()) == selection


def test_test_split_or_inexact_score_cannot_enter_selection() -> None:
    payload = _metric(17, 1, 0.1, 0.2).model_dump()
    with pytest.raises(ValidationError):
        ValidationCheckpointMetric.model_validate({**payload, "split": "test"})
    with pytest.raises(ValidationError, match="must equal"):
        ValidationCheckpointMetric.model_validate({**payload, "score": 99.0})
    with pytest.raises(ValidationError):
        ValidationCheckpointMetric.model_validate({**payload, "test_error": 0.0})


def test_selection_rejects_identity_missing_seed_and_duplicate_epoch() -> None:
    complete = (
        _metric(17, 1, 0.1, 0.1),
        _metric(23, 1, 0.2, 0.2),
        _metric(31, 1, 0.3, 0.3),
    )
    with pytest.raises(ArtifactIntegrityError, match="identities differ"):
        freeze_validation_selection(
            (*complete, _metric(31, 2, 0.1, 0.1, experiment_id="f" * 20)),
            expected_seeds=(17, 23, 31),
        )
    with pytest.raises(ArtifactIntegrityError, match="cover every seed"):
        freeze_validation_selection(complete[:2], expected_seeds=(17, 23, 31))
    with pytest.raises(ArtifactIntegrityError, match="duplicate epoch"):
        freeze_validation_selection(
            (*complete, _metric(17, 1, 0.05, 0.05)),
            expected_seeds=(17, 23, 31),
        )
    with pytest.raises(ArtifactIntegrityError, match="validation metrics"):
        freeze_validation_selection(
            cast(Any, (object(),)),
            expected_seeds=(17, 23, 31),
        )
