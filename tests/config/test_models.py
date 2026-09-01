from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from soufflerie.config import (
    Range,
    ServiceConfig,
    SweepConfig,
    TrainingConfig,
    ValidationConfig,
    load_config,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_range_and_canonical_sweep_constraints() -> None:
    with pytest.raises(ValidationError, match="minimum must be less"):
        Range(minimum=1.0, maximum=1.0)

    sweep = load_config(PROJECT_ROOT / "configs/sweeps/mvp-v1.yaml", SweepConfig)
    assert sweep.samples == 1_000
    assert sweep.split_counts == (600, 200, 200)
    assert sweep.grid.shape == (640, 512)
    assert sweep.aspect_ratio.minimum == 0.5
    assert len(sweep.config_digest) == 64

    with pytest.raises(ValidationError, match="canonical"):
        SweepConfig.model_validate(
            {**sweep.model_dump(), "reynolds": {"minimum": 50.0, "maximum": 300.0}}
        )
    with pytest.raises(ValidationError, match="canonical"):
        SweepConfig.model_validate(
            {**sweep.model_dump(), "aspect_ratio": {"minimum": 0.3, "maximum": 1.0}}
        )


def test_training_config_requires_distinct_seeds_and_learning_rate_order() -> None:
    training = load_config(PROJECT_ROOT / "configs/training/fno-v1.yaml", TrainingConfig)
    with pytest.raises(ValidationError, match="three distinct"):
        TrainingConfig.model_validate({**training.model_dump(), "seeds": (17, 17, 31)})
    with pytest.raises(ValidationError, match="must not exceed"):
        TrainingConfig.model_validate(
            {**training.model_dump(), "learning_rate": 1e-5, "min_learning_rate": 1e-3}
        )


def test_validation_and_service_identity_and_budget_controls() -> None:
    validation = load_config(PROJECT_ROOT / "configs/validation/release-v1.yaml", ValidationConfig)
    with pytest.raises(ValidationError, match="must be distinct"):
        ValidationConfig.model_validate(
            {
                **validation.model_dump(),
                "baseline_ids": (validation.ensemble_model_ids[0], validation.baseline_ids[1]),
            }
        )

    service = load_config(PROJECT_ROOT / "configs/service/demo-v1.yaml", ServiceConfig)
    assert service.solve_enabled is False
    assert service.solve_concurrency == 0
    with pytest.raises(ValidationError, match="must be zero"):
        ServiceConfig.model_validate({**service.model_dump(), "solve_concurrency": 1})
    with pytest.raises(ValidationError, match="less than or equal to 20"):
        ServiceConfig.model_validate({**service.model_dump(), "solves_per_day_global": 21})


def test_models_are_frozen() -> None:
    service = load_config(PROJECT_ROOT / "configs/service/demo-v1.yaml", ServiceConfig)
    with pytest.raises(ValidationError, match="frozen"):
        service.port = 9000
