from __future__ import annotations

import importlib
import importlib.util
from typing import Any, cast

import pytest

from soufflerie.surrogate.fno import FnoPredictor
from soufflerie.surrogate.preprocessing import MODEL_SPATIAL_SHAPE, PredictionBatch


@pytest.mark.slow
def test_one_batch_fixture_reduces_loss_by_at_least_ninety_percent() -> None:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("physicsnemo") is None:
        pytest.skip("the optional ml runtime is not installed")
    torch = cast(Any, importlib.import_module("torch"))
    torch.manual_seed(7)
    torch.set_num_threads(4)
    predictor = FnoPredictor()
    batch = PredictionBatch(
        inputs=torch.zeros((1, 2, *MODEL_SPATIAL_SHAPE), dtype=torch.float32),
        fluid_mask=torch.ones((1, 1, *MODEL_SPATIAL_SHAPE), dtype=torch.bool),
        design_params=torch.zeros((1, 4), dtype=torch.float32),
    )
    target_fields = torch.full((1, 3, *MODEL_SPATIAL_SHAPE), 0.25, dtype=torch.float32)
    target_cd = torch.tensor([0.5], dtype=torch.float32)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-2)

    def fixture_loss() -> Any:
        result = predictor(batch)
        return torch.nn.functional.mse_loss(
            result.fields_normalized,
            target_fields,
        ) + torch.nn.functional.mse_loss(result.cd_head, target_cd)

    initial = float(fixture_loss().detach().item())
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = fixture_loss()
        loss.backward()
        optimizer.step()
    final = float(fixture_loss().detach().item())

    assert final < 0.1 * initial, (initial, final)
