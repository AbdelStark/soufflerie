from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from infra.validation_pipeline import _flow_fields


def test_flow_fields_derive_model_grid_mask_from_persisted_sdf() -> None:
    sdf = np.asarray(
        [
            [-1.0, -0.0, 0.5],
            [1.0, 2.0, -0.25],
        ],
        dtype=np.float16,
    )
    sample = SimpleNamespace(
        sdf=sdf,
        # Curation nearest-samples this source-grid mask independently, so it
        # may disagree with the quantized model-grid SDF at boundary cells.
        obstacle_mask=np.zeros(sdf.shape, dtype=np.uint8),
    )
    physical = np.ones((1, 3, *sdf.shape), dtype=np.float32)

    fields = _flow_fields(sample, physical, 0)

    np.testing.assert_array_equal(fields.obstacle_mask, sdf <= np.float16(0.0))
    np.testing.assert_array_equal(fields.sdf, sdf.astype(np.float32))
