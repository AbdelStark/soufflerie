from __future__ import annotations

import numpy as np
import numpy.typing as npt

from soufflerie.surrogate.bundle import (
    EXPECTED_MODEL_TENSORS,
    CompatibilityRange,
    ModelBundle,
    ModelCardGate,
    ModelCardMetadata,
    build_model_bundle,
)
from soufflerie.surrogate.preprocessing import (
    MODEL_CELL_COUNT,
    OutputChannelStatistics,
    OutputNormalizationStatistics,
    PreprocessingStatistics,
)

DATASET_ID = "d" * 20
DATASET_SHA256 = "d" * 64
EXPERIMENT_ID = "e" * 20
CODE_REVISION = "a" * 40
LOCK_DIGEST = "b" * 64


def preprocessing_statistics(*, dataset_id: str = DATASET_ID) -> PreprocessingStatistics:
    channel = OutputChannelStatistics(
        mean=0.0,
        raw_standard_deviation=1.0,
        standard_deviation=1.0,
        floored=False,
    )
    return PreprocessingStatistics(
        dataset_id=dataset_id,
        training_case_count=1,
        training_cell_count=MODEL_CELL_COUNT,
        outputs=OutputNormalizationStatistics(
            u_mean=channel,
            v_mean=channel,
            rho_delta=channel,
        ),
    )


def model_card() -> ModelCardMetadata:
    return ModelCardMetadata(
        display_name="Soufflerie test FNO",
        summary="A deterministic fixture for the safe bundle contract.",
        intended_uses=("Contract and inference round-trip testing.",),
        limitations=("This fixture is not release validation evidence.",),
        gates=(
            ModelCardGate(
                name="Velocity relative L2",
                status="not_evaluated",
                threshold="median below 0.10",
            ),
            ModelCardGate(
                name="Drag relative error",
                status="green",
                threshold="median below 0.10",
                measured="median 0.04",
                evidence_sha256="c" * 64,
            ),
        ),
    )


def zero_weights() -> dict[str, npt.NDArray[np.float32]]:
    weights = {
        descriptor.name: np.zeros(descriptor.shape, dtype=np.float32)
        for descriptor in EXPECTED_MODEL_TENSORS
    }
    weights["core.decoder_net.final_layer.linear.bias"][:] = np.array(
        [0.25, -0.5, 0.75], dtype=np.float32
    )
    weights["cd_head.4.bias"][:] = np.float32(0.125)
    return weights


def make_test_bundle(
    *,
    weights: dict[str, npt.NDArray[np.float32]] | None = None,
    preprocessing: PreprocessingStatistics | None = None,
    compatibility: CompatibilityRange | None = None,
) -> ModelBundle:
    return build_model_bundle(
        weights=weights if weights is not None else zero_weights(),
        preprocessing=preprocessing or preprocessing_statistics(),
        dataset_sha256=DATASET_SHA256,
        experiment_id=EXPERIMENT_ID,
        seed=7,
        selected_epoch=12,
        code_revision=CODE_REVISION,
        lock_digest=LOCK_DIGEST,
        model_card=model_card(),
        compatibility=compatibility,
    )
