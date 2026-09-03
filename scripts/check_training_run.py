"""Verify the canonical three-seed training handoff without opening trusted state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Direct script execution exposes ``scripts/`` rather than the repository root.
# The remote execution contracts intentionally live in the uninstalled ``infra``
# package, so make that operator entrypoint boundary explicit before importing it.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.train_validate_execution import TrainingRunIndex  # noqa: E402
from soufflerie.config import TrainingConfig, load_config  # noqa: E402
from soufflerie.errors import ArtifactIntegrityError  # noqa: E402
from soufflerie.schemas import ArtifactRef  # noqa: E402

REFERENCE_GPU = "L40S"
REFERENCE_PRECISION = "bf16"
REFERENCE_EPOCHS = 100
MAX_SEED_WALL_SECONDS = 60 * 60


def load_training_index(path: Path) -> TrainingRunIndex:
    """Load one bounded, canonical index and re-run every schema invariant."""

    try:
        content = path.read_bytes()
    except OSError as error:
        raise ArtifactIntegrityError("TRAIN-RUN-1 READ: index cannot be read") from error
    if not content or len(content) > 1024 * 1024:
        raise ArtifactIntegrityError("TRAIN-RUN-1 SIZE: index violates the 1 MiB cap")
    try:
        index = TrainingRunIndex.model_validate_json(content)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError("TRAIN-RUN-1 SCHEMA: index is invalid") from error
    expected = (json.dumps(index.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
    if content != expected:
        raise ArtifactIntegrityError("TRAIN-RUN-1 ENCODING: index is not canonical pretty JSON")
    return index


def load_dataset_reference(path: Path) -> ArtifactRef:
    """Read the exact dataset reference retained by the canonical sweep summary."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))["dataset_reference"]
        reference = ArtifactRef.model_validate(payload)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            "TRAIN-RUN-2 DATASET: sweep summary has no valid dataset reference"
        ) from error
    if reference.artifact_type != "dataset":
        raise ArtifactIntegrityError("TRAIN-RUN-2 DATASET: sweep artifact is not a dataset")
    return reference


def check_training_run(
    index: TrainingRunIndex,
    *,
    config: TrainingConfig,
    dataset: ArtifactRef,
) -> None:
    """Enforce the release experiment, resource budget, and selection handoff."""

    request = index.request
    if request.config != config or request.config.config_digest != config.config_digest:
        raise ArtifactIntegrityError("TRAIN-RUN-3 CONFIG: index does not bind the frozen config")
    if request.dataset != dataset or request.dataset.sha256 != dataset.sha256:
        raise ArtifactIntegrityError("TRAIN-RUN-4 DATASET: index does not bind the frozen dataset")
    if (
        request.requested_device_class != REFERENCE_GPU
        or request.config.precision != REFERENCE_PRECISION
        or request.config.epochs != REFERENCE_EPOCHS
    ):
        raise ArtifactIntegrityError(
            "TRAIN-RUN-5 IDENTITY: run is not the 100-epoch L40S bf16 experiment"
        )
    if len({receipt.model.sha256 for receipt in index.receipts}) != 3:
        raise ArtifactIntegrityError(
            "TRAIN-RUN-6 MODELS: three distinct full model digests required"
        )
    if any(
        receipt.accounting.wall_seconds >= MAX_SEED_WALL_SECONDS
        or "l40s" not in receipt.accounting.device_name.casefold().replace(" ", "")
        for receipt in index.receipts
    ):
        raise ArtifactIntegrityError(
            "TRAIN-RUN-7 BUDGET: every seed must finish within 60 minutes on L40S"
        )
    if index.selected_model_id not in {receipt.model.artifact_id for receipt in index.receipts}:
        raise ArtifactIntegrityError(
            "TRAIN-RUN-8 SELECTION: deployable model is absent from the seed receipts"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="canonical TrainingRunIndex JSON")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "fno-v1.yaml",
        help="frozen TrainingConfig",
    )
    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dataset" / "sweep-summary.json",
        help="canonical sweep summary",
    )
    args = parser.parse_args()
    try:
        index = load_training_index(args.index)
        config = load_config(args.config, TrainingConfig)
        dataset = load_dataset_reference(args.dataset_summary)
        check_training_run(index, config=config, dataset=dataset)
    except (ArtifactIntegrityError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(
        "training_run=PASS "
        f"experiment={index.request.experiment_id} "
        f"models={len(index.receipts)} "
        f"selected={index.selected_model_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
