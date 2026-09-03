from __future__ import annotations

from pathlib import Path

import pytest

from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ArtifactRef
from soufflerie.training import ManifestDataset, open_manifest_dataset
from tests.training.helpers import TrainingHarness, build_harness


@pytest.fixture(scope="module")
def harness() -> TrainingHarness:
    return build_harness()


@pytest.fixture
def dataset(
    harness: TrainingHarness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ManifestDataset:
    harness.opened.clear()
    harness.tamper_reference = False
    harness.tamper_metadata = False
    return harness.open(tmp_path, monkeypatch)


def test_loader_opens_only_manifest_membership_without_directory_listing(
    harness: TrainingHarness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stray = tmp_path / "runs" / "not-membership"
    stray.mkdir(parents=True)
    (stray / "fields.npz").write_bytes(b"untrusted")
    harness.install(monkeypatch)

    def no_listing(_path: Path) -> object:
        raise AssertionError("membership must not use filesystem listing")

    monkeypatch.setattr(Path, "iterdir", no_listing)
    loaded = open_manifest_dataset(tmp_path, harness.published.reference)
    train = loaded.split_rows("train")
    sample = loaded.load_sample(train[0])

    assert len(train) == 600
    assert len(loaded.split_rows("validation")) == 200
    assert len(loaded.split_rows("test")) == 200
    assert sample.case_id == train[0].case_id
    assert harness.opened == [train[0].run_digest]
    assert "not-membership" not in {row.case_id for row in train}


def test_epoch_order_and_batch_partition_are_exact_and_deterministic(
    dataset: ManifestDataset,
) -> None:
    first = dataset.ordered_rows("train", seed=17, epoch=0)
    repeated = dataset.ordered_rows("train", seed=17, epoch=0)
    next_epoch = dataset.ordered_rows("train", seed=17, epoch=1)
    validation_a = dataset.ordered_rows("validation", seed=17, epoch=0)
    validation_b = dataset.ordered_rows("validation", seed=99, epoch=100)

    assert first == repeated
    assert first != next_epoch
    assert {row.design_id for row in first} == {
        row.design_id for row in dataset.split_rows("train")
    }
    assert validation_a == validation_b

    batches = dataset.batch_rows("train", batch_size=64, seed=17, epoch=0)
    assert len(batches) == 10
    assert [len(batch.rows) for batch in batches] == [64] * 9 + [24]
    assert tuple(row for batch in batches for row in batch.rows) == first


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"batch_size": 0, "seed": 17, "epoch": 0}, "batch_size"),
        ({"batch_size": 65, "seed": 17, "epoch": 0}, "batch_size"),
        ({"batch_size": 8, "seed": -1, "epoch": 0}, "seed"),
        ({"batch_size": 8, "seed": 17, "epoch": -1}, "epoch"),
    ],
)
def test_batch_contract_rejects_unbounded_or_invalid_ordering_inputs(
    dataset: ManifestDataset,
    arguments: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ArtifactIntegrityError, match=message):
        dataset.batch_rows("train", **arguments)


def test_iter_batches_opens_only_the_bounded_slice_and_binds_membership(
    dataset: ManifestDataset,
    harness: TrainingHarness,
) -> None:
    expected = dataset.ordered_rows("train", seed=23, epoch=4)[:3]

    batch = next(
        dataset.iter_batches(
            harness.statistics,
            "train",
            batch_size=3,
            seed=23,
            epoch=4,
        )
    )

    assert batch.membership.rows == expected
    assert batch.membership.ordinal == 0
    assert batch.data.inputs.shape == (3, 2, 320, 256)
    assert batch.data.fields_normalized.shape == (3, 3, 320, 256)
    assert harness.opened == [row.run_digest for row in expected]


def test_preloaded_splits_are_verified_once_and_reused_without_store_io(
    dataset: ManifestDataset,
    harness: TrainingHarness,
) -> None:
    expected = tuple(
        row.run_digest for split in ("train", "validation") for row in dataset.split_rows(split)
    )

    assert dataset.preload_splits(("train", "validation")) == 800
    assert harness.opened == list(expected)
    harness.opened.clear()

    for split in ("train", "validation"):
        rows = dataset.split_rows(split)
        assert dataset.load_sample(rows[0]).case_id == rows[0].case_id
    assert dataset.preload_splits(("validation",)) == 800
    assert harness.opened == []


@pytest.mark.parametrize("splits", [(), ("train", "train"), ("invalid",)])
def test_preload_rejects_empty_duplicate_and_unknown_splits(
    dataset: ManifestDataset,
    splits: tuple[str, ...],
) -> None:
    with pytest.raises(ArtifactIntegrityError, match="splits must be"):
        dataset.preload_splits(splits)  # type: ignore[arg-type]


def test_loader_rejects_nonmembership_rebinding_and_wrong_statistics(
    dataset: ManifestDataset,
    harness: TrainingHarness,
) -> None:
    row = dataset.split_rows("train")[0]
    forged = row.model_copy(update={"cd": row.cd + 1.0})
    with pytest.raises(ArtifactIntegrityError, match="not dataset membership"):
        dataset.load_sample(forged)

    wrong = harness.statistics.model_copy(update={"dataset_id": "f" * 20})
    with pytest.raises(ArtifactIntegrityError, match="statistics and dataset"):
        next(dataset.iter_batches(wrong, "train", batch_size=1, seed=17, epoch=0))

    harness.tamper_reference = True
    with pytest.raises(ArtifactIntegrityError, match="opened run reference changed"):
        dataset.load_sample(row)
    harness.tamper_reference = False
    harness.tamper_metadata = True
    with pytest.raises(ArtifactIntegrityError, match="run content disagrees"):
        dataset.load_sample(row)


def test_public_factory_and_constructor_enforce_verified_dataset_boundary(
    harness: TrainingHarness,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="open_manifest_dataset"):
        ManifestDataset(harness.published, object(), _verification=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="root must be a Path"):
        open_manifest_dataset("not-a-path", harness.published.reference)  # type: ignore[arg-type]
    wrong = ArtifactRef(
        artifact_type="run",
        artifact_id="a" * 20,
        sha256="a" * 64,
        size_bytes=1,
        uri=f"runs/{'a' * 20}/{'a' * 64}",
    )
    with pytest.raises(ArtifactIntegrityError, match="dataset ArtifactRef"):
        open_manifest_dataset(tmp_path, wrong)


def test_iter_batches_requires_statistics_record(
    dataset: ManifestDataset,
) -> None:
    with pytest.raises(TypeError, match="PreprocessingStatistics"):
        next(
            dataset.iter_batches(
                object(),  # type: ignore[arg-type]
                "train",
                batch_size=1,
                seed=17,
                epoch=0,
            )
        )
