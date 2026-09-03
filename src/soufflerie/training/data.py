"""Verified manifest membership and deterministic bounded training batches."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from soufflerie.datagen.manifest import (
    LocalDatasetArtifactStore,
    ManifestRow,
    PublishedDataset,
)
from soufflerie.datagen.run_artifact import LocalRunArtifactStore, RunArtifactStore
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ArtifactRef, Split
from soufflerie.surrogate.preprocessing import (
    PreprocessedBatch,
    PreprocessingSample,
    PreprocessingStatistics,
    preprocess_batch,
)

MAX_TRAINING_BATCH_SIZE = 64
_UINT64_MAX = 2**64 - 1
_VERIFIED_DATASET = object()


@dataclass(frozen=True, slots=True)
class ManifestRowBatch:
    """One bounded deterministic slice of authoritative manifest rows."""

    split: Split
    epoch: int
    ordinal: int
    rows: tuple[ManifestRow, ...]

    def __post_init__(self) -> None:
        if not self.rows or len(self.rows) > MAX_TRAINING_BATCH_SIZE:
            raise ArtifactIntegrityError(
                "TRAIN-2 BATCH: row batch must contain between 1 and 64 rows"
            )
        if any(row.split != self.split for row in self.rows):
            raise ArtifactIntegrityError("TRAIN-2 BATCH: row batch crosses split membership")
        if self.epoch < 0 or self.ordinal < 0:
            raise ArtifactIntegrityError("TRAIN-2 BATCH: epoch and ordinal must be nonnegative")


@dataclass(frozen=True, slots=True)
class ManifestBatch:
    """Preprocessed tensors bound to one authoritative row batch."""

    membership: ManifestRowBatch
    data: PreprocessedBatch

    def __post_init__(self) -> None:
        if int(self.data.inputs.shape[0]) != len(self.membership.rows):
            raise ArtifactIntegrityError(
                "TRAIN-2 BATCH: preprocessed batch size does not match membership"
            )


class ManifestDataset:
    """A fully verified dataset whose membership comes only from its manifest."""

    __slots__ = ("_published", "_run_store", "_sample_cache")

    def __init__(
        self,
        published: PublishedDataset,
        run_store: RunArtifactStore,
        *,
        _verification: object,
    ) -> None:
        if _verification is not _VERIFIED_DATASET:
            raise TypeError("use open_manifest_dataset() to verify dataset membership")
        if not isinstance(published, PublishedDataset):
            raise TypeError("published must be a verified PublishedDataset")
        self._published = published
        self._run_store = run_store
        self._sample_cache: dict[str, PreprocessingSample] = {}

    @property
    def reference(self) -> ArtifactRef:
        return self._published.reference

    @property
    def dataset_sha256(self) -> str:
        return self._published.manifest.metadata.dataset_sha256

    @property
    def parent_run_sha256(self) -> tuple[str, ...]:
        """Return the verified canonical solver-run lineage in manifest order."""

        return self._published.manifest.metadata.parent_run_sha256

    def split_rows(self, split: Split) -> tuple[ManifestRow, ...]:
        """Return immutable manifest membership in canonical design-ID order."""

        if split not in {"train", "validation", "test"}:
            raise ArtifactIntegrityError("TRAIN-1 MANIFEST: unsupported split")
        rows = tuple(row for row in self._published.manifest.rows if row.split == split)
        expected = {"train": 600, "validation": 200, "test": 200}[split]
        if len(rows) != expected:
            raise ArtifactIntegrityError("TRAIN-1 MANIFEST: split count changed after verification")
        return rows

    def ordered_rows(self, split: Split, *, seed: int, epoch: int) -> tuple[ManifestRow, ...]:
        """Return stable train shuffling or canonical evaluation ordering."""

        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _UINT64_MAX:
            raise ArtifactIntegrityError("TRAIN-2 BATCH: seed must be an unsigned 64-bit integer")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ArtifactIntegrityError("TRAIN-2 BATCH: epoch must be a nonnegative integer")
        rows = self.split_rows(split)
        if split != "train":
            return rows

        def order_key(row: ManifestRow) -> tuple[bytes, str]:
            payload = (
                f"training-order-v1\0{self.dataset_sha256}\0{seed}\0{epoch}\0{row.design_id}"
            ).encode("ascii")
            return hashlib.sha256(payload).digest(), row.design_id

        return tuple(sorted(rows, key=order_key))

    def batch_rows(
        self,
        split: Split,
        *,
        batch_size: int,
        seed: int,
        epoch: int,
    ) -> tuple[ManifestRowBatch, ...]:
        """Partition every row exactly once; the final partial batch is retained."""

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= MAX_TRAINING_BATCH_SIZE
        ):
            raise ArtifactIntegrityError("TRAIN-2 BATCH: batch_size must be in [1,64]")
        ordered = self.ordered_rows(split, seed=seed, epoch=epoch)
        return tuple(
            ManifestRowBatch(
                split=split,
                epoch=epoch,
                ordinal=ordinal,
                rows=ordered[start : start + batch_size],
            )
            for ordinal, start in enumerate(range(0, len(ordered), batch_size))
        )

    def load_sample(self, row: ManifestRow) -> PreprocessingSample:
        """Open one exact manifest parent and bind its verified fields back to the row."""

        if not isinstance(row, ManifestRow):
            raise TypeError("row must be a ManifestRow")
        if row.dataset_id != self.reference.artifact_id or row not in self._published.manifest.rows:
            raise ArtifactIntegrityError("TRAIN-1 MANIFEST: row is not dataset membership")
        cached = self._sample_cache.get(row.run_digest)
        if cached is not None:
            return cached
        expected = ArtifactRef(
            artifact_type="run",
            artifact_id=row.run_digest[:20],
            sha256=row.run_digest,
            size_bytes=row.bytes,
            uri=row.run_uri,
        )
        artifact = self._run_store.open_run(expected)
        metadata = artifact.metadata
        if artifact.reference != expected:
            raise ArtifactIntegrityError("TRAIN-1 MANIFEST: opened run reference changed")
        if (
            metadata.case_id != row.case_id
            or metadata.design_id != row.design_id
            or metadata.split != row.split
            or metadata.artifact_digest != row.run_digest
            or metadata.cd != row.cd
            or metadata.case.shape.aspect_ratio != row.aspect_ratio
            or metadata.case.shape.rotation_deg != row.rotation_deg
            or metadata.case.shape.scale != row.scale
            or metadata.case.reynolds != row.reynolds
        ):
            raise ArtifactIntegrityError(
                "TRAIN-1 MANIFEST: run content disagrees with manifest row"
            )
        fields = artifact.fields
        return PreprocessingSample(
            dataset_id=row.dataset_id,
            case_id=row.case_id,
            split=row.split,
            aspect_ratio=row.aspect_ratio,
            rotation_deg=row.rotation_deg,
            scale=row.scale,
            reynolds=row.reynolds,
            cd=row.cd,
            u_mean=fields.u_mean,
            v_mean=fields.v_mean,
            rho_mean=fields.rho_mean,
            sdf=fields.sdf,
            obstacle_mask=fields.obstacle_mask,
        )

    def preload_splits(self, splits: tuple[Split, ...]) -> int:
        """Verify and retain bounded immutable samples for repeated epoch access."""

        if (
            not isinstance(splits, tuple)
            or not splits
            or len(set(splits)) != len(splits)
            or any(split not in {"train", "validation", "test"} for split in splits)
        ):
            raise ArtifactIntegrityError(
                "TRAIN-3 CACHE: splits must be a nonempty distinct split tuple"
            )
        rows = tuple(row for split in splits for row in self.split_rows(split))
        staged = {
            row.run_digest: self.load_sample(row)
            for row in rows
            if row.run_digest not in self._sample_cache
        }
        if len(staged) + len(self._sample_cache) > 1_000:
            raise ArtifactIntegrityError("TRAIN-3 CACHE: sample cache exceeds manifest bounds")
        self._sample_cache.update(staged)
        return len(self._sample_cache)

    def iter_samples(self, split: Split) -> Iterator[PreprocessingSample]:
        """Stream verified samples in canonical design-ID order."""

        for row in self.split_rows(split):
            yield self.load_sample(row)

    def iter_batches(
        self,
        statistics: PreprocessingStatistics,
        split: Split,
        *,
        batch_size: int,
        seed: int,
        epoch: int,
    ) -> Iterator[ManifestBatch]:
        """Open and preprocess at most ``batch_size`` verified parents at a time."""

        if not isinstance(statistics, PreprocessingStatistics):
            raise TypeError("statistics must be PreprocessingStatistics")
        if statistics.dataset_id != self.reference.artifact_id:
            raise ArtifactIntegrityError("TRAIN-2 BATCH: statistics and dataset identities differ")
        for membership in self.batch_rows(
            split,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
        ):
            samples = tuple(self.load_sample(row) for row in membership.rows)
            yield ManifestBatch(membership=membership, data=preprocess_batch(samples, statistics))


def open_manifest_dataset(root: Path, reference: ArtifactRef) -> ManifestDataset:
    """Open one committed dataset and establish the run-artifact trust boundary."""

    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    if not isinstance(reference, ArtifactRef) or reference.artifact_type != "dataset":
        raise ArtifactIntegrityError("TRAIN-1 MANIFEST: expected a dataset ArtifactRef")
    published = LocalDatasetArtifactStore(root).open(reference)
    return ManifestDataset(
        published,
        LocalRunArtifactStore(root),
        _verification=_VERIFIED_DATASET,
    )


__all__ = [
    "MAX_TRAINING_BATCH_SIZE",
    "ManifestBatch",
    "ManifestDataset",
    "ManifestRowBatch",
    "open_manifest_dataset",
]
