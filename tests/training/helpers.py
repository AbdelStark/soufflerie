from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import pytest

import soufflerie.training.data as data_module
from soufflerie.config import SweepConfig, load_config
from soufflerie.datagen.manifest import ManifestRow, PublishedDataset, load_manifest
from soufflerie.datagen.run_artifact import CuratedRunFields, RunArtifact, RunMetadata
from soufflerie.schemas import ArtifactRef, CaseConfig, ShapeParams
from soufflerie.surrogate.preprocessing import (
    MODEL_SPATIAL_SHAPE,
    OutputChannelStatistics,
    OutputNormalizationStatistics,
    PredictionBatch,
    PreprocessingStatistics,
)
from soufflerie.training import ManifestDataset, open_manifest_dataset

MANIFEST_FIXTURE = Path("tests/fixtures/dataset/manifest.parquet")
SWEEP_CONFIG = Path("configs/sweeps/mvp-v1.yaml")


def _readonly(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    value.flags.writeable = False
    return value


def _fields() -> CuratedRunFields:
    return CuratedRunFields(
        u_mean=_readonly(np.full(MODEL_SPATIAL_SHAPE, 0.25, dtype=np.float16)),
        v_mean=_readonly(np.full(MODEL_SPATIAL_SHAPE, -0.5, dtype=np.float16)),
        rho_mean=_readonly(np.full(MODEL_SPATIAL_SHAPE, 1.125, dtype=np.float16)),
        sdf=_readonly(np.full(MODEL_SPATIAL_SHAPE, 8.0, dtype=np.float16)),
        obstacle_mask=_readonly(np.zeros(MODEL_SPATIAL_SHAPE, dtype=np.uint8)),
        force_steps=_readonly(np.asarray([100], dtype=np.int64)),
        cd_history=_readonly(np.asarray([1.0], dtype=np.float32)),
        cl_history=_readonly(np.asarray([0.0], dtype=np.float32)),
    )


class ArrayTensor:
    def __init__(self, value: np.ndarray[Any, Any], *, device: str = "cpu") -> None:
        self.value = np.ascontiguousarray(value)
        self.device = device
        self.dtype = "torch.bool" if self.value.dtype == np.dtype(np.bool_) else "torch.float32"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def is_contiguous(self) -> bool:
        return self.value.flags.c_contiguous

    def isfinite(self) -> ArrayTensor:
        return ArrayTensor(np.isfinite(self.value), device=self.device)

    def all(self) -> ArrayTensor:
        return ArrayTensor(np.asarray(self.value.all(), dtype=np.bool_), device=self.device)

    def item(self) -> object:
        return self.value.item()

    def detach(self) -> Self:
        return self

    def cpu(self) -> ArrayTensor:
        return ArrayTensor(self.value, device="cpu")

    def contiguous(self) -> Self:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.value

    def new_tensor(self, value: np.ndarray[Any, Any]) -> ArrayTensor:
        return ArrayTensor(np.asarray(value), device=self.device)


def prediction_batch(model_design: np.ndarray[Any, Any]) -> PredictionBatch:
    design = np.ascontiguousarray(model_design, dtype=np.float32)
    batch_size = int(design.shape[0])
    return PredictionBatch(
        inputs=ArrayTensor(np.zeros((batch_size, 2, *MODEL_SPATIAL_SHAPE), dtype=np.float32)),
        fluid_mask=ArrayTensor(np.ones((batch_size, 1, *MODEL_SPATIAL_SHAPE), dtype=np.bool_)),
        design_params=ArrayTensor(design),
    )


@dataclass(frozen=True, slots=True)
class FakeMetadata:
    case_id: str
    design_id: str
    split: str
    artifact_digest: str
    cd: float
    case: CaseConfig


@dataclass(frozen=True, slots=True)
class FakeRecord:
    reference: ArtifactRef
    metadata: FakeMetadata


def _record(row: ManifestRow, config: SweepConfig) -> FakeRecord:
    case = CaseConfig(
        shape=ShapeParams(
            aspect_ratio=row.aspect_ratio,
            rotation_deg=row.rotation_deg,
            scale=row.scale,
        ),
        reynolds=row.reynolds,
        nx=config.grid.nx,
        ny=config.grid.ny,
        steps=config.run.steps,
        warmup_steps=config.run.warmup_steps,
        inlet_velocity_lu=config.run.inlet_velocity_lu,
        seed=config.seed,
    )
    assert case.case_id == row.case_id
    reference = ArtifactRef(
        artifact_type="run",
        artifact_id=row.run_digest[:20],
        sha256=row.run_digest,
        size_bytes=row.bytes,
        uri=row.run_uri,
    )
    return FakeRecord(
        reference=reference,
        metadata=FakeMetadata(
            case_id=row.case_id,
            design_id=row.design_id,
            split=row.split,
            artifact_digest=row.run_digest,
            cd=row.cd,
            case=case,
        ),
    )


@dataclass(slots=True)
class TrainingHarness:
    published: PublishedDataset
    records: dict[str, FakeRecord]
    fields: CuratedRunFields
    statistics: PreprocessingStatistics
    opened: list[str] = field(default_factory=list)
    tamper_reference: bool = False
    tamper_metadata: bool = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harness = self

        class FakeDatasetStore:
            def __init__(self, root: Path) -> None:
                assert isinstance(root, Path)

            def open(self, reference: ArtifactRef) -> PublishedDataset:
                assert reference == harness.published.reference
                return harness.published

        class FakeRunStore:
            def __init__(self, root: Path) -> None:
                assert isinstance(root, Path)

            def open_run(self, reference: ArtifactRef) -> RunArtifact:
                harness.opened.append(reference.sha256)
                record = harness.records[reference.sha256]
                artifact_reference = record.reference
                if harness.tamper_reference:
                    artifact_reference = artifact_reference.model_copy(
                        update={"size_bytes": artifact_reference.size_bytes + 1}
                    )
                metadata = record.metadata
                if harness.tamper_metadata:
                    metadata = replace(metadata, cd=metadata.cd + 1.0)
                return RunArtifact(
                    reference=artifact_reference,
                    metadata=cast(RunMetadata, metadata),
                    fields=harness.fields,
                    metadata_sha256="f" * 64,
                )

        monkeypatch.setattr(data_module, "LocalDatasetArtifactStore", FakeDatasetStore)
        monkeypatch.setattr(data_module, "LocalRunArtifactStore", FakeRunStore)

    def open(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> ManifestDataset:
        self.install(monkeypatch)
        return open_manifest_dataset(root, self.published.reference)


def build_harness() -> TrainingHarness:
    config = load_config(SWEEP_CONFIG, SweepConfig)
    manifest = load_manifest(MANIFEST_FIXTURE)
    records = tuple(_record(row, config) for row in manifest.rows)
    reference = ArtifactRef(
        artifact_type="dataset",
        artifact_id=manifest.metadata.dataset_id,
        sha256=manifest.metadata.dataset_sha256,
        size_bytes=manifest.metadata.total_payload_bytes,
        uri=f"datasets/{manifest.metadata.dataset_id}",
    )
    channel = OutputChannelStatistics(
        mean=0.0,
        raw_standard_deviation=1.0,
        standard_deviation=1.0,
        floored=False,
    )
    statistics = PreprocessingStatistics(
        dataset_id=manifest.metadata.dataset_id,
        training_case_count=600,
        training_cell_count=600 * MODEL_SPATIAL_SHAPE[0] * MODEL_SPATIAL_SHAPE[1],
        outputs=OutputNormalizationStatistics(
            u_mean=channel,
            v_mean=channel,
            rho_delta=channel,
        ),
    )
    return TrainingHarness(
        published=PublishedDataset(reference=reference, manifest=manifest),
        records={record.reference.sha256: record for record in records},
        fields=_fields(),
        statistics=statistics,
    )


__all__ = [
    "ArrayTensor",
    "TrainingHarness",
    "build_harness",
    "prediction_batch",
]
