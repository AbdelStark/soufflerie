from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import soufflerie.datagen.design as design_module
import soufflerie.datagen.manifest as manifest_module
from soufflerie.config import SweepConfig, load_config
from soufflerie.datagen.design import DesignPoint, assign_splits, case_config_for_point
from soufflerie.datagen.manifest import DatasetManifest, VerifiedRunRecord
from soufflerie.datagen.run_artifact import (
    QuantizationStatistic,
    RunMetadata,
    run_member_descriptors,
)
from soufflerie.schemas import (
    ArtifactRef,
    Provenance,
    SolverDiagnostics,
    canonical_sha256,
    sha256_bytes,
)

CONFIG_PATH = Path("configs/sweeps/mvp-v1.yaml")


def canonical_config() -> SweepConfig:
    return load_config(CONFIG_PATH, SweepConfig)


def canonical_points(config: SweepConfig) -> tuple[DesignPoint, ...]:
    selection = design_module._select_candidate(config)
    unsplit = design_module._unsplit_points(config, selection.normalized)
    return assign_splits(unsplit, config.seed)


def synthetic_verified_runs(
    config: SweepConfig,
    points: tuple[DesignPoint, ...],
) -> tuple[VerifiedRunRecord, ...]:
    started = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)
    diagnostics = SolverDiagnostics(
        steps_completed=config.run.steps,
        sample_count=1,
        initial_mass=100.0,
        final_mass=100.01,
        mass_drift_ratio=0.0001,
        min_rho=0.99,
        max_rho=1.01,
        max_speed_lu=0.06,
        converged=True,
        valid=True,
        messages=(),
    )
    descriptors = run_member_descriptors(1)
    quantization = {
        name: QuantizationStatistic(max_abs_error=0.0, mean_abs_error=0.0)
        for name in ("u_mean", "v_mean", "rho_mean", "sdf")
    }
    records: list[VerifiedRunRecord] = []
    for index, point in enumerate(points):
        case = case_config_for_point(point, config)
        provenance = Provenance(
            source_revision="a" * 40,
            source_dirty=False,
            python_version="3.11.14",
            lock_sha256="b" * 64,
            packages={"numpy": "2.2.6", "soufflerie": "0.1.0"},
            os="linux",
            architecture="x86_64",
            device_class="synthetic-fixture",
            dtype_policy="solver-fp32-storage-fp16",
            config_sha256=case.sha256,
            parent_sha256={},
            seeds=(config.seed,),
            deterministic=True,
            started_at=started,
            completed_at=started + timedelta(seconds=1),
            gpu_seconds=1.0,
        )
        fields_sha256 = sha256_bytes(f"synthetic-fields-{index}".encode())
        draft = RunMetadata.model_construct(
            schema_version=1,
            case_id=case.case_id,
            design_id=point.design_id,
            split=point.split,
            case=case,
            cd=1.2 + index / 100_000.0,
            cl_mean=-0.02 + index / 1_000_000.0,
            strouhal=None if index % 10 == 0 else 0.15 + index / 100_000.0,
            diagnostics=diagnostics,
            field_members=descriptors,
            quantization=quantization,
            provenance=provenance,
            fields_sha256=fields_sha256,
            artifact_digest="0" * 64,
        )
        metadata = RunMetadata.model_validate(
            draft.model_dump(mode="python")
            | {"artifact_digest": canonical_sha256(draft.logical_identity())}
        )
        reference = ArtifactRef(
            artifact_type="run",
            artifact_id=metadata.artifact_digest[:20],
            sha256=metadata.artifact_digest,
            size_bytes=1_000_000 + index,
            uri=f"runs/{case.case_id}/{metadata.artifact_digest}",
        )
        records.append(VerifiedRunRecord(reference=reference, metadata=metadata))
    return tuple(records)


def synthetic_manifest() -> DatasetManifest:
    config = canonical_config()
    points = canonical_points(config)
    runs = synthetic_verified_runs(config, points)
    return manifest_module._assemble_manifest(runs, config=config, points=points)


__all__ = [
    "canonical_config",
    "canonical_points",
    "synthetic_manifest",
    "synthetic_verified_runs",
]
