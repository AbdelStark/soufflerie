from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import soufflerie.datagen.manifest as manifest_module
from soufflerie.datagen.manifest import (
    DATASET_PAYLOAD_LIMIT_BYTES,
    MANIFEST_COLUMN_TYPES,
    MANIFEST_ROW_GROUP_SIZE,
    MANIFEST_SCHEMA_SHA256,
    DatasetManifest,
    LocalDatasetArtifactStore,
    ManifestRow,
    VerifiedRunRecord,
    build_manifest,
    load_manifest,
)
from soufflerie.datagen.run_artifact import RunArtifact
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ArtifactRef, canonical_sha256
from tests.datagen.manifest_helpers import (
    canonical_config,
    canonical_points,
    synthetic_manifest,
    synthetic_verified_runs,
)

FIXTURE = Path("tests/fixtures/dataset/manifest.parquet")


@pytest.fixture(scope="module")
def manifest() -> DatasetManifest:
    return synthetic_manifest()


def test_manifest_has_fixed_schema_order_groups_identity_and_statistics(
    manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.parquet"
    path.write_bytes(manifest.parquet_bytes)
    loaded = load_manifest(path)
    reader = pq.ParquetFile(pa.BufferReader(manifest.parquet_bytes))

    assert loaded == manifest
    assert tuple(reader.schema_arrow.names) == tuple(MANIFEST_COLUMN_TYPES)
    assert tuple(
        reader.metadata.row_group(index).num_rows for index in range(reader.metadata.num_row_groups)
    ) == (256, 256, 256, 232)
    assert MANIFEST_ROW_GROUP_SIZE == 256
    assert manifest.metadata.manifest_schema_sha256 == MANIFEST_SCHEMA_SHA256
    assert manifest.metadata.dataset_id == "83d400f135848978b152"
    assert (
        manifest.metadata.dataset_sha256
        == "83d400f135848978b1525535a554c3e769f6b9015eef145bc2d6893d95ea889c"
    )
    assert len(manifest.rows) == 1_000
    assert manifest.metadata.split_counts.model_dump() == {
        "train": 600,
        "validation": 200,
        "test": 200,
    }
    assert manifest.metadata.total_payload_bytes == 1_000_499_500
    assert manifest.statistics.columns.strouhal.count == 900
    assert manifest.statistics.columns.strouhal.null_count == 100


def test_builder_opens_every_explicit_parent_and_never_lists_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = canonical_config()
    points = canonical_points(config)
    records = synthetic_verified_runs(config, points)
    by_digest = {record.reference.sha256: record for record in records}
    opened: list[str] = []

    class FakeRunStore:
        def __init__(self, root: Path) -> None:
            assert root == Path("/artifact-root")

        def open_run(self, reference: ArtifactRef) -> RunArtifact:
            opened.append(reference.sha256)
            record = by_digest[reference.sha256]
            return RunArtifact(
                reference=record.reference,
                metadata=record.metadata,
                fields=None,  # type: ignore[arg-type]  # Parent was verified by this fake boundary.
                metadata_sha256="f" * 64,
            )

    monkeypatch.setattr(manifest_module, "LocalRunArtifactStore", FakeRunStore)
    monkeypatch.setattr(manifest_module, "sample_design", lambda config: points)
    built = build_manifest(
        Path("/artifact-root"),
        config=config,
        run_references=tuple(record.reference for record in reversed(records)),
    )

    assert len(opened) == 1_000
    assert set(opened) == set(by_digest)
    assert built.metadata.dataset_id == "83d400f135848978b152"
    assert tuple(row.design_id for row in built.rows) == tuple(
        sorted(point.design_id for point in points)
    )


def test_logical_dataset_identity_is_equal_across_physical_roots(
    manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    first_store = LocalDatasetArtifactStore(tmp_path / "first")
    second_store = LocalDatasetArtifactStore(tmp_path / "second")
    first = first_store.publish(manifest)
    second = second_store.publish(manifest)

    assert first == second
    assert first_store.open(first).manifest == second_store.open(second).manifest
    assert first.uri == f"datasets/{manifest.metadata.dataset_id}"
    assert first.sha256 == manifest.metadata.dataset_sha256


def test_builder_rejects_duplicates_wrong_split_provenance_and_payload_size() -> None:
    config = canonical_config()
    points = canonical_points(config)
    records = synthetic_verified_runs(config, points)

    with pytest.raises(ArtifactIntegrityError, match="duplicate design"):
        manifest_module._assemble_manifest(
            (*records[:-1], records[-2]),
            config=config,
            points=points,
        )

    wrong_split = "test" if records[-1].metadata.split != "test" else "train"
    draft_split_metadata = records[-1].metadata.model_copy(
        update={"split": wrong_split, "artifact_digest": "0" * 64}
    )
    changed_split_metadata = records[-1].metadata.model_validate(
        draft_split_metadata.model_dump(mode="python")
        | {"artifact_digest": canonical_sha256(draft_split_metadata.logical_identity())}
    )
    changed_split_reference = ArtifactRef(
        artifact_type="run",
        artifact_id=changed_split_metadata.artifact_digest[:20],
        sha256=changed_split_metadata.artifact_digest,
        size_bytes=records[-1].reference.size_bytes,
        uri=(f"runs/{changed_split_metadata.case_id}/{changed_split_metadata.artifact_digest}"),
    )
    with pytest.raises(ArtifactIntegrityError, match="run split"):
        manifest_module._assemble_manifest(
            (
                *records[:-1],
                VerifiedRunRecord(changed_split_reference, changed_split_metadata),
            ),
            config=config,
            points=points,
        )

    mismatched_source = records[-1].metadata.provenance.model_copy(
        update={"source_revision": "c" * 40}
    )
    draft_metadata = records[-1].metadata.model_copy(
        update={"provenance": mismatched_source, "artifact_digest": "0" * 64}
    )
    changed_metadata = records[-1].metadata.model_validate(
        draft_metadata.model_dump(mode="python")
        | {"artifact_digest": canonical_sha256(draft_metadata.logical_identity())}
    )
    changed_reference = ArtifactRef(
        artifact_type="run",
        artifact_id=changed_metadata.artifact_digest[:20],
        sha256=changed_metadata.artifact_digest,
        size_bytes=records[-1].reference.size_bytes,
        uri=(f"runs/{changed_metadata.case_id}/{changed_metadata.artifact_digest}"),
    )
    with pytest.raises(ArtifactIntegrityError, match="source revision"):
        manifest_module._assemble_manifest(
            (*records[:-1], VerifiedRunRecord(changed_reference, changed_metadata)),
            config=config,
            points=points,
        )

    oversized = tuple(
        replace(
            record,
            reference=record.reference.model_copy(update={"size_bytes": 3_000_000}),
        )
        for record in records
    )
    with pytest.raises(ArtifactIntegrityError, match="below 2 GiB"):
        manifest_module._assemble_manifest(oversized, config=config, points=points)

    with pytest.raises(ArtifactIntegrityError, match="exactly 1,000"):
        build_manifest(Path("/unused"), config=config, run_references=())
    duplicate_references = tuple(record.reference for record in (*records[:-1], records[-2]))
    with pytest.raises(ArtifactIntegrityError, match="must be unique"):
        build_manifest(Path("/unused"), config=config, run_references=duplicate_references)


def test_loader_rejects_row_reordering_dataset_tampering_and_wrong_groups(
    manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    reader = pq.ParquetFile(pa.BufferReader(manifest.parquet_bytes))
    metadata = reader.schema_arrow.metadata

    reversed_path = tmp_path / "reversed.parquet"
    reversed_path.write_bytes(
        manifest_module._encode_parquet(tuple(reversed(manifest.rows)), metadata=metadata or {})
    )
    with pytest.raises(ArtifactIntegrityError, match="rows must sort"):
        load_manifest(reversed_path)

    tampered_rows = (
        manifest.rows[0].model_copy(update={"dataset_id": "f" * 20}),
        *manifest.rows[1:],
    )
    tampered_path = tmp_path / "tampered.parquet"
    tampered_path.write_bytes(
        manifest_module._encode_parquet(tampered_rows, metadata=metadata or {})
    )
    with pytest.raises(ArtifactIntegrityError, match="share one dataset ID"):
        load_manifest(tampered_path)

    table = pq.read_table(pa.BufferReader(manifest.parquet_bytes), use_threads=False)
    wrong_groups = tmp_path / "wrong-groups.parquet"
    pq.write_table(table, wrong_groups, row_group_size=500, compression="zstd")
    with pytest.raises(ArtifactIntegrityError, match="row groups"):
        load_manifest(wrong_groups)


@pytest.mark.parametrize("fault_stage", ("members_written", "verified", "committed"))
def test_dataset_publication_hides_pre_rename_failures(
    fault_stage: str,
    manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    def inject(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError(f"forced {stage}")

    store = LocalDatasetArtifactStore(tmp_path, fault_injector=inject)
    with pytest.raises(RuntimeError, match="forced"):
        store.publish(manifest)
    datasets = tmp_path / "datasets"
    assert not datasets.exists() or list(datasets.rglob("COMMITTED")) == []


def test_dataset_publication_rejects_tampered_member(
    manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    store = LocalDatasetArtifactStore(tmp_path)
    reference = store.publish(manifest)
    path = tmp_path / reference.uri / "manifest.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        store.open(reference)


def test_checked_fixture_is_the_canonical_synthetic_contract(manifest: DatasetManifest) -> None:
    assert FIXTURE.read_bytes() == manifest.parquet_bytes
    loaded = load_manifest(FIXTURE)
    assert loaded.metadata.dataset_id == "83d400f135848978b152"
    assert loaded.metadata.total_payload_bytes < DATASET_PAYLOAD_LIMIT_BYTES


def test_manifest_row_requires_canonical_relative_run_uri(manifest: DatasetManifest) -> None:
    payload = manifest.rows[0].model_dump(mode="python")
    with pytest.raises(ValueError, match="run_uri"):
        ManifestRow.model_validate(payload | {"run_uri": "/absolute/run"})
