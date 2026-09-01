from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from soufflerie.datagen.run_artifact import RUN_MAX_FILE_BYTES, LocalRunArtifactStore
from soufflerie.errors import ArtifactIntegrityError, SchemaVersionError
from soufflerie.schemas import ArtifactRef, CaseConfig, SolverResult, sha256_bytes

DESIGN_ID = "d" * 20


def _publish(
    store: LocalRunArtifactStore,
    case: CaseConfig,
    result: SolverResult,
    *,
    attempt_id: str = "attempt-1",
) -> ArtifactRef:
    return store.publish_run(
        attempt_id=attempt_id,
        design_id=DESIGN_ID,
        split="train",
        case=case,
        result=result,
    )


def test_publish_open_and_matching_duplicate_are_idempotent(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    store = LocalRunArtifactStore(tmp_path)
    first = _publish(store, run_case, solver_result)
    second = _publish(store, run_case, solver_result, attempt_id="attempt-2")
    loaded = store.open_run(first)

    assert first == second == loaded.reference
    assert first.uri == f"runs/{run_case.case_id}/{first.sha256}"
    assert loaded.metadata.case_id == run_case.case_id
    assert loaded.metadata.fields_sha256 == sha256_bytes(
        (tmp_path / first.uri / "fields.npz").read_bytes()
    )
    marker = (tmp_path / first.uri / "COMMITTED").read_text(encoding="ascii")
    assert marker == loaded.metadata_sha256 + "\n"
    assert loaded.fields.u_mean.shape == (128, 256)


@pytest.mark.parametrize(
    "fault_stage",
    ("fields_written", "metadata_written", "verified", "committed"),
)
def test_fault_before_atomic_rename_never_exposes_partial_run(
    fault_stage: str,
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    def inject(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError(f"forced failure at {stage}")

    store = LocalRunArtifactStore(tmp_path, fault_injector=inject)
    with pytest.raises(RuntimeError, match="forced failure"):
        _publish(store, run_case, solver_result)
    runs = tmp_path / "runs"
    assert not runs.exists() or list(runs.rglob("COMMITTED")) == []


def test_failure_after_rename_leaves_a_complete_recoverable_run(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    def inject(stage: str) -> None:
        if stage == "published":
            raise RuntimeError("forced failure after publish")

    failing = LocalRunArtifactStore(tmp_path, fault_injector=inject)
    with pytest.raises(RuntimeError, match="after publish"):
        _publish(failing, run_case, solver_result)

    recovered = _publish(LocalRunArtifactStore(tmp_path), run_case, solver_result)
    assert LocalRunArtifactStore(tmp_path).open_run(recovered).reference == recovered


def test_matching_concurrent_publisher_wins_without_overwrite(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    winner: ArtifactRef | None = None

    def publish_competitor(stage: str) -> None:
        nonlocal winner
        if stage == "committed":
            winner = _publish(
                LocalRunArtifactStore(tmp_path),
                run_case,
                solver_result,
                attempt_id="competitor",
            )

    returned = _publish(
        LocalRunArtifactStore(tmp_path, fault_injector=publish_competitor),
        run_case,
        solver_result,
    )
    assert winner is not None
    assert returned == winner
    assert LocalRunArtifactStore(tmp_path).open_run(returned).reference == winner


def test_missing_marker_and_digest_tampering_are_rejected(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    store = LocalRunArtifactStore(tmp_path)
    reference = _publish(store, run_case, solver_result)
    run_root = tmp_path / reference.uri
    fields = run_root / "fields.npz"
    fields.write_bytes(fields.read_bytes() + b"tamper")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        store.open_run(reference)

    fresh_root = tmp_path / "fresh"
    fresh_store = LocalRunArtifactStore(fresh_root)
    fresh = _publish(fresh_store, run_case, solver_result)
    (fresh_root / fresh.uri / "COMMITTED").unlink()
    with pytest.raises(ArtifactIntegrityError, match="unable to open"):
        fresh_store.open_run(fresh)


def test_wrong_metadata_schema_and_reference_size_fail_closed(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    store = LocalRunArtifactStore(tmp_path)
    reference = _publish(store, run_case, solver_result)
    with pytest.raises(ArtifactIntegrityError, match="byte count"):
        store.open_run(reference.model_copy(update={"size_bytes": reference.size_bytes + 1}))

    run_root = tmp_path / reference.uri
    metadata_path = run_root / "metadata.json"
    metadata = metadata_path.read_text(encoding="utf-8").replace(
        '"schema_version": 1', '"schema_version": 2', 1
    )
    metadata_path.write_text(metadata, encoding="utf-8")
    (run_root / "COMMITTED").write_text(
        sha256_bytes(metadata.encode("utf-8")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(SchemaVersionError):
        store.open_run(reference)


def test_reader_enforces_run_specific_archive_cap(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    store = LocalRunArtifactStore(tmp_path)
    reference = _publish(store, run_case, solver_result)
    fields_path = tmp_path / reference.uri / "fields.npz"
    with fields_path.open("ab") as handle:
        handle.truncate(RUN_MAX_FILE_BYTES + 1)

    with pytest.raises(ArtifactIntegrityError, match="byte limit"):
        store.open_run(reference)


def test_writer_rejects_symlinked_store_prefix(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    root = tmp_path / "store"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "runs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactIntegrityError, match="not a real directory"):
        _publish(LocalRunArtifactStore(root), run_case, solver_result)
    assert list(outside.iterdir()) == []


def test_logical_run_digest_excludes_attempt_timing_and_gpu_telemetry(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    first_store = LocalRunArtifactStore(tmp_path)
    first = _publish(first_store, run_case, solver_result)
    provenance = solver_result.provenance.model_copy(
        update={
            "started_at": solver_result.provenance.started_at + timedelta(hours=1),
            "completed_at": solver_result.provenance.completed_at + timedelta(hours=1),
            "gpu_seconds": 1.75,
        }
    )
    rerun = SolverResult(
        case_id=solver_result.case_id,
        fields=solver_result.fields,
        cd=solver_result.cd,
        cl_mean=solver_result.cl_mean,
        strouhal=solver_result.strouhal,
        force_steps=solver_result.force_steps,
        cd_history=solver_result.cd_history,
        cl_history=solver_result.cl_history,
        diagnostics=solver_result.diagnostics,
        provenance=provenance,
    )
    second = _publish(first_store, run_case, rerun, attempt_id="attempt-2")
    assert first == second


def test_store_rejects_invalid_ids_and_non_run_references(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    store = LocalRunArtifactStore(tmp_path)
    with pytest.raises(ArtifactIntegrityError, match="attempt_id"):
        _publish(store, run_case, solver_result, attempt_id="../escape")
    with pytest.raises(ValidationError):
        ArtifactRef(
            artifact_type="run",
            artifact_id="a" * 20,
            sha256="a" * 64,
            size_bytes=0,
            uri="../escape",
        )
    not_run = ArtifactRef(
        artifact_type="report",
        artifact_id="a" * 20,
        sha256="a" * 64,
        size_bytes=0,
        uri="reports/a",
    )
    with pytest.raises(ArtifactIntegrityError, match="run ArtifactRef"):
        store.open_run(not_run)
