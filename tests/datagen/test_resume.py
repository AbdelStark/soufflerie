from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from soufflerie.datagen import LocalRunArtifactStore, LocalSweepStateStore, build_resume_plan
from soufflerie.errors import ArtifactIntegrityError, ConfigurationError
from soufflerie.schemas import CaseConfig, SolverResult

SWEEP_DIGEST = "f" * 64
DESIGN_ID = "d" * 20
NOW = datetime(2026, 9, 1, 13, 0, tzinfo=UTC)


def _claim(
    store: LocalSweepStateStore,
    case_id: str,
    *,
    attempt_id: str,
    owner: str,
    now: datetime,
) -> None:
    claimed = store.claim_case(
        case_id,
        attempt_id=attempt_id,
        lease_owner=owner,
        now=now,
    )
    assert claimed is not None


def test_resume_reaps_expiry_rehashes_success_and_partitions_only_expected_cases(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    states = LocalSweepStateStore(tmp_path, sweep_digest=SWEEP_DIGEST)
    artifacts = LocalRunArtifactStore(tmp_path)
    succeeded_id = run_case.case_id
    pending_id = "1" * 20
    active_id = "2" * 20
    expired_id = "3" * 20
    failed_id = "4" * 20
    stray_id = "5" * 20
    expected = (succeeded_id, pending_id, active_id, expired_id, failed_id)
    for case_id in (*expected, stray_id):
        states.initialize_case(case_id, now=NOW - timedelta(minutes=20))

    _claim(
        states,
        succeeded_id,
        attempt_id="success-attempt",
        owner="worker-success",
        now=NOW - timedelta(minutes=2),
    )
    reference = artifacts.publish_run(
        attempt_id="success-attempt",
        design_id=DESIGN_ID,
        split="train",
        case=run_case,
        result=solver_result,
    )
    states.succeed_case(
        succeeded_id,
        attempt_id="success-attempt",
        lease_owner="worker-success",
        run_digest=reference.sha256,
        artifact_store=artifacts,
        now=NOW - timedelta(minutes=1),
    )

    _claim(
        states,
        active_id,
        attempt_id="active-attempt",
        owner="worker-active",
        now=NOW - timedelta(minutes=5),
    )
    _claim(
        states,
        expired_id,
        attempt_id="expired-attempt",
        owner="worker-expired",
        now=NOW - timedelta(minutes=11),
    )
    _claim(
        states,
        failed_id,
        attempt_id="failed-attempt",
        owner="worker-failed",
        now=NOW - timedelta(minutes=2),
    )
    states.fail_case(
        failed_id,
        attempt_id="failed-attempt",
        lease_owner="worker-failed",
        error=ConfigurationError("invalid case"),
        now=NOW - timedelta(minutes=1),
    )

    plan = build_resume_plan(
        case_ids=reversed(expected),
        state_store=states,
        artifact_store=artifacts,
        now=NOW,
    )
    assert plan.claimable_case_ids == tuple(sorted((pending_id, expired_id)))
    assert plan.active_case_ids == (active_id,)
    assert plan.failed_case_ids == (failed_id,)
    assert tuple(item.case_id for item in plan.succeeded_runs) == (succeeded_id,)
    assert plan.succeeded_runs[0].reference == reference
    assert states.read_case(expired_id).error_code == "LEASE_EXPIRED"
    assert states.read_case(stray_id).state == "pending"


def test_resume_fails_closed_when_a_successful_run_no_longer_verifies(
    tmp_path: Path,
    run_case: CaseConfig,
    solver_result: SolverResult,
) -> None:
    states = LocalSweepStateStore(tmp_path, sweep_digest=SWEEP_DIGEST)
    artifacts = LocalRunArtifactStore(tmp_path)
    states.initialize_case(run_case.case_id, now=NOW - timedelta(minutes=2))
    _claim(
        states,
        run_case.case_id,
        attempt_id="success-attempt",
        owner="worker-success",
        now=NOW - timedelta(minutes=2),
    )
    reference = artifacts.publish_run(
        attempt_id="success-attempt",
        design_id=DESIGN_ID,
        split="train",
        case=run_case,
        result=solver_result,
    )
    states.succeed_case(
        run_case.case_id,
        attempt_id="success-attempt",
        lease_owner="worker-success",
        run_digest=reference.sha256,
        artifact_store=artifacts,
        now=NOW - timedelta(minutes=1),
    )
    fields = tmp_path / reference.uri / "fields.npz"
    fields.write_bytes(fields.read_bytes() + b"tamper")

    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        build_resume_plan(
            case_ids=(run_case.case_id,),
            state_store=states,
            artifact_store=artifacts,
            now=NOW,
        )
