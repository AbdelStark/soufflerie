from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from soufflerie.datagen import (
    LEASE_DURATION,
    CaseState,
    LocalSweepStateStore,
    RunArtifactStore,
)
from soufflerie.errors import (
    ArtifactIntegrityError,
    ConfigurationError,
    InternalInvariantError,
    RemoteExecutionError,
)
from soufflerie.schemas import ArtifactRef

CASE_ID = "a" * 20
SWEEP_DIGEST = "b" * 64
RUN_DIGEST = "c" * 64
STARTED = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


class _Verifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def verify_run(self, *, case_id: str, run_digest: str) -> ArtifactRef:
        self.calls.append((case_id, run_digest))
        return ArtifactRef(
            artifact_type="run",
            artifact_id=run_digest[:20],
            sha256=run_digest,
            size_bytes=1,
            uri=f"runs/{case_id}/{run_digest}",
        )


def _store(tmp_path: Path) -> LocalSweepStateStore:
    return LocalSweepStateStore(tmp_path, sweep_digest=SWEEP_DIGEST)


def test_case_state_schema_rejects_incoherent_states() -> None:
    with pytest.raises(ValidationError, match="succeeded state"):
        CaseState(
            sweep_digest=SWEEP_DIGEST,
            case_id=CASE_ID,
            state="succeeded",
            revision=1,
            attempt=1,
            attempt_id="attempt-1",
            updated_at=STARTED,
        )
    with pytest.raises(ValidationError, match="complete lease"):
        CaseState(
            sweep_digest=SWEEP_DIGEST,
            case_id=CASE_ID,
            state="running",
            revision=1,
            attempt=1,
            attempt_id="attempt-1",
            lease_owner="worker-1",
            updated_at=STARTED,
        )


def test_claim_and_renew_are_lease_fenced_and_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    initial = store.initialize_case(CASE_ID, now=STARTED)
    claimed = store.claim_case(
        CASE_ID,
        attempt_id="attempt-1",
        lease_owner="worker-1",
        now=STARTED,
    )
    assert claimed is not None
    assert initial.state == "pending" and initial.revision == 0
    assert claimed.state == "running" and claimed.attempt == 1
    assert claimed.lease_expires_at == STARTED + LEASE_DURATION

    duplicate = store.claim_case(
        CASE_ID,
        attempt_id="attempt-1",
        lease_owner="worker-1",
        now=STARTED + timedelta(minutes=1),
    )
    assert duplicate == claimed
    assert (
        store.claim_case(
            CASE_ID,
            attempt_id="attempt-2",
            lease_owner="worker-2",
            now=STARTED + timedelta(minutes=1),
        )
        is None
    )

    renewed_at = STARTED + timedelta(minutes=5)
    renewed = store.renew_lease(
        CASE_ID,
        attempt_id="attempt-1",
        lease_owner="worker-1",
        now=renewed_at,
    )
    assert renewed.revision == claimed.revision + 1
    assert renewed.lease_expires_at == renewed_at + LEASE_DURATION
    with pytest.raises(InternalInvariantError, match="active lease"):
        store.renew_lease(
            CASE_ID,
            attempt_id="attempt-1",
            lease_owner="worker-2",
            now=renewed_at + timedelta(minutes=1),
        )


def test_expired_leases_are_reclaimed_and_third_expiry_is_terminal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize_case(CASE_ID, now=STARTED)
    first = store.claim_case(
        CASE_ID,
        attempt_id="attempt-1",
        lease_owner="worker-1",
        now=STARTED,
    )
    assert first is not None
    second = store.claim_case(
        CASE_ID,
        attempt_id="attempt-2",
        lease_owner="worker-2",
        now=STARTED + timedelta(minutes=10),
    )
    assert second is not None and second.attempt == 2
    third = store.claim_case(
        CASE_ID,
        attempt_id="attempt-3",
        lease_owner="worker-3",
        now=STARTED + timedelta(minutes=20),
    )
    assert third is not None and third.attempt == 3

    terminal = store.reap_expired(CASE_ID, now=STARTED + timedelta(minutes=30))
    assert terminal.state == "failed"
    assert terminal.error_code == "LEASE_EXPIRED"
    assert terminal.failure_codes == ("LEASE_EXPIRED",) * 3
    assert (
        store.claim_case(
            CASE_ID,
            attempt_id="attempt-4",
            lease_owner="worker-4",
            now=STARTED + timedelta(minutes=31),
        )
        is None
    )


def test_failure_policy_retries_only_transient_errors_and_never_more_than_three(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize_case(CASE_ID, now=STARTED)
    first = store.claim_case(
        CASE_ID,
        attempt_id="attempt-1",
        lease_owner="worker-1",
        now=STARTED,
    )
    assert first is not None
    transient = RemoteExecutionError("preempted")
    retry = store.fail_case(
        CASE_ID,
        attempt_id="attempt-1",
        lease_owner="worker-1",
        error=transient,
        now=STARTED + timedelta(minutes=1),
    )
    assert retry.state == "pending" and retry.error_code == "REMOTE_EXECUTION"
    assert retry.failure_codes == ("REMOTE_EXECUTION",)
    duplicate = store.fail_case(
        CASE_ID,
        attempt_id="attempt-1",
        lease_owner="worker-1",
        error=transient,
        now=STARTED + timedelta(minutes=2),
    )
    assert duplicate == retry

    second = store.claim_case(
        CASE_ID,
        attempt_id="attempt-2",
        lease_owner="worker-2",
        now=STARTED + timedelta(minutes=2),
    )
    assert second is not None
    deterministic = store.fail_case(
        CASE_ID,
        attempt_id="attempt-2",
        lease_owner="worker-2",
        error=ConfigurationError("bad configuration", retryable=True),
        now=STARTED + timedelta(minutes=3),
    )
    assert deterministic.state == "failed"
    assert deterministic.error_code == "CONFIG_INVALID"
    assert deterministic.failure_codes == ("REMOTE_EXECUTION", "CONFIG_INVALID")

    exhausted_case = "d" * 20
    store.initialize_case(exhausted_case, now=STARTED)
    final: CaseState | None = None
    for number in range(1, 4):
        owner = f"worker-{number}"
        attempt_id = f"retry-{number}"
        claim = store.claim_case(
            exhausted_case,
            attempt_id=attempt_id,
            lease_owner=owner,
            now=STARTED + timedelta(minutes=number),
        )
        assert claim is not None
        final = store.fail_case(
            exhausted_case,
            attempt_id=attempt_id,
            lease_owner=owner,
            error=RemoteExecutionError("transient"),
            now=STARTED + timedelta(minutes=number, seconds=1),
        )
    assert final is not None and final.state == "failed" and final.attempt == 3
    assert final.failure_codes == ("REMOTE_EXECUTION",) * 3


def test_success_is_verified_immutable_and_matching_duplicates_are_noops(tmp_path: Path) -> None:
    store = _store(tmp_path)
    verifier = _Verifier()
    artifact_store = cast(RunArtifactStore, verifier)
    store.initialize_case(CASE_ID, now=STARTED)
    with pytest.raises(InternalInvariantError, match="active lease"):
        store.succeed_case(
            CASE_ID,
            attempt_id="attempt-1",
            lease_owner="worker-1",
            run_digest=RUN_DIGEST,
            artifact_store=artifact_store,
            now=STARTED,
        )
    assert verifier.calls == []
    store.claim_case(
        CASE_ID,
        attempt_id="attempt-1",
        lease_owner="worker-1",
        now=STARTED,
    )
    succeeded = store.succeed_case(
        CASE_ID,
        attempt_id="attempt-1",
        lease_owner="worker-1",
        run_digest=RUN_DIGEST,
        artifact_store=artifact_store,
        now=STARTED + timedelta(minutes=1),
    )
    assert succeeded.state == "succeeded" and succeeded.run_digest == RUN_DIGEST

    duplicate = store.succeed_case(
        CASE_ID,
        attempt_id="duplicate",
        lease_owner="other-worker",
        run_digest=RUN_DIGEST,
        artifact_store=artifact_store,
        now=STARTED + timedelta(minutes=2),
    )
    assert duplicate == succeeded
    assert verifier.calls == [(CASE_ID, RUN_DIGEST), (CASE_ID, RUN_DIGEST)]
    with pytest.raises(ArtifactIntegrityError, match="DIVERGENCE"):
        store.succeed_case(
            CASE_ID,
            attempt_id="duplicate",
            lease_owner="other-worker",
            run_digest="e" * 64,
            artifact_store=artifact_store,
            now=STARTED + timedelta(minutes=2),
        )


def test_atomic_faults_leave_the_old_or_complete_new_revision(tmp_path: Path) -> None:
    base = _store(tmp_path)
    initial = base.initialize_case(CASE_ID, now=STARTED)

    def fail_before_replace(stage: str) -> None:
        if stage == "state_staged":
            raise RuntimeError("before replace")

    before = LocalSweepStateStore(
        tmp_path,
        sweep_digest=SWEEP_DIGEST,
        fault_injector=fail_before_replace,
    )
    with pytest.raises(RuntimeError, match="before replace"):
        before.claim_case(
            CASE_ID,
            attempt_id="attempt-1",
            lease_owner="worker-1",
            now=STARTED,
        )
    assert base.read_case(CASE_ID) == initial

    def fail_after_replace(stage: str) -> None:
        if stage == "state_published":
            raise RuntimeError("after replace")

    after = LocalSweepStateStore(
        tmp_path,
        sweep_digest=SWEEP_DIGEST,
        fault_injector=fail_after_replace,
    )
    with pytest.raises(RuntimeError, match="after replace"):
        after.claim_case(
            CASE_ID,
            attempt_id="attempt-1",
            lease_owner="worker-1",
            now=STARTED,
        )
    recovered = base.read_case(CASE_ID)
    assert recovered.state == "running" and recovered.attempt_id == "attempt-1"


def test_case_lock_serializes_competing_claims(tmp_path: Path) -> None:
    first_store = _store(tmp_path)
    second_store = _store(tmp_path)
    first_store.initialize_case(CASE_ID, now=STARTED)

    def claim(store: LocalSweepStateStore, attempt_id: str) -> CaseState | None:
        return store.claim_case(
            CASE_ID,
            attempt_id=attempt_id,
            lease_owner=attempt_id,
            now=STARTED,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(
            pool.map(
                lambda item: claim(*item),
                ((first_store, "worker-1"), (second_store, "worker-2")),
            )
        )
    assert sum(outcome is not None for outcome in outcomes) == 1
    assert first_store.read_case(CASE_ID).attempt == 1


def test_store_rejects_symlinked_state_prefix(tmp_path: Path) -> None:
    root = tmp_path / "store"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "sweeps").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="not a real directory"):
        LocalSweepStateStore(root, sweep_digest=SWEEP_DIGEST)
    assert list(outside.iterdir()) == []


def test_state_cannot_be_swapped_between_sweep_roots(tmp_path: Path) -> None:
    source = _store(tmp_path)
    source.initialize_case(CASE_ID, now=STARTED)
    other_digest = "f" * 64
    target = LocalSweepStateStore(tmp_path, sweep_digest=other_digest)
    source_path = tmp_path / "sweeps" / SWEEP_DIGEST / "state" / f"{CASE_ID}.json"
    target_path = tmp_path / "sweeps" / other_digest / "state" / f"{CASE_ID}.json"
    target_path.write_bytes(source_path.read_bytes())

    with pytest.raises(ArtifactIntegrityError, match="another sweep"):
        target.read_case(CASE_ID)
