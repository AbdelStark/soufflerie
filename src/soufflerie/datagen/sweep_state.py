"""Lease-fenced, resumable sweep state with durable local compare-before-write updates."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Protocol, Self

from pydantic import Field, StringConstraints, model_validator

from soufflerie.artifacts import DEFAULT_READER_LIMITS, ReaderLimits, safe_read_json
from soufflerie.datagen._local_files import ensure_real_directory, fsync_directory
from soufflerie.datagen.run_artifact import RunArtifactStore
from soufflerie.errors import ArtifactIntegrityError, InternalInvariantError, SoufflerieError
from soufflerie.schemas import ArtifactRef, ContentId, RunState, Sha256, VersionedModel

LEASE_DURATION = timedelta(minutes=10)
MAX_SWEEP_ATTEMPTS = 3
STATE_MAX_BYTES = 64 * 1024
LEASE_EXPIRED_CODE = "LEASE_EXPIRED"
RETRYABLE_SWEEP_ERROR_CODES = frozenset({"CAPACITY_EXHAUSTED", "REMOTE_EXECUTION"})

AttemptId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
LeaseOwner = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")]
ErrorCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]

_CONTENT_ID_PATTERN = re.compile(r"^[0-9a-f]{20}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_LEASE_OWNER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


def _canonical_time(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("sweep timestamps must be datetime instances")
    if value.utcoffset() is None:
        raise ValueError("sweep timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _lock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


class CaseState(VersionedModel):
    """One durable, revisioned case state with an attempt-token lease fence."""

    sweep_digest: Sha256
    case_id: ContentId
    state: RunState
    revision: int = Field(ge=0)
    attempt: int = Field(ge=0, le=MAX_SWEEP_ATTEMPTS)
    attempt_id: AttemptId | None = None
    lease_owner: LeaseOwner | None = None
    lease_expires_at: datetime | None = None
    run_digest: Sha256 | None = None
    error_code: ErrorCode | None = None
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_timestamps(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("lease_expires_at", "updated_at"):
            timestamp = normalized.get(name)
            if isinstance(timestamp, str):
                with suppress(ValueError):
                    normalized[name] = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return normalized

    @model_validator(mode="after")
    def _state_is_coherent(self) -> Self:
        if self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        if self.lease_expires_at is not None and self.lease_expires_at.utcoffset() is None:
            raise ValueError("lease_expires_at must be timezone-aware")

        leased = self.lease_owner is not None or self.lease_expires_at is not None
        if self.state == "running":
            if self.attempt < 1 or self.attempt_id is None:
                raise ValueError("running state requires an active attempt")
            if self.lease_owner is None or self.lease_expires_at is None:
                raise ValueError("running state requires a complete lease")
            if self.lease_expires_at <= self.updated_at:
                raise ValueError("running lease must expire after updated_at")
            if self.run_digest is not None or self.error_code is not None:
                raise ValueError("running state cannot contain a result or error")
            return self

        if leased:
            raise ValueError("only running state may contain lease fields")
        if self.state == "pending":
            if self.run_digest is not None:
                raise ValueError("pending state cannot contain a run digest")
            if self.attempt == 0:
                if self.attempt_id is not None or self.error_code is not None:
                    raise ValueError("initial pending state cannot contain attempt evidence")
            elif self.attempt >= MAX_SWEEP_ATTEMPTS:
                raise ValueError("retryable pending state must have another attempt available")
            elif self.attempt_id is None or self.error_code is None:
                raise ValueError("retryable pending state requires prior attempt evidence")
            return self

        if self.attempt < 1 or self.attempt_id is None:
            raise ValueError("terminal state requires attempt evidence")
        if self.state == "succeeded":
            if self.run_digest is None or self.error_code is not None:
                raise ValueError("succeeded state requires only a run digest")
        elif self.state == "failed" and (self.error_code is None or self.run_digest is not None):
            raise ValueError("failed state requires only an error code")
        return self


def _state_bytes(state: CaseState) -> bytes:
    payload = state.model_dump(mode="json")
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _transition(current: CaseState, *, now: datetime, **changes: object) -> CaseState:
    timestamp = _canonical_time(now)
    if timestamp < current.updated_at.astimezone(UTC):
        raise InternalInvariantError("state transition timestamp precedes the current revision")
    payload = current.model_dump(mode="python")
    payload.update(changes)
    payload["revision"] = current.revision + 1
    payload["updated_at"] = timestamp
    return CaseState.model_validate(payload)


class SweepStateStore(Protocol):
    """State multiplicity boundary required by local and remote sweep adapters."""

    def initialize_case(self, case_id: str, *, now: datetime) -> CaseState: ...

    def read_case(self, case_id: str) -> CaseState: ...

    def claim_case(
        self,
        case_id: str,
        *,
        attempt_id: str,
        lease_owner: str,
        now: datetime,
    ) -> CaseState | None: ...

    def renew_lease(
        self,
        case_id: str,
        *,
        attempt_id: str,
        lease_owner: str,
        now: datetime,
    ) -> CaseState: ...

    def succeed_case(
        self,
        case_id: str,
        *,
        attempt_id: str,
        lease_owner: str,
        run_digest: str,
        artifact_store: RunArtifactStore,
        now: datetime,
    ) -> CaseState: ...

    def fail_case(
        self,
        case_id: str,
        *,
        attempt_id: str,
        lease_owner: str,
        error: SoufflerieError,
        now: datetime,
    ) -> CaseState: ...

    def reap_expired(self, case_id: str, *, now: datetime) -> CaseState: ...


class LocalSweepStateStore:
    """JSON state adapter serialized by per-case advisory locks and atomic replace."""

    def __init__(
        self,
        root: Path,
        *,
        sweep_digest: str,
        limits: ReaderLimits = DEFAULT_READER_LIMITS,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self._validate("sweep_digest", sweep_digest, _SHA256_PATTERN)
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sweep_digest = sweep_digest
        sweep_root = ensure_real_directory(self.root, "sweeps", sweep_digest)
        self._state_root = ensure_real_directory(sweep_root, "state")
        self._lock_root = ensure_real_directory(sweep_root, "locks")
        self._limits = limits.model_copy(
            update={
                "max_file_bytes": min(limits.max_file_bytes, STATE_MAX_BYTES),
                "max_json_bytes": min(limits.max_json_bytes, STATE_MAX_BYTES),
            }
        )
        self._fault_injector = fault_injector

    @staticmethod
    def _validate(name: str, value: str, pattern: re.Pattern[str]) -> None:
        if pattern.fullmatch(value) is None:
            raise ArtifactIntegrityError(f"SWEEP-1 IDENTITY: invalid {name}")

    def _inject(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)

    def _state_path(self, case_id: str) -> Path:
        return self._state_root / f"{case_id}.json"

    @contextmanager
    def _case_lock(self, case_id: str) -> Iterator[None]:
        lock_path = self._lock_root / f"{case_id}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise ArtifactIntegrityError("SWEEP-2 LOCK: unable to open case lock") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ArtifactIntegrityError("SWEEP-2 LOCK: case lock is not a regular file")
            try:
                _lock_descriptor(descriptor)
            except OSError as error:
                raise ArtifactIntegrityError("SWEEP-2 LOCK: unable to acquire case lock") from error
            try:
                yield
            finally:
                try:
                    _unlock_descriptor(descriptor)
                except OSError as error:
                    raise ArtifactIntegrityError(
                        "SWEEP-2 LOCK: unable to release case lock"
                    ) from error
        finally:
            os.close(descriptor)

    def _read_unlocked(self, case_id: str) -> CaseState:
        state = safe_read_json(
            self._state_root,
            f"{case_id}.json",
            model=CaseState,
            limits=self._limits,
        )
        if state.case_id != case_id:
            raise ArtifactIntegrityError("SWEEP-1 IDENTITY: state does not match its case path")
        if state.sweep_digest != self.sweep_digest:
            raise ArtifactIntegrityError("SWEEP-1 IDENTITY: state belongs to another sweep")
        return state

    def _write_unlocked(self, state: CaseState) -> None:
        content = _state_bytes(state)
        if len(content) > STATE_MAX_BYTES:
            raise ArtifactIntegrityError("SWEEP-3 SIZE: case state exceeds its byte cap")
        descriptor = -1
        temporary_path: Path | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{state.case_id}.", suffix=".tmp", dir=self._state_root
            )
            temporary_path = Path(temporary)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            self._inject("state_staged")
            os.replace(temporary_path, self._state_path(state.case_id))
            temporary_path = None
            fsync_directory(self._state_root)
            self._inject("state_published")
        except OSError as error:
            raise ArtifactIntegrityError(
                "SWEEP-3 COMMIT: atomic state publication failed"
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def initialize_case(self, case_id: str, *, now: datetime) -> CaseState:
        """Create one initial pending state, or return its verified existing state."""

        self._validate("case_id", case_id, _CONTENT_ID_PATTERN)
        timestamp = _canonical_time(now)
        with self._case_lock(case_id):
            path = self._state_path(case_id)
            if path.exists() or path.is_symlink():
                return self._read_unlocked(case_id)
            state = CaseState(
                sweep_digest=self.sweep_digest,
                case_id=case_id,
                state="pending",
                revision=0,
                attempt=0,
                updated_at=timestamp,
            )
            self._write_unlocked(state)
            return state

    def read_case(self, case_id: str) -> CaseState:
        self._validate("case_id", case_id, _CONTENT_ID_PATTERN)
        with self._case_lock(case_id):
            return self._read_unlocked(case_id)

    @staticmethod
    def _require_active_lease(
        state: CaseState,
        *,
        attempt_id: str,
        lease_owner: str,
        now: datetime,
    ) -> None:
        timestamp = _canonical_time(now)
        if timestamp < state.updated_at.astimezone(UTC):
            raise InternalInvariantError("lease operation timestamp precedes the current revision")
        if (
            state.state != "running"
            or state.attempt_id != attempt_id
            or state.lease_owner != lease_owner
            or state.lease_expires_at is None
            or state.lease_expires_at <= timestamp
        ):
            raise InternalInvariantError("attempt does not own an active lease for this case")

    def _expire_unlocked(self, state: CaseState, *, now: datetime) -> CaseState:
        if state.state != "running" or state.lease_expires_at is None:
            return state
        timestamp = _canonical_time(now)
        if state.lease_expires_at > timestamp:
            return state
        if state.attempt >= MAX_SWEEP_ATTEMPTS:
            expired = _transition(
                state,
                now=timestamp,
                state="failed",
                lease_owner=None,
                lease_expires_at=None,
                error_code=LEASE_EXPIRED_CODE,
            )
        else:
            expired = _transition(
                state,
                now=timestamp,
                state="pending",
                lease_owner=None,
                lease_expires_at=None,
                error_code=LEASE_EXPIRED_CODE,
            )
        self._write_unlocked(expired)
        return expired

    def claim_case(
        self,
        case_id: str,
        *,
        attempt_id: str,
        lease_owner: str,
        now: datetime,
    ) -> CaseState | None:
        """Claim pending or expired work; a matching live claim is an idempotent no-op."""

        self._validate("case_id", case_id, _CONTENT_ID_PATTERN)
        self._validate("attempt_id", attempt_id, _ATTEMPT_ID_PATTERN)
        self._validate("lease_owner", lease_owner, _LEASE_OWNER_PATTERN)
        timestamp = _canonical_time(now)
        with self._case_lock(case_id):
            state = self._read_unlocked(case_id)
            if state.state == "running" and state.lease_expires_at is not None:
                if state.lease_expires_at > timestamp:
                    if state.attempt_id == attempt_id and state.lease_owner == lease_owner:
                        return state
                    if state.attempt_id == attempt_id:
                        raise InternalInvariantError("attempt ID is already bound to another owner")
                    return None
                state = self._expire_unlocked(state, now=timestamp)
            if state.state != "pending":
                return None
            if state.attempt_id == attempt_id:
                raise InternalInvariantError("expired or failed attempt IDs cannot be reused")
            claimed = _transition(
                state,
                now=timestamp,
                state="running",
                attempt=state.attempt + 1,
                attempt_id=attempt_id,
                lease_owner=lease_owner,
                lease_expires_at=timestamp + LEASE_DURATION,
                run_digest=None,
                error_code=None,
            )
            self._write_unlocked(claimed)
            return claimed

    def renew_lease(
        self,
        case_id: str,
        *,
        attempt_id: str,
        lease_owner: str,
        now: datetime,
    ) -> CaseState:
        """Renew exactly the current attempt's live lease for another ten minutes."""

        self._validate("case_id", case_id, _CONTENT_ID_PATTERN)
        self._validate("attempt_id", attempt_id, _ATTEMPT_ID_PATTERN)
        self._validate("lease_owner", lease_owner, _LEASE_OWNER_PATTERN)
        timestamp = _canonical_time(now)
        with self._case_lock(case_id):
            state = self._read_unlocked(case_id)
            self._require_active_lease(
                state,
                attempt_id=attempt_id,
                lease_owner=lease_owner,
                now=timestamp,
            )
            deadline = timestamp + LEASE_DURATION
            if state.lease_expires_at == deadline:
                return state
            renewed = _transition(state, now=timestamp, lease_expires_at=deadline)
            self._write_unlocked(renewed)
            return renewed

    def succeed_case(
        self,
        case_id: str,
        *,
        attempt_id: str,
        lease_owner: str,
        run_digest: str,
        artifact_store: RunArtifactStore,
        now: datetime,
    ) -> CaseState:
        """Commit immutable success only after the deterministic run root verifies."""

        self._validate("case_id", case_id, _CONTENT_ID_PATTERN)
        self._validate("attempt_id", attempt_id, _ATTEMPT_ID_PATTERN)
        self._validate("lease_owner", lease_owner, _LEASE_OWNER_PATTERN)
        self._validate("run_digest", run_digest, _SHA256_PATTERN)
        timestamp = _canonical_time(now)
        with self._case_lock(case_id):
            state = self._read_unlocked(case_id)
            if state.state == "succeeded":
                if state.run_digest != run_digest:
                    raise ArtifactIntegrityError(
                        "SWEEP-4 DIVERGENCE: successful case has a different full run digest"
                    )
                artifact_store.verify_run(case_id=case_id, run_digest=run_digest)
                return state
            self._require_active_lease(
                state,
                attempt_id=attempt_id,
                lease_owner=lease_owner,
                now=timestamp,
            )
            artifact_store.verify_run(case_id=case_id, run_digest=run_digest)
            succeeded = _transition(
                state,
                now=timestamp,
                state="succeeded",
                lease_owner=None,
                lease_expires_at=None,
                run_digest=run_digest,
                error_code=None,
            )
            self._write_unlocked(succeeded)
            return succeeded

    def fail_case(
        self,
        case_id: str,
        *,
        attempt_id: str,
        lease_owner: str,
        error: SoufflerieError,
        now: datetime,
    ) -> CaseState:
        """Record a retryable transient failure or an immutable terminal failure."""

        if not isinstance(error, SoufflerieError):
            raise TypeError("error must be a SoufflerieError instance")
        self._validate("case_id", case_id, _CONTENT_ID_PATTERN)
        self._validate("attempt_id", attempt_id, _ATTEMPT_ID_PATTERN)
        self._validate("lease_owner", lease_owner, _LEASE_OWNER_PATTERN)
        timestamp = _canonical_time(now)
        retryable = error.retryable and error.code in RETRYABLE_SWEEP_ERROR_CODES
        with self._case_lock(case_id):
            state = self._read_unlocked(case_id)
            if state.state in {"pending", "failed"} and state.attempt_id == attempt_id:
                expected_state = (
                    "pending" if retryable and state.attempt < MAX_SWEEP_ATTEMPTS else "failed"
                )
                if state.error_code == error.code and state.state == expected_state:
                    return state
                raise InternalInvariantError("duplicate attempt reported a divergent failure")
            self._require_active_lease(
                state,
                attempt_id=attempt_id,
                lease_owner=lease_owner,
                now=timestamp,
            )
            next_state: RunState = (
                "pending" if retryable and state.attempt < MAX_SWEEP_ATTEMPTS else "failed"
            )
            failed = _transition(
                state,
                now=timestamp,
                state=next_state,
                lease_owner=None,
                lease_expires_at=None,
                run_digest=None,
                error_code=error.code,
            )
            self._write_unlocked(failed)
            return failed

    def reap_expired(self, case_id: str, *, now: datetime) -> CaseState:
        """Persist the retryable or terminal consequence of an expired lease."""

        self._validate("case_id", case_id, _CONTENT_ID_PATTERN)
        timestamp = _canonical_time(now)
        with self._case_lock(case_id):
            state = self._read_unlocked(case_id)
            return self._expire_unlocked(state, now=timestamp)


@dataclass(frozen=True, slots=True)
class VerifiedCaseRun:
    case_id: str
    reference: ArtifactRef


@dataclass(frozen=True, slots=True)
class ResumePlan:
    """Deterministically partition expected cases after reaping and verification."""

    claimable_case_ids: tuple[str, ...]
    active_case_ids: tuple[str, ...]
    succeeded_runs: tuple[VerifiedCaseRun, ...]
    failed_case_ids: tuple[str, ...]


def build_resume_plan(
    *,
    case_ids: Iterable[str],
    state_store: SweepStateStore,
    artifact_store: RunArtifactStore,
    now: datetime,
) -> ResumePlan:
    """Re-hash every success and return only genuinely incomplete claimable work."""

    expected = tuple(case_ids)
    if not expected:
        raise ValueError("resume requires at least one expected case")
    if len(set(expected)) != len(expected):
        raise ArtifactIntegrityError("SWEEP-1 IDENTITY: expected case IDs must be unique")
    timestamp = _canonical_time(now)
    claimable: list[str] = []
    active: list[str] = []
    succeeded: list[VerifiedCaseRun] = []
    failed: list[str] = []
    for case_id in sorted(expected):
        state = state_store.reap_expired(case_id, now=timestamp)
        if state.state == "pending":
            claimable.append(case_id)
        elif state.state == "running":
            active.append(case_id)
        elif state.state == "failed":
            failed.append(case_id)
        else:
            if state.run_digest is None:
                raise InternalInvariantError("successful state is missing its run digest")
            reference = artifact_store.verify_run(
                case_id=state.case_id,
                run_digest=state.run_digest,
            )
            succeeded.append(VerifiedCaseRun(case_id=state.case_id, reference=reference))
    return ResumePlan(
        claimable_case_ids=tuple(claimable),
        active_case_ids=tuple(active),
        succeeded_runs=tuple(succeeded),
        failed_case_ids=tuple(failed),
    )


__all__ = [
    "LEASE_DURATION",
    "LEASE_EXPIRED_CODE",
    "MAX_SWEEP_ATTEMPTS",
    "CaseState",
    "LocalSweepStateStore",
    "ResumePlan",
    "SweepStateStore",
    "VerifiedCaseRun",
    "build_resume_plan",
]
