"""Bounded in-process reference-solve jobs with replayable event state."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from soufflerie.config import ServiceConfig
from soufflerie.errors import (
    CapacityError,
    ConfigurationError,
    EventCursorError,
    IdempotencyConflictError,
    InternalInvariantError,
    JobNotFoundError,
    RemoteExecutionError,
    SoufflerieError,
)
from soufflerie.observability import new_correlation_id
from soufflerie.schemas import canonical_sha256
from soufflerie.service.contracts import (
    JobState,
    PredictionRequest,
    PublicError,
    PublicErrorCode,
    SolveAccepted,
    SolveEvent,
    SolveEventData,
    SolveResultResponse,
    SolveStatus,
)

TERMINAL_RETENTION_SECONDS = 60 * 60
DEFAULT_HEARTBEAT_SECONDS = 15.0
MAX_RETAINED_JOBS = 1_024
MAX_PROGRESS_EVENTS = 2_048
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")

ProgressCallback = Callable[[float], Awaitable[None]]
AdmissionCheck = Callable[[], None]
Clock = Callable[[], datetime]
IdFactory = Callable[[], str]
SolveEventStream = AsyncGenerator[SolveEvent | None, None]


class SolveExecutor(Protocol):
    """Infrastructure adapter invoked exactly once for an admitted job."""

    async def execute(
        self,
        request: PredictionRequest,
        *,
        job_id: str,
        case_id: str,
        correlation_id: str,
        progress: ProgressCallback,
    ) -> SolveResultResponse: ...


class SolveJobBackend(Protocol):
    """HTTP-facing job manager contract isolated from execution infrastructure."""

    async def submit(
        self,
        request: PredictionRequest,
        *,
        correlation_id: str,
        idempotency_key: str | None,
        admission_check: AdmissionCheck | None = None,
    ) -> SolveAccepted: ...

    async def status(self, job_id: str) -> SolveStatus: ...

    async def open_events(self, job_id: str, *, after: int) -> SolveEventStream: ...


@dataclass(slots=True)
class _JobRecord:
    request: PredictionRequest
    request_sha256: str
    correlation_id: str
    accepted: SolveAccepted
    idempotency_key: str | None
    state: JobState = "queued"
    progress: float = 0.0
    result: SolveResultResponse | None = None
    error: PublicError | None = None
    sequence: int = 0
    events: list[SolveEvent] = field(default_factory=list)
    terminal_at: datetime | None = None
    progress_event_count: int = 0


_LEGAL_TRANSITIONS: frozenset[tuple[JobState, JobState]] = frozenset(
    {
        ("queued", "running"),
        ("queued", "failed"),
        ("running", "running"),
        ("running", "succeeded"),
        ("running", "failed"),
        ("succeeded", "expired"),
        ("failed", "expired"),
    }
)


def validate_job_transition(current: JobState, target: JobState) -> None:
    """Reject state regression, terminal mutation, and skipped phases."""

    if (current, target) not in _LEGAL_TRANSITIONS:
        raise ValueError(f"illegal solve job transition: {current} -> {target}")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _aware_now(now: Clock) -> datetime:
    value = now()
    if value.utcoffset() is None:
        raise InternalInvariantError("solve job clock must return timezone-aware timestamps")
    return value.astimezone(UTC)


def _public_failure(error: Exception, *, correlation_id: str) -> PublicError:
    if isinstance(error, TimeoutError):
        return PublicError(
            code="REMOTE_EXECUTION",
            message="reference solve timed out",
            retryable=True,
            correlation_id=correlation_id,
        )
    if isinstance(error, SoufflerieError):
        return PublicError(
            code=cast(PublicErrorCode, error.code),
            message="reference solve failed",
            retryable=error.retryable,
            correlation_id=correlation_id,
        )
    return PublicError(
        code="INTERNAL_ERROR",
        message="reference solve failed",
        retryable=False,
        correlation_id=correlation_id,
    )


class SolveJobManager:
    """One-process bounded job store; persistence and remote execution stay external."""

    def __init__(
        self,
        *,
        config: ServiceConfig,
        executor: SolveExecutor,
        now: Clock = _utc_now,
        id_factory: IdFactory = new_correlation_id,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        retention_seconds: int = TERMINAL_RETENTION_SECONDS,
        timeout_seconds: float | None = None,
        max_retained_jobs: int = MAX_RETAINED_JOBS,
    ) -> None:
        if not config.solve_enabled or config.solve_concurrency <= 0:
            raise ConfigurationError("solve job manager requires enabled solve capacity")
        if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0.0:
            raise ConfigurationError("heartbeat_seconds must be finite and positive")
        resolved_timeout = (
            float(config.solve_timeout_seconds) if timeout_seconds is None else timeout_seconds
        )
        if not math.isfinite(resolved_timeout) or resolved_timeout <= 0.0:
            raise ConfigurationError("timeout_seconds must be finite and positive")
        if retention_seconds != TERMINAL_RETENTION_SECONDS:
            raise ConfigurationError("terminal retention must be exactly 60 minutes")
        if max_retained_jobs < config.solve_concurrency + config.solve_queue_capacity:
            raise ConfigurationError("retained-job cap must fit active and queued capacity")

        self._config = config
        self._executor = executor
        self._now = now
        self._id_factory = id_factory
        self._heartbeat_seconds = heartbeat_seconds
        self._retention_seconds = retention_seconds
        self._timeout_seconds = resolved_timeout
        self._max_retained_jobs = max_retained_jobs
        self._semaphore = asyncio.Semaphore(config.solve_concurrency)
        self._condition = asyncio.Condition()
        self._records: dict[str, _JobRecord] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    async def submit(
        self,
        request: PredictionRequest,
        *,
        correlation_id: str,
        idempotency_key: str | None,
        admission_check: AdmissionCheck | None = None,
    ) -> SolveAccepted:
        if (
            idempotency_key is not None
            and IDEMPOTENCY_KEY_PATTERN.fullmatch(idempotency_key) is None
        ):
            raise ConfigurationError("idempotency key is invalid")
        request_sha256 = canonical_sha256(request)

        async with self._condition:
            if self._closed:
                raise RemoteExecutionError("solve job manager is closed", retryable=False)
            self._expire_and_trim_locked()
            if idempotency_key is not None and idempotency_key in self._idempotency:
                bound_sha256, job_id = self._idempotency[idempotency_key]
                if bound_sha256 != request_sha256:
                    raise IdempotencyConflictError("idempotency key is bound to another request")
                return self._record_locked(job_id).accepted

            active_or_queued = sum(
                record.state in {"queued", "running"} for record in self._records.values()
            )
            total_capacity = self._config.solve_concurrency + self._config.solve_queue_capacity
            if active_or_queued >= total_capacity or len(self._records) >= self._max_retained_jobs:
                raise CapacityError("solve job capacity is exhausted")

            job_id = self._id_factory()
            if job_id in self._records:
                raise InternalInvariantError("job ID factory returned a duplicate")
            case_id = request_sha256[:20]
            created_at = _aware_now(self._now)
            batches_until_terminal = active_or_queued // self._config.solve_concurrency + 1
            expires_at = created_at + timedelta(
                seconds=(batches_until_terminal * self._timeout_seconds + self._retention_seconds)
            )
            accepted = SolveAccepted(
                job_id=job_id,
                case_id=case_id,
                state="queued",
                status_url=f"/solve/{job_id}",
                events_url=f"/solve/{job_id}/events",
                expires_at=expires_at,
            )
            if admission_check is not None:
                admission_check()
            record = _JobRecord(
                request=request,
                request_sha256=request_sha256,
                correlation_id=correlation_id,
                accepted=accepted,
                idempotency_key=idempotency_key,
            )
            self._records[job_id] = record
            if idempotency_key is not None:
                self._idempotency[idempotency_key] = (request_sha256, job_id)
            self._append_event_locked(record, event="queued")
            task = asyncio.create_task(self._execute(job_id), name=f"soufflerie-solve-{job_id}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            self._condition.notify_all()
            return accepted

    async def status(self, job_id: str) -> SolveStatus:
        async with self._condition:
            record = self._record_locked(job_id)
            self._expire_record_locked(record)
            return self._snapshot_locked(record)

    async def open_events(self, job_id: str, *, after: int) -> SolveEventStream:
        if after < 0:
            raise EventCursorError("event cursor must be nonnegative")
        async with self._condition:
            record = self._record_locked(job_id)
            self._expire_record_locked(record)
            if record.state == "expired":
                raise JobNotFoundError("solve event history has expired")
            if after > record.sequence:
                raise EventCursorError("event cursor is ahead of retained state")
        return self._event_stream(job_id, after=after)

    async def aclose(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            tasks = tuple(self._tasks)
            for task in tasks:
                task.cancel()
            self._condition.notify_all()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute(self, job_id: str) -> None:
        try:
            async with self._semaphore:
                await self._start(job_id)

                async def progress(value: float) -> None:
                    await self._record_progress(job_id, value)

                async with asyncio.timeout(self._timeout_seconds):
                    request, accepted, correlation_id = await self._execution_inputs(job_id)
                    result = await self._executor.execute(
                        request,
                        job_id=accepted.job_id,
                        case_id=accepted.case_id,
                        correlation_id=correlation_id,
                        progress=progress,
                    )
                await self._complete(job_id, result)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._fail(job_id, error)

    async def _execution_inputs(self, job_id: str) -> tuple[PredictionRequest, SolveAccepted, str]:
        async with self._condition:
            record = self._record_locked(job_id)
            return record.request, record.accepted, record.correlation_id

    async def _start(self, job_id: str) -> None:
        async with self._condition:
            record = self._record_locked(job_id)
            validate_job_transition(record.state, "running")
            record.state = "running"
            self._append_event_locked(record, event="running")
            self._condition.notify_all()

    async def _record_progress(self, job_id: str, value: float) -> None:
        if not math.isfinite(value) or value < 0.0 or value >= 1.0:
            raise InternalInvariantError("worker progress must be finite in [0, 1)")
        async with self._condition:
            record = self._record_locked(job_id)
            validate_job_transition(record.state, "running")
            if value < record.progress:
                raise InternalInvariantError("worker progress must be monotonic")
            if value == record.progress:
                return
            if record.progress_event_count >= MAX_PROGRESS_EVENTS:
                raise InternalInvariantError("worker emitted too many progress events")
            record.progress = value
            record.progress_event_count += 1
            self._append_event_locked(record, event="progress")
            self._condition.notify_all()

    async def _complete(self, job_id: str, result: SolveResultResponse) -> None:
        async with self._condition:
            record = self._record_locked(job_id)
            if (
                result.job_id != record.accepted.job_id
                or result.case_id != record.accepted.case_id
                or result.correlation_id != record.correlation_id
            ):
                raise InternalInvariantError("executor result identities do not match admitted job")
            validate_job_transition(record.state, "succeeded")
            record.state = "succeeded"
            record.progress = 1.0
            record.result = result
            record.terminal_at = _aware_now(self._now)
            self._append_event_locked(record, event="completed")
            self._condition.notify_all()

    async def _fail(self, job_id: str, error: Exception) -> None:
        async with self._condition:
            record = self._record_locked(job_id)
            if record.state in {"succeeded", "failed", "expired"}:
                return
            validate_job_transition(record.state, "failed")
            record.state = "failed"
            record.error = _public_failure(error, correlation_id=record.correlation_id)
            record.terminal_at = _aware_now(self._now)
            self._append_event_locked(record, event="failed")
            self._condition.notify_all()

    async def _event_stream(self, job_id: str, *, after: int) -> SolveEventStream:
        cursor = after
        while True:
            event: SolveEvent | None = None
            heartbeat = False
            async with self._condition:
                record = self._record_locked(job_id)
                self._expire_record_locked(record)
                if record.state == "expired":
                    return
                available = next((item for item in record.events if item.sequence > cursor), None)
                if available is not None:
                    event = available
                    cursor = available.sequence
                elif record.state in {"succeeded", "failed"} or self._closed:
                    return
                else:
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(), timeout=self._heartbeat_seconds
                        )
                    except TimeoutError:
                        heartbeat = True
            if event is not None:
                yield event
            elif heartbeat:
                yield None

    def _record_locked(self, job_id: str) -> _JobRecord:
        try:
            return self._records[job_id]
        except KeyError as error:
            raise JobNotFoundError("solve job is unknown or expired") from error

    def _snapshot_locked(self, record: _JobRecord) -> SolveStatus:
        return SolveStatus(
            job_id=record.accepted.job_id,
            case_id=record.accepted.case_id,
            state=record.state,
            progress=record.progress,
            result=record.result,
            error=record.error,
            sequence=record.sequence,
        )

    def _append_event_locked(
        self,
        record: _JobRecord,
        *,
        event: str,
    ) -> None:
        record.sequence += 1
        record.events.append(
            SolveEvent.model_validate(
                {
                    "sequence": record.sequence,
                    "job_id": record.accepted.job_id,
                    "timestamp": _aware_now(self._now),
                    "event": event,
                    "data": SolveEventData(
                        case_id=record.accepted.case_id,
                        state=record.state,
                        progress=record.progress,
                        result=record.result,
                        error=record.error,
                    ),
                }
            )
        )

    def _expire_record_locked(self, record: _JobRecord) -> None:
        if record.state not in {"succeeded", "failed"} or record.terminal_at is None:
            return
        if _aware_now(self._now) < record.terminal_at + timedelta(seconds=self._retention_seconds):
            return
        validate_job_transition(record.state, "expired")
        record.state = "expired"
        record.progress = 1.0
        record.result = None
        record.error = None
        record.events.clear()
        if record.idempotency_key is not None:
            self._idempotency.pop(record.idempotency_key, None)

    def _expire_and_trim_locked(self) -> None:
        for record in self._records.values():
            self._expire_record_locked(record)
        expired = [job_id for job_id, record in self._records.items() if record.state == "expired"]
        while len(self._records) >= self._max_retained_jobs and expired:
            self._records.pop(expired.pop(0))


__all__ = [
    "DEFAULT_HEARTBEAT_SECONDS",
    "IDEMPOTENCY_KEY_PATTERN",
    "MAX_PROGRESS_EVENTS",
    "MAX_RETAINED_JOBS",
    "TERMINAL_RETENTION_SECONDS",
    "AdmissionCheck",
    "Clock",
    "IdFactory",
    "ProgressCallback",
    "SolveEventStream",
    "SolveExecutor",
    "SolveJobBackend",
    "SolveJobManager",
    "validate_job_transition",
]
