from __future__ import annotations

import asyncio

import pytest

from soufflerie.errors import (
    CapacityError,
    EventCursorError,
    IdempotencyConflictError,
    JobNotFoundError,
)
from soufflerie.observability import new_correlation_id
from soufflerie.service.contracts import JobState
from soufflerie.service.jobs import SolveJobManager, validate_job_transition
from tests.service.helpers import (
    NOW,
    FakeClock,
    IdFactory,
    ManualExecutor,
    NeverExecutor,
    prediction_request,
    service_config,
)


async def _wait_for_state(manager: SolveJobManager, job_id: str, state: JobState) -> None:
    async with asyncio.timeout(1.0):
        while (await manager.status(job_id)).state != state:
            await asyncio.sleep(0.001)


def test_transition_table_accepts_only_legal_monotonic_states() -> None:
    legal: set[tuple[JobState, JobState]] = {
        ("queued", "running"),
        ("queued", "failed"),
        ("running", "running"),
        ("running", "succeeded"),
        ("running", "failed"),
        ("succeeded", "expired"),
        ("failed", "expired"),
    }
    states: tuple[JobState, ...] = ("queued", "running", "succeeded", "failed", "expired")
    for current in states:
        for target in states:
            if (current, target) in legal:
                validate_job_transition(current, target)
            else:
                with pytest.raises(ValueError, match="illegal solve job transition"):
                    validate_job_transition(current, target)


@pytest.mark.anyio
async def test_manager_bounds_active_and_queued_jobs_and_releases_capacity() -> None:
    executor = ManualExecutor()
    manager = SolveJobManager(
        config=service_config(queue_capacity=1),
        executor=executor,
        now=FakeClock(),
        id_factory=IdFactory(),
    )
    correlation_id = new_correlation_id(timestamp=NOW)
    try:
        first = await manager.submit(
            prediction_request(), correlation_id=correlation_id, idempotency_key="first"
        )
        assert await executor.entered.get() == first.job_id
        second = await manager.submit(
            prediction_request(reynolds=101.0),
            correlation_id=correlation_id,
            idempotency_key="second",
        )
        assert (await manager.status(second.job_id)).state == "queued"

        with pytest.raises(CapacityError):
            await manager.submit(
                prediction_request(reynolds=102.0),
                correlation_id=correlation_id,
                idempotency_key="third",
            )

        executor.finish(first.job_id)
        await _wait_for_state(manager, first.job_id, "succeeded")
        assert await executor.entered.get() == second.job_id
        executor.finish(second.job_id)
        await _wait_for_state(manager, second.job_id, "succeeded")

        third = await manager.submit(
            prediction_request(reynolds=102.0),
            correlation_id=correlation_id,
            idempotency_key="third",
        )
        assert third.state == "queued"
    finally:
        await manager.aclose()


@pytest.mark.anyio
async def test_idempotency_reuses_exact_request_and_rejects_rebinding() -> None:
    executor = ManualExecutor()
    manager = SolveJobManager(
        config=service_config(), executor=executor, now=FakeClock(), id_factory=IdFactory()
    )
    correlation_id = new_correlation_id(timestamp=NOW)
    try:
        first = await manager.submit(
            prediction_request(), correlation_id=correlation_id, idempotency_key="stable-key"
        )
        duplicate = await manager.submit(
            prediction_request(), correlation_id=correlation_id, idempotency_key="stable-key"
        )
        assert duplicate == first
        assert await executor.entered.get() == first.job_id
        await asyncio.sleep(0)
        assert executor.calls == [first.job_id]

        with pytest.raises(IdempotencyConflictError):
            await manager.submit(
                prediction_request(reynolds=200.0),
                correlation_id=correlation_id,
                idempotency_key="stable-key",
            )
    finally:
        await manager.aclose()


@pytest.mark.anyio
async def test_timeout_becomes_one_immutable_failed_terminal_event() -> None:
    manager = SolveJobManager(
        config=service_config(),
        executor=NeverExecutor(),
        now=FakeClock(),
        id_factory=IdFactory(),
        timeout_seconds=0.01,
    )
    try:
        accepted = await manager.submit(
            prediction_request(),
            correlation_id=new_correlation_id(timestamp=NOW),
            idempotency_key=None,
        )
        await _wait_for_state(manager, accepted.job_id, "failed")
        first = await manager.status(accepted.job_id)
        await asyncio.sleep(0.02)
        second = await manager.status(accepted.job_id)
        assert second == first
        assert first.error is not None
        assert first.error.code == "REMOTE_EXECUTION"
        assert first.error.retryable is True
        stream = await manager.open_events(accepted.job_id, after=0)
        events = [event async for event in stream if event is not None]
        assert [event.event for event in events] == ["queued", "running", "failed"]
        assert [event.sequence for event in events] == [1, 2, 3]
    finally:
        await manager.aclose()


@pytest.mark.anyio
async def test_terminal_events_replay_for_sixty_minutes_then_status_expires() -> None:
    clock = FakeClock()
    executor = ManualExecutor()
    manager = SolveJobManager(
        config=service_config(), executor=executor, now=clock, id_factory=IdFactory()
    )
    try:
        accepted = await manager.submit(
            prediction_request(),
            correlation_id=new_correlation_id(timestamp=NOW),
            idempotency_key=None,
        )
        assert await executor.entered.get() == accepted.job_id
        executor.finish(accepted.job_id)
        await _wait_for_state(manager, accepted.job_id, "succeeded")
        terminal = await manager.status(accepted.job_id)

        clock.advance(seconds=3_599)
        assert (await manager.status(accepted.job_id)) == terminal
        replay = await manager.open_events(accepted.job_id, after=2)
        replayed = [event async for event in replay if event is not None]
        assert [event.sequence for event in replayed] == [3, 4, 5]

        clock.advance(seconds=2)
        expired = await manager.status(accepted.job_id)
        assert expired.state == "expired"
        assert expired.result is None
        with pytest.raises(JobNotFoundError):
            await manager.open_events(accepted.job_id, after=terminal.sequence + 1)
    finally:
        await manager.aclose()


@pytest.mark.anyio
async def test_future_cursor_is_rejected_before_opening_stream() -> None:
    manager = SolveJobManager(
        config=service_config(),
        executor=NeverExecutor(),
        now=FakeClock(),
        id_factory=IdFactory(),
    )
    try:
        accepted = await manager.submit(
            prediction_request(),
            correlation_id=new_correlation_id(timestamp=NOW),
            idempotency_key=None,
        )
        with pytest.raises(EventCursorError):
            await manager.open_events(accepted.job_id, after=99)
    finally:
        await manager.aclose()
