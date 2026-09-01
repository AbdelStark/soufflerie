from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from soufflerie.observability import (
    InMemoryEventSink,
    OperationTimer,
    TimingSummary,
    operation_context,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_timer_distinguishes_synchronized_compute_queue_io_and_wall() -> None:
    clock = FakeClock()
    synchronizations: list[float] = []

    def synchronize() -> None:
        synchronizations.append(clock())

    timer = OperationTimer(clock=clock, synchronize=synchronize)
    with timer.segment("queue"):
        clock.advance(2.0)
    clock.advance(1.0)
    with timer.segment("io"):
        clock.advance(3.0)
    with timer.segment("compute"):
        clock.advance(5.0)
    clock.advance(4.0)

    summary = timer.finish()
    assert summary == TimingSummary(
        wall_seconds=15.0,
        compute_seconds=5.0,
        queue_seconds=2.0,
        io_seconds=3.0,
    )
    assert synchronizations == [106.0, 111.0]
    assert timer.finish() == summary


def test_compute_timer_includes_pending_gpu_work_after_the_body() -> None:
    clock = FakeClock()
    calls = 0

    def synchronize() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            clock.advance(2.5)

    timer = OperationTimer(clock=clock, synchronize=synchronize)
    with timer.segment("compute"):
        clock.advance(1.0)

    assert timer.finish().compute_seconds == 3.5
    assert calls == 2


def test_timer_rejects_overlap_active_finish_reuse_and_backwards_clock() -> None:
    clock = FakeClock()
    timer = OperationTimer(clock=clock)
    with timer.segment("queue"):
        with pytest.raises(RuntimeError, match="cannot overlap"), timer.segment("io"):
            pass
        with pytest.raises(RuntimeError, match="cannot finish"):
            timer.finish()
        clock.advance(1.0)

    timer.finish()
    with pytest.raises(RuntimeError, match="already finished"), timer.segment("compute"):
        pass

    backwards = OperationTimer(clock=clock)
    clock.advance(-1.0)
    with pytest.raises(RuntimeError, match="moved backwards"):
        backwards.finish()

    invalid = OperationTimer(clock=clock)
    with (
        pytest.raises(ValueError, match="unknown timing segment"),
        invalid.segment(
            "network"  # type: ignore[arg-type]
        ),
    ):
        pass


def test_timer_recovers_if_post_compute_synchronization_fails() -> None:
    clock = FakeClock()
    calls = 0

    def synchronize() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("device synchronization failed")

    timer = OperationTimer(clock=clock, synchronize=synchronize)
    with (
        pytest.raises(RuntimeError, match="device synchronization failed"),
        timer.segment("compute"),
    ):
        clock.advance(1.0)
    # The failed segment is not recorded, but the timer is not stranded active.
    assert timer.finish().compute_seconds == 0.0


def test_timing_summary_rejects_overlapping_totals() -> None:
    with pytest.raises(ValidationError, match="must not exceed wall"):
        TimingSummary(
            wall_seconds=1.0,
            compute_seconds=0.6,
            queue_seconds=0.3,
            io_seconds=0.2,
        )


def test_operation_context_adds_all_timing_fields_to_terminal_event() -> None:
    clock = FakeClock()
    timer = OperationTimer(clock=clock)
    sink = InMemoryEventSink()
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    with operation_context(
        "operation_started",
        component="training",
        event_sink=sink,
        timer=timer,
        utc_clock=lambda: now,
    ) as operation:
        with operation.timer.segment("queue"):
            clock.advance(0.5)
        with operation.timer.segment("compute"):
            clock.advance(1.25)
        with operation.timer.segment("io"):
            clock.advance(0.25)
        clock.advance(0.75)

    terminal = sink.events[-1]
    assert terminal.event == "operation_completed"
    assert terminal.component == "training"
    assert terminal.fields == {
        "wall_seconds": 2.75,
        "compute_seconds": 1.25,
        "queue_seconds": 0.5,
        "io_seconds": 0.25,
        "outcome": "succeeded",
    }
