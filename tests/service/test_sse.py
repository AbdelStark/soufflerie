from __future__ import annotations

import asyncio
import json
from datetime import datetime

import httpx
import pytest

from soufflerie.observability import new_correlation_id
from soufflerie.service import SolveEvent, create_app
from soufflerie.service.jobs import SolveJobManager
from tests.service.helpers import (
    JOB_IDS,
    NOW,
    FakeClock,
    IdFactory,
    ManualExecutor,
    prediction_request,
    readiness_probe,
    service_config,
)


async def _wait_for_terminal(manager: SolveJobManager, job_id: str) -> None:
    for _ in range(100):
        if (await manager.status(job_id)).state in {"succeeded", "failed"}:
            return
        await asyncio.sleep(0)
    raise AssertionError("job did not terminate")


def _request_payload(*, reynolds: float = 100.0) -> dict[str, object]:
    return prediction_request(reynolds=reynolds).model_dump(mode="json")


def _parse_sse(body: str) -> list[SolveEvent]:
    events: list[SolveEvent] = []
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        assert lines[0].startswith("id: ")
        assert lines[1].startswith("event: ")
        assert lines[2].startswith("data: ")
        payload = json.loads(lines[2].removeprefix("data: "))
        payload["timestamp"] = datetime.fromisoformat(payload["timestamp"])
        event = SolveEvent.model_validate(payload)
        assert lines[0] == f"id: {event.sequence}"
        assert lines[1] == f"event: {event.event}"
        events.append(event)
    return events


@pytest.mark.anyio
async def test_http_submit_poll_idempotency_and_terminal_sse_replay() -> None:
    config = service_config()
    executor = ManualExecutor()
    manager = SolveJobManager(
        config=config, executor=executor, now=FakeClock(), id_factory=IdFactory()
    )
    app = create_app(
        config=config,
        readiness=readiness_probe(),
        package_version="0.1.0",
        solve_jobs=manager,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            submitted = await client.post(
                "/solve", json=_request_payload(), headers={"Idempotency-Key": "same-request"}
            )
            assert submitted.status_code == 202
            job_id = submitted.json()["job_id"]
            assert job_id == JOB_IDS[0]
            assert await executor.entered.get() == job_id

            duplicate = await client.post(
                "/solve", json=_request_payload(), headers={"Idempotency-Key": "same-request"}
            )
            assert duplicate.status_code == 202
            assert duplicate.json() == submitted.json()

            conflict = await client.post(
                "/solve",
                json=_request_payload(reynolds=200.0),
                headers={"Idempotency-Key": "same-request"},
            )
            assert conflict.status_code == 409
            assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"

            running = await client.get(f"/solve/{job_id}")
            assert running.status_code == 200
            assert running.json()["state"] == "running"

            executor.finish(job_id)
            await _wait_for_terminal(manager, job_id)
            terminal = await client.get(f"/solve/{job_id}")
            assert terminal.json()["state"] == "succeeded"

            replay = await client.get(f"/solve/{job_id}/events", headers={"Last-Event-ID": "2"})
            assert replay.status_code == 200
            assert replay.headers["content-type"].startswith("text/event-stream")
            assert replay.headers["cache-control"] == "no-cache"
            events = _parse_sse(replay.text)
            assert [event.event for event in events] == ["progress", "progress", "completed"]
            assert [event.sequence for event in events] == [3, 4, 5]
    finally:
        await manager.aclose()


@pytest.mark.anyio
async def test_invalid_and_future_cursors_fail_as_json_before_streaming() -> None:
    config = service_config()
    manager = SolveJobManager(
        config=config,
        executor=ManualExecutor(),
        now=FakeClock(),
        id_factory=IdFactory(),
    )
    app = create_app(
        config=config,
        readiness=readiness_probe(),
        package_version="0.1.0",
        solve_jobs=manager,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            submitted = await client.post("/solve", json=_request_payload())
            job_id = submitted.json()["job_id"]
            for cursor in ("", "-1", "00", "not-an-integer", "11111111111111111111", "99"):
                response = await client.get(
                    f"/solve/{job_id}/events", headers={"Last-Event-ID": cursor}
                )
                assert response.status_code == 409
                assert response.headers["content-type"].startswith("application/json")
                assert response.json()["error"]["code"] == "EVENT_CURSOR_INVALID"

            unknown = await client.get(f"/solve/{JOB_IDS[-1]}/events")
            assert unknown.status_code == 404
            assert unknown.json()["error"]["code"] == "JOB_NOT_FOUND"
    finally:
        await manager.aclose()


@pytest.mark.anyio
async def test_http_heartbeat_is_comment_only_and_does_not_advance_replay_cursor() -> None:
    config = service_config()
    executor = ManualExecutor()
    manager = SolveJobManager(
        config=config,
        executor=executor,
        now=FakeClock(),
        id_factory=IdFactory(),
        heartbeat_seconds=0.01,
    )
    app = create_app(
        config=config,
        readiness=readiness_probe(),
        package_version="0.1.0",
        solve_jobs=manager,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            submitted = await client.post("/solve", json=_request_payload())
            job_id = submitted.json()["job_id"]
            assert await executor.entered.get() == job_id
            cursor = (await manager.status(job_id)).sequence

            async def finish_after_heartbeat() -> None:
                await asyncio.sleep(0.05)
                executor.finish(job_id)

            finisher = asyncio.create_task(finish_after_heartbeat())
            response = await client.get(
                f"/solve/{job_id}/events", headers={"Last-Event-ID": str(cursor)}
            )
            await finisher

            frames = response.text.strip().split("\n\n")
            heartbeat_frames = [frame for frame in frames if frame.startswith(":")]
            data_frames = [frame for frame in frames if not frame.startswith(":")]
            assert heartbeat_frames
            assert set(heartbeat_frames) == {": heartbeat"}
            assert all("data:" not in frame and "id:" not in frame for frame in heartbeat_frames)
            events = _parse_sse("\n\n".join(data_frames))
            assert [event.sequence for event in events] == [cursor + 1, cursor + 2]
            assert [event.event for event in events] == ["progress", "completed"]
    finally:
        await manager.aclose()


@pytest.mark.anyio
async def test_heartbeat_has_no_data_and_subscriber_disconnect_does_not_cancel_job() -> None:
    executor = ManualExecutor()
    manager = SolveJobManager(
        config=service_config(),
        executor=executor,
        now=FakeClock(),
        id_factory=IdFactory(),
        heartbeat_seconds=0.01,
    )
    try:
        accepted = await manager.submit(
            prediction_request(),
            correlation_id=new_correlation_id(timestamp=NOW),
            idempotency_key=None,
        )
        assert await executor.entered.get() == accepted.job_id
        running = await manager.status(accepted.job_id)
        stream = await manager.open_events(accepted.job_id, after=running.sequence)
        assert await anext(stream) is None
        await stream.aclose()

        executor.finish(accepted.job_id)
        await _wait_for_terminal(manager, accepted.job_id)
        assert (await manager.status(accepted.job_id)).state == "succeeded"
    finally:
        await manager.aclose()
