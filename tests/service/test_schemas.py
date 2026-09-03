from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from soufflerie.observability import new_correlation_id
from soufflerie.schemas import sha256_bytes
from soufflerie.service import (
    ConsistencyFlags,
    EncodedArtifact,
    PredictionRequest,
    PublicError,
    ShapeRequest,
    SolveAccepted,
    SolveEvent,
    SolveEventData,
    SolveStatus,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
JOB_ID = new_correlation_id(timestamp=NOW)


def _request() -> PredictionRequest:
    return PredictionRequest(
        shape=ShapeRequest(aspect_ratio=0.75, rotation_deg=12.0, scale=1.0),
        reynolds=100.0,
    )


def _error() -> PublicError:
    return PublicError(
        code="REMOTE_EXECUTION",
        message="reference solve failed",
        retryable=True,
        correlation_id=JOB_ID,
    )


def test_prediction_request_is_closed_strict_finite_and_domain_bounded() -> None:
    request = _request()
    assert request.schema_version == 1
    assert request.shape.aspect_ratio == 0.75

    invalid: tuple[dict[str, object], ...] = (
        {**request.model_dump(), "unknown": 1},
        {**request.model_dump(), "reynolds": True},
        {**request.model_dump(), "reynolds": float("nan")},
        {**request.model_dump(), "reynolds": float("inf")},
        {**request.model_dump(), "reynolds": 39.999},
        {**request.model_dump(), "reynolds": 300.001},
        {
            **request.model_dump(),
            "shape": {**request.shape.model_dump(), "aspect_ratio": False},
        },
        {
            **request.model_dump(),
            "shape": {**request.shape.model_dump(), "rotation_deg": 30.001},
        },
    )
    for payload in invalid:
        with pytest.raises(ValidationError):
            PredictionRequest.model_validate(payload)

    for reynolds in (40.0, 300.0):
        assert request.model_copy(update={"reynolds": reynolds}).reynolds == reynolds


def test_encoded_artifact_binds_media_bytes_base64_and_digest() -> None:
    content = b"safe deterministic artifact"
    encoded = EncodedArtifact(
        media_type="image/png",
        encoding="base64",
        data=base64.b64encode(content).decode("ascii"),
        sha256=sha256_bytes(content),
        bytes=len(content),
    )
    assert encoded.decoded_bytes == content

    for change in (
        {"data": "not-base64"},
        {"sha256": "f" * 64},
        {"bytes": len(content) + 1},
    ):
        with pytest.raises(ValidationError):
            EncodedArtifact.model_validate({**encoded.model_dump(), **change})


def test_consistency_statuses_cannot_disagree_with_fixed_thresholds() -> None:
    flags = ConsistencyFlags(
        head_field_gap_pct=7.0,
        head_field_gap="green",
        divergence_ratio_to_solver_baseline=2.0,
        divergence="green",
        obstacle_velocity_ratio=0.005,
        obstacle_compliance="green",
        ood=False,
    )
    with pytest.raises(ValidationError, match="head_field_gap"):
        ConsistencyFlags.model_validate(
            {**flags.model_dump(), "head_field_gap_pct": 10.1, "head_field_gap": "green"}
        )
    with pytest.raises(ValidationError, match="divergence"):
        ConsistencyFlags.model_validate(
            {
                **flags.model_dump(),
                "divergence_ratio_to_solver_baseline": 3.0,
                "divergence": "green",
            }
        )
    with pytest.raises(ValidationError, match="obstacle_compliance"):
        ConsistencyFlags.model_validate(
            {
                **flags.model_dump(),
                "obstacle_velocity_ratio": 0.01,
                "obstacle_compliance": "green",
            }
        )


def test_solve_records_enforce_urls_timestamps_and_terminal_coherence() -> None:
    accepted = SolveAccepted(
        job_id=JOB_ID,
        case_id="a" * 20,
        state="queued",
        status_url=f"/solve/{JOB_ID}",
        events_url=f"/solve/{JOB_ID}/events",
        expires_at=NOW + timedelta(hours=1),
    )
    assert accepted.job_id in accepted.events_url

    with pytest.raises(ValidationError, match="status_url"):
        SolveAccepted.model_validate({**accepted.model_dump(), "status_url": "/solve/other"})
    with pytest.raises(ValidationError, match="timezone-aware"):
        SolveAccepted.model_validate(
            {**accepted.model_dump(), "expires_at": datetime(2026, 9, 3, 13, 0)}
        )

    queued = SolveStatus(
        job_id=JOB_ID,
        case_id="a" * 20,
        state="queued",
        progress=0.0,
        result=None,
        error=None,
        sequence=1,
    )
    with pytest.raises(ValidationError, match="failed status"):
        SolveStatus.model_validate({**queued.model_dump(), "state": "failed"})
    with pytest.raises(ValidationError, match="non-terminal"):
        SolveStatus.model_validate({**queued.model_dump(), "error": _error().model_dump()})

    failed = queued.model_copy(update={"state": "failed", "error": _error()})
    SolveStatus.model_validate(failed.model_dump())
    with pytest.raises(ValidationError, match="terminal"):
        SolveStatus.model_validate({**failed.model_dump(), "state": "succeeded"})


def test_solve_event_name_state_and_sequence_are_coherent() -> None:
    event = SolveEvent(
        sequence=2,
        job_id=JOB_ID,
        timestamp=NOW,
        event="progress",
        data=SolveEventData(
            case_id="a" * 20,
            state="running",
            progress=0.5,
            result=None,
            error=None,
        ),
    )
    assert event.data.progress == 0.5
    with pytest.raises(ValidationError, match="event does not match"):
        SolveEvent.model_validate(
            {**event.model_dump(), "event": "completed", "data": event.data.model_dump()}
        )
    with pytest.raises(ValidationError):
        SolveEvent.model_validate({**event.model_dump(), "sequence": 0})
