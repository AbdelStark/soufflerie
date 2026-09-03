from __future__ import annotations

import base64
import importlib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest

from soufflerie.demo import (
    DEBOUNCE_SECONDS,
    DEMO_CSS,
    ControlValues,
    DemoIdentity,
    PredictionCoordinator,
    PredictionOutcome,
    build_demo,
    prediction_view,
)
from soufflerie.errors import DependencyUnavailableError
from soufflerie.schemas import sha256_bytes
from soufflerie.service import (
    ConsistencyFlags,
    EncodedArtifact,
    PredictionRequest,
    PredictionResponse,
)

MODEL_ID = "1" * 20
DATASET_ID = "2" * 20
REPORT_ID = "3" * 20
SESSION_ID = "browser_session_0123456789"
CORRELATION_ID = "0199a1b2-c3d4-7e5f-8a9b-000000000001"


def _artifact(media_type: str, content: bytes) -> EncodedArtifact:
    return EncodedArtifact(
        media_type=cast(Any, media_type),
        encoding="base64",
        data=base64.b64encode(content).decode("ascii"),
        sha256=sha256_bytes(content),
        bytes=len(content),
    )


def _response(
    *,
    validation_status: str = "red",
    model_id: str = MODEL_ID,
    reynolds: float = 100.0,
) -> PredictionResponse:
    red = reynolds > 150.0
    return PredictionResponse(
        correlation_id=CORRELATION_ID,
        case_id="4" * 20,
        fields_png=_artifact("image/png", b"png"),
        fields_npz=_artifact("application/x-npz", b"npz"),
        cd_head=1.23456,
        cd_field=1.11111,
        consistency=ConsistencyFlags(
            head_field_gap_pct=12.5 if red else 8.0,
            head_field_gap="red" if red else "green",
            divergence_ratio_to_solver_baseline=2.0,
            divergence="green",
            obstacle_velocity_ratio=0.02 if red else 0.005,
            obstacle_compliance="red" if red else "green",
            ood=False,
        ),
        validation_status=cast(Any, validation_status),
        model_id=model_id,
        dataset_id=DATASET_ID,
        report_id=REPORT_ID,
        inference_ms=31.25,
        request_ms=42.5,
    )


def _identity(*, validation_status: str = "red") -> DemoIdentity:
    return DemoIdentity(
        model_id=MODEL_ID,
        dataset_id=DATASET_ID,
        report_id=REPORT_ID,
        validation_status=cast(Any, validation_status),
        report_href="/validation/report",
    )


class RecordingPredictor:
    def __init__(self) -> None:
        self.requests: list[PredictionRequest] = []

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        self.requests.append(request)
        return _response(reynolds=request.reynolds)


class BrowserRequest:
    session_hash = SESSION_ID


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_controls_use_canonical_defaults_and_request_bounds() -> None:
    controls = ControlValues()

    request = controls.to_request()
    assert request.shape.aspect_ratio == 0.65
    assert request.shape.rotation_deg == 0.0
    assert request.shape.scale == 1.0
    assert request.reynolds == 100.0

    for values in (
        {"aspect_ratio": 0.49},
        {"rotation_deg": 30.01},
        {"scale": 1.26},
        {"reynolds": 301.0},
        {"reynolds": float("nan")},
        {"reynolds": cast(Any, True)},
    ):
        with pytest.raises(ValueError):
            ControlValues(**values)


def test_identity_and_coordinator_configuration_fail_closed() -> None:
    invalid_identities: tuple[Callable[[], DemoIdentity], ...] = (
        lambda: DemoIdentity("z" * 20, DATASET_ID, REPORT_ID, "red", "/report"),
        lambda: DemoIdentity(
            MODEL_ID,
            DATASET_ID,
            REPORT_ID,
            cast(Any, "amber"),
            "/report",
        ),
        lambda: DemoIdentity(MODEL_ID, DATASET_ID, REPORT_ID, "red", "report"),
        lambda: DemoIdentity(
            MODEL_ID,
            DATASET_ID,
            REPORT_ID,
            "red",
            "https://user@example.com/report",
        ),
    )
    for factory in invalid_identities:
        with pytest.raises(ValueError):
            factory()

    predictor = RecordingPredictor()
    for arguments in (
        {"debounce_seconds": 0.0},
        {"session_ttl_seconds": float("inf")},
        {"max_sessions": cast(Any, True)},
    ):
        with pytest.raises(ValueError):
            PredictionCoordinator(predictor, _identity(), **arguments)

    coordinator = PredictionCoordinator(predictor, _identity())
    with pytest.raises(ValueError, match="session_id"):
        coordinator.reserve("invalid/session", ControlValues())
    with pytest.raises(TypeError, match="controls"):
        coordinator.reserve(SESSION_ID, cast(Any, object()))
    with pytest.raises(TypeError, match="ticket"):
        coordinator.resolve(cast(Any, object()))


def test_prediction_outcome_rejects_incoherent_payloads() -> None:
    with pytest.raises(ValueError, match="positive"):
        PredictionOutcome(sequence=0, phase="stale", response=None, message=None)
    with pytest.raises(ValueError, match="requires only a response"):
        PredictionOutcome(sequence=1, phase="ready", response=None, message=None)
    with pytest.raises(ValueError, match="requires only a message"):
        PredictionOutcome(
            sequence=1,
            phase="error",
            response=_response(),
            message="failed",
        )
    with pytest.raises(ValueError, match="cannot contain display data"):
        PredictionOutcome(sequence=1, phase="stale", response=None, message="stale")


def test_debounce_coalesces_commits_and_sequences_are_monotonic() -> None:
    predictor = RecordingPredictor()
    clock = FakeClock()
    coordinator = PredictionCoordinator(
        predictor,
        _identity(),
        clock=clock,
        sleeper=clock.sleep,
    )

    first = coordinator.reserve(SESSION_ID, ControlValues(reynolds=100.0))
    second = coordinator.reserve(SESSION_ID, ControlValues(reynolds=120.0))

    assert first.sequence == 1
    assert second.sequence == 2
    assert second.eligible_at - clock.value == pytest.approx(DEBOUNCE_SECONDS)
    assert coordinator.resolve(first).phase == "stale"
    latest = coordinator.resolve(second)
    assert latest.phase == "ready"
    assert latest.applied is True
    assert [request.reynolds for request in predictor.requests] == [120.0]
    assert sum(clock.sleeps) == pytest.approx(DEBOUNCE_SECONDS)


def test_in_flight_older_result_cannot_overwrite_newer_commit() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingPredictor:
        def predict(self, request: PredictionRequest) -> PredictionResponse:
            if request.reynolds == 100.0:
                entered.set()
                assert release.wait(timeout=2.0)
            return _response(reynolds=request.reynolds)

    coordinator = PredictionCoordinator(
        BlockingPredictor(),
        _identity(),
        debounce_seconds=0.001,
    )
    first = coordinator.reserve(SESSION_ID, ControlValues(reynolds=100.0))
    with ThreadPoolExecutor(max_workers=2) as executor:
        older_future = executor.submit(coordinator.resolve, first)
        assert entered.wait(timeout=2.0)
        second = coordinator.reserve(SESSION_ID, ControlValues(reynolds=200.0))
        latest = coordinator.resolve(second)
        release.set()
        older = older_future.result(timeout=2.0)

    assert latest.phase == "ready"
    assert latest.response is not None
    assert latest.response.consistency.head_field_gap == "red"
    assert older.phase == "stale"
    assert older.applied is False


def test_failures_and_identity_drift_are_sanitized_for_retry() -> None:
    class FailingPredictor:
        def predict(self, request: PredictionRequest) -> PredictionResponse:
            del request
            raise RuntimeError("secret path /private/model and token")

    class DriftedPredictor:
        def predict(self, request: PredictionRequest) -> PredictionResponse:
            return _response(model_id="9" * 20, reynolds=request.reynolds)

    clock = FakeClock()
    for predictor in (FailingPredictor(), DriftedPredictor()):
        coordinator = PredictionCoordinator(
            predictor,
            _identity(),
            clock=clock,
            sleeper=clock.sleep,
        )
        outcome = coordinator.resolve(coordinator.reserve(SESSION_ID, ControlValues()))
        assert outcome.phase == "error"
        assert outcome.applied is True
        assert outcome.message == "Prediction unavailable. Retry the current design."
        assert "secret" not in (outcome.message or "")


def test_session_registry_is_bounded_and_expired_sequences_restart() -> None:
    predictor = RecordingPredictor()
    clock = FakeClock()
    coordinator = PredictionCoordinator(
        predictor,
        _identity(),
        clock=clock,
        sleeper=clock.sleep,
        session_ttl_seconds=1.0,
        max_sessions=2,
    )
    oldest = coordinator.reserve("browser_session_0000000001", ControlValues())
    coordinator.reserve("browser_session_0000000002", ControlValues())
    coordinator.reserve("browser_session_0000000003", ControlValues())

    assert coordinator.resolve(oldest).phase == "stale"
    clock.advance(1.0)
    restarted = coordinator.reserve("browser_session_0000000002", ControlValues())
    assert restarted.sequence == 1


def test_prediction_view_keeps_validation_and_consistency_explicit() -> None:
    response = _response(reynolds=200.0)

    view = prediction_view(_identity(), response, ControlValues(reynolds=200.0))

    assert "VALIDATION BLOCKED" in view.validation_html
    assert "Surrogate unvalidated: one or more required gates failed" in view.validation_html
    assert view.validation_html.index("Model") < view.validation_html.index("validation report")
    assert 'role="status"' in view.validation_html
    assert "FAIL — model-head and field Cd agreement" in view.consistency_html
    assert "Measured 12.500% gap; required ≤ 10.000%." in view.consistency_html
    assert "PASS — velocity divergence" in view.consistency_html
    assert "FAIL — obstacle velocity compliance" in view.consistency_html
    assert "data:image/png;base64," in view.fields_html
    assert 'alt="Surrogate prediction with velocity magnitude' in view.fields_html
    assert "Cd, model head" in view.metrics_html
    assert "42.5 ms" in view.status_html

    green_view = prediction_view(
        _identity(validation_status="green"),
        _response(validation_status="green"),
        ControlValues(),
    )
    assert "Validated against all v0.1 gates" in green_view.validation_html
    assert "PASS" in green_view.validation_html


def test_demo_component_tree_is_responsive_labeled_and_release_only() -> None:
    predictor = RecordingPredictor()

    demo = build_demo(predictor, _identity())
    config = cast(dict[str, Any], demo.get_config_file())
    components = cast(list[dict[str, Any]], config["components"])
    by_element = {
        component.get("props", {}).get("elem_id"): component
        for component in components
        if component.get("props", {}).get("elem_id")
    }

    expected_sliders = {
        "control-aspect-ratio": ("Ellipse aspect ratio", 0.5, 1.0, 0.65, 0.01),
        "control-rotation": ("Rotation (degrees)", 0.0, 30.0, 0.0, 0.5),
        "control-scale": ("Ellipse scale", 0.75, 1.25, 1.0, 0.01),
        "control-reynolds": ("Reynolds number", 40.0, 300.0, 100.0, 1.0),
    }
    for element_id, expected in expected_sliders.items():
        component = by_element[element_id]
        props = component["props"]
        assert component["type"] == "slider"
        assert (
            props["label"],
            props["minimum"],
            props["maximum"],
            props["value"],
            props["step"],
        ) == expected
        assert props["info"]

    component_ids = [component.get("props", {}).get("elem_id") for component in components]
    assert component_ids.index("validation-banner") < component_ids.index("demo-heading")
    assert component_ids.index("validation-banner") < component_ids.index("prediction-fields")
    assert "VALIDATION BLOCKED" in by_element["validation-banner"]["props"]["value"]
    assert "@media (max-width: 760px)" in cast(str, config["css"])
    assert ":focus-visible" in cast(str, config["css"])
    assert config["css"] == DEMO_CSS

    dependencies = cast(list[dict[str, Any]], config["dependencies"])
    targets = [target for dependency in dependencies for target in dependency["targets"]]
    assert sum(event == "release" for _, event in targets) == 4
    assert sum(event == "click" for _, event in targets) == 1
    assert not {"change", "input"}.intersection(event for _, event in targets)
    for function in list(demo.fns.values())[:5]:
        assert function.trigger_mode == "multiple"
        assert function.show_progress == "hidden"
        assert function.concurrency_limit == 8
        assert function.concurrency_id == "soufflerie-prediction"
        assert function.api_visibility == "private"


def test_demo_stream_exposes_loading_ready_and_retry_without_clearing_results() -> None:
    predictor = RecordingPredictor()
    demo = build_demo(predictor, _identity())
    submit = cast(Any, demo.fns[0].fn)

    loading, ready = tuple(submit(0.65, 0.0, 1.0, 100.0, BrowserRequest()))
    assert "in progress" in loading[1]
    assert loading[5]["interactive"] is False
    assert "Prediction ready" in ready[1]
    assert "Prediction case" in ready[2]
    assert ready[5]["value"] == "Predict current design"

    class FailingPredictor:
        def predict(self, request: PredictionRequest) -> PredictionResponse:
            del request
            raise RuntimeError("private failure")

    failing_demo = build_demo(FailingPredictor(), _identity())
    failing_submit = cast(Any, failing_demo.fns[0].fn)
    _, retry = tuple(failing_submit(0.65, 0.0, 1.0, 100.0, BrowserRequest()))
    assert "Retry the current design" in retry[1]
    assert "previous successful result remains visible" in retry[1]
    assert retry[5]["value"] == "Retry prediction"
    assert all(item.get("__type__") == "update" for item in retry[2:5])


def test_missing_gradio_extra_has_an_actionable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def unavailable(name: str, package: str | None = None) -> Any:
        if name == "gradio":
            raise ImportError("not installed")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", unavailable)

    with pytest.raises(DependencyUnavailableError, match=r"soufflerie\[serve\]"):
        build_demo(RecordingPredictor(), _identity())
