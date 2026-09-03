"""Accessible Gradio prediction UI with deterministic stale-response rejection."""

from __future__ import annotations

import html
import importlib
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from soufflerie.errors import ArtifactIntegrityError, DependencyUnavailableError
from soufflerie.service.contracts import (
    PredictionRequest,
    PredictionResponse,
    ShapeRequest,
    ValidationStatus,
)

DEBOUNCE_SECONDS = 0.150
SESSION_TTL_SECONDS = 3_600.0
MAX_DEMO_SESSIONS = 1_024
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_CONTENT_ID_PATTERN = re.compile(r"^[0-9a-f]{20}$")

DEMO_CSS = """
#soufflerie-demo { max-width: 1280px; margin: 0 auto; }
#validation-banner .validation-card {
  align-items: flex-start;
  border: 2px solid var(--validation-border);
  border-radius: 12px;
  display: flex;
  gap: 0.8rem;
  padding: 0.9rem 1rem;
}
#validation-banner .validation-card--green {
  --validation-border: #18794e;
  background: #ecfdf3;
  color: #0f5132;
}
#validation-banner .validation-card--red {
  --validation-border: #b42318;
  background: #fff1f0;
  color: #7a271a;
}
#validation-banner .validation-mark {
  border: 2px solid currentColor;
  border-radius: 50%;
  flex: 0 0 1.6rem;
  font-weight: 800;
  line-height: 1.35rem;
  text-align: center;
}
#validation-banner p { margin: 0.25rem 0 0; }
#soufflerie-controls { display: grid; gap: 0.75rem; grid-template-columns: repeat(4, 1fr); }
#prediction-layout {
  display: grid;
  gap: 1rem;
  grid-template-columns: minmax(0, 2fr) minmax(18rem, 1fr);
}
#prediction-fields img { display: block; height: auto; max-width: 100%; width: 100%; }
.result-card, .consistency-card, .prediction-state {
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  padding: 0.8rem;
}
.prediction-state[role="alert"] { border-color: #b42318; }
.metric-grid { display: grid; gap: 0.45rem 1rem; grid-template-columns: auto 1fr; margin: 0; }
.metric-grid dt { font-weight: 700; }
.metric-grid dd { margin: 0; text-align: right; }
.consistency-list { display: grid; gap: 0.55rem; list-style: none; margin: 0.6rem 0 0; padding: 0; }
.consistency-item { border-left: 5px solid currentColor; padding: 0.35rem 0.55rem; }
.consistency-item--green { color: #0f6b43; }
.consistency-item--red { color: #9f2118; }
.consistency-label { color: #172033; display: block; font-weight: 700; }
#soufflerie-demo :focus-visible { outline: 3px solid #2563eb !important; outline-offset: 3px; }
@media (max-width: 760px) {
  #soufflerie-controls { grid-template-columns: 1fr; }
  #prediction-layout { grid-template-columns: 1fr; }
  #soufflerie-demo { padding-inline: 0.25rem; }
}
""".strip()


class DemoPredictor(Protocol):
    """Prediction boundary implemented by local and deployed adapters."""

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Return one validated response without performing implicit retry."""


@dataclass(frozen=True, slots=True)
class DemoIdentity:
    """Immutable model/report identity displayed before any prediction."""

    model_id: str
    dataset_id: str
    report_id: str
    validation_status: ValidationStatus
    report_href: str

    def __post_init__(self) -> None:
        for name in ("model_id", "dataset_id", "report_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or _CONTENT_ID_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be 20 lowercase hexadecimal characters")
        if self.validation_status not in {"green", "red"}:
            raise ValueError("validation_status must be green or red")
        if not isinstance(self.report_href, str) or not 1 <= len(self.report_href) <= 512:
            raise ValueError("report_href must contain 1..512 characters")
        parsed = urlsplit(self.report_href)
        is_local = (
            not parsed.scheme
            and not parsed.netloc
            and self.report_href.startswith("/")
            and not self.report_href.startswith("//")
        )
        is_https = (
            parsed.scheme == "https"
            and bool(parsed.netloc)
            and parsed.username is None
            and parsed.password is None
        )
        if not (is_local or is_https) or any(character.isspace() for character in self.report_href):
            raise ValueError("report_href must be a root-relative or credential-free HTTPS URL")


@dataclass(frozen=True, slots=True)
class ControlValues:
    """The four committed controls in their public units."""

    aspect_ratio: float = 0.65
    rotation_deg: float = 0.0
    scale: float = 1.0
    reynolds: float = 100.0

    def __post_init__(self) -> None:
        self.to_request()

    def to_request(self) -> PredictionRequest:
        """Validate controls through the canonical HTTP request contract."""

        return PredictionRequest(
            shape=ShapeRequest(
                aspect_ratio=self.aspect_ratio,
                rotation_deg=self.rotation_deg,
                scale=self.scale,
            ),
            reynolds=self.reynolds,
        )


@dataclass(frozen=True, slots=True)
class PredictionTicket:
    """One monotonic session request reserved at control commit time."""

    session_id: str
    sequence: int
    request: PredictionRequest
    eligible_at: float


PredictionPhase = Literal["ready", "error", "stale"]


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """Result of resolving a ticket; stale outcomes must never update the UI."""

    sequence: int
    phase: PredictionPhase
    response: PredictionResponse | None
    message: str | None

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("prediction outcome sequence must be positive")
        if self.phase == "ready" and (self.response is None or self.message is not None):
            raise ValueError("ready prediction outcome requires only a response")
        if self.phase == "error" and (self.response is not None or not self.message):
            raise ValueError("error prediction outcome requires only a message")
        if self.phase == "stale" and (self.response is not None or self.message is not None):
            raise ValueError("stale prediction outcome cannot contain display data")

    @property
    def applied(self) -> bool:
        return self.phase != "stale"


@dataclass(slots=True)
class _SessionRecord:
    sequence: int
    touched_at: float


class PredictionCoordinator:
    """Bounded per-browser debounce and last-commit-wins prediction coordinator."""

    def __init__(
        self,
        predictor: DemoPredictor,
        identity: DemoIdentity,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        debounce_seconds: float = DEBOUNCE_SECONDS,
        session_ttl_seconds: float = SESSION_TTL_SECONDS,
        max_sessions: int = MAX_DEMO_SESSIONS,
    ) -> None:
        if not callable(getattr(predictor, "predict", None)):
            raise TypeError("predictor must implement predict(request)")
        if not isinstance(identity, DemoIdentity):
            raise TypeError("identity must be DemoIdentity")
        for name, value in (
            ("debounce_seconds", debounce_seconds),
            ("session_ttl_seconds", session_ttl_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        self._predictor = predictor
        self._identity = identity
        self._clock = clock
        self._sleeper = sleeper
        self._debounce_seconds = float(debounce_seconds)
        self._session_ttl_seconds = float(session_ttl_seconds)
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, _SessionRecord] = OrderedDict()
        self._lock = threading.Lock()

    def _now(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RuntimeError("demo clock must return a finite monotonic timestamp")
        return float(value)

    def _prune(self, now: float) -> None:
        expired = [
            session_id
            for session_id, record in self._sessions.items()
            if now - record.touched_at >= self._session_ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def reserve(self, session_id: str, controls: ControlValues) -> PredictionTicket:
        """Reserve the next sequence and begin the coalescing window."""

        if not isinstance(session_id, str) or _SESSION_ID_PATTERN.fullmatch(session_id) is None:
            raise ValueError("session_id does not match the bounded browser-session format")
        if not isinstance(controls, ControlValues):
            raise TypeError("controls must be ControlValues")
        request = controls.to_request()
        now = self._now()
        with self._lock:
            self._prune(now)
            existing = self._sessions.pop(session_id, None)
            sequence = 1 if existing is None else existing.sequence + 1
            if existing is None and len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            self._sessions[session_id] = _SessionRecord(sequence=sequence, touched_at=now)
        return PredictionTicket(
            session_id=session_id,
            sequence=sequence,
            request=request,
            eligible_at=now + self._debounce_seconds,
        )

    def _is_latest(self, ticket: PredictionTicket, *, now: float) -> bool:
        with self._lock:
            record = self._sessions.get(ticket.session_id)
            if record is None or record.sequence != ticket.sequence:
                return False
            record.touched_at = now
            self._sessions.move_to_end(ticket.session_id)
            return True

    def _identities_match(self, response: PredictionResponse) -> bool:
        return (
            response.model_id == self._identity.model_id
            and response.dataset_id == self._identity.dataset_id
            and response.report_id == self._identity.report_id
            and response.validation_status == self._identity.validation_status
        )

    def resolve(self, ticket: PredictionTicket) -> PredictionOutcome:
        """Resolve a ticket once; superseded work returns no display payload."""

        if not isinstance(ticket, PredictionTicket):
            raise TypeError("ticket must be PredictionTicket")
        remaining = ticket.eligible_at - self._now()
        if remaining > 0.0:
            self._sleeper(remaining)
        if not self._is_latest(ticket, now=self._now()):
            return PredictionOutcome(
                sequence=ticket.sequence,
                phase="stale",
                response=None,
                message=None,
            )
        try:
            response = self._predictor.predict(ticket.request)
            if not isinstance(response, PredictionResponse):
                raise ArtifactIntegrityError("demo predictor returned an invalid response type")
            if not self._identities_match(response):
                raise ArtifactIntegrityError("demo prediction identities differ from loaded state")
        except Exception:
            if not self._is_latest(ticket, now=self._now()):
                return PredictionOutcome(
                    sequence=ticket.sequence,
                    phase="stale",
                    response=None,
                    message=None,
                )
            return PredictionOutcome(
                sequence=ticket.sequence,
                phase="error",
                response=None,
                message="Prediction unavailable. Retry the current design.",
            )
        if not self._is_latest(ticket, now=self._now()):
            return PredictionOutcome(
                sequence=ticket.sequence,
                phase="stale",
                response=None,
                message=None,
            )
        return PredictionOutcome(
            sequence=ticket.sequence,
            phase="ready",
            response=response,
            message=None,
        )


@dataclass(frozen=True, slots=True)
class PredictionView:
    """Sanitized HTML fragments for one applied prediction state."""

    validation_html: str
    status_html: str
    fields_html: str
    metrics_html: str
    consistency_html: str


def _validation_banner(identity: DemoIdentity) -> str:
    is_green = identity.validation_status == "green"
    headline = (
        "Validated against all v0.1 gates"
        if is_green
        else "Surrogate unvalidated: one or more required gates failed"
    )
    qualifier = "PASS" if is_green else "VALIDATION BLOCKED"
    mark = "✓" if is_green else "!"
    css_status = "green" if is_green else "red"
    return (
        f'<section class="validation-card validation-card--{css_status}" role="status" '
        'aria-live="polite">'
        f'<span class="validation-mark" aria-hidden="true">{mark}</span><div>'
        f"<strong>{qualifier} — {headline}</strong>"
        f"<p>Model <code>{html.escape(identity.model_id)}</code> · "
        f'<a href="{html.escape(identity.report_href, quote=True)}">'
        f"validation report {html.escape(identity.report_id)}</a></p></div></section>"
    )


def _prediction_image(response: PredictionResponse, controls: ControlValues) -> str:
    alt_text = (
        "Surrogate prediction with velocity magnitude, pressure proxy, and vorticity panels "
        f"for aspect ratio {controls.aspect_ratio:.2f}, rotation {controls.rotation_deg:.1f} "
        f"degrees, scale {controls.scale:.2f}, and Reynolds number {controls.reynolds:.0f}."
    )
    return (
        '<figure class="result-card"><img '
        f'src="data:image/png;base64,{response.fields_png.data}" '
        f'alt="{html.escape(alt_text, quote=True)}" />'
        f"<figcaption>Prediction case <code>{html.escape(response.case_id)}</code></figcaption>"
        "</figure>"
    )


def _prediction_metrics(response: PredictionResponse) -> str:
    return (
        '<section class="result-card" aria-labelledby="prediction-metrics-heading">'
        '<h3 id="prediction-metrics-heading">Prediction summary</h3><dl class="metric-grid">'
        f"<dt>Cd, model head</dt><dd>{response.cd_head:.5f}</dd>"
        f"<dt>Cd, field estimate</dt><dd>{response.cd_field:.5f}</dd>"
        f"<dt>Inference</dt><dd>{response.inference_ms:.1f} ms</dd>"
        f"<dt>Total request</dt><dd>{response.request_ms:.1f} ms</dd>"
        f"<dt>Model</dt><dd><code>{html.escape(response.model_id)}</code></dd>"
        f"<dt>Dataset</dt><dd><code>{html.escape(response.dataset_id)}</code></dd>"
        "</dl></section>"
    )


def _flag_item(*, label: str, status: str, measured: str, threshold: str) -> str:
    passed = status == "green"
    word = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "!"
    css_status = "green" if passed else "red"
    return (
        f'<li class="consistency-item consistency-item--{css_status}">'
        f'<span class="consistency-label">{symbol} {word} — {html.escape(label)}</span>'
        f"Measured {html.escape(measured)}; required {html.escape(threshold)}.</li>"
    )


def _consistency_panel(response: PredictionResponse) -> str:
    flags = response.consistency
    items = (
        _flag_item(
            label="model-head and field Cd agreement",
            status=flags.head_field_gap,
            measured=f"{flags.head_field_gap_pct:.3f}% gap",
            threshold="≤ 10.000%",
        ),
        _flag_item(
            label="velocity divergence",
            status=flags.divergence,
            measured=f"{flags.divergence_ratio_to_solver_baseline:.4f} times solver baseline",
            threshold="< 3.0000 times solver baseline",
        ),
        _flag_item(
            label="obstacle velocity compliance",
            status=flags.obstacle_compliance,
            measured=f"{flags.obstacle_velocity_ratio:.6f} ratio",
            threshold="< 0.010000 ratio",
        ),
    )
    return (
        '<section class="consistency-card" aria-labelledby="consistency-heading">'
        '<h3 id="consistency-heading">Per-case consistency</h3>'
        "<p>Each state includes text, a symbol, its measured value, and the fixed threshold.</p>"
        f'<ul class="consistency-list">{"".join(items)}</ul></section>'
    )


def prediction_view(
    identity: DemoIdentity,
    response: PredictionResponse,
    controls: ControlValues,
) -> PredictionView:
    """Build one escaped, non-color-only ready state from validated contracts."""

    if not isinstance(identity, DemoIdentity):
        raise TypeError("identity must be DemoIdentity")
    if not isinstance(response, PredictionResponse):
        raise TypeError("response must be PredictionResponse")
    if not isinstance(controls, ControlValues):
        raise TypeError("controls must be ControlValues")
    if (
        response.model_id != identity.model_id
        or response.dataset_id != identity.dataset_id
        or response.report_id != identity.report_id
        or response.validation_status != identity.validation_status
    ):
        raise ArtifactIntegrityError("prediction view identities differ from loaded demo state")
    return PredictionView(
        validation_html=_validation_banner(identity),
        status_html=(
            '<div class="prediction-state" role="status" aria-live="polite">'
            f"Prediction ready in {response.request_ms:.1f} ms.</div>"
        ),
        fields_html=_prediction_image(response, controls),
        metrics_html=_prediction_metrics(response),
        consistency_html=_consistency_panel(response),
    )


def _initial_view(identity: DemoIdentity) -> PredictionView:
    return PredictionView(
        validation_html=_validation_banner(identity),
        status_html=(
            '<div class="prediction-state" role="status" aria-live="polite">'
            "Commit a control or press Predict current design.</div>"
        ),
        fields_html=(
            '<section class="result-card" aria-label="Prediction fields">'
            "Prediction fields will appear here.</section>"
        ),
        metrics_html=(
            '<section class="result-card" aria-label="Prediction summary">'
            "Prediction metrics will appear here.</section>"
        ),
        consistency_html=(
            '<section class="consistency-card" aria-label="Per-case consistency">'
            "Consistency measurements will appear here.</section>"
        ),
    )


def _gradio() -> Any:
    try:
        return importlib.import_module("gradio")
    except ImportError as error:
        raise DependencyUnavailableError(
            "interactive demo requires the 'serve' extra; install soufflerie[serve]"
        ) from error


def build_demo(
    predictor: DemoPredictor,
    identity: DemoIdentity,
) -> Any:
    """Build the reusable Gradio component tree without launching a server."""

    if not callable(getattr(predictor, "predict", None)):
        raise TypeError("predictor must implement predict(request)")
    if not isinstance(identity, DemoIdentity):
        raise TypeError("identity must be DemoIdentity")
    gradio = _gradio()
    active_coordinator = PredictionCoordinator(predictor, identity)
    initial = _initial_view(identity)

    with gradio.Blocks(title="Soufflerie wind tunnel") as demo:
        with gradio.Column(elem_id="soufflerie-demo"):
            validation = gradio.HTML(
                initial.validation_html,
                elem_id="validation-banner",
                container=False,
            )
            gradio.Markdown(
                "# Soufflerie\nExplore one bounded ellipse design with the loaded surrogate.",
                elem_id="demo-heading",
            )
            with gradio.Row(elem_id="soufflerie-controls"):
                aspect_ratio = gradio.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.65,
                    step=0.01,
                    label="Ellipse aspect ratio",
                    info="0.50 to 1.00",
                    elem_id="control-aspect-ratio",
                )
                rotation_deg = gradio.Slider(
                    minimum=0.0,
                    maximum=30.0,
                    value=0.0,
                    step=0.5,
                    label="Rotation (degrees)",
                    info="0 to 30 degrees",
                    elem_id="control-rotation",
                )
                scale = gradio.Slider(
                    minimum=0.75,
                    maximum=1.25,
                    value=1.0,
                    step=0.01,
                    label="Ellipse scale",
                    info="0.75 to 1.25",
                    elem_id="control-scale",
                )
                reynolds = gradio.Slider(
                    minimum=40.0,
                    maximum=300.0,
                    value=100.0,
                    step=1.0,
                    label="Reynolds number",
                    info="40 to 300",
                    elem_id="control-reynolds",
                )
            predict_button = gradio.Button(
                "Predict current design",
                variant="primary",
                elem_id="predict-button",
            )
            status = gradio.HTML(initial.status_html, elem_id="prediction-status", container=False)
            with gradio.Row(elem_id="prediction-layout"):
                fields = gradio.HTML(
                    initial.fields_html,
                    elem_id="prediction-fields",
                    container=False,
                )
                with gradio.Column():
                    metrics = gradio.HTML(
                        initial.metrics_html,
                        elem_id="prediction-metrics",
                        container=False,
                    )
                    consistency = gradio.HTML(
                        initial.consistency_html,
                        elem_id="prediction-consistency",
                        container=False,
                    )

        inputs = [aspect_ratio, rotation_deg, scale, reynolds]
        outputs = [validation, status, fields, metrics, consistency, predict_button]

        def submit(
            aspect: float,
            rotation: float,
            ellipse_scale: float,
            reynolds_number: float,
            browser_request: Any,
        ) -> Iterator[tuple[Any, ...]]:
            try:
                raw_session = getattr(browser_request, "session_hash", None)
                browser_session = raw_session if isinstance(raw_session, str) else ""
                controls = ControlValues(
                    aspect_ratio=aspect,
                    rotation_deg=rotation,
                    scale=ellipse_scale,
                    reynolds=reynolds_number,
                )
                ticket = active_coordinator.reserve(browser_session, controls)
            except (TypeError, ValueError):
                yield (
                    gradio.skip(),
                    (
                        '<div class="prediction-state" role="alert">'
                        "Invalid controls. Restore values within the documented ranges.</div>"
                    ),
                    gradio.skip(),
                    gradio.skip(),
                    gradio.skip(),
                    gradio.update(value="Retry prediction", interactive=True),
                )
                return
            yield (
                gradio.skip(),
                (
                    '<div class="prediction-state" role="status" aria-live="polite">'
                    f"Prediction {ticket.sequence} in progress…</div>"
                ),
                gradio.skip(),
                gradio.skip(),
                gradio.skip(),
                gradio.update(value="Predicting…", interactive=False),
            )
            outcome = active_coordinator.resolve(ticket)
            if not outcome.applied:
                return
            if outcome.phase == "error":
                yield (
                    gradio.skip(),
                    (
                        '<div class="prediction-state" role="alert">'
                        f"{html.escape(outcome.message or 'Prediction unavailable.')} "
                        "The previous successful result remains visible.</div>"
                    ),
                    gradio.skip(),
                    gradio.skip(),
                    gradio.skip(),
                    gradio.update(value="Retry prediction", interactive=True),
                )
                return
            response = outcome.response
            if response is None:
                raise RuntimeError("ready prediction outcome lost its response")
            view = prediction_view(identity, response, controls)
            yield (
                view.validation_html,
                view.status_html,
                view.fields_html,
                view.metrics_html,
                view.consistency_html,
                gradio.update(value="Predict current design", interactive=True),
            )

        submit.__annotations__["browser_request"] = gradio.Request
        for control in (aspect_ratio, rotation_deg, scale, reynolds):
            control.release(
                submit,
                inputs=inputs,
                outputs=outputs,
                api_visibility="private",
                show_progress="hidden",
                trigger_mode="multiple",
                concurrency_limit=8,
                concurrency_id="soufflerie-prediction",
            )
        predict_button.click(
            submit,
            inputs=inputs,
            outputs=outputs,
            api_visibility="private",
            show_progress="hidden",
            trigger_mode="multiple",
            concurrency_limit=8,
            concurrency_id="soufflerie-prediction",
        )

    demo.css = DEMO_CSS
    return demo.queue(max_size=8, default_concurrency_limit=8)


__all__ = [
    "DEBOUNCE_SECONDS",
    "DEMO_CSS",
    "MAX_DEMO_SESSIONS",
    "SESSION_TTL_SECONDS",
    "ControlValues",
    "DemoIdentity",
    "DemoPredictor",
    "PredictionCoordinator",
    "PredictionOutcome",
    "PredictionPhase",
    "PredictionTicket",
    "PredictionView",
    "build_demo",
    "prediction_view",
]
