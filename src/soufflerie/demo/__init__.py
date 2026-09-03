"""Deterministic field visualization adapters."""

from soufflerie.demo.rendering import (
    PanelMetadata,
    PngArtifact,
    RenderSpec,
    render_comparison,
    render_fields,
)
from soufflerie.demo.ui import (
    DEBOUNCE_SECONDS,
    DEMO_CSS,
    MAX_DEMO_SESSIONS,
    SESSION_TTL_SECONDS,
    ControlValues,
    DemoIdentity,
    DemoPredictor,
    PredictionCoordinator,
    PredictionOutcome,
    PredictionPhase,
    PredictionTicket,
    PredictionView,
    build_demo,
    prediction_view,
)

__all__ = [
    "DEBOUNCE_SECONDS",
    "DEMO_CSS",
    "MAX_DEMO_SESSIONS",
    "SESSION_TTL_SECONDS",
    "ControlValues",
    "DemoIdentity",
    "DemoPredictor",
    "PanelMetadata",
    "PngArtifact",
    "PredictionCoordinator",
    "PredictionOutcome",
    "PredictionPhase",
    "PredictionTicket",
    "PredictionView",
    "RenderSpec",
    "build_demo",
    "prediction_view",
    "render_comparison",
    "render_fields",
]
