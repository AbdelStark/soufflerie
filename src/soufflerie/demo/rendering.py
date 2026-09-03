"""Deterministic, non-mutating field and comparison PNG rendering."""

from __future__ import annotations

import importlib
import io
import math
import struct
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from soufflerie.errors import ArtifactIntegrityError, DependencyUnavailableError
from soufflerie.schemas import FlowFields, canonical_sha256, sha256_bytes

RenderRole = Literal["standalone", "reference", "prediction", "error"]
RenderVariable = Literal["velocity_magnitude", "pressure_proxy", "vorticity"]
ColorMap = Literal["viridis", "coolwarm", "RdBu_r", "magma"]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
RgbaArray = npt.NDArray[np.uint8]

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_TEXT_CHARACTERS = 240
_MAX_ALT_TEXT_CHARACTERS = 768
_MAX_PIXELS = 4_000_000
_MIN_SCALE = float(np.finfo(np.float32).eps)
_RENDER_LOCK = threading.Lock()
_ROLES = frozenset({"standalone", "reference", "prediction", "error"})
_VARIABLES = frozenset({"velocity_magnitude", "pressure_proxy", "vorticity"})
_COLORMAPS = frozenset({"viridis", "coolwarm", "RdBu_r", "magma"})


def _clean_text(value: object, *, label: str, maximum: int = _MAX_TEXT_CHARACTERS) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    cleaned = " ".join(value.split())
    if not cleaned or len(cleaned) > maximum or not cleaned.isprintable():
        raise ValueError(f"{label} must be 1..{maximum} printable characters")
    return cleaned


@dataclass(frozen=True, slots=True)
class RenderSpec:
    """Closed v0.1 raster style and bounded annotation contract."""

    schema_version: Literal[1] = 1
    width_px: int = 1_200
    height_px: int = 400
    dpi: int = 100
    colormap_velocity: Literal["viridis"] = "viridis"
    colormap_pressure: Literal["coolwarm"] = "coolwarm"
    colormap_vorticity: Literal["RdBu_r"] = "RdBu_r"
    colormap_error: Literal["magma"] = "magma"
    spacing_lu: float = 2.0
    annotation: str = "Ellipse flow | x-y lattice coordinates"
    provenance: str = "Source: verified Soufflerie fields"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("render schema_version must equal 1")
        for name, value, lower, upper in (
            ("width_px", self.width_px, 600, 2_400),
            ("height_px", self.height_px, 300, 1_600),
            ("dpi", self.dpi, 72, 300),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if self.width_px * self.height_px > _MAX_PIXELS:
            raise ValueError(f"render output must contain at most {_MAX_PIXELS} pixels")
        expected_colormaps = {
            "colormap_velocity": "viridis",
            "colormap_pressure": "coolwarm",
            "colormap_vorticity": "RdBu_r",
            "colormap_error": "magma",
        }
        for name, expected in expected_colormaps.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must remain {expected!r} in schema v1")
        if (
            isinstance(self.spacing_lu, bool)
            or not isinstance(self.spacing_lu, (int, float))
            or not math.isfinite(self.spacing_lu)
            or self.spacing_lu <= 0.0
        ):
            raise ValueError("spacing_lu must be finite and positive")
        object.__setattr__(self, "annotation", _clean_text(self.annotation, label="annotation"))
        object.__setattr__(self, "provenance", _clean_text(self.provenance, label="provenance"))


@dataclass(frozen=True, slots=True)
class PanelMetadata:
    """Auditable variable scale and accessible label for one raster panel."""

    role: RenderRole
    variable: RenderVariable
    title: str
    units: str
    colormap: ColorMap
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ArtifactIntegrityError("render panel role is unsupported")
        if self.variable not in _VARIABLES:
            raise ArtifactIntegrityError("render panel variable is unsupported")
        if self.colormap not in _COLORMAPS:
            raise ArtifactIntegrityError("render panel colormap is unsupported")
        object.__setattr__(self, "title", _clean_text(self.title, label="panel title"))
        object.__setattr__(self, "units", _clean_text(self.units, label="panel units"))
        if (
            isinstance(self.minimum, bool)
            or not isinstance(self.minimum, (int, float))
            or isinstance(self.maximum, bool)
            or not isinstance(self.maximum, (int, float))
            or not math.isfinite(self.minimum)
            or not math.isfinite(self.maximum)
        ):
            raise ArtifactIntegrityError("render panel scale must be finite")
        if self.minimum >= self.maximum:
            raise ArtifactIntegrityError("render panel scale minimum must precede maximum")


@dataclass(frozen=True, slots=True)
class PngArtifact:
    """Self-verifying raster plus semantic rendering evidence."""

    data: bytes = field(repr=False)
    sha256: str
    contract_sha256: str
    width_px: int
    height_px: int
    alt_text: str
    panels: tuple[PanelMetadata, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.width_px, bool)
            or not isinstance(self.width_px, int)
            or isinstance(self.height_px, bool)
            or not isinstance(self.height_px, int)
            or self.width_px <= 0
            or self.height_px <= 0
        ):
            raise ArtifactIntegrityError("render artifact dimensions must be positive integers")
        if (
            not isinstance(self.panels, tuple)
            or len(self.panels) not in {3, 9}
            or not all(isinstance(panel, PanelMetadata) for panel in self.panels)
        ):
            raise ArtifactIntegrityError("render artifact must describe three or nine panels")
        if not isinstance(self.data, bytes) or not self.data.startswith(_PNG_SIGNATURE):
            raise ArtifactIntegrityError("render artifact must contain PNG bytes")
        if self.sha256 != sha256_bytes(self.data):
            raise ArtifactIntegrityError("render PNG digest does not match its bytes")
        if len(self.data) < 24 or self.data[12:16] != b"IHDR":
            raise ArtifactIntegrityError("render artifact has no canonical PNG header")
        width, height = struct.unpack(">II", self.data[16:24])
        if (width, height) != (self.width_px, self.height_px):
            raise ArtifactIntegrityError("render PNG dimensions do not match metadata")
        object.__setattr__(
            self,
            "alt_text",
            _clean_text(self.alt_text, label="alt_text", maximum=_MAX_ALT_TEXT_CHARACTERS),
        )
        expected_contract = canonical_sha256(
            {
                "schema_version": 1,
                "width_px": self.width_px,
                "height_px": self.height_px,
                "alt_text": self.alt_text,
                "panels": [asdict(panel) for panel in self.panels],
            }
        )
        if self.contract_sha256 != expected_contract:
            raise ArtifactIntegrityError("render contract digest does not match metadata")

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class _FieldSnapshot:
    u: FloatArray
    v: FloatArray
    rho: FloatArray
    obstacle_mask: BoolArray


@dataclass(frozen=True, slots=True)
class _Panel:
    data: FloatArray
    mask: BoolArray
    metadata: PanelMetadata


def _snapshot(fields: FlowFields, *, label: str) -> _FieldSnapshot:
    if not isinstance(fields, FlowFields):
        raise TypeError(f"{label} must be FlowFields")
    shape = fields.shape
    if len(shape) != 2 or min(shape) < 3:
        raise ArtifactIntegrityError(f"{label} fields must be at least 3x3")
    arrays: dict[str, FloatArray] = {}
    for name in ("u", "v", "rho", "sdf"):
        value = getattr(fields, name)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != np.dtype(np.float32)
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise ArtifactIntegrityError(
                f"{label} {name} must remain finite contiguous float32 at the declared shape"
            )
        arrays[name] = np.array(value, dtype=np.float64, order="C", copy=True)
    mask_value = fields.obstacle_mask
    if (
        not isinstance(mask_value, np.ndarray)
        or mask_value.shape != shape
        or mask_value.dtype != np.dtype(np.bool_)
        or not mask_value.flags.c_contiguous
    ):
        raise ArtifactIntegrityError(f"{label} obstacle mask is invalid")
    mask = np.array(mask_value, dtype=np.bool_, order="C", copy=True)
    if not np.all(arrays["rho"] > 0.0):
        raise ArtifactIntegrityError(f"{label} density must remain strictly positive")
    if not np.array_equal(mask, arrays["sdf"] <= 0.0):
        raise ArtifactIntegrityError(f"{label} obstacle mask no longer matches the SDF")
    if not np.any(~mask):
        raise ArtifactIntegrityError(f"{label} has no fluid cells to render")
    return _FieldSnapshot(
        u=arrays["u"],
        v=arrays["v"],
        rho=arrays["rho"],
        obstacle_mask=mask,
    )


def _velocity(fields: _FieldSnapshot) -> FloatArray:
    return np.sqrt(fields.u * fields.u + fields.v * fields.v)


def _pressure(fields: _FieldSnapshot) -> FloatArray:
    return (fields.rho - 1.0) / 3.0


def _vorticity(fields: _FieldSnapshot, *, spacing_lu: float) -> FloatArray:
    dv_dx = np.gradient(fields.v, spacing_lu, axis=1, edge_order=2)
    du_dy = np.gradient(fields.u, spacing_lu, axis=0, edge_order=2)
    return np.ascontiguousarray(dv_dx - du_dy, dtype=np.float64)


def _positive_percentile(values: FloatArray, mask: BoolArray, percentile: float) -> float:
    fluid = values[~mask]
    if not np.isfinite(fluid).all():
        raise ArtifactIntegrityError("render scale input contains NaN or infinity")
    return max(float(np.percentile(fluid, percentile, method="linear")), _MIN_SCALE)


def _absolute_limit(values: FloatArray, mask: BoolArray, percentile: float | None) -> float:
    absolute = np.abs(values[~mask])
    if not np.isfinite(absolute).all():
        raise ArtifactIntegrityError("render scale input contains NaN or infinity")
    limit = (
        float(np.max(absolute))
        if percentile is None
        else float(np.percentile(absolute, percentile, method="linear"))
    )
    return max(limit, _MIN_SCALE)


def _base_panels(
    fields: _FieldSnapshot,
    spec: RenderSpec,
    *,
    role: Literal["standalone", "reference"],
) -> tuple[_Panel, _Panel, _Panel]:
    velocity = _velocity(fields)
    pressure = _pressure(fields)
    vorticity = _vorticity(fields, spacing_lu=spec.spacing_lu)
    velocity_max = _positive_percentile(velocity, fields.obstacle_mask, 99.5)
    pressure_abs = _absolute_limit(pressure, fields.obstacle_mask, None)
    vorticity_abs = _absolute_limit(vorticity, fields.obstacle_mask, 99.0)
    prefix = "" if role == "standalone" else "Reference · "
    return (
        _Panel(
            velocity,
            fields.obstacle_mask,
            PanelMetadata(
                role=role,
                variable="velocity_magnitude",
                title=f"{prefix}Velocity magnitude",
                units="lattice velocity",
                colormap=spec.colormap_velocity,
                minimum=0.0,
                maximum=velocity_max,
            ),
        ),
        _Panel(
            pressure,
            fields.obstacle_mask,
            PanelMetadata(
                role=role,
                variable="pressure_proxy",
                title=f"{prefix}Pressure proxy",
                units="(rho - 1) / 3",
                colormap=spec.colormap_pressure,
                minimum=-pressure_abs,
                maximum=pressure_abs,
            ),
        ),
        _Panel(
            vorticity,
            fields.obstacle_mask,
            PanelMetadata(
                role=role,
                variable="vorticity",
                title=f"{prefix}Vorticity",
                units="inverse lattice unit",
                colormap=spec.colormap_vorticity,
                minimum=-vorticity_abs,
                maximum=vorticity_abs,
            ),
        ),
    )


def _comparison_panels(
    prediction: _FieldSnapshot,
    reference: _FieldSnapshot,
    spec: RenderSpec,
) -> tuple[_Panel, ...]:
    if prediction.u.shape != reference.u.shape:
        raise ArtifactIntegrityError("prediction and reference render shapes differ")
    if not np.array_equal(prediction.obstacle_mask, reference.obstacle_mask):
        raise ArtifactIntegrityError("prediction and reference obstacle masks differ")
    reference_panels = _base_panels(reference, spec, role="reference")
    prediction_values = (
        _velocity(prediction),
        _pressure(prediction),
        _vorticity(prediction, spacing_lu=spec.spacing_lu),
    )
    labels = ("Velocity magnitude", "Pressure proxy", "Vorticity")
    prediction_panels = tuple(
        _Panel(
            data,
            prediction.obstacle_mask,
            PanelMetadata(
                role="prediction",
                variable=reference_panel.metadata.variable,
                title=f"Prediction · {label}",
                units=reference_panel.metadata.units,
                colormap=reference_panel.metadata.colormap,
                minimum=reference_panel.metadata.minimum,
                maximum=reference_panel.metadata.maximum,
            ),
        )
        for data, label, reference_panel in zip(
            prediction_values, labels, reference_panels, strict=True
        )
    )
    error_values = (
        np.sqrt((prediction.u - reference.u) ** 2 + (prediction.v - reference.v) ** 2),
        prediction_values[1] - reference_panels[1].data,
        prediction_values[2] - reference_panels[2].data,
    )
    velocity_error_max = _positive_percentile(error_values[0], reference.obstacle_mask, 99.5)
    pressure_error_abs = _absolute_limit(error_values[1], reference.obstacle_mask, None)
    vorticity_error_abs = _absolute_limit(error_values[2], reference.obstacle_mask, 99.0)
    error_limits: tuple[tuple[float, float, ColorMap, str], ...] = (
        (0.0, velocity_error_max, spec.colormap_error, "vector magnitude error"),
        (-pressure_error_abs, pressure_error_abs, spec.colormap_pressure, "signed proxy error"),
        (
            -vorticity_error_abs,
            vorticity_error_abs,
            spec.colormap_vorticity,
            "signed inverse lattice unit error",
        ),
    )
    error_panels = tuple(
        _Panel(
            data,
            reference.obstacle_mask,
            PanelMetadata(
                role="error",
                variable=reference_panel.metadata.variable,
                title=f"Error · {label}",
                units=units,
                colormap=colormap,
                minimum=minimum,
                maximum=maximum,
            ),
        )
        for data, label, reference_panel, (minimum, maximum, colormap, units) in zip(
            error_values, labels, reference_panels, error_limits, strict=True
        )
    )
    return (*reference_panels, *prediction_panels, *error_panels)


def _matplotlib() -> Any:
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg", force=True)
        pyplot = importlib.import_module("matplotlib.pyplot")
    except ImportError as error:
        raise DependencyUnavailableError(
            "field rendering requires the 'viz' extra; install soufflerie[viz]"
        ) from error
    return pyplot


def _obstacle_overlay(mask: BoolArray) -> RgbaArray:
    """Return a neutral opaque obstacle layer with transparent fluid cells."""

    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask] = (75, 85, 99, 255)
    return overlay


def _render(
    panels: tuple[_Panel, ...],
    spec: RenderSpec,
    *,
    rows: int,
    alt_text: str,
) -> PngArtifact:
    pyplot = _matplotlib()
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#5b6470",
        "text.color": "#1f2933",
        "axes.labelcolor": "#374151",
        "xtick.color": "#4b5563",
        "ytick.color": "#4b5563",
        "savefig.facecolor": "white",
    }
    output = io.BytesIO()
    with _RENDER_LOCK, pyplot.rc_context(style):
        figure, axes = pyplot.subplots(
            rows,
            3,
            figsize=(spec.width_px / spec.dpi, spec.height_px / spec.dpi),
            dpi=spec.dpi,
            squeeze=False,
        )
        try:
            for axis, panel in zip(axes.flat, panels, strict=True):
                ny, nx = panel.data.shape
                extent = (0.0, nx * spec.spacing_lu, 0.0, ny * spec.spacing_lu)
                image = axis.imshow(
                    panel.data,
                    origin="lower",
                    interpolation="nearest",
                    cmap=panel.metadata.colormap,
                    vmin=panel.metadata.minimum,
                    vmax=panel.metadata.maximum,
                    extent=extent,
                    aspect="equal",
                    rasterized=True,
                )
                axis.imshow(
                    _obstacle_overlay(panel.mask),
                    origin="lower",
                    interpolation="nearest",
                    extent=extent,
                    aspect="equal",
                    alpha=1.0,
                    rasterized=True,
                )
                axis.set_title(panel.metadata.title)
                axis.set_xlabel("x [lattice units]")
                axis.set_ylabel("y [lattice units]")
                axis.annotate(
                    "flow",
                    xy=(0.18, 0.91),
                    xytext=(0.04, 0.91),
                    xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 1.2},
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "#111827",
                        "edgecolor": "white",
                        "alpha": 0.88,
                        "linewidth": 0.4,
                    },
                    color="white",
                    fontsize=6,
                    fontweight="bold",
                    ha="left",
                    va="center",
                )
                colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
                colorbar.set_label(panel.metadata.units, fontsize=6)
                colorbar.ax.tick_params(labelsize=5, length=2)
            figure.suptitle(spec.annotation, fontsize=10, fontweight="bold", y=0.985)
            figure.text(
                0.5,
                0.012,
                spec.provenance,
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#4b5563",
            )
            figure.subplots_adjust(
                left=0.045,
                right=0.985,
                bottom=0.16 if rows == 1 else 0.09,
                top=0.84 if rows == 1 else 0.91,
                wspace=0.34,
                hspace=0.48,
            )
            figure.savefig(
                output,
                format="png",
                dpi=spec.dpi,
                metadata={
                    "Title": spec.annotation,
                    "Description": alt_text,
                    "Software": "Soufflerie 0.1.0",
                },
                pil_kwargs={"compress_level": 9},
            )
        finally:
            pyplot.close(figure)
    data = output.getvalue()
    metadata = tuple(panel.metadata for panel in panels)
    cleaned_alt = _clean_text(alt_text, label="alt_text", maximum=_MAX_ALT_TEXT_CHARACTERS)
    contract_sha256 = canonical_sha256(
        {
            "schema_version": 1,
            "width_px": spec.width_px,
            "height_px": spec.height_px,
            "alt_text": cleaned_alt,
            "panels": [asdict(panel) for panel in metadata],
        }
    )
    return PngArtifact(
        data=data,
        sha256=sha256_bytes(data),
        contract_sha256=contract_sha256,
        width_px=spec.width_px,
        height_px=spec.height_px,
        alt_text=cleaned_alt,
        panels=metadata,
    )


def render_fields(fields: FlowFields, spec: RenderSpec) -> PngArtifact:
    """Render velocity, pressure proxy, and vorticity without changing raw fields."""

    if not isinstance(spec, RenderSpec):
        raise TypeError("spec must be a RenderSpec")
    snapshot = _snapshot(fields, label="standalone")
    panels = _base_panels(snapshot, spec, role="standalone")
    alt_text = (
        "Three-panel ellipse flow field showing velocity magnitude, pressure proxy, "
        f"and vorticity. {spec.annotation}. {spec.provenance}."
    )
    return _render(panels, spec, rows=1, alt_text=alt_text)


def render_comparison(
    prediction: FlowFields,
    reference: FlowFields,
    spec: RenderSpec,
) -> PngArtifact:
    """Render reference, prediction, and honest signed/magnitude field errors."""

    if not isinstance(spec, RenderSpec):
        raise TypeError("spec must be a RenderSpec")
    predicted = _snapshot(prediction, label="prediction")
    solved = _snapshot(reference, label="reference")
    panels = _comparison_panels(predicted, solved, spec)
    alt_text = (
        "Nine-panel ellipse flow comparison with reference, prediction, and error rows "
        "for velocity magnitude, pressure proxy, and vorticity. Reference and prediction "
        f"share scales. {spec.annotation}. {spec.provenance}."
    )
    return _render(panels, spec, rows=3, alt_text=alt_text)


__all__ = [
    "PanelMetadata",
    "PngArtifact",
    "RenderSpec",
    "render_comparison",
    "render_fields",
]
