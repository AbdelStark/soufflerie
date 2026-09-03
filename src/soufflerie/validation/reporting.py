"""Deterministic validation JSON, Markdown, SVG plots, and publication checks."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from types import MappingProxyType
from typing import Annotated, Literal, Self, TypeAlias
from xml.sax.saxutils import escape

from pydantic import Field, StringConstraints, model_validator

from soufflerie.artifacts import ReaderLimits, safe_read_json, validate_release_provenance
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import (
    ContentId,
    Sha256,
    StrictFrozenModel,
    VersionedModel,
    canonical_sha256,
    sha256_bytes,
)
from soufflerie.validation.gates import ValidationReport
from soufflerie.validation.plot_data import FieldComparisonData

GENERATOR_VERSION = "validation-report-v1"
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_RENDERED_FILE_BYTES = 16 * 1024 * 1024
PlotKind: TypeAlias = Literal[
    "representative_fields",
    "worst_fields",
    "baseline_comparison",
    "error_by_design",
    "head_vs_field",
    "divergence_compliance",
    "ood_variance",
    "sensitivity",
]
PlotFilename = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9-]*\.svg$", min_length=5, max_length=96),
]

PLOT_DEFINITIONS: tuple[tuple[PlotKind, str, str], ...] = (
    ("representative_fields", "representative-fields.svg", "Representative flow fields"),
    ("worst_fields", "worst-fields.svg", "Worst-case flow fields"),
    ("baseline_comparison", "baseline-comparison.svg", "Model and baseline comparison"),
    ("error_by_design", "error-by-design.svg", "Error by design parameter"),
    ("head_vs_field", "head-vs-field.svg", "Cd head and field consistency"),
    (
        "divergence_compliance",
        "divergence-compliance.svg",
        "Divergence and obstacle compliance",
    ),
    ("ood_variance", "ood-variance.svg", "OOD ensemble variance"),
    ("sensitivity", "sensitivity.svg", "Rotation sensitivity agreement"),
)

_REPORT_PARENT_ROLES = (
    "dataset",
    "solver",
    "ensemble_model_0",
    "ensemble_model_1",
    "ensemble_model_2",
    "baseline_0",
    "baseline_1",
)
_WIDTH = 960
_HEIGHT = 540
_BACKGROUND = "#f7f9fc"
_INK = "#172033"
_MUTED = "#667085"
_GRID = "#d7deea"
_BLUE = "#246bfd"
_CYAN = "#15aabf"
_ORANGE = "#f08c46"
_RED = "#c92a2a"
_GREEN = "#2b8a3e"


class PlotArtifact(StrictFrozenModel):
    """Digest-bound metadata for one deterministic report plot."""

    kind: PlotKind
    filename: PlotFilename
    title: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    sha256: Sha256
    size_bytes: int = Field(ge=1, le=MAX_RENDERED_FILE_BYTES)


class PlotManifest(VersionedModel):
    """Self-verifying manifest for the exact RFC-0008 plot set."""

    generator_version: Literal["validation-report-v1"] = "validation-report-v1"
    report_id: ContentId
    report_sha256: Sha256
    overall_status: Literal["green", "red"]
    plots: tuple[PlotArtifact, ...]
    manifest_sha256: Sha256

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_plots(cls, value: object) -> object:
        if isinstance(value, Mapping) and isinstance(value.get("plots"), list):
            return {**value, "plots": tuple(value["plots"])}
        return value

    def logical_identity(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"manifest_sha256"})

    @model_validator(mode="after")
    def _manifest_is_complete(self) -> Self:
        observed = tuple((item.kind, item.filename, item.title) for item in self.plots)
        if observed != PLOT_DEFINITIONS:
            raise ValueError("plot manifest must contain the exact canonical plot set")
        if len({item.sha256 for item in self.plots}) != len(self.plots):
            raise ValueError("plot artifacts must have distinct content digests")
        if self.manifest_sha256 != canonical_sha256(self.logical_identity()):
            raise ValueError("plot manifest digest does not bind its complete contents")
        return self

    @classmethod
    def create(
        cls,
        *,
        report: ValidationReport,
        plots: Mapping[str, bytes],
    ) -> PlotManifest:
        if set(plots) != {item[1] for item in PLOT_DEFINITIONS}:
            raise ArtifactIntegrityError("plot bytes must contain the exact canonical plot set")
        artifacts = tuple(
            PlotArtifact(
                kind=kind,
                filename=filename,
                title=title,
                sha256=sha256_bytes(plots[filename]),
                size_bytes=len(plots[filename]),
            )
            for kind, filename, title in PLOT_DEFINITIONS
        )
        values: dict[str, object] = {
            "generator_version": GENERATOR_VERSION,
            "report_id": report.report_id,
            "report_sha256": report.report_sha256,
            "overall_status": report.overall_status,
            "plots": artifacts,
        }
        draft = cls.model_construct(
            generator_version=GENERATOR_VERSION,
            report_id=report.report_id,
            report_sha256=report.report_sha256,
            overall_status=report.overall_status,
            plots=artifacts,
            manifest_sha256="0" * 64,
        )
        return cls.model_validate(
            {
                **values,
                "manifest_sha256": canonical_sha256(draft.logical_identity()),
            }
        )


@dataclass(frozen=True, slots=True)
class RenderedValidationArtifacts:
    """In-memory artifact set; the plot manifest is always written last."""

    report_json: bytes
    markdown: bytes
    plot_manifest_json: bytes
    plots: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if tuple(self.plots) != tuple(item[1] for item in PLOT_DEFINITIONS):
            raise ArtifactIntegrityError("rendered plots do not match the canonical plot set")
        for name, content in (
            ("report JSON", self.report_json),
            ("Markdown", self.markdown),
            ("plot manifest", self.plot_manifest_json),
            *self.plots.items(),
        ):
            if (
                not isinstance(content, bytes)
                or not content
                or len(content) > MAX_RENDERED_FILE_BYTES
            ):
                raise ArtifactIntegrityError(f"rendered {name} violates the byte contract")


def report_parent_sha256(report: ValidationReport) -> dict[str, str]:
    """Return the exact required parent map after checking ID/digest alignment."""

    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    parents = report.provenance.parent_sha256
    if tuple(sorted(parents)) != tuple(sorted(_REPORT_PARENT_ROLES)):
        raise ArtifactIntegrityError(
            "VAL-8 LINEAGE: report parents must be dataset, solver, three models, and two baselines"
        )
    if report.dataset_id != parents["dataset"][:20]:
        raise ArtifactIntegrityError("VAL-8 LINEAGE: dataset ID does not match its full digest")
    ensemble_digests = tuple(parents[f"ensemble_model_{index}"] for index in range(3))
    if tuple(digest[:20] for digest in ensemble_digests) != report.ensemble_model_ids:
        raise ArtifactIntegrityError("VAL-8 LINEAGE: ensemble model IDs do not match full digests")
    baseline_digests = tuple(parents[f"baseline_{index}"] for index in range(2))
    if tuple(digest[:20] for digest in baseline_digests) != report.baseline_ids:
        raise ArtifactIntegrityError("VAL-8 LINEAGE: baseline IDs do not match full digests")

    if report.ood is not None:
        ensemble_by_id = dict(zip(report.ensemble_model_ids, ensemble_digests, strict=True))
        expected_ood = tuple(ensemble_by_id[model_id] for model_id in report.ood.model_ids)
        if report.ood.model_sha256s != expected_ood:
            raise ArtifactIntegrityError(
                "VAL-8 LINEAGE: OOD model digests differ from report parents"
            )
    if report.sensitivity is not None:
        selected_index = report.ensemble_model_ids.index(report.selected_model_id)
        if report.sensitivity.model.model_sha256 != ensemble_digests[selected_index]:
            raise ArtifactIntegrityError(
                "VAL-8 LINEAGE: sensitivity model digest differs from the selected parent"
            )
    return dict(parents)


def validate_report_publication(
    report: ValidationReport,
    *,
    expected_source_revision: str,
    expected_lock_sha256: str,
    expected_config_sha256: str,
    expected_packages: Mapping[str, str],
    expected_parent_sha256: Mapping[str, str],
) -> None:
    """Bind a report to externally reviewed source, lock, config, packages, and parents."""

    observed_parents = report_parent_sha256(report)
    if observed_parents != dict(expected_parent_sha256):
        raise ArtifactIntegrityError("VAL-8 LINEAGE: report parents differ from reviewed artifacts")
    validate_release_provenance(
        report.provenance,
        expected_source_revision=expected_source_revision,
        expected_lock_sha256=expected_lock_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_packages=expected_packages,
        required_parent_sha256=expected_parent_sha256,
    )


def _pretty_json_bytes(model: VersionedModel) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _markdown_cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _format_scalar(value: float | int | bool | None) -> str:
    if value is None:
        return "invalid"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def render_validation_markdown(
    report: ValidationReport,
    *,
    plot_directory: str = "validation.plots",
) -> bytes:
    """Render report values verbatim; no metric, gate, or status is recomputed."""

    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", plot_directory
    ) is None or plot_directory in {".", ".."}:
        raise ValueError("plot_directory must be one safe path component")
    status = report.overall_status.upper()
    lines = [
        f"# Validation status: {status}",
        "",
        f"- Overall status: **{status}**",
        f"- Report ID: `{report.report_id}`",
        f"- Dataset ID: `{report.dataset_id}`",
        f"- Selected model ID: `{report.selected_model_id}`",
        f"- Ensemble model IDs: {', '.join(f'`{item}`' for item in report.ensemble_model_ids)}",
        f"- Baseline IDs: {', '.join(f'`{item}`' for item in report.baseline_ids)}",
        f"- Generator: `{report.generator_version}`",
        "",
    ]
    if report.overall_status == "red":
        lines.extend(
            (
                "> **Release blocked:** one or more required gates are red. The plots below",
                "> are diagnostic evidence and do not override this status.",
                "",
            )
        )
    lines.extend(
        (
            "## Required gates",
            "",
            "| Gate | Status | Value | Operator | Threshold | Units | Evidence |",
            "|---|---|---:|:---:|---:|---|---|",
        )
    )
    for gate in report.gates:
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(gate.name),
                    gate.status.upper(),
                    _format_scalar(gate.value),
                    gate.operator,
                    _format_scalar(gate.threshold),
                    _markdown_cell(gate.units),
                    _markdown_cell("; ".join(gate.evidence)),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Metric distributions",
            "",
            "| Metric | Status | Count | Median | P90 | P95 | Maximum | Bootstrap median 95% |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        )
    )
    metric_details: list[str] = []
    for name, summary in sorted(report.metrics.items()):
        interval = (
            "invalid"
            if summary.bootstrap_median_95 is None
            else " - ".join(_format_scalar(item) for item in summary.bootstrap_median_95)
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(name),
                    summary.status,
                    str(summary.count),
                    _format_scalar(summary.median),
                    _format_scalar(summary.p90),
                    _format_scalar(summary.p95),
                    _format_scalar(summary.maximum),
                    interval,
                )
            )
            + " |"
        )
        if summary.invalid_case_ids:
            metric_details.append(
                f"\nInvalid `{name}` cases: "
                + ", ".join(f"`{item}`" for item in summary.invalid_case_ids)
                + "."
            )
        elif summary.worst_case_ids:
            metric_details.append(
                f"\nWorst `{name}` cases: "
                + ", ".join(f"`{item}`" for item in summary.worst_case_ids)
                + "."
            )
    lines.extend(metric_details)
    if report.ood is not None:
        lines.extend(
            (
                "",
                "## OOD heuristic",
                "",
                f"- Status: `{report.ood.status}`",
                "- Median OOD normalized variance: "
                f"`{_format_scalar(report.ood.median_ood_variance)}`",
                "- Median ID-boundary normalized variance: "
                f"`{_format_scalar(report.ood.median_id_boundary_variance)}`",
                f"- OOD / ID-boundary ratio: `{_format_scalar(report.ood.variance_ratio)}`",
            )
        )
    if report.sensitivity is not None:
        lines.extend(
            (
                "",
                "## Rotation sensitivity",
                "",
                f"Agreed signs: **{report.sensitivity.agreement_count} of 10**.",
                "",
                "| Case | Autograd Cd/degree | Central difference Cd/degree | Agrees |",
                "|---|---:|---:|:---:|",
            )
        )
        lines.extend(
            "| "
            f"`{item.geometry.source_case_id}` | {_format_scalar(item.autograd_cd_per_degree)} | "
            f"{_format_scalar(item.central_difference_cd_per_degree)} | "
            f"{'yes' if item.agrees else 'no'} |"
            for item in report.sensitivity.results
        )
    lines.extend(("", "## Diagnostic plots", ""))
    lines.extend(
        f"- [{title}]({plot_directory}/{filename})" for _, filename, title in PLOT_DEFINITIONS
    )
    lines.extend(
        (
            "",
            "## Provenance",
            "",
            f"- Source revision: `{report.provenance.source_revision}`",
            f"- Source dirty: `{str(report.provenance.source_dirty).lower()}`",
            f"- Lock SHA-256: `{report.provenance.lock_sha256}`",
            f"- Config SHA-256: `{report.provenance.config_sha256}`",
            f"- Device class: `{_markdown_cell(report.provenance.device_class)}`",
            f"- Report SHA-256: `{report.report_sha256}`",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _number(value: float) -> str:
    if abs(value) >= 1_000 or (0 < abs(value) < 0.001):
        return f"{value:.2e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


class _Svg:
    def __init__(self, title: str, report: ValidationReport) -> None:
        status_color = _GREEN if report.overall_status == "green" else _RED
        self._items = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" '
                f'viewBox="0 0 {_WIDTH} {_HEIGHT}" role="img" aria-labelledby="title desc">'
            ),
            f'<title id="title">{escape(title)}</title>',
            (
                f'<desc id="desc">Soufflerie validation report {report.report_id}; '
                f"overall status {report.overall_status}.</desc>"
            ),
            f'<rect width="{_WIDTH}" height="{_HEIGHT}" fill="{_BACKGROUND}"/>',
            f'<rect width="{_WIDTH}" height="8" fill="{status_color}"/>',
        ]
        self.text(28, 35, title, size=22, weight=700)
        self.text(
            932,
            34,
            f"{report.overall_status.upper()} · {report.report_id}",
            size=12,
            color=status_color,
            anchor="end",
            weight=700,
        )

    def raw(self, value: str) -> None:
        self._items.append(value)

    def text(
        self,
        x: float,
        y: float,
        value: object,
        *,
        size: int = 11,
        color: str = _INK,
        anchor: Literal["start", "middle", "end"] = "start",
        weight: int = 400,
    ) -> None:
        self.raw(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-family="DejaVu Sans, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">'
            f"{escape(str(value))}</text>"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = _GRID,
        width: float = 1.0,
        dash: str | None = None,
    ) -> None:
        dashed = "" if dash is None else f' stroke-dasharray="{dash}"'
        self.raw(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width:.2f}"{dashed}/>'
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        stroke: str | None = None,
    ) -> None:
        border = "" if stroke is None else f' stroke="{stroke}" stroke-width="1"'
        self.raw(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'fill="{fill}"{border}/>'
        )

    def circle(self, x: float, y: float, *, radius: float, fill: str) -> None:
        self.raw(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}"/>')

    def finish(self) -> bytes:
        self._items.append("</svg>")
        return ("\n".join(self._items) + "\n").encode("utf-8")


def _limits(values: Sequence[float], *, include_zero: bool = False) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    padding = max(abs(low) * 0.1, 1.0) if math.isclose(low, high) else (high - low) * 0.08
    padded_low = 0.0 if include_zero and low == 0.0 else low - padding
    return padded_low, high + padding


def _axes(
    svg: _Svg,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    x_values: Sequence[float],
    y_values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    x_zero: bool = False,
    y_zero: bool = False,
    show_x_limits: bool = True,
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    x_low, x_high = _limits(x_values, include_zero=x_zero)
    y_low, y_high = _limits(y_values, include_zero=y_zero)
    svg.rect(x, y, width, height, fill="#ffffff", stroke=_GRID)
    for fraction in (0.25, 0.5, 0.75):
        svg.line(x, y + height * fraction, x + width, y + height * fraction)
    svg.text(x, y - 10, title, size=13, weight=700)
    svg.text(x + width / 2, y + height + 34, x_label, anchor="middle", color=_MUTED)
    if show_x_limits:
        svg.text(x, y + height + 17, _number(x_low), size=9, color=_MUTED)
        svg.text(
            x + width,
            y + height + 17,
            _number(x_high),
            size=9,
            color=_MUTED,
            anchor="end",
        )
    svg.text(x - 7, y + 4, _number(y_high), size=9, color=_MUTED, anchor="end")
    svg.text(x - 7, y + height, _number(y_low), size=9, color=_MUTED, anchor="end")
    svg.text(x + width, y - 10, y_label, size=9, color=_MUTED, anchor="end")

    def map_x(value: float) -> float:
        return x + (value - x_low) * width / (x_high - x_low)

    def map_y(value: float) -> float:
        return y + height - (value - y_low) * height / (y_high - y_low)

    return map_x, map_y


def _heat_color(value: float, low: float, high: float, *, error: bool) -> str:
    fraction = 0.5 if math.isclose(low, high) else (value - low) / (high - low)
    fraction = min(1.0, max(0.0, fraction))
    start = (255, 247, 237) if error else (239, 246, 255)
    end = (201, 42, 42) if error else (36, 107, 253)
    channels = tuple(
        round(left + fraction * (right - left)) for left, right in zip(start, end, strict=True)
    )
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _field_plot(
    report: ValidationReport,
    data: FieldComparisonData,
    title: str,
) -> bytes:
    svg = _Svg(title, report)
    error = tuple(
        tuple(
            abs(predicted - expected)
            for predicted, expected in zip(pred_row, solver_row, strict=True)
        )
        for pred_row, solver_row in zip(data.surrogate, data.solver, strict=True)
    )
    shared = tuple(value for grid in (data.solver, data.surrogate) for row in grid for value in row)
    errors = tuple(value for row in error for value in row)
    field_low, field_high = min(shared), max(shared)
    error_low, error_high = min(errors), max(errors)
    panels = (
        ("Solver", data.solver, field_low, field_high, False),
        ("Surrogate", data.surrogate, field_low, field_high, False),
        ("Absolute error", error, error_low, error_high, True),
    )
    panel_width = 280.0
    panel_height = 350.0
    for index, (label, grid, low, high, is_error) in enumerate(panels):
        left = 25.0 + index * 312.0
        top = 85.0
        rows = len(grid)
        columns = len(grid[0])
        cell_width = panel_width / columns
        cell_height = panel_height / rows
        svg.text(left, top - 12, label, size=13, weight=700)
        for row_index, row in enumerate(grid):
            for column_index, value in enumerate(row):
                svg.rect(
                    left + column_index * cell_width,
                    top + row_index * cell_height,
                    cell_width + 0.02,
                    cell_height + 0.02,
                    fill=_heat_color(value, low, high, error=is_error),
                )
        svg.rect(left, top, panel_width, panel_height, fill="none", stroke=_GRID)
        svg.text(left, top + panel_height + 22, _number(low), size=9, color=_MUTED)
        svg.text(
            left + panel_width,
            top + panel_height + 22,
            _number(high),
            size=9,
            color=_MUTED,
            anchor="end",
        )
    svg.text(25, 495, f"Case {data.case_id} · {data.quantity} · {data.units}", color=_MUTED)
    return svg.finish()


def _baseline_plot(report: ValidationReport) -> bytes:
    assert report.plot_data is not None
    svg = _Svg("Model and baseline comparison", report)
    labels = ("Selected FNO", "Mean field", "Nearest design")
    colors = (_BLUE, _ORANGE, _CYAN)
    series = report.plot_data.baselines
    for panel_index, (attribute, title, unit) in enumerate(
        (
            ("median_velocity_rel_l2", "Median velocity relative L2", "ratio"),
            ("median_cd_pct", "Median Cd error", "percent"),
        )
    ):
        left = 65.0 + panel_index * 465.0
        top = 90.0
        width = 390.0
        height = 340.0
        values = tuple(float(getattr(item, attribute)) for item in series)
        maximum = max(values) * 1.15 if max(values) > 0 else 1.0
        svg.rect(left, top, width, height, fill="#ffffff", stroke=_GRID)
        svg.text(left, top - 12, title, size=14, weight=700)
        svg.text(left, top + height + 48, unit, color=_MUTED)
        for index, (label, color, value) in enumerate(zip(labels, colors, values, strict=True)):
            bar_width = 72.0
            bar_left = left + 38.0 + index * 120.0
            bar_height = value * (height - 35.0) / maximum
            svg.rect(
                bar_left,
                top + height - bar_height,
                bar_width,
                bar_height,
                fill=color,
            )
            svg.text(
                bar_left + bar_width / 2,
                top + height - bar_height - 8,
                _number(value),
                anchor="middle",
            )
            svg.text(bar_left + bar_width / 2, top + height + 20, label, size=9, anchor="middle")
    return svg.finish()


def _error_by_design_plot(report: ValidationReport) -> bytes:
    assert report.plot_data is not None
    svg = _Svg("Error by design parameter", report)
    cases = report.plot_data.cases
    panels = (
        ("aspect_ratio", "Aspect ratio"),
        ("rotation_deg", "Rotation (degrees)"),
        ("scale", "Scale"),
        ("reynolds", "Reynolds number"),
    )
    for index, (attribute, label) in enumerate(panels):
        left = 58.0 + (index % 2) * 465.0
        top = 82.0 + (index // 2) * 225.0
        x_values = tuple(float(getattr(item, attribute)) for item in cases)
        y_values = tuple(item.velocity_rel_l2 for item in cases)
        map_x, map_y = _axes(
            svg,
            x=left,
            y=top,
            width=390,
            height=155,
            x_values=x_values,
            y_values=y_values,
            title=label,
            x_label=label,
            y_label="velocity L2",
        )
        for x_value, y_value in zip(x_values, y_values, strict=True):
            svg.circle(map_x(x_value), map_y(y_value), radius=3.5, fill=_BLUE)
    return svg.finish()


def _head_vs_field_plot(report: ValidationReport) -> bytes:
    assert report.plot_data is not None
    svg = _Svg("Cd head and field consistency", report)
    cases = report.plot_data.cases
    x_values = tuple(item.cd_field for item in cases)
    y_values = tuple(item.cd_head for item in cases)
    combined = (*x_values, *y_values)
    map_x, map_y = _axes(
        svg,
        x=105,
        y=85,
        width=750,
        height=365,
        x_values=combined,
        y_values=combined,
        title="Per-case Cd estimates",
        x_label="Field-derived Cd",
        y_label="Head Cd",
    )
    low, high = min(combined), max(combined)
    svg.line(map_x(low), map_y(low), map_x(high), map_y(high), color=_MUTED, dash="6 5")
    for point in cases:
        deviation = abs(point.cd_head - point.cd_field)
        color = _RED if deviation > 0.1 * max(abs(point.cd_solver), 0.1) else _BLUE
        svg.circle(map_x(point.cd_field), map_y(point.cd_head), radius=4, fill=color)
    svg.text(105, 495, "Dashed line: exact head/field agreement", color=_MUTED)
    return svg.finish()


def _divergence_compliance_plot(report: ValidationReport) -> bytes:
    assert report.plot_data is not None
    svg = _Svg("Divergence and obstacle compliance", report)
    cases = report.plot_data.cases
    x_values = tuple(float(index + 1) for index in range(len(cases)))
    panels = (
        (
            tuple(sorted(item.prediction_div_mean_abs for item in cases)),
            "Prediction divergence",
            "inverse lattice unit",
            _BLUE,
        ),
        (
            tuple(sorted(item.obstacle_ratio for item in cases)),
            "Obstacle velocity ratio",
            "ratio",
            _ORANGE,
        ),
    )
    for index, (values, title, unit, color) in enumerate(panels):
        left = 68.0 + index * 465.0
        map_x, map_y = _axes(
            svg,
            x=left,
            y=90,
            width=390,
            height=340,
            x_values=x_values,
            y_values=values,
            title=title,
            x_label="Cases sorted by value",
            y_label=unit,
            y_zero=True,
            show_x_limits=False,
        )
        previous: tuple[float, float] | None = None
        for x_value, value in zip(x_values, values, strict=True):
            current = (map_x(x_value), map_y(value))
            if previous is not None:
                svg.line(*previous, *current, color=color, width=1.5)
            svg.circle(*current, radius=3, fill=color)
            previous = current
    return svg.finish()


def _ood_plot(report: ValidationReport) -> bytes:
    assert report.ood is not None
    svg = _Svg("OOD ensemble variance", report)
    reynolds_values = (20, 40, 300, 400)
    grouped = {
        reynolds: tuple(
            item.normalized_ensemble_variance
            for item in report.ood.results
            if item.reynolds == reynolds
        )
        for reynolds in reynolds_values
    }
    all_values = tuple(value for values in grouped.values() for value in values)
    map_x, map_y = _axes(
        svg,
        x=105,
        y=85,
        width=750,
        height=365,
        x_values=tuple(float(index) for index in range(4)),
        y_values=all_values,
        title="Normalized ensemble variance by Reynolds value",
        x_label="Reynolds value",
        y_label="normalized variance",
        y_zero=True,
        show_x_limits=False,
    )
    for group_index, reynolds in enumerate(reynolds_values):
        color = _ORANGE if reynolds in (20, 400) else _BLUE
        values = grouped[reynolds]
        for value_index, value in enumerate(values):
            offset = (value_index - (len(values) - 1) / 2.0) * 0.018
            svg.circle(map_x(group_index + offset), map_y(value), radius=3.5, fill=color)
        middle = median(values)
        svg.line(
            map_x(group_index - 0.13),
            map_y(middle),
            map_x(group_index + 0.13),
            map_y(middle),
            color=_INK,
            width=2.5,
        )
        svg.text(map_x(group_index), 475, f"Re={reynolds}", anchor="middle", color=_MUTED)
    svg.text(105, 505, "Orange: OOD · Blue: in-domain boundary · Black: median", color=_MUTED)
    return svg.finish()


def _sensitivity_plot(report: ValidationReport) -> bytes:
    assert report.sensitivity is not None
    svg = _Svg("Rotation sensitivity agreement", report)
    results = report.sensitivity.results
    x_values = tuple(float(index + 1) for index in range(len(results)))
    autograd = tuple(item.autograd_cd_per_degree for item in results)
    central = tuple(item.central_difference_cd_per_degree for item in results)
    map_x, map_y = _axes(
        svg,
        x=105,
        y=85,
        width=750,
        height=365,
        x_values=x_values,
        y_values=(*autograd, *central),
        title="Autograd and h=0.25 degree central differences",
        x_label="Canonical probe index",
        y_label="Cd per degree",
        y_zero=True,
        show_x_limits=False,
    )
    svg.line(map_x(1), map_y(0), map_x(len(results)), map_y(0), color=_MUTED, dash="5 4")
    for x_value, first, second, item in zip(x_values, autograd, central, results, strict=True):
        svg.line(map_x(x_value), map_y(first), map_x(x_value), map_y(second), color=_GRID, width=2)
        svg.circle(map_x(x_value) - 3, map_y(first), radius=4, fill=_BLUE)
        svg.circle(map_x(x_value) + 3, map_y(second), radius=4, fill=_ORANGE)
        if not item.agrees:
            svg.circle(map_x(x_value), map_y((first + second) / 2.0), radius=2.5, fill=_RED)
    svg.text(
        105,
        495,
        "Blue: autograd · Orange: central difference · Red: sign disagreement",
        color=_MUTED,
    )
    return svg.finish()


def _render_plots(report: ValidationReport) -> Mapping[str, bytes]:
    assert report.plot_data is not None
    plots = {
        "representative-fields.svg": _field_plot(
            report,
            report.plot_data.representative_fields,
            "Representative flow fields",
        ),
        "worst-fields.svg": _field_plot(
            report,
            report.plot_data.worst_fields,
            "Worst-case flow fields",
        ),
        "baseline-comparison.svg": _baseline_plot(report),
        "error-by-design.svg": _error_by_design_plot(report),
        "head-vs-field.svg": _head_vs_field_plot(report),
        "divergence-compliance.svg": _divergence_compliance_plot(report),
        "ood-variance.svg": _ood_plot(report),
        "sensitivity.svg": _sensitivity_plot(report),
    }
    return MappingProxyType(plots)


def render_validation_artifacts(
    report: ValidationReport,
    *,
    plot_directory: str = "validation.plots",
) -> RenderedValidationArtifacts:
    """Render the complete artifact set only from one already-evaluated report."""

    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    if report.generator_version != GENERATOR_VERSION:
        raise ArtifactIntegrityError("VAL-8 REPORT: unsupported generator version")
    if report.ood is None or report.sensitivity is None or report.plot_data is None:
        raise ArtifactIntegrityError(
            "VAL-8 REPORT: publication requires OOD, sensitivity, and complete plot data"
        )
    if report.provenance.source_dirty or not report.provenance.deterministic:
        raise ArtifactIntegrityError(
            "VAL-8 REPORT: publication requires clean deterministic evidence"
        )
    report_parent_sha256(report)
    plots = _render_plots(report)
    manifest = PlotManifest.create(report=report, plots=plots)
    return RenderedValidationArtifacts(
        report_json=_pretty_json_bytes(report),
        markdown=render_validation_markdown(report, plot_directory=plot_directory),
        plot_manifest_json=_pretty_json_bytes(manifest),
        plots=plots,
    )


def _artifact_paths(report_path: Path) -> tuple[Path, Path, Path]:
    return (
        report_path.with_suffix(".md"),
        report_path.with_suffix(".plots.json"),
        report_path.with_suffix(".plots"),
    )


def _owned_publication_chain(path: Path, trusted_root: Path | None) -> tuple[Path, ...]:
    absolute = path.absolute()
    if trusted_root is None:
        return (absolute, *absolute.parents)
    boundary = trusted_root.absolute()
    try:
        absolute.relative_to(boundary)
    except ValueError as error:
        raise ArtifactIntegrityError(
            f"publication target {path} escapes trusted root {trusted_root}"
        ) from error
    chain = [absolute]
    while chain[-1] != boundary:
        chain.append(chain[-1].parent)
    return tuple(chain)


def _read_bounded(
    path: Path,
    *,
    maximum: int = MAX_RENDERED_FILE_BYTES,
    trusted_root: Path | None = None,
) -> bytes:
    if any(item.is_symlink() for item in _owned_publication_chain(path, trusted_root)):
        raise ArtifactIntegrityError(f"validation artifact {path} must not be a symbolic link")
    try:
        stat = path.stat()
    except OSError as error:
        raise ArtifactIntegrityError(f"unable to read validation artifact {path}") from error
    if not path.is_file() or stat.st_size < 1 or stat.st_size > maximum:
        raise ArtifactIntegrityError(f"validation artifact {path} violates the byte contract")
    content = path.read_bytes()
    if len(content) != stat.st_size:
        raise ArtifactIntegrityError(f"validation artifact {path} changed while reading")
    return content


def load_validation_report(
    path: Path,
    *,
    trusted_root: Path | None = None,
) -> ValidationReport:
    """Load one bounded strict report and require its canonical checked-in encoding."""

    content = _read_bounded(path, maximum=MAX_REPORT_BYTES, trusted_root=trusted_root)
    report = safe_read_json(
        path.parent,
        path.name,
        model=ValidationReport,
        limits=ReaderLimits(
            max_file_bytes=MAX_REPORT_BYTES,
            max_json_bytes=MAX_REPORT_BYTES,
        ),
    )
    if content != _pretty_json_bytes(report):
        raise ArtifactIntegrityError("VAL-8 REPORT: JSON is not in canonical rendered form")
    return report


def check_validation_artifacts(
    report_path: Path,
    artifacts: RenderedValidationArtifacts,
    *,
    trusted_root: Path | None = None,
) -> tuple[str, ...]:
    """Compare the complete checked-in artifact set without mutating it."""

    markdown_path, manifest_path, plots_path = _artifact_paths(report_path)
    expected_files = {
        report_path: artifacts.report_json,
        markdown_path: artifacts.markdown,
        manifest_path: artifacts.plot_manifest_json,
    }
    errors: list[str] = []
    for path, expected in expected_files.items():
        try:
            actual = _read_bounded(path, trusted_root=trusted_root)
        except ArtifactIntegrityError as error:
            errors.append(str(error))
            continue
        if actual != expected:
            errors.append(f"validation artifact {path} is stale")
    if plots_path.is_symlink() or not plots_path.is_dir():
        errors.append(f"validation plot directory {plots_path} is missing or unsafe")
        return tuple(errors)
    observed_names = {path.name for path in plots_path.iterdir()}
    expected_names = set(artifacts.plots)
    if observed_names != expected_names:
        errors.append(f"validation plot directory {plots_path} has an incomplete or extra file set")
    for name, expected in artifacts.plots.items():
        path = plots_path / name
        try:
            actual = _read_bounded(path, trusted_root=trusted_root)
        except ArtifactIntegrityError as error:
            errors.append(str(error))
            continue
        if actual != expected:
            errors.append(f"validation plot {path} is stale")
    return tuple(errors)


def _atomic_write(path: Path, content: bytes, *, trusted_root: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(item.is_symlink() for item in _owned_publication_chain(path, trusted_root)):
        raise ArtifactIntegrityError(f"refusing symbolic-link publication target {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_validation_artifacts(
    report_path: Path,
    artifacts: RenderedValidationArtifacts,
    *,
    trusted_root: Path | None = None,
) -> None:
    """Write members atomically and commit the plot manifest last.

    A trusted root limits symlink checks to the caller-owned storage namespace;
    provider-managed mount ancestors remain outside that trust boundary.
    """

    markdown_path, manifest_path, plots_path = _artifact_paths(report_path)
    if plots_path.exists() and (plots_path.is_symlink() or not plots_path.is_dir()):
        raise ArtifactIntegrityError("validation plot target exists but is not a safe directory")
    plots_path.mkdir(parents=True, exist_ok=True)
    extra = {path.name for path in plots_path.iterdir()} - set(artifacts.plots)
    if extra:
        raise ArtifactIntegrityError("validation plot target contains unrecognized files")
    _atomic_write(report_path, artifacts.report_json, trusted_root=trusted_root)
    _atomic_write(markdown_path, artifacts.markdown, trusted_root=trusted_root)
    for name, content in artifacts.plots.items():
        _atomic_write(plots_path / name, content, trusted_root=trusted_root)
    _atomic_write(manifest_path, artifacts.plot_manifest_json, trusted_root=trusted_root)


__all__ = [
    "GENERATOR_VERSION",
    "MAX_REPORT_BYTES",
    "PLOT_DEFINITIONS",
    "PlotArtifact",
    "PlotKind",
    "PlotManifest",
    "RenderedValidationArtifacts",
    "check_validation_artifacts",
    "load_validation_report",
    "render_validation_artifacts",
    "render_validation_markdown",
    "report_parent_sha256",
    "validate_report_publication",
    "write_validation_artifacts",
]
