"""Bounded, schema-versioned source data for deterministic validation plots."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import median
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, model_validator

from soufflerie.schemas import ContentId, StrictFrozenModel, VersionedModel

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
FiniteNonnegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
ScalarGrid: TypeAlias = tuple[tuple[FiniteNonnegative, ...], ...]
BaselineKind: TypeAlias = Literal["selected_fno", "mean_field", "nearest_design"]


class FieldComparisonData(StrictFrozenModel):
    """A bounded scalar field pair for one representative or worst test case."""

    selection: Literal["representative", "worst"]
    case_id: ContentId
    quantity: Literal["velocity_magnitude"] = "velocity_magnitude"
    units: Literal["lattice_velocity"] = "lattice_velocity"
    solver: ScalarGrid
    surrogate: ScalarGrid

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_grids(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("solver", "surrogate"):
            grid = normalized.get(name)
            if isinstance(grid, list):
                normalized[name] = tuple(
                    tuple(row) if isinstance(row, list) else row for row in grid
                )
        return normalized

    @model_validator(mode="after")
    def _grids_are_bounded_and_aligned(self) -> Self:
        if len(self.solver) < 2 or len(self.solver) > 128:
            raise ValueError("plot field height must be between 2 and 128")
        widths = {len(row) for row in self.solver}
        if len(widths) != 1 or not 2 <= next(iter(widths)) <= 128:
            raise ValueError("plot field rows must share a width between 2 and 128")
        if len(self.surrogate) != len(self.solver) or any(
            len(row) != next(iter(widths)) for row in self.surrogate
        ):
            raise ValueError("solver and surrogate plot fields must share one shape")
        return self


class CasePlotPoint(StrictFrozenModel):
    """One test case's design, drag, and metric coordinates for report plots."""

    case_id: ContentId
    aspect_ratio: float = Field(ge=0.5, le=1.0, allow_inf_nan=False)
    rotation_deg: float = Field(ge=0.0, le=30.0, allow_inf_nan=False)
    scale: float = Field(ge=0.75, le=1.25, allow_inf_nan=False)
    reynolds: float = Field(ge=40.0, le=300.0, allow_inf_nan=False)
    velocity_rel_l2: FiniteNonnegative
    cd_head_pct: FiniteNonnegative
    cd_field_pct: FiniteNonnegative
    prediction_div_mean_abs: FiniteNonnegative
    obstacle_ratio: FiniteNonnegative
    cd_head: FiniteFloat
    cd_field: FiniteFloat
    cd_solver: FiniteFloat


class BaselinePlotSeries(StrictFrozenModel):
    """One selected-model or baseline aggregate in the comparison plot."""

    kind: BaselineKind
    artifact_id: ContentId
    median_velocity_rel_l2: FiniteNonnegative
    median_cd_pct: FiniteNonnegative


class ValidationPlotData(VersionedModel):
    """Complete bounded inputs for the RFC-0008 deterministic plot set."""

    representative_fields: FieldComparisonData
    worst_fields: FieldComparisonData
    cases: tuple[CasePlotPoint, ...] = Field(min_length=3, max_length=10_000)
    baselines: tuple[BaselinePlotSeries, BaselinePlotSeries, BaselinePlotSeries]

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("cases", "baselines"):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized

    @model_validator(mode="after")
    def _plot_sources_are_complete(self) -> Self:
        case_ids = tuple(item.case_id for item in self.cases)
        if tuple(sorted(case_ids)) != case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("plot cases must have distinct identities in canonical order")
        expected_kinds: tuple[BaselineKind, ...] = (
            "selected_fno",
            "mean_field",
            "nearest_design",
        )
        if tuple(item.kind for item in self.baselines) != expected_kinds:
            raise ValueError("plot baseline series must use the fixed canonical order")
        if len({item.artifact_id for item in self.baselines}) != len(self.baselines):
            raise ValueError("plot model and baseline identities must be distinct")

        point_by_id = {item.case_id: item for item in self.cases}
        if self.representative_fields.selection != "representative":
            raise ValueError("representative field data has the wrong selection label")
        if self.worst_fields.selection != "worst":
            raise ValueError("worst field data has the wrong selection label")
        if self.representative_fields.case_id == self.worst_fields.case_id:
            raise ValueError("representative and worst field cases must be distinct")
        if (
            self.representative_fields.case_id not in point_by_id
            or self.worst_fields.case_id not in point_by_id
        ):
            raise ValueError("field plot cases must occur in the per-case plot evidence")

        middle = median(item.velocity_rel_l2 for item in self.cases)
        representative = min(
            self.cases,
            key=lambda item: (abs(item.velocity_rel_l2 - middle), item.case_id),
        )
        worst = min(self.cases, key=lambda item: (-item.velocity_rel_l2, item.case_id))
        if self.representative_fields.case_id != representative.case_id:
            raise ValueError("representative field case must be closest to the error median")
        if self.worst_fields.case_id != worst.case_id:
            raise ValueError("worst field case must have the largest velocity error")
        if not math.isfinite(middle):  # pragma: no cover - scalar models already reject this
            raise ValueError("plot metric median is not finite")
        return self


__all__ = [
    "BaselineKind",
    "BaselinePlotSeries",
    "CasePlotPoint",
    "FieldComparisonData",
    "ScalarGrid",
    "ValidationPlotData",
]
