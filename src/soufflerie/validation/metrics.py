"""Fixed fp64 validation metrics and deterministic distribution summaries."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Annotated, Literal, Self, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from pydantic import Field, StringConstraints, model_validator

from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ContentId, StrictFrozenModel

MetricName: TypeAlias = Literal[
    "velocity_rel_l2",
    "cd_head_pct",
    "cd_field_pct",
    "head_field_gap_pct",
    "prediction_div_mean_abs",
    "solver_div_mean_abs",
    "obstacle_ratio",
]
MetricUnits: TypeAlias = Literal["ratio", "percent", "inverse_lattice_unit"]
FiniteNonnegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
FailureReason = Annotated[str, StringConstraints(min_length=1, max_length=512)]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

_METRIC_CONTRACT: tuple[tuple[MetricName, MetricUnits], ...] = (
    ("velocity_rel_l2", "ratio"),
    ("cd_head_pct", "percent"),
    ("cd_field_pct", "percent"),
    ("head_field_gap_pct", "percent"),
    ("prediction_div_mean_abs", "inverse_lattice_unit"),
    ("solver_div_mean_abs", "inverse_lattice_unit"),
    ("obstacle_ratio", "ratio"),
)


class MetricObservation(StrictFrozenModel):
    """One valid value or one explicit invalid-metric reason."""

    name: MetricName
    status: Literal["valid", "invalid"]
    value: FiniteNonnegative | None
    units: MetricUnits
    failure: FailureReason | None = None

    @model_validator(mode="after")
    def _state_is_coherent(self) -> Self:
        if self.status == "valid" and (self.value is None or self.failure is not None):
            raise ValueError("valid metric observations require a value and no failure")
        if self.status == "invalid" and (self.value is not None or self.failure is None):
            raise ValueError("invalid metric observations require a failure and no value")
        return self


class CaseMetrics(StrictFrozenModel):
    """All fixed per-case validation observations in canonical order."""

    case_id: ContentId
    velocity_rel_l2: MetricObservation
    cd_head_pct: MetricObservation
    cd_field_pct: MetricObservation
    head_field_gap_pct: MetricObservation
    prediction_div_mean_abs: MetricObservation
    solver_div_mean_abs: MetricObservation
    obstacle_ratio: MetricObservation

    @model_validator(mode="after")
    def _observations_match_fields(self) -> Self:
        for name, units in _METRIC_CONTRACT:
            observation = cast(MetricObservation, getattr(self, name))
            if observation.name != name or observation.units != units:
                raise ValueError(f"{name} observation does not match its field contract")
        return self


class MetricSummary(StrictFrozenModel):
    """Deterministic median/tail/bootstrap evidence without invalid-case skipping."""

    name: MetricName
    status: Literal["valid", "invalid"]
    count: int = Field(ge=1)
    valid_count: int = Field(ge=0)
    invalid_case_ids: tuple[ContentId, ...]
    median: FiniteNonnegative | None
    p90: FiniteNonnegative | None
    p95: FiniteNonnegative | None
    maximum: FiniteNonnegative | None
    bootstrap_median_95: tuple[FiniteNonnegative, FiniteNonnegative] | None
    worst_case_ids: tuple[ContentId, ...] = Field(max_length=20)
    bootstrap_resamples: int = Field(ge=100, le=100_000)
    report_seed: int = Field(ge=0, le=2**64 - 1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("invalid_case_ids", "bootstrap_median_95", "worst_case_ids"):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized

    @model_validator(mode="after")
    def _summary_is_coherent(self) -> Self:
        if self.valid_count + len(self.invalid_case_ids) != self.count:
            raise ValueError("valid and invalid metric counts must cover the distribution")
        statistics = (
            self.median,
            self.p90,
            self.p95,
            self.maximum,
            self.bootstrap_median_95,
        )
        if self.status == "invalid":
            if not self.invalid_case_ids or any(item is not None for item in statistics):
                raise ValueError("invalid summaries require case evidence and no aggregate values")
            if self.worst_case_ids:
                raise ValueError("invalid summaries must not rank a partial distribution")
        else:
            if self.invalid_case_ids or self.valid_count != self.count:
                raise ValueError("valid summaries cannot contain invalid cases")
            if any(item is None for item in statistics):
                raise ValueError("valid summaries require every aggregate value")
            if not self.worst_case_ids:
                raise ValueError("valid summaries require worst-case identities")
            if len(set(self.worst_case_ids)) != min(self.count, 20):
                raise ValueError("worst-case identities must be unique and complete")
            assert self.median is not None
            assert self.p90 is not None
            assert self.p95 is not None
            assert self.maximum is not None
            assert self.bootstrap_median_95 is not None
            lower, upper = self.bootstrap_median_95
            if not (self.median <= self.p90 <= self.p95 <= self.maximum and lower <= upper):
                raise ValueError("metric summary quantiles are not ordered")
        if tuple(sorted(set(self.invalid_case_ids))) != self.invalid_case_ids:
            raise ValueError("invalid case identities must be unique and sorted")
        return self


def _valid(name: MetricName, value: float, units: MetricUnits) -> MetricObservation:
    if not math.isfinite(value) or value < 0.0:
        return _invalid(name, units, "VAL-1 NONFINITE: metric reduction is not finite")
    return MetricObservation(name=name, status="valid", value=value, units=units)


def _invalid(name: MetricName, units: MetricUnits, reason: str) -> MetricObservation:
    return MetricObservation(name=name, status="invalid", value=None, units=units, failure=reason)


def _validate_array_contract(
    name: str,
    array: object,
    *,
    dtype: np.dtype[np.generic],
    shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ArtifactIntegrityError(f"VAL-1 SHAPE: {name} must be a two-dimensional array")
    if array.dtype != dtype:
        raise ArtifactIntegrityError(f"VAL-1 DTYPE: {name} must use {dtype.name}")
    if not array.flags.c_contiguous:
        raise ArtifactIntegrityError(f"VAL-1 LAYOUT: {name} must be C-contiguous")
    observed = cast(tuple[int, int], array.shape)
    if min(observed) <= 0 or (shape is not None and observed != shape):
        raise ArtifactIntegrityError(f"VAL-1 SHAPE: {name} has an incompatible shape")
    return observed


def _finite_scalar(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _relative_velocity(
    prediction_u: Float32Array,
    prediction_v: Float32Array,
    solver_u: Float32Array,
    solver_v: Float32Array,
    fluid_mask: BoolArray,
) -> MetricObservation:
    name: MetricName = "velocity_rel_l2"
    if not all(
        np.isfinite(array).all() for array in (prediction_u, prediction_v, solver_u, solver_v)
    ):
        return _invalid(name, "ratio", "VAL-1 NONFINITE: velocity input contains NaN or infinity")
    if not np.any(fluid_mask):
        return _invalid(name, "ratio", "VAL-2 NO_VALID_CELLS: fluid mask is empty")
    predicted_u64 = prediction_u.astype(np.float64, copy=False)
    predicted_v64 = prediction_v.astype(np.float64, copy=False)
    solver_u64 = solver_u.astype(np.float64, copy=False)
    solver_v64 = solver_v.astype(np.float64, copy=False)
    du = predicted_u64[fluid_mask] - solver_u64[fluid_mask]
    dv = predicted_v64[fluid_mask] - solver_v64[fluid_mask]
    numerator = math.sqrt(float(np.sum(du * du + dv * dv, dtype=np.float64)))
    denominator = math.sqrt(
        float(
            np.sum(
                solver_u64[fluid_mask] ** 2 + solver_v64[fluid_mask] ** 2,
                dtype=np.float64,
            )
        )
    )
    return _valid(name, numerator / max(denominator, 1e-8), "ratio")


def _drag_observation(
    name: Literal["cd_head_pct", "cd_field_pct", "head_field_gap_pct"],
    left: object,
    right: object,
    solver_cd: object,
) -> MetricObservation:
    values = (_finite_scalar(left), _finite_scalar(right), _finite_scalar(solver_cd))
    if any(value is None for value in values):
        return _invalid(name, "percent", "VAL-1 NONFINITE: drag input is not finite")
    left_value, right_value, solver_value = cast(tuple[float, float, float], values)
    return _valid(
        name,
        100.0 * abs(left_value - right_value) / max(abs(solver_value), 0.1),
        "percent",
    )


def _divergence(
    name: Literal["prediction_div_mean_abs", "solver_div_mean_abs"],
    u: Float32Array,
    v: Float32Array,
    fluid_mask: BoolArray,
    *,
    spacing_lu: float,
) -> MetricObservation:
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        return _invalid(
            name,
            "inverse_lattice_unit",
            "VAL-1 NONFINITE: divergence input contains NaN or infinity",
        )
    stencil = (
        fluid_mask[1:-1, 1:-1]
        & fluid_mask[1:-1, :-2]
        & fluid_mask[1:-1, 2:]
        & fluid_mask[:-2, 1:-1]
        & fluid_mask[2:, 1:-1]
    )
    if not np.any(stencil):
        return _invalid(
            name,
            "inverse_lattice_unit",
            "VAL-2 NO_VALID_CELLS: no central-difference fluid stencil exists",
        )
    u64 = u.astype(np.float64, copy=False)
    v64 = v.astype(np.float64, copy=False)
    inverse_two_spacing = 1.0 / (2.0 * spacing_lu)
    divergence = (u64[1:-1, 2:] - u64[1:-1, :-2]) * inverse_two_spacing + (
        v64[2:, 1:-1] - v64[:-2, 1:-1]
    ) * inverse_two_spacing
    return _valid(
        name,
        float(np.mean(np.abs(divergence[stencil]), dtype=np.float64)),
        "inverse_lattice_unit",
    )


def _obstacle_compliance(
    prediction_u: Float32Array,
    prediction_v: Float32Array,
    obstacle_mask: BoolArray,
    inlet_velocity_lu: object,
) -> MetricObservation:
    name: MetricName = "obstacle_ratio"
    inlet = _finite_scalar(inlet_velocity_lu)
    if inlet is None or inlet <= 0.0:
        return _invalid(name, "ratio", "VAL-1 NONFINITE: inlet velocity is not finite positive")
    if not np.isfinite(prediction_u).all() or not np.isfinite(prediction_v).all():
        return _invalid(name, "ratio", "VAL-1 NONFINITE: obstacle velocity is not finite")
    if not np.any(obstacle_mask):
        return _invalid(name, "ratio", "VAL-2 NO_VALID_CELLS: obstacle mask is empty")
    u64 = prediction_u.astype(np.float64, copy=False)[obstacle_mask]
    v64 = prediction_v.astype(np.float64, copy=False)[obstacle_mask]
    speed = np.sqrt(u64 * u64 + v64 * v64)
    return _valid(name, float(np.mean(speed, dtype=np.float64)) / inlet, "ratio")


def evaluate_case_metrics(
    *,
    case_id: str,
    prediction_u: Float32Array,
    prediction_v: Float32Array,
    solver_u: Float32Array,
    solver_v: Float32Array,
    fluid_mask: BoolArray,
    obstacle_mask: BoolArray,
    cd_head: object,
    cd_field: object,
    cd_solver: object,
    inlet_velocity_lu: object,
    spacing_lu: float = 2.0,
) -> CaseMetrics:
    """Compute every RFC-0008 per-case metric from de-normalized float32 fields."""

    arrays = {
        "prediction_u": prediction_u,
        "prediction_v": prediction_v,
        "solver_u": solver_u,
        "solver_v": solver_v,
    }
    shape: tuple[int, int] | None = None
    for name, array in arrays.items():
        shape = _validate_array_contract(name, array, dtype=np.dtype(np.float32), shape=shape)
    assert shape is not None
    _validate_array_contract("fluid_mask", fluid_mask, dtype=np.dtype(np.bool_), shape=shape)
    _validate_array_contract("obstacle_mask", obstacle_mask, dtype=np.dtype(np.bool_), shape=shape)
    if np.any(fluid_mask & obstacle_mask):
        raise ArtifactIntegrityError("VAL-1 MASK: fluid and obstacle masks must not overlap")
    spacing = _finite_scalar(spacing_lu)
    if spacing is None or spacing <= 0.0:
        raise ArtifactIntegrityError("VAL-1 SPACING: spacing_lu must be finite positive")

    return CaseMetrics(
        case_id=case_id,
        velocity_rel_l2=_relative_velocity(
            prediction_u, prediction_v, solver_u, solver_v, fluid_mask
        ),
        cd_head_pct=_drag_observation("cd_head_pct", cd_head, cd_solver, cd_solver),
        cd_field_pct=_drag_observation("cd_field_pct", cd_field, cd_solver, cd_solver),
        head_field_gap_pct=_drag_observation("head_field_gap_pct", cd_head, cd_field, cd_solver),
        prediction_div_mean_abs=_divergence(
            "prediction_div_mean_abs",
            prediction_u,
            prediction_v,
            fluid_mask,
            spacing_lu=spacing,
        ),
        solver_div_mean_abs=_divergence(
            "solver_div_mean_abs", solver_u, solver_v, fluid_mask, spacing_lu=spacing
        ),
        obstacle_ratio=_obstacle_compliance(
            prediction_u, prediction_v, obstacle_mask, inlet_velocity_lu
        ),
    )


def summarize_metric(
    name: MetricName,
    observations: Mapping[str, MetricObservation],
    *,
    report_seed: int,
    bootstrap_resamples: int,
) -> MetricSummary:
    """Summarize one complete distribution; never aggregate around invalid cases."""

    if not observations:
        raise ArtifactIntegrityError("VAL-3 SUMMARY: at least one observation is required")
    if (
        isinstance(report_seed, bool)
        or not isinstance(report_seed, int)
        or not 0 <= report_seed < 2**64
    ):
        raise ArtifactIntegrityError("VAL-3 SUMMARY: report seed must be unsigned 64-bit")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or not 100 <= bootstrap_resamples <= 100_000
    ):
        raise ArtifactIntegrityError("VAL-3 SUMMARY: bootstrap count is outside policy")
    ordered = tuple(sorted(observations.items()))
    if any(
        not isinstance(case_id, str)
        or re.fullmatch(r"[0-9a-f]{20}", case_id) is None
        or not isinstance(observation, MetricObservation)
        or observation.name != name
        for case_id, observation in ordered
    ):
        raise ArtifactIntegrityError("VAL-3 SUMMARY: observation identity or metric differs")
    invalid = tuple(case_id for case_id, item in ordered if item.status == "invalid")
    valid = tuple((case_id, item.value) for case_id, item in ordered if item.value is not None)
    if invalid:
        return MetricSummary(
            name=name,
            status="invalid",
            count=len(ordered),
            valid_count=len(valid),
            invalid_case_ids=invalid,
            median=None,
            p90=None,
            p95=None,
            maximum=None,
            bootstrap_median_95=None,
            worst_case_ids=(),
            bootstrap_resamples=bootstrap_resamples,
            report_seed=report_seed,
        )

    values = np.asarray([value for _, value in valid], dtype=np.float64)
    median = float(np.median(values))
    p90, p95 = (float(value) for value in np.quantile(values, (0.9, 0.95), method="linear"))
    generator = np.random.default_rng(report_seed)
    bootstrap = np.empty(bootstrap_resamples, dtype=np.float64)
    for index in range(bootstrap_resamples):
        sample = generator.integers(0, len(values), size=len(values))
        bootstrap[index] = np.median(values[sample])
    lower, upper = (
        float(value) for value in np.quantile(bootstrap, (0.025, 0.975), method="linear")
    )
    worst = tuple(
        case_id for case_id, _ in sorted(valid, key=lambda item: (-item[1], item[0]))[:20]
    )
    return MetricSummary(
        name=name,
        status="valid",
        count=len(ordered),
        valid_count=len(valid),
        invalid_case_ids=(),
        median=median,
        p90=p90,
        p95=p95,
        maximum=float(np.max(values)),
        bootstrap_median_95=(lower, upper),
        worst_case_ids=worst,
        bootstrap_resamples=bootstrap_resamples,
        report_seed=report_seed,
    )


__all__ = [
    "CaseMetrics",
    "MetricName",
    "MetricObservation",
    "MetricSummary",
    "evaluate_case_metrics",
    "summarize_metric",
]
