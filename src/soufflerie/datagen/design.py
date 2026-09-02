"""Deterministic maximin design generation and immutable split assignment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator, model_validator
from scipy.spatial.distance import pdist, squareform  # type: ignore[import-untyped]

from soufflerie.config import SweepConfig
from soufflerie.geometry import validate_geometry
from soufflerie.numerics import derive_lattice
from soufflerie.schemas import (
    CaseConfig,
    JsonValue,
    ShapeParams,
    Split,
    StrictFrozenModel,
    VersionedModel,
    canonical_json_bytes,
    canonical_sha256,
)

Float64Matrix = npt.NDArray[np.float64]

CANDIDATE_COUNT = 32
DESIGN_SAMPLE_COUNT = 1_000
TRAIN_COUNT = 600
VALIDATION_COUNT = 200
TEST_COUNT = 200
SPLIT_SALT = b"split-v1"
DIMENSION_NAMES: tuple[
    Literal["aspect_ratio"],
    Literal["rotation_deg"],
    Literal["scale"],
    Literal["reynolds"],
] = ("aspect_ratio", "rotation_deg", "scale", "reynolds")
_CONTENT_ID = re.compile(r"^[0-9a-f]{20}$")


def _design_payload(
    *,
    aspect_ratio: float,
    rotation_deg: float,
    scale: float,
    reynolds: float,
) -> dict[str, JsonValue]:
    return {
        "design_schema_version": 1,
        "shape_family": "ellipse",
        "shape": {
            "aspect_ratio": aspect_ratio,
            "rotation_deg": rotation_deg,
            "scale": scale,
        },
        "reynolds": reynolds,
    }


def _design_id(payload: dict[str, JsonValue]) -> str:
    return canonical_sha256(payload)[:20]


@dataclass(frozen=True, slots=True)
class UnsplitDesignPoint:
    """One physical design point before immutable split assignment."""

    index: int
    aspect_ratio: float
    rotation_deg: float
    scale: float
    reynolds: float
    design_id: str

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("design point index must be a nonnegative integer")
        bounds = {
            "aspect_ratio": (self.aspect_ratio, 0.5, 1.0),
            "rotation_deg": (self.rotation_deg, 0.0, 30.0),
            "scale": (self.scale, 0.75, 1.25),
            "reynolds": (self.reynolds, 40.0, 300.0),
        }
        for name, (value, minimum, maximum) in bounds.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, float)
                or not math.isfinite(value)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be a finite float in [{minimum}, {maximum}]")
        if _CONTENT_ID.fullmatch(self.design_id) is None:
            raise ValueError("design_id must be 20 lowercase hexadecimal characters")
        if self.design_id != _design_id(self.canonical_design()):
            raise ValueError("design_id does not match the canonical physical parameters")

    def canonical_design(self) -> dict[str, JsonValue]:
        """Return the path- and solver-independent design identity payload."""

        return _design_payload(
            aspect_ratio=self.aspect_ratio,
            rotation_deg=self.rotation_deg,
            scale=self.scale,
            reynolds=self.reynolds,
        )

    @property
    def shape(self) -> ShapeParams:
        return ShapeParams(
            aspect_ratio=self.aspect_ratio,
            rotation_deg=self.rotation_deg,
            scale=self.scale,
        )


@dataclass(frozen=True, slots=True)
class DesignPoint(UnsplitDesignPoint):
    """One physical point with a frozen design-level split."""

    split: Split

    def __post_init__(self) -> None:
        super(DesignPoint, self).__post_init__()
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split {self.split!r}")


class DistributionSummary(StrictFrozenModel):
    """Finite deterministic summary of one scalar vector."""

    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)
    mean: float = Field(allow_inf_nan=False)
    q05: float = Field(allow_inf_nan=False)
    q25: float = Field(allow_inf_nan=False)
    q50: float = Field(allow_inf_nan=False)
    q75: float = Field(allow_inf_nan=False)
    q95: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _quantiles_are_ordered(self) -> DistributionSummary:
        ordered = (
            self.minimum,
            self.q05,
            self.q25,
            self.q50,
            self.q75,
            self.q95,
            self.maximum,
        )
        if any(left > right for left, right in pairwise(ordered)):
            raise ValueError("distribution quantiles must be ordered within the observed range")
        if not self.minimum <= self.mean <= self.maximum:
            raise ValueError("distribution mean must lie within the observed range")
        return self


class DesignDimensionSummaries(StrictFrozenModel):
    aspect_ratio: DistributionSummary
    rotation_deg: DistributionSummary
    scale: DistributionSummary
    reynolds: DistributionSummary


class DesignSplitCounts(StrictFrozenModel):
    train: Literal[600] = 600
    validation: Literal[200] = 200
    test: Literal[200] = 200


CorrelationRow = tuple[float, float, float, float]


class DesignSummary(VersionedModel):
    """Checked, compact evidence for one immutable maximin design."""

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    seed: int = Field(ge=0, le=2**64 - 1)
    samples: Literal[1000] = 1000
    dimensions: tuple[
        Literal["aspect_ratio"],
        Literal["rotation_deg"],
        Literal["scale"],
        Literal["reynolds"],
    ] = DIMENSION_NAMES
    candidate_count: Literal[32] = 32
    candidate_child_seeds: tuple[int, ...] = Field(min_length=32, max_length=32)
    candidate_seeds_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_minimum_distances: tuple[float, ...] = Field(min_length=32, max_length=32)
    selected_candidate_index: int = Field(ge=0, lt=32)
    selected_minimum_distance: float = Field(gt=0.0, allow_inf_nan=False)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_counts: DesignSplitCounts
    dimension_statistics: DesignDimensionSummaries
    pairwise_correlation: tuple[CorrelationRow, CorrelationRow, CorrelationRow, CorrelationRow]
    nearest_neighbor_distance: DistributionSummary
    lattice_preflight_passed: Literal[1000] = 1000
    geometry_preflight_passed: Literal[1000] = 1000
    all_preflights_passed: Literal[True] = True

    @field_validator(
        "dimensions",
        "candidate_child_seeds",
        "candidate_minimum_distances",
        mode="before",
    )
    @classmethod
    def _json_arrays_to_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("pairwise_correlation", mode="before")
    @classmethod
    def _json_matrix_to_fixed_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(row) if isinstance(row, list) else row for row in value)
        return value

    @model_validator(mode="after")
    def _selection_and_seed_evidence_are_coherent(self) -> DesignSummary:
        if len(set(self.candidate_child_seeds)) != CANDIDATE_COUNT:
            raise ValueError("candidate child seeds must be unique")
        if any(seed < 0 or seed > 2**64 - 1 for seed in self.candidate_child_seeds):
            raise ValueError("candidate child seeds must be unsigned 64-bit integers")
        if self.candidate_seeds_sha256 != canonical_sha256(self.candidate_child_seeds):
            raise ValueError("candidate seed digest does not match the recorded child seeds")
        distances = self.candidate_minimum_distances
        if any(not math.isfinite(value) or value <= 0.0 for value in distances):
            raise ValueError("candidate minimum distances must be finite and positive")
        selected = distances[self.selected_candidate_index]
        if selected != self.selected_minimum_distance:
            raise ValueError("selected minimum distance does not match its candidate")
        expected_index = max(range(CANDIDATE_COUNT), key=lambda index: (distances[index], -index))
        if self.selected_candidate_index != expected_index:
            raise ValueError("selected candidate is not maximin with index tie-breaking")
        for row_index, row in enumerate(self.pairwise_correlation):
            for column_index, value in enumerate(row):
                if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                    raise ValueError("correlations must be finite and within [-1, 1]")
                if row_index == column_index and not math.isclose(value, 1.0, abs_tol=1e-14):
                    raise ValueError("correlation diagonal must equal one")
                if not math.isclose(
                    value,
                    self.pairwise_correlation[column_index][row_index],
                    rel_tol=0.0,
                    abs_tol=1e-14,
                ):
                    raise ValueError("correlation matrix must be symmetric")
        return self


@dataclass(frozen=True, slots=True)
class _CandidateSelection:
    normalized: Float64Matrix
    child_seeds: tuple[int, ...]
    minimum_distances: tuple[float, ...]
    selected_index: int


@dataclass(frozen=True, slots=True)
class _GeneratedDesign:
    points: tuple[DesignPoint, ...]
    selection: _CandidateSelection
    lattice_preflight_passed: int
    geometry_preflight_passed: int


def _candidate_child_seeds(seed: int) -> tuple[int, ...]:
    master = np.random.Generator(np.random.PCG64(seed))
    raw = np.asarray(master.bit_generator.random_raw(CANDIDATE_COUNT), dtype=np.uint64)
    return tuple(int(item) for item in raw)


def _lhs_candidate(*, samples: int, dimensions: int, seed: int) -> Float64Matrix:
    rng = np.random.Generator(np.random.PCG64(seed))
    strata = np.arange(samples, dtype=np.float64)
    candidate = np.empty((samples, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values = (strata + rng.random(samples)) / float(samples)
        candidate[:, dimension] = values[rng.permutation(samples)]
    return candidate


def _select_maximin_index(minimum_distances: Sequence[float]) -> int:
    if len(minimum_distances) != CANDIDATE_COUNT:
        raise ValueError(f"maximin selection requires exactly {CANDIDATE_COUNT} candidates")
    if any(not math.isfinite(value) or value < 0.0 for value in minimum_distances):
        raise ValueError("candidate distances must be finite and nonnegative")
    return max(
        range(CANDIDATE_COUNT),
        key=lambda index: (minimum_distances[index], -index),
    )


def _select_candidate(config: SweepConfig) -> _CandidateSelection:
    child_seeds = _candidate_child_seeds(config.seed)
    candidates: list[Float64Matrix] = []
    minimum_distances: list[float] = []
    for seed in child_seeds:
        candidate = _lhs_candidate(samples=config.samples, dimensions=4, seed=seed)
        distances = pdist(candidate, metric="euclidean")
        candidates.append(candidate)
        minimum_distances.append(float(np.min(distances)))
    selected_index = _select_maximin_index(minimum_distances)
    selected = np.ascontiguousarray(candidates[selected_index], dtype=np.float64)
    selected.flags.writeable = False
    return _CandidateSelection(
        normalized=selected,
        child_seeds=child_seeds,
        minimum_distances=tuple(minimum_distances),
        selected_index=selected_index,
    )


def _scale(value: float, *, minimum: float, maximum: float) -> float:
    return minimum + value * (maximum - minimum)


def _unsplit_points(
    config: SweepConfig,
    normalized: Float64Matrix,
) -> tuple[UnsplitDesignPoint, ...]:
    points: list[UnsplitDesignPoint] = []
    for index, row in enumerate(normalized):
        aspect_ratio = _scale(
            float(row[0]),
            minimum=config.aspect_ratio.minimum,
            maximum=config.aspect_ratio.maximum,
        )
        rotation_deg = _scale(
            float(row[1]),
            minimum=config.rotation_deg.minimum,
            maximum=config.rotation_deg.maximum,
        )
        scale = _scale(
            float(row[2]),
            minimum=config.scale.minimum,
            maximum=config.scale.maximum,
        )
        reynolds = _scale(
            float(row[3]),
            minimum=config.reynolds.minimum,
            maximum=config.reynolds.maximum,
        )
        payload = _design_payload(
            aspect_ratio=aspect_ratio,
            rotation_deg=rotation_deg,
            scale=scale,
            reynolds=reynolds,
        )
        points.append(
            UnsplitDesignPoint(
                index=index,
                aspect_ratio=aspect_ratio,
                rotation_deg=rotation_deg,
                scale=scale,
                reynolds=reynolds,
                design_id=_design_id(payload),
            )
        )
    return tuple(points)


def assign_splits(
    points: Sequence[UnsplitDesignPoint],
    seed: int,
) -> tuple[DesignPoint, ...]:
    """Hash-rank exactly 1,000 points into immutable 600/200/200 splits."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**64 - 1:
        raise ValueError("split seed must be an unsigned 64-bit integer")
    if len(points) != DESIGN_SAMPLE_COUNT:
        raise ValueError(f"split assignment requires exactly {DESIGN_SAMPLE_COUNT} points")
    if any(type(point) is not UnsplitDesignPoint for point in points):
        raise TypeError("split assignment requires UnsplitDesignPoint instances")
    if {point.index for point in points} != set(range(DESIGN_SAMPLE_COUNT)):
        raise ValueError("design point indices must be unique and contiguous from zero")
    if len({point.design_id for point in points}) != DESIGN_SAMPLE_COUNT:
        raise ValueError("design IDs must be unique")

    seed_bytes = seed.to_bytes(8, byteorder="big", signed=False)
    ranked: list[tuple[str, str, UnsplitDesignPoint]] = []
    for point in points:
        digest = hashlib.sha256(
            SPLIT_SALT + seed_bytes + canonical_json_bytes(point.canonical_design())
        ).hexdigest()
        ranked.append((digest, point.design_id, point))
    ranked.sort(key=lambda item: (item[0], item[1]))

    membership: dict[str, Split] = {}
    for rank, (_, design_id, _) in enumerate(ranked):
        if rank < TRAIN_COUNT:
            split: Split = "train"
        elif rank < TRAIN_COUNT + VALIDATION_COUNT:
            split = "validation"
        else:
            split = "test"
        membership[design_id] = split

    return tuple(
        DesignPoint(
            index=point.index,
            aspect_ratio=point.aspect_ratio,
            rotation_deg=point.rotation_deg,
            scale=point.scale,
            reynolds=point.reynolds,
            design_id=point.design_id,
            split=membership[point.design_id],
        )
        for point in points
    )


def case_config_for_point(point: UnsplitDesignPoint, config: SweepConfig) -> CaseConfig:
    """Bind one physical design identity to the sweep numerical controls."""

    if not isinstance(point, UnsplitDesignPoint):
        raise TypeError("point must be an UnsplitDesignPoint")
    if not isinstance(config, SweepConfig):
        raise TypeError("config must be a SweepConfig")
    return CaseConfig(
        shape=point.shape,
        reynolds=point.reynolds,
        nx=config.grid.nx,
        ny=config.grid.ny,
        steps=config.run.steps,
        warmup_steps=config.run.warmup_steps,
        inlet_velocity_lu=config.run.inlet_velocity_lu,
        seed=config.seed,
    )


def _generate(config: SweepConfig) -> _GeneratedDesign:
    if not isinstance(config, SweepConfig):
        raise TypeError("config must be a SweepConfig")
    selection = _select_candidate(config)
    unsplit = _unsplit_points(config, selection.normalized)
    points = assign_splits(unsplit, config.seed)
    lattice_passed = 0
    geometry_passed = 0
    for point in points:
        case = case_config_for_point(point, config)
        derive_lattice(case, sample_interval=config.run.sample_interval)
        lattice_passed += 1
        validate_geometry(case.shape, case.grid)
        geometry_passed += 1
    return _GeneratedDesign(
        points=points,
        selection=selection,
        lattice_preflight_passed=lattice_passed,
        geometry_preflight_passed=geometry_passed,
    )


def sample_design(config: SweepConfig) -> tuple[DesignPoint, ...]:
    """Generate, preflight, and split the complete canonical design."""

    return _generate(config).points


def _distribution(values: npt.NDArray[np.float64]) -> DistributionSummary:
    quantiles = np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95), method="linear")
    return DistributionSummary(
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values, dtype=np.float64)),
        q05=float(quantiles[0]),
        q25=float(quantiles[1]),
        q50=float(quantiles[2]),
        q75=float(quantiles[3]),
        q95=float(quantiles[4]),
    )


def design_sha256(points: Sequence[DesignPoint]) -> str:
    """Hash the canonical ordered physical design and its presentation IDs."""

    if len(points) != DESIGN_SAMPLE_COUNT:
        raise ValueError(f"design digest requires exactly {DESIGN_SAMPLE_COUNT} points")
    payload: list[JsonValue] = []
    for point in sorted(points, key=lambda item: item.index):
        payload.append(
            {
                "index": point.index,
                "design_id": point.design_id,
                **point.canonical_design(),
            }
        )
    return canonical_sha256(payload)


def split_sha256(points: Sequence[DesignPoint]) -> str:
    """Hash design-level split membership independent of execution order."""

    if len(points) != DESIGN_SAMPLE_COUNT:
        raise ValueError(f"split digest requires exactly {DESIGN_SAMPLE_COUNT} points")
    payload: list[JsonValue] = [
        {"design_id": point.design_id, "split": point.split}
        for point in sorted(points, key=lambda item: item.design_id)
    ]
    return canonical_sha256(payload)


def generate_design_summary(config: SweepConfig) -> DesignSummary:
    """Generate complete design evidence after all allocation-free and geometry preflights."""

    generated = _generate(config)
    points = generated.points
    physical = np.ascontiguousarray(
        [(point.aspect_ratio, point.rotation_deg, point.scale, point.reynolds) for point in points],
        dtype=np.float64,
    )
    condensed = pdist(generated.selection.normalized, metric="euclidean")
    distances = squareform(condensed)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    correlations = np.corrcoef(physical, rowvar=False, dtype=np.float64)
    counts = Counter(point.split for point in points)
    if counts != Counter({"train": 600, "validation": 200, "test": 200}):
        raise ValueError("generated split counts do not match the canonical design")
    if (
        generated.lattice_preflight_passed != DESIGN_SAMPLE_COUNT
        or generated.geometry_preflight_passed != DESIGN_SAMPLE_COUNT
    ):
        raise ValueError("all canonical points must pass lattice and geometry preflight")

    rows = cast(
        tuple[CorrelationRow, CorrelationRow, CorrelationRow, CorrelationRow],
        tuple(tuple(float(value) for value in row) for row in correlations),
    )
    selection = generated.selection
    return DesignSummary(
        name=config.name,
        seed=config.seed,
        candidate_child_seeds=selection.child_seeds,
        candidate_seeds_sha256=canonical_sha256(selection.child_seeds),
        candidate_minimum_distances=selection.minimum_distances,
        selected_candidate_index=selection.selected_index,
        selected_minimum_distance=selection.minimum_distances[selection.selected_index],
        config_sha256=config.config_digest,
        design_sha256=design_sha256(points),
        split_sha256=split_sha256(points),
        split_counts=DesignSplitCounts(),
        dimension_statistics=DesignDimensionSummaries(
            aspect_ratio=_distribution(physical[:, 0]),
            rotation_deg=_distribution(physical[:, 1]),
            scale=_distribution(physical[:, 2]),
            reynolds=_distribution(physical[:, 3]),
        ),
        pairwise_correlation=rows,
        nearest_neighbor_distance=_distribution(nearest),
        lattice_preflight_passed=cast(Literal[1000], generated.lattice_preflight_passed),
        geometry_preflight_passed=cast(Literal[1000], generated.geometry_preflight_passed),
        all_preflights_passed=True,
    )


def render_design_summary(summary: DesignSummary) -> str:
    """Render a canonical review-friendly JSON representation."""

    return json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


__all__ = [
    "CANDIDATE_COUNT",
    "DESIGN_SAMPLE_COUNT",
    "DIMENSION_NAMES",
    "SPLIT_SALT",
    "DesignPoint",
    "DesignSummary",
    "DistributionSummary",
    "UnsplitDesignPoint",
    "assign_splits",
    "case_config_for_point",
    "design_sha256",
    "generate_design_summary",
    "render_design_summary",
    "sample_design",
    "split_sha256",
]
