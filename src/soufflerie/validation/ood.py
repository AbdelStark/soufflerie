"""Deterministic OOD/ID probe selection and normalized ensemble variance."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from soufflerie.datagen.manifest import ManifestRow
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import ContentId, Sha256, StrictFrozenModel, canonical_sha256

PROBE_COUNT = 10
ProbeReynolds: TypeAlias = Literal[20, 40, 300, 400]
OOD_REYNOLDS: tuple[Literal[20], Literal[400]] = (20, 400)
ID_BOUNDARY_REYNOLDS: tuple[Literal[40], Literal[300]] = (40, 300)
ALL_PROBE_REYNOLDS: tuple[Literal[20], Literal[400], Literal[40], Literal[300]] = (
    *OOD_REYNOLDS,
    *ID_BOUNDARY_REYNOLDS,
)
FiniteNonnegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


class ProbeGeometry(StrictFrozenModel):
    """One source-bound test geometry selected by its complete rank digest."""

    selection: Literal["ood", "sensitivity"]
    dataset_id: ContentId
    source_case_id: ContentId
    source_design_id: ContentId
    source_split: Literal["test"] = "test"
    aspect_ratio: float = Field(ge=0.5, le=1.0, allow_inf_nan=False)
    rotation_deg: float = Field(ge=0.0, le=30.0, allow_inf_nan=False)
    scale: float = Field(ge=0.75, le=1.25, allow_inf_nan=False)
    source_reynolds: float = Field(ge=40.0, le=300.0, allow_inf_nan=False)
    rank_sha256: Sha256

    def rank_identity(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"rank_sha256"})

    @model_validator(mode="after")
    def _rank_binds_source_geometry(self) -> Self:
        if self.rank_sha256 != canonical_sha256(self.rank_identity()):
            raise ValueError("probe rank digest does not bind the source geometry")
        if self.selection == "sensitivity" and not 1.0 <= self.rotation_deg <= 29.0:
            raise ValueError("sensitivity geometry must remain away from rotation boundaries")
        return self

    @classmethod
    def from_manifest(
        cls,
        row: ManifestRow,
        *,
        selection: Literal["ood", "sensitivity"],
    ) -> ProbeGeometry:
        if row.split != "test":
            raise ArtifactIntegrityError("VAL-5 SELECT: probe source must belong to the test split")
        values: dict[str, object] = {
            "selection": selection,
            "dataset_id": row.dataset_id,
            "source_case_id": row.case_id,
            "source_design_id": row.design_id,
            "source_split": "test",
            "aspect_ratio": row.aspect_ratio,
            "rotation_deg": row.rotation_deg,
            "scale": row.scale,
            "source_reynolds": row.reynolds,
        }
        return cls.model_validate({"rank_sha256": canonical_sha256(values), **values})


def select_probe_geometries(
    rows: Sequence[ManifestRow],
    *,
    selection: Literal["ood", "sensitivity"],
) -> tuple[ProbeGeometry, ...]:
    """Select the ten lowest complete rank digests without changing membership."""

    materialized = tuple(rows)
    if not materialized or any(not isinstance(row, ManifestRow) for row in materialized):
        raise ArtifactIntegrityError("VAL-5 SELECT: manifest rows are required")
    dataset_ids = {row.dataset_id for row in materialized}
    if len(dataset_ids) != 1:
        raise ArtifactIntegrityError("VAL-5 SELECT: rows must share one dataset identity")
    if len({row.design_id for row in materialized}) != len(materialized):
        raise ArtifactIntegrityError("VAL-5 SELECT: source design identities must be unique")
    candidates = tuple(
        ProbeGeometry.from_manifest(row, selection=selection)
        for row in materialized
        if row.split == "test" and (selection == "ood" or 1.0 <= row.rotation_deg <= 29.0)
    )
    if len(candidates) < PROBE_COUNT:
        raise ArtifactIntegrityError("VAL-5 SELECT: fewer than ten eligible test geometries")
    return tuple(
        sorted(candidates, key=lambda probe: (probe.rank_sha256, probe.source_design_id))[
            :PROBE_COUNT
        ]
    )


class ProbeModelIdentity(StrictFrozenModel):
    model_id: ContentId
    model_sha256: Sha256

    @model_validator(mode="after")
    def _id_prefixes_digest(self) -> Self:
        if self.model_id != self.model_sha256[:20]:
            raise ValueError("probe model ID must prefix its full digest")
        return self


@dataclass(frozen=True, slots=True)
class EnsembleFieldPrediction:
    """One verified model's de-normalized field prediction for a shared probe."""

    model: ProbeModelIdentity
    fields: Float32Array
    fluid_mask: BoolArray

    def __post_init__(self) -> None:
        if not isinstance(self.model, ProbeModelIdentity):
            raise TypeError("model must be a ProbeModelIdentity")
        if (
            not isinstance(self.fields, np.ndarray)
            or self.fields.dtype != np.dtype(np.float32)
            or self.fields.ndim != 3
            or self.fields.shape[0] != 3
            or not self.fields.flags.c_contiguous
            or not np.isfinite(self.fields).all()
        ):
            raise ArtifactIntegrityError(
                "VAL-6 ENSEMBLE: fields must be finite C-contiguous float32[3,H,W]"
            )
        expected_mask_shape = cast(tuple[int, int], self.fields.shape[1:])
        if (
            not isinstance(self.fluid_mask, np.ndarray)
            or self.fluid_mask.dtype != np.dtype(np.bool_)
            or self.fluid_mask.shape != expected_mask_shape
            or not self.fluid_mask.flags.c_contiguous
            or not np.any(self.fluid_mask)
        ):
            raise ArtifactIntegrityError(
                "VAL-6 ENSEMBLE: fluid mask must be nonempty C-contiguous bool[H,W]"
            )


class OodProbeResult(StrictFrozenModel):
    """Normalized three-model field variance for one geometry/Re probe."""

    geometry: ProbeGeometry
    reynolds: ProbeReynolds
    regime: Literal["ood", "id_boundary"]
    model_ids: tuple[ContentId, ContentId, ContentId]
    model_sha256s: tuple[Sha256, Sha256, Sha256]
    normalized_ensemble_variance: FiniteNonnegative

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_models(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("model_ids", "model_sha256s"):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized

    @model_validator(mode="after")
    def _probe_is_coherent(self) -> Self:
        if self.geometry.selection != "ood":
            raise ValueError("OOD result requires an OOD-selected geometry")
        expected_regime = "ood" if self.reynolds in OOD_REYNOLDS else "id_boundary"
        if self.regime != expected_regime:
            raise ValueError("probe regime does not match Reynolds value")
        if tuple(sorted(self.model_ids)) != self.model_ids or len(set(self.model_ids)) != 3:
            raise ValueError("probe requires three distinct sorted model IDs")
        if len(set(self.model_sha256s)) != 3:
            raise ValueError("probe requires three distinct full model digests")
        if any(
            model_id != digest[:20]
            for model_id, digest in zip(self.model_ids, self.model_sha256s, strict=True)
        ):
            raise ValueError("probe model IDs do not prefix their full digests")
        return self


def evaluate_ensemble_variance(
    geometry: ProbeGeometry,
    *,
    reynolds: ProbeReynolds,
    predictions: Sequence[EnsembleFieldPrediction],
    training_output_variance: tuple[float, float, float],
) -> OodProbeResult:
    """Compute population ensemble variance normalized per training output channel."""

    if not isinstance(geometry, ProbeGeometry) or geometry.selection != "ood":
        raise ArtifactIntegrityError("VAL-6 ENSEMBLE: an OOD-selected geometry is required")
    materialized = tuple(predictions)
    if len(materialized) != 3 or any(
        not isinstance(item, EnsembleFieldPrediction) for item in materialized
    ):
        raise ArtifactIntegrityError("VAL-6 ENSEMBLE: exactly three predictions are required")
    ordered = tuple(sorted(materialized, key=lambda item: item.model.model_id))
    identities = tuple(item.model for item in ordered)
    if (
        len({item.model_id for item in identities}) != 3
        or len({item.model_sha256 for item in identities}) != 3
    ):
        raise ArtifactIntegrityError("VAL-6 ENSEMBLE: model identities must be distinct")
    reference_shape = ordered[0].fields.shape
    reference_mask = ordered[0].fluid_mask
    if any(
        item.fields.shape != reference_shape or not np.array_equal(item.fluid_mask, reference_mask)
        for item in ordered[1:]
    ):
        raise ArtifactIntegrityError("VAL-6 ENSEMBLE: prediction shapes or fluid masks differ")
    if len(training_output_variance) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in training_output_variance
    ):
        raise ArtifactIntegrityError(
            "VAL-6 ENSEMBLE: training output variances must be three finite positive values"
        )
    stacked = np.stack([item.fields for item in ordered]).astype(np.float64, copy=False)
    variance = np.var(stacked, axis=0, dtype=np.float64, ddof=0)
    normalized = variance / np.asarray(training_output_variance, dtype=np.float64)[:, None, None]
    fluid_values = normalized[:, reference_mask]
    value = float(np.mean(fluid_values, dtype=np.float64))
    if not math.isfinite(value) or value < 0.0:
        raise ArtifactIntegrityError("VAL-6 ENSEMBLE: normalized variance is not finite")
    return OodProbeResult(
        geometry=geometry,
        reynolds=reynolds,
        regime="ood" if reynolds in OOD_REYNOLDS else "id_boundary",
        model_ids=cast(tuple[str, str, str], tuple(item.model_id for item in identities)),
        model_sha256s=cast(tuple[str, str, str], tuple(item.model_sha256 for item in identities)),
        normalized_ensemble_variance=value,
    )


class OodEvaluation(StrictFrozenModel):
    """Complete 10-geometry, four-Re OOD heuristic evidence."""

    status: Literal["valid", "invalid"]
    model_ids: tuple[ContentId, ContentId, ContentId]
    model_sha256s: tuple[Sha256, Sha256, Sha256]
    results: tuple[OodProbeResult, ...]
    median_ood_variance: FiniteNonnegative | None
    median_id_boundary_variance: FiniteNonnegative | None
    variance_ratio: FiniteNonnegative | None
    failure: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name in ("model_ids", "model_sha256s", "results"):
            item = normalized.get(name)
            if isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized

    @model_validator(mode="after")
    def _evaluation_is_complete(self) -> Self:
        if len(self.results) != PROBE_COUNT * len(ALL_PROBE_REYNOLDS):
            raise ValueError("OOD evaluation requires ten geometries at four Reynolds values")
        geometry_ids = {item.geometry.source_design_id for item in self.results}
        if len(geometry_ids) != PROBE_COUNT:
            raise ValueError("OOD evaluation requires ten distinct source geometries")
        if len({item.geometry.dataset_id for item in self.results}) != 1:
            raise ValueError("OOD evaluation geometries must share one dataset identity")
        for geometry_id in geometry_ids:
            matching = tuple(
                item for item in self.results if item.geometry.source_design_id == geometry_id
            )
            if len({item.geometry for item in matching}) != 1:
                raise ValueError("every OOD Reynolds probe must use the same source geometry")
            reynolds = {item.reynolds for item in matching}
            if reynolds != set(ALL_PROBE_REYNOLDS):
                raise ValueError("every OOD geometry requires all four Reynolds probes")
        if any(
            item.model_ids != self.model_ids or item.model_sha256s != self.model_sha256s
            for item in self.results
        ):
            raise ValueError("OOD result model identities differ")
        values = (
            self.median_ood_variance,
            self.median_id_boundary_variance,
            self.variance_ratio,
        )
        if self.status == "valid" and (any(item is None for item in values) or self.failure):
            raise ValueError("valid OOD evaluation requires complete aggregates")
        if self.status == "invalid" and (
            any(item is not None for item in values) or not self.failure
        ):
            raise ValueError("invalid OOD evaluation requires only failure evidence")
        return self


def summarize_ood_evaluation(results: Sequence[OodProbeResult]) -> OodEvaluation:
    """Freeze complete per-probe values and the median OOD/ID boundary ratio."""

    materialized = tuple(results)
    if not materialized or any(not isinstance(item, OodProbeResult) for item in materialized):
        raise ArtifactIntegrityError("VAL-6 ENSEMBLE: OOD probe results are required")
    ordered = tuple(
        sorted(
            materialized,
            key=lambda item: (
                item.geometry.rank_sha256,
                item.geometry.source_design_id,
                item.reynolds,
            ),
        )
    )
    models = ordered[0].model_ids
    digests = ordered[0].model_sha256s
    ood = np.asarray(
        [item.normalized_ensemble_variance for item in ordered if item.regime == "ood"],
        dtype=np.float64,
    )
    in_domain = np.asarray(
        [item.normalized_ensemble_variance for item in ordered if item.regime == "id_boundary"],
        dtype=np.float64,
    )
    if len(ood) != 2 * PROBE_COUNT or len(in_domain) != 2 * PROBE_COUNT:
        raise ArtifactIntegrityError("VAL-6 ENSEMBLE: OOD/ID probe counts are incomplete")
    median_ood = float(np.median(ood))
    median_id = float(np.median(in_domain))
    common: dict[str, Any] = {
        "model_ids": models,
        "model_sha256s": digests,
        "results": ordered,
    }
    if median_id <= 0.0:
        return OodEvaluation(
            status="invalid",
            median_ood_variance=None,
            median_id_boundary_variance=None,
            variance_ratio=None,
            failure="VAL-6 ENSEMBLE: median ID boundary variance is zero",
            **common,
        )
    return OodEvaluation(
        status="valid",
        median_ood_variance=median_ood,
        median_id_boundary_variance=median_id,
        variance_ratio=median_ood / median_id,
        failure=None,
        **common,
    )


__all__ = [
    "ALL_PROBE_REYNOLDS",
    "ID_BOUNDARY_REYNOLDS",
    "OOD_REYNOLDS",
    "PROBE_COUNT",
    "EnsembleFieldPrediction",
    "OodEvaluation",
    "OodProbeResult",
    "ProbeGeometry",
    "ProbeModelIdentity",
    "ProbeReynolds",
    "evaluate_ensemble_variance",
    "select_probe_geometries",
    "summarize_ood_evaluation",
]
