from __future__ import annotations

import json
from typing import Any, Literal, cast

import numpy as np
import pytest

from soufflerie.datagen.manifest import ManifestRow
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.validation import (
    ALL_PROBE_REYNOLDS,
    EnsembleFieldPrediction,
    OodEvaluation,
    OodProbeResult,
    ProbeGeometry,
    ProbeModelIdentity,
    evaluate_ensemble_variance,
    evaluate_gate,
    ood_gate_evidence,
    select_probe_geometries,
    summarize_ood_evaluation,
)
from soufflerie.validation.gates import GATE_DEFINITIONS_BY_NAME


def _row(
    index: int,
    *,
    split: Literal["train", "validation", "test"] = "test",
    rotation: float = 15.0,
) -> ManifestRow:
    case_id = f"{index:020x}"
    design_id = f"{index + 1000:020x}"
    digest = f"{index + 1:064x}"
    return ManifestRow(
        dataset_id="d" * 20,
        case_id=case_id,
        design_id=design_id,
        split=split,
        aspect_ratio=0.75,
        rotation_deg=rotation,
        scale=1.0,
        reynolds=100.0,
        run_uri=f"runs/{case_id}/{digest}",
        run_digest=digest,
        bytes=100,
        cd=1.0,
        cl_mean=0.0,
        strouhal=0.17,
    )


def _models() -> tuple[ProbeModelIdentity, ProbeModelIdentity, ProbeModelIdentity]:
    return (
        ProbeModelIdentity(model_id="1" * 20, model_sha256="1" * 64),
        ProbeModelIdentity(model_id="2" * 20, model_sha256="2" * 64),
        ProbeModelIdentity(model_id="3" * 20, model_sha256="3" * 64),
    )


def test_probe_selection_reproduces_by_full_hash_and_preserves_test_membership() -> None:
    rows = (
        *(_row(index) for index in range(15)),
        _row(100, split="train"),
        _row(101, split="validation"),
    )
    before = tuple(row.model_dump_json() for row in rows)
    expected = tuple(
        sorted(
            (
                ProbeGeometry.from_manifest(row, selection="ood")
                for row in rows
                if row.split == "test"
            ),
            key=lambda probe: (probe.rank_sha256, probe.source_design_id),
        )[:10]
    )

    selected = select_probe_geometries(tuple(reversed(rows)), selection="ood")
    assert selected == expected
    assert all(probe.source_split == "test" for probe in selected)
    assert tuple(row.model_dump_json() for row in rows) == before
    assert tuple(probe.rank_sha256 for probe in selected) == tuple(
        sorted(probe.rank_sha256 for probe in selected)
    )


def test_sensitivity_selection_filters_rotation_boundaries_before_hash_ranking() -> None:
    rows = tuple(_row(index, rotation=float(index)) for index in range(16))
    selected = select_probe_geometries(rows, selection="sensitivity")
    assert len(selected) == 10
    assert all(1.0 <= probe.rotation_deg <= 29.0 for probe in selected)
    assert all(probe.selection == "sensitivity" for probe in selected)


def test_probe_selection_rejects_incomplete_or_ambiguous_manifest_evidence() -> None:
    with pytest.raises(ArtifactIntegrityError, match="manifest rows"):
        select_probe_geometries((), selection="ood")
    with pytest.raises(ArtifactIntegrityError, match="fewer than ten"):
        select_probe_geometries(tuple(_row(index) for index in range(9)), selection="ood")

    rows = tuple(_row(index) for index in range(10))
    mixed_dataset = (*rows[:-1], rows[-1].model_copy(update={"dataset_id": "e" * 20}))
    with pytest.raises(ArtifactIntegrityError, match="one dataset"):
        select_probe_geometries(mixed_dataset, selection="ood")

    duplicate_design = (*rows[:-1], rows[-1].model_copy(update={"design_id": rows[0].design_id}))
    with pytest.raises(ArtifactIntegrityError, match="must be unique"):
        select_probe_geometries(duplicate_design, selection="ood")

    with pytest.raises(ArtifactIntegrityError, match="test split"):
        ProbeGeometry.from_manifest(_row(100, split="train"), selection="ood")


def test_normalized_ensemble_variance_matches_hand_calculation_and_requires_distinct_models() -> (
    None
):
    geometry = select_probe_geometries(tuple(_row(index) for index in range(10)), selection="ood")[
        0
    ]
    mask = np.array([[True, False], [True, True]], dtype=np.bool_)
    models = _models()
    predictions = tuple(
        EnsembleFieldPrediction(
            model=model,
            fields=np.full((3, 2, 2), float(index), dtype=np.float32),
            fluid_mask=mask,
        )
        for index, model in enumerate(models)
    )
    result = evaluate_ensemble_variance(
        geometry,
        reynolds=20,
        predictions=predictions,
        training_output_variance=(2.0, 2.0, 2.0),
    )
    assert result.regime == "ood"
    assert result.normalized_ensemble_variance == pytest.approx(1 / 3)
    assert result.model_ids == tuple(model.model_id for model in models)

    with pytest.raises(ArtifactIntegrityError, match="identities must be distinct"):
        evaluate_ensemble_variance(
            geometry,
            reynolds=20,
            predictions=(predictions[0], predictions[0], predictions[2]),
            training_output_variance=(2.0, 2.0, 2.0),
        )
    with pytest.raises(ArtifactIntegrityError, match="variances"):
        evaluate_ensemble_variance(
            geometry,
            reynolds=20,
            predictions=predictions,
            training_output_variance=(2.0, 0.0, 2.0),
        )

    different_mask = EnsembleFieldPrediction(
        model=models[1],
        fields=predictions[1].fields,
        fluid_mask=np.array([[True, True], [True, True]], dtype=np.bool_),
    )
    with pytest.raises(ArtifactIntegrityError, match="fluid masks differ"):
        evaluate_ensemble_variance(
            geometry,
            reynolds=20,
            predictions=(predictions[0], different_mask, predictions[2]),
            training_output_variance=(2.0, 2.0, 2.0),
        )


def test_prediction_and_model_identity_contracts_reject_invalid_arrays() -> None:
    with pytest.raises(ValueError, match="prefix"):
        ProbeModelIdentity(model_id="a" * 20, model_sha256="b" * 64)

    mask = np.ones((2, 2), dtype=np.bool_)
    with pytest.raises(ArtifactIntegrityError, match="float32"):
        EnsembleFieldPrediction(
            model=_models()[0],
            fields=cast(Any, np.ones((3, 2, 2), dtype=np.float64)),
            fluid_mask=mask,
        )
    with pytest.raises(ArtifactIntegrityError, match="nonempty"):
        EnsembleFieldPrediction(
            model=_models()[0],
            fields=np.ones((3, 2, 2), dtype=np.float32),
            fluid_mask=np.zeros((2, 2), dtype=np.bool_),
        )


def _probe_results(*, id_value: float) -> tuple[OodProbeResult, ...]:
    geometries = select_probe_geometries(tuple(_row(index) for index in range(10)), selection="ood")
    models = _models()
    model_ids = (models[0].model_id, models[1].model_id, models[2].model_id)
    model_sha256s = (
        models[0].model_sha256,
        models[1].model_sha256,
        models[2].model_sha256,
    )
    return tuple(
        OodProbeResult(
            geometry=geometry,
            reynolds=reynolds,
            regime="ood" if reynolds in (20, 400) else "id_boundary",
            model_ids=model_ids,
            model_sha256s=model_sha256s,
            normalized_ensemble_variance=(3.0 if reynolds in (20, 400) else id_value),
        )
        for geometry in geometries
        for reynolds in ALL_PROBE_REYNOLDS
    )


def test_complete_ood_evaluation_records_ratio_gate_and_zero_id_failure() -> None:
    evaluation = summarize_ood_evaluation(tuple(reversed(_probe_results(id_value=1.0))))
    assert evaluation.status == "valid"
    assert evaluation.variance_ratio == 3.0
    gate = evaluate_gate(
        GATE_DEFINITIONS_BY_NAME["ood_variance_increase"],
        ood_gate_evidence(evaluation),
    )
    assert gate.status == "green"
    assert OodEvaluation.model_validate_json(evaluation.model_dump_json()) == evaluation

    invalid = summarize_ood_evaluation(_probe_results(id_value=0.0))
    assert invalid.status == "invalid"
    assert invalid.variance_ratio is None
    assert (
        evaluate_gate(
            GATE_DEFINITIONS_BY_NAME["ood_variance_increase"],
            ood_gate_evidence(invalid),
        ).status
        == "red"
    )

    with pytest.raises(ArtifactIntegrityError, match="counts are incomplete"):
        summarize_ood_evaluation(_probe_results(id_value=1.0)[:-1])
    with pytest.raises(ArtifactIntegrityError, match="results are required"):
        summarize_ood_evaluation(cast(Any, (object(),)))


def test_probe_rank_and_result_models_reject_tampering() -> None:
    geometry = select_probe_geometries(tuple(_row(index) for index in range(10)), selection="ood")[
        0
    ]
    payload = json.loads(geometry.model_dump_json())
    with pytest.raises(ValueError, match="rank digest"):
        ProbeGeometry.model_validate({**payload, "scale": 1.1})

    with pytest.raises(ValueError, match="three distinct full model digests"):
        OodProbeResult(
            geometry=geometry,
            reynolds=20,
            regime="ood",
            model_ids=("1" * 20, "2" * 20, "3" * 20),
            model_sha256s=("1" * 64, "2" * 64, "2" * 64),
            normalized_ensemble_variance=1.0,
        )


def test_ood_evaluation_binds_one_dataset_and_one_geometry_per_design() -> None:
    results = _probe_results(id_value=1.0)
    original = results[0]
    source_index = int(original.geometry.source_design_id, 16) - 1000
    source = _row(source_index, rotation=original.geometry.rotation_deg)

    mixed_dataset_geometry = ProbeGeometry.from_manifest(
        source.model_copy(update={"dataset_id": "e" * 20}),
        selection="ood",
    )
    mixed_dataset = OodProbeResult.model_validate(
        {**original.model_dump(mode="python"), "geometry": mixed_dataset_geometry}
    )
    with pytest.raises(ValueError, match="one dataset identity"):
        summarize_ood_evaluation((mixed_dataset, *results[1:]))

    changed_geometry = ProbeGeometry.from_manifest(
        source.model_copy(update={"rotation_deg": original.geometry.rotation_deg + 0.5}),
        selection="ood",
    )
    changed_result = OodProbeResult.model_validate(
        {**original.model_dump(mode="python"), "geometry": changed_geometry}
    )
    with pytest.raises(ValueError, match="same source geometry"):
        summarize_ood_evaluation((changed_result, *results[1:]))

    with pytest.raises(ValueError, match="regime"):
        OodProbeResult(
            geometry=original.geometry,
            reynolds=20,
            regime="id_boundary",
            model_ids=("1" * 20, "2" * 20, "3" * 20),
            model_sha256s=("1" * 64, "2" * 64, "3" * 64),
            normalized_ensemble_variance=1.0,
        )
