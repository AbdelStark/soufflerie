from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypedDict, cast
from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

from soufflerie.datagen.manifest import ManifestRow
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import Provenance, canonical_sha256, sha256_bytes
from soufflerie.validation import (
    ALL_PROBE_REYNOLDS,
    REQUIRED_GATE_DEFINITIONS,
    BaselinePlotSeries,
    CasePlotPoint,
    FieldComparisonData,
    GateEvidence,
    GateName,
    GateResult,
    MetricName,
    MetricObservation,
    MetricSummary,
    OodEvaluation,
    OodProbeResult,
    PlotManifest,
    ProbeModelIdentity,
    RenderedValidationArtifacts,
    SensitivityEvaluation,
    SensitivityProbeResult,
    ValidationPlotData,
    ValidationReport,
    check_validation_artifacts,
    evaluate_required_gates,
    load_validation_report,
    render_validation_artifacts,
    render_validation_markdown,
    report_parent_sha256,
    select_probe_geometries,
    summarize_metric,
    summarize_ood_evaluation,
    validate_report_publication,
    write_validation_artifacts,
)
from soufflerie.validation.metrics import MetricUnits

PROJECT_ROOT = Path(__file__).parents[2]
FIXTURE = PROJECT_ROOT / "tests/fixtures/report.json"


def _digest(character: str) -> str:
    return character * 64


def _manifest_row(index: int) -> ManifestRow:
    case_id = f"{index + 100:020x}"
    design_id = f"{index + 200:020x}"
    return ManifestRow(
        dataset_id="1" * 20,
        case_id=case_id,
        design_id=design_id,
        split="test",
        aspect_ratio=0.55 + index * 0.025,
        rotation_deg=2.0 + index,
        scale=0.8 + index * 0.02,
        reynolds=50.0 + index * 20.0,
        run_uri=f"runs/{case_id}/{index + 10:064x}",
        run_digest=f"{index + 10:064x}",
        bytes=100,
        cd=1.0 + index * 0.02,
        cl_mean=0.0,
        strouhal=0.17,
    )


def _models() -> tuple[ProbeModelIdentity, ProbeModelIdentity, ProbeModelIdentity]:
    return (
        ProbeModelIdentity(model_id="3" * 20, model_sha256=_digest("3")),
        ProbeModelIdentity(model_id="4" * 20, model_sha256=_digest("4")),
        ProbeModelIdentity(model_id="5" * 20, model_sha256=_digest("5")),
    )


def _ood_evaluation() -> OodEvaluation:
    probes = select_probe_geometries(
        tuple(_manifest_row(index) for index in range(10)), selection="ood"
    )
    models = _models()
    results = tuple(
        OodProbeResult(
            geometry=probe,
            reynolds=reynolds,
            regime="ood" if reynolds in (20, 400) else "id_boundary",
            model_ids=(models[0].model_id, models[1].model_id, models[2].model_id),
            model_sha256s=(
                models[0].model_sha256,
                models[1].model_sha256,
                models[2].model_sha256,
            ),
            normalized_ensemble_variance=3.0 if reynolds in (20, 400) else 1.0,
        )
        for probe in probes
        for reynolds in ALL_PROBE_REYNOLDS
    )
    return summarize_ood_evaluation(results)


def _sensitivity_evaluation() -> SensitivityEvaluation:
    probes = select_probe_geometries(
        tuple(_manifest_row(index) for index in range(10)),
        selection="sensitivity",
    )
    model = _models()[0]
    results = tuple(
        SensitivityProbeResult(
            geometry=probe,
            model=model,
            cd_minus=0.75,
            cd_center=1.0,
            cd_plus=1.25,
            autograd_cd_per_degree=-1.0 if index == 9 else 1.0,
            central_difference_cd_per_degree=1.0,
            agrees=index != 9,
        )
        for index, probe in enumerate(probes)
    )
    return SensitivityEvaluation(model=model, results=results, agreement_count=9)


def _case_points() -> tuple[CasePlotPoint, ...]:
    return tuple(
        CasePlotPoint(
            case_id=f"{index:020x}",
            aspect_ratio=0.5 + index * 0.04,
            rotation_deg=2.0 + index * 2.0,
            scale=0.75 + index * 0.04,
            reynolds=40.0 + index * 23.0,
            velocity_rel_l2=0.03 + index * 0.01,
            cd_head_pct=0.5 * index,
            cd_field_pct=0.4 * index,
            prediction_div_mean_abs=0.1 * index,
            obstacle_ratio=0.0008 * index,
            cd_head=0.8 + index * 0.05,
            cd_field=0.81 + index * 0.048,
            cd_solver=0.82 + index * 0.049,
        )
        for index in range(1, 12)
    )


def _field_data(
    *,
    selection: Literal["representative", "worst"],
    case_id: str,
    offset: float,
) -> FieldComparisonData:
    solver = tuple(tuple(0.1 * row + 0.03 * column for column in range(5)) for row in range(4))
    surrogate = tuple(tuple(value + offset for value in row) for row in solver)
    return FieldComparisonData.model_validate(
        {
            "selection": selection,
            "case_id": case_id,
            "solver": solver,
            "surrogate": surrogate,
        }
    )


def _plot_data(points: tuple[CasePlotPoint, ...]) -> ValidationPlotData:
    return ValidationPlotData(
        representative_fields=_field_data(
            selection="representative",
            case_id=points[5].case_id,
            offset=0.01,
        ),
        worst_fields=_field_data(
            selection="worst",
            case_id=points[-1].case_id,
            offset=0.08,
        ),
        cases=points,
        baselines=(
            BaselinePlotSeries(
                kind="selected_fno",
                artifact_id="3" * 20,
                median_velocity_rel_l2=0.09,
                median_cd_pct=3.0,
            ),
            BaselinePlotSeries(
                kind="mean_field",
                artifact_id="6" * 20,
                median_velocity_rel_l2=0.15,
                median_cd_pct=7.0,
            ),
            BaselinePlotSeries(
                kind="nearest_design",
                artifact_id="7" * 20,
                median_velocity_rel_l2=0.12,
                median_cd_pct=5.5,
            ),
        ),
    )


def _metric_summaries(points: tuple[CasePlotPoint, ...]) -> dict[str, MetricSummary]:
    values: dict[MetricName, tuple[float, ...]] = {
        "velocity_rel_l2": tuple(item.velocity_rel_l2 for item in points),
        "cd_head_pct": tuple(item.cd_head_pct for item in points),
        "cd_field_pct": tuple(item.cd_field_pct for item in points),
        "head_field_gap_pct": tuple(
            100.0 * abs(item.cd_head - item.cd_field) / max(abs(item.cd_solver), 0.1)
            for item in points
        ),
        "prediction_div_mean_abs": tuple(item.prediction_div_mean_abs for item in points),
        "solver_div_mean_abs": tuple(item.prediction_div_mean_abs / 2.0 for item in points),
        "obstacle_ratio": tuple(item.obstacle_ratio for item in points),
    }
    units: dict[MetricName, MetricUnits] = {
        "velocity_rel_l2": "ratio",
        "cd_head_pct": "percent",
        "cd_field_pct": "percent",
        "head_field_gap_pct": "percent",
        "prediction_div_mean_abs": "inverse_lattice_unit",
        "solver_div_mean_abs": "inverse_lattice_unit",
        "obstacle_ratio": "ratio",
    }
    return {
        name: summarize_metric(
            name,
            {
                point.case_id: MetricObservation(
                    name=name,
                    status="valid",
                    value=value,
                    units=units[name],
                )
                for point, value in zip(points, observations, strict=True)
            },
            report_seed=20260903,
            bootstrap_resamples=100,
        )
        for name, observations in values.items()
    }


def _gates() -> tuple[GateResult, ...]:
    values: dict[GateName, float | int | bool] = {
        "field_error": 0.09,
        "cd_head_error": 3.0,
        "head_field_consistency": 1.0,
        "divergence": 2.0,
        "obstacle_compliance": 0.0084,
        "mean_baseline_field": 0.09,
        "nearest_baseline_field": 0.09,
        "mean_baseline_cd": 3.0,
        "nearest_baseline_cd": 3.0,
        "ood_variance_increase": 3.0,
        "sensitivity_sign": 9,
        "evidence_integrity": True,
    }
    dynamic: dict[GateName, float] = {
        "mean_baseline_field": 0.15,
        "nearest_baseline_field": 0.12,
        "mean_baseline_cd": 7.0,
        "nearest_baseline_cd": 5.5,
    }
    return evaluate_required_gates(
        tuple(
            GateEvidence(
                name=definition.name,
                value=values[definition.name],
                comparison_threshold=dynamic.get(definition.name),
                evidence=(f"fixture:{definition.metric}",),
            )
            for definition in REQUIRED_GATE_DEFINITIONS
        )
    )


def build_report() -> ValidationReport:
    points = _case_points()
    started = datetime(2026, 9, 3, 8, 0, tzinfo=UTC)
    provenance = Provenance(
        source_revision="a" * 40,
        source_dirty=False,
        python_version="3.11.14",
        lock_sha256=_digest("b"),
        packages={"numpy": "2.2.6", "soufflerie": "0.1.0"},
        os="linux",
        architecture="x86_64",
        device_class="L40S-fixture",
        dtype_policy="fields-fp32-metrics-fp64",
        config_sha256=_digest("c"),
        parent_sha256={
            "dataset": _digest("1"),
            "solver": _digest("2"),
            "ensemble_model_0": _digest("3"),
            "ensemble_model_1": _digest("4"),
            "ensemble_model_2": _digest("5"),
            "baseline_0": _digest("6"),
            "baseline_1": _digest("7"),
        },
        seeds=(17, 23, 31),
        deterministic=True,
        started_at=started,
        completed_at=started + timedelta(minutes=4),
        gpu_seconds=240.0,
    )
    return ValidationReport.create(
        dataset_id="1" * 20,
        selected_model_id="3" * 20,
        ensemble_model_ids=("3" * 20, "4" * 20, "5" * 20),
        baseline_ids=("6" * 20, "7" * 20),
        metrics=_metric_summaries(points),
        gates=_gates(),
        overall_status="red",
        provenance=provenance,
        ood=_ood_evaluation(),
        sensitivity=_sensitivity_evaluation(),
        plot_data=_plot_data(points),
    )


class _Expectations(TypedDict):
    expected_source_revision: str
    expected_lock_sha256: str
    expected_config_sha256: str
    expected_packages: dict[str, str]
    expected_parent_sha256: dict[str, str]


def _expectations(report: ValidationReport) -> _Expectations:
    return {
        "expected_source_revision": report.provenance.source_revision,
        "expected_lock_sha256": report.provenance.lock_sha256,
        "expected_config_sha256": report.provenance.config_sha256,
        "expected_packages": report.provenance.packages,
        "expected_parent_sha256": report.provenance.parent_sha256,
    }


def test_checked_in_report_fixture_and_every_generated_artifact_are_current() -> None:
    report = load_validation_report(FIXTURE)
    assert report == build_report()
    artifacts = render_validation_artifacts(report, plot_directory="report.plots")
    assert check_validation_artifacts(FIXTURE, artifacts) == ()

    completed = subprocess.run(
        [sys.executable, "scripts/render_validation.py", "--check", str(FIXTURE)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "validation_report=PASS" in completed.stdout


def test_markdown_is_red_first_complete_and_does_not_recompute_status() -> None:
    report = build_report()
    markdown = render_validation_markdown(report).decode()
    assert markdown.startswith("# Validation status: RED\n\n- Overall status: **RED**")
    assert markdown.index("Release blocked") < markdown.index("## Diagnostic plots")
    assert all(definition.name in markdown for definition in REQUIRED_GATE_DEFINITIONS)
    assert report.dataset_id in markdown
    assert report.selected_model_id in markdown
    assert report.generator_version in markdown

    bypassed = report.model_copy(update={"overall_status": "green"})
    assert render_validation_markdown(bypassed).decode().startswith("# Validation status: GREEN")


def test_svg_and_manifest_bytes_are_deterministic_and_digest_bound() -> None:
    report = build_report()
    first = render_validation_artifacts(report)
    second = render_validation_artifacts(report)
    assert first == second
    assert len(first.plots) == 8
    for content in first.plots.values():
        root = ElementTree.fromstring(content)
        assert root.tag.endswith("svg")
        assert report.report_id.encode() in content

    manifest = PlotManifest.model_validate_json(first.plot_manifest_json)
    assert manifest.report_id == report.report_id
    assert tuple(item.filename for item in manifest.plots) == tuple(first.plots)
    assert all(item.sha256 == sha256_bytes(first.plots[item.filename]) for item in manifest.plots)


def test_publication_lineage_requires_external_reviewed_values() -> None:
    report = build_report()
    expectations = _expectations(report)
    validate_report_publication(report, **expectations)
    with pytest.raises(ArtifactIntegrityError, match="source revision"):
        validate_report_publication(
            report,
            expected_source_revision="d" * 40,
            expected_lock_sha256=expectations["expected_lock_sha256"],
            expected_config_sha256=expectations["expected_config_sha256"],
            expected_packages=expectations["expected_packages"],
            expected_parent_sha256=expectations["expected_parent_sha256"],
        )
    with pytest.raises(ArtifactIntegrityError, match="reviewed artifacts"):
        validate_report_publication(
            report,
            expected_source_revision=expectations["expected_source_revision"],
            expected_lock_sha256=expectations["expected_lock_sha256"],
            expected_config_sha256=expectations["expected_config_sha256"],
            expected_packages=expectations["expected_packages"],
            expected_parent_sha256={
                **report.provenance.parent_sha256,
                "solver": _digest("d"),
            },
        )


def test_report_loader_rejects_identity_tampering_and_noncanonical_json(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["report_id"] = "f" * 20
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="does not match ValidationReport"):
        load_validation_report(tampered)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(FIXTURE.read_text(encoding="utf-8").strip(), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="canonical rendered form"):
        load_validation_report(noncanonical)


def test_atomic_writer_and_checker_fail_closed_on_stale_or_extra_files(tmp_path: Path) -> None:
    report = build_report()
    target = tmp_path / "report.json"
    artifacts = render_validation_artifacts(report, plot_directory="report.plots")
    write_validation_artifacts(target, artifacts)
    assert check_validation_artifacts(target, artifacts) == ()

    target.with_suffix(".md").write_text("stale\n", encoding="utf-8")
    assert any("stale" in item for item in check_validation_artifacts(target, artifacts))
    target.with_suffix(".plots").joinpath("extra.svg").write_text("extra", encoding="utf-8")
    assert any("extra file set" in item for item in check_validation_artifacts(target, artifacts))
    with pytest.raises(ArtifactIntegrityError, match="unrecognized"):
        write_validation_artifacts(target, artifacts)


def test_atomic_writer_trusts_only_provider_mount_ancestors(tmp_path: Path) -> None:
    report = build_report()
    artifacts = render_validation_artifacts(report, plot_directory="report.plots")
    provider_storage = tmp_path / "provider-storage"
    provider_storage.mkdir()
    mount = tmp_path / "mounted-volume"
    mount.symlink_to(provider_storage, target_is_directory=True)
    trusted_root = mount / "soufflerie" / "v1"
    target = trusted_root / "validation" / "report.json"

    write_validation_artifacts(target, artifacts, trusted_root=trusted_root)
    assert check_validation_artifacts(target, artifacts, trusted_root=trusted_root) == ()

    outside = tmp_path / "outside"
    with pytest.raises(ArtifactIntegrityError, match="escapes trusted root"):
        write_validation_artifacts(outside / "report.json", artifacts, trusted_root=trusted_root)

    unsafe_root = mount / "unsafe"
    unsafe_root.mkdir()
    unsafe_target = unsafe_root / "validation"
    unsafe_target.symlink_to(trusted_root / "validation", target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="symbolic-link publication"):
        write_validation_artifacts(
            unsafe_target / "report.json",
            artifacts,
            trusted_root=unsafe_root,
        )


def test_plot_data_and_report_gate_order_reject_ambiguous_evidence() -> None:
    report = build_report()
    assert report.plot_data is not None
    plot_payload = report.plot_data.model_dump(mode="python")
    with pytest.raises(ValidationError, match="canonical order"):
        ValidationPlotData.model_validate(
            {**plot_payload, "cases": tuple(reversed(report.plot_data.cases))}
        )
    with pytest.raises(ValidationError, match="largest velocity error"):
        ValidationPlotData.model_validate(
            {
                **plot_payload,
                "worst_fields": report.plot_data.worst_fields.model_copy(
                    update={"case_id": report.plot_data.cases[-2].case_id}
                ),
            }
        )
    with pytest.raises(ValidationError, match="immutable RFC order"):
        ValidationReport.model_validate(
            {
                **report.model_dump(mode="python"),
                "gates": tuple(reversed(report.gates)),
            }
        )


def test_plot_data_rejects_malformed_grids_labels_and_baselines() -> None:
    report = build_report()
    assert report.plot_data is not None
    fields = report.plot_data.representative_fields.model_dump(mode="python")
    for update, message in (
        ({"solver": ((1.0, 2.0),)}, "height"),
        ({"solver": ((1.0,), (2.0,))}, "width"),
        ({"surrogate": ((1.0, 2.0), (3.0, 4.0))}, "share one shape"),
    ):
        with pytest.raises(ValidationError, match=message):
            FieldComparisonData.model_validate({**fields, **update})

    payload = report.plot_data.model_dump(mode="python")
    baselines = report.plot_data.baselines
    plot_updates: tuple[tuple[dict[str, object], str], ...] = (
        ({"baselines": (baselines[0], baselines[2], baselines[1])}, "canonical order"),
        (
            {
                "baselines": (
                    baselines[0],
                    baselines[1],
                    baselines[2].model_copy(update={"artifact_id": baselines[1].artifact_id}),
                )
            },
            "must be distinct",
        ),
        (
            {
                "representative_fields": report.plot_data.representative_fields.model_copy(
                    update={"selection": "worst"}
                )
            },
            "representative field data",
        ),
        (
            {
                "worst_fields": report.plot_data.worst_fields.model_copy(
                    update={"selection": "representative"}
                )
            },
            "worst field data",
        ),
        (
            {
                "worst_fields": report.plot_data.worst_fields.model_copy(
                    update={"case_id": report.plot_data.representative_fields.case_id}
                )
            },
            "must be distinct",
        ),
        (
            {
                "representative_fields": report.plot_data.representative_fields.model_copy(
                    update={"case_id": "f" * 20}
                )
            },
            "must occur",
        ),
        (
            {
                "representative_fields": report.plot_data.representative_fields.model_copy(
                    update={"case_id": report.plot_data.cases[4].case_id}
                )
            },
            "closest to the error median",
        ),
    )
    for plot_update, message in plot_updates:
        with pytest.raises(ValidationError, match=message):
            ValidationPlotData.model_validate({**payload, **plot_update})


def test_reporting_helpers_reject_incomplete_lineage_and_unsafe_outputs(tmp_path: Path) -> None:
    report = build_report()
    assert report.ood is not None
    assert report.sensitivity is not None
    artifacts = render_validation_artifacts(report)
    with pytest.raises(TypeError, match="ValidationReport"):
        report_parent_sha256(cast(Any, object()))

    for changed, message in (
        (
            report.model_copy(
                update={"provenance": report.provenance.model_copy(update={"parent_sha256": {}})}
            ),
            "report parents",
        ),
        (report.model_copy(update={"dataset_id": "f" * 20}), "dataset ID"),
        (
            report.model_copy(update={"ensemble_model_ids": ("f" * 20, "4" * 20, "5" * 20)}),
            "ensemble model IDs",
        ),
        (report.model_copy(update={"baseline_ids": ("f" * 20, "7" * 20)}), "baseline IDs"),
        (
            report.model_copy(
                update={"ood": report.ood.model_copy(update={"model_sha256s": (_digest("f"),) * 3})}
            ),
            "OOD model digests",
        ),
        (
            report.model_copy(
                update={
                    "sensitivity": report.sensitivity.model_copy(
                        update={
                            "model": ProbeModelIdentity(
                                model_id="3" * 20,
                                model_sha256="3" * 20 + "f" * 44,
                            )
                        }
                    )
                }
            ),
            "sensitivity model digest",
        ),
    ):
        with pytest.raises(ArtifactIntegrityError, match=message):
            report_parent_sha256(changed)

    with pytest.raises(ArtifactIntegrityError, match="exact canonical plot set"):
        PlotManifest.create(report=report, plots={})
    manifest = PlotManifest.model_validate_json(artifacts.plot_manifest_json)
    manifest_payload = manifest.model_dump(mode="python")
    with pytest.raises(ValidationError, match="digest does not bind"):
        PlotManifest.model_validate({**manifest_payload, "manifest_sha256": "f" * 64})
    duplicate_plots = tuple(
        item.model_copy(update={"sha256": manifest.plots[0].sha256}) for item in manifest.plots
    )
    duplicate_payload = {**manifest_payload, "plots": duplicate_plots}
    duplicate_payload["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in duplicate_payload.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValidationError, match="distinct content digests"):
        PlotManifest.model_validate(duplicate_payload)

    with pytest.raises(ArtifactIntegrityError, match="canonical plot set"):
        RenderedValidationArtifacts(
            report_json=artifacts.report_json,
            markdown=artifacts.markdown,
            plot_manifest_json=artifacts.plot_manifest_json,
            plots=MappingProxyType(dict(reversed(tuple(artifacts.plots.items())))),
        )
    with pytest.raises(ValueError, match="safe path component"):
        render_validation_markdown(report, plot_directory="../plots")
    with pytest.raises(ArtifactIntegrityError, match="complete plot data"):
        render_validation_artifacts(report.model_copy(update={"plot_data": None}))
    with pytest.raises(ArtifactIntegrityError, match="clean deterministic"):
        render_validation_artifacts(
            report.model_copy(
                update={"provenance": report.provenance.model_copy(update={"source_dirty": True})}
            )
        )

    missing_target = tmp_path / "missing/report.json"
    missing_errors = check_validation_artifacts(missing_target, artifacts)
    assert any("missing or unsafe" in item for item in missing_errors)
    unsafe_target = tmp_path / "unsafe.json"
    unsafe_target.with_suffix(".plots").write_text("not a directory", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="not a safe directory"):
        write_validation_artifacts(unsafe_target, artifacts)
