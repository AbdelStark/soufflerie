"""Provider-neutral immutable validation and report publication pipeline."""

from __future__ import annotations

import importlib
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal, cast

import numpy as np

from infra.runtime_manifest import RuntimeBuildManifest
from infra.train_validate_execution import (
    ExecutionAccounting,
    RemoteValidationRequest,
    ValidationReceipt,
    assert_report_matches_request,
    assert_request_matches_build,
    utc_now,
)
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.geometry import OUTPUT_GRID_NX, OUTPUT_GRID_NY
from soufflerie.schemas import ArtifactRef, FlowFields, Provenance, canonical_sha256
from soufflerie.solver import field_drag_coefficient
from soufflerie.surrogate import (
    LocalModelBundleStore,
    PredictionBatch,
    denormalize_fields,
    fit_preprocessing_statistics,
    instantiate_bundle_predictor,
    prediction_batch_to_torch,
)
from soufflerie.surrogate.fno import FnoPredictor
from soufflerie.training import fit_baselines, open_manifest_dataset
from soufflerie.validation import (
    ALL_PROBE_REYNOLDS,
    BaselinePlotSeries,
    CaseMetrics,
    CasePlotPoint,
    EnsembleFieldPrediction,
    FieldComparisonData,
    GateEvidence,
    MetricName,
    MetricObservation,
    MetricSummary,
    ProbeGeometry,
    ProbeModelIdentity,
    ValidationPlotData,
    ValidationReport,
    check_validation_artifacts,
    divergence_gate_evidence,
    evaluate_case_metrics,
    evaluate_ensemble_variance,
    evaluate_required_gates,
    evaluate_sensitivity_probes,
    head_field_consistency_gate_evidence,
    load_validation_report,
    ood_gate_evidence,
    overall_gate_status,
    render_validation_artifacts,
    select_probe_geometries,
    sensitivity_gate_evidence,
    summarize_metric,
    summarize_ood_evaluation,
    write_validation_artifacts,
)
from soufflerie.validation.sensitivity import AutogradCdResult

_INLET_VELOCITY_LU = 0.05
_REFERENCE_DIAMETER_LU = 16.0
_SPONGE_START_X = 224
_VALIDATION_BATCH_SIZE = 8


@dataclass(slots=True)
class _GpuMeter:
    torch: Any
    device_index: int = 0
    seconds: float = 0.0

    def measure(self, function: Callable[[], Any]) -> Any:
        self.torch.cuda.synchronize(self.device_index)
        started = time.perf_counter()
        value = function()
        self.torch.cuda.synchronize(self.device_index)
        self.seconds += time.perf_counter() - started
        return value


@dataclass(frozen=True, slots=True)
class _SelectedCase:
    metrics: CaseMetrics
    point: CasePlotPoint
    solver_speed: np.ndarray[Any, Any]
    surrogate_speed: np.ndarray[Any, Any]


def _predict(
    predictor: Any,
    batch: PredictionBatch,
    *,
    torch: Any,
    precision: Literal["bf16", "fp16"],
    meter: _GpuMeter,
) -> Any:
    if not isinstance(predictor, FnoPredictor):
        return predictor.predict(batch)

    def run() -> Any:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=dtype, enabled=True),
        ):
            return predictor.forward(batch)

    return meter.measure(run)


def _flow_fields(sample: Any, physical: np.ndarray[Any, Any], index: int) -> FlowFields:
    sdf = np.ascontiguousarray(sample.sdf, dtype=np.float32)
    # Preprocessing and every learned prediction use the sign of the persisted,
    # quantized model-grid SDF as their fluid membership. The separately stored
    # mask is nearest-sampled from the solver grid and can differ at boundary
    # cells, so deriving it here keeps validation on the exact trained geometry.
    obstacle = np.ascontiguousarray(sdf <= np.float32(0.0), dtype=np.bool_)
    return FlowFields(
        u=np.ascontiguousarray(physical[index, 0], dtype=np.float32),
        v=np.ascontiguousarray(physical[index, 1], dtype=np.float32),
        rho=np.ascontiguousarray(physical[index, 2], dtype=np.float32),
        sdf=sdf,
        obstacle_mask=obstacle,
    )


def _observation_value(observation: MetricObservation) -> float:
    if observation.status != "valid" or observation.value is None:
        raise ArtifactIntegrityError(
            f"REMOTE_VALIDATE_METRIC_INVALID: {observation.name} is invalid"
        )
    return observation.value


def _evaluate_test_set(
    request: RemoteValidationRequest,
    *,
    dataset: Any,
    statistics: Any,
    selected: FnoPredictor,
    baselines: tuple[Any, Any],
    torch: Any,
    meter: _GpuMeter,
) -> tuple[
    dict[str, tuple[CaseMetrics, ...]],
    tuple[_SelectedCase, ...],
]:
    by_predictor: dict[str, list[CaseMetrics]] = {
        "selected": [],
        "mean": [],
        "nearest": [],
    }
    selected_cases: list[_SelectedCase] = []
    predictors = (selected, *baselines)
    names = ("selected", "mean", "nearest")
    for manifest_batch in dataset.iter_batches(
        statistics,
        "test",
        batch_size=_VALIDATION_BATCH_SIZE,
        seed=request.config.report_seed,
        epoch=0,
    ):
        tensor_batch = prediction_batch_to_torch(manifest_batch.data, device="cuda:0")
        outputs = tuple(
            _predict(
                predictor,
                tensor_batch,
                torch=torch,
                precision=request.precision,
                meter=meter,
            )
            for predictor in predictors
        )
        physical_outputs = tuple(
            denormalize_fields(
                np.ascontiguousarray(
                    output.fields_normalized.detach().cpu().numpy(), dtype=np.float32
                ),
                statistics,
            )
            for output in outputs
        )
        cd_outputs = tuple(
            np.asarray(output.cd_head.detach().cpu().numpy(), dtype=np.float64)
            for output in outputs
        )
        for index, row in enumerate(manifest_batch.membership.rows):
            sample = dataset.load_sample(row)
            solver = _flow_fields(
                sample,
                np.stack((sample.u_mean, sample.v_mean, sample.rho_mean), axis=0)[None, ...],
                0,
            )
            for name, physical, cd_values in zip(names, physical_outputs, cd_outputs, strict=True):
                prediction = _flow_fields(sample, physical, index)
                cd_head = float(cd_values[index])
                cd_field = field_drag_coefficient(
                    prediction,
                    sponge_start_x=_SPONGE_START_X,
                    inlet_velocity_lu=_INLET_VELOCITY_LU,
                    reference_diameter_lu=_REFERENCE_DIAMETER_LU,
                ).cd
                metrics = evaluate_case_metrics(
                    case_id=row.case_id,
                    prediction_u=prediction.u,
                    prediction_v=prediction.v,
                    solver_u=solver.u,
                    solver_v=solver.v,
                    fluid_mask=np.ascontiguousarray(sample.sdf > 0, dtype=np.bool_),
                    obstacle_mask=solver.obstacle_mask,
                    cd_head=cd_head,
                    cd_field=cd_field,
                    cd_solver=row.cd,
                    inlet_velocity_lu=_INLET_VELOCITY_LU,
                )
                by_predictor[name].append(metrics)
                if name == "selected":
                    point = CasePlotPoint(
                        case_id=row.case_id,
                        aspect_ratio=row.aspect_ratio,
                        rotation_deg=row.rotation_deg,
                        scale=row.scale,
                        reynolds=row.reynolds,
                        velocity_rel_l2=_observation_value(metrics.velocity_rel_l2),
                        cd_head_pct=_observation_value(metrics.cd_head_pct),
                        cd_field_pct=_observation_value(metrics.cd_field_pct),
                        prediction_div_mean_abs=_observation_value(metrics.prediction_div_mean_abs),
                        obstacle_ratio=_observation_value(metrics.obstacle_ratio),
                        cd_head=cd_head,
                        cd_field=cd_field,
                        cd_solver=row.cd,
                    )
                    selected_cases.append(
                        _SelectedCase(
                            metrics=metrics,
                            point=point,
                            solver_speed=np.hypot(solver.u, solver.v),
                            surrogate_speed=np.hypot(prediction.u, prediction.v),
                        )
                    )
    if any(len(values) != 200 for values in by_predictor.values()):
        raise ArtifactIntegrityError("REMOTE_VALIDATE_TEST_INCOMPLETE: test membership changed")
    return (
        {name: tuple(values) for name, values in by_predictor.items()},
        tuple(sorted(selected_cases, key=lambda item: item.point.case_id)),
    )


def _summaries(
    request: RemoteValidationRequest,
    cases: Mapping[str, tuple[CaseMetrics, ...]],
) -> dict[str, MetricSummary]:
    names: tuple[MetricName, ...] = (
        "velocity_rel_l2",
        "cd_head_pct",
        "cd_field_pct",
        "head_field_gap_pct",
        "prediction_div_mean_abs",
        "solver_div_mean_abs",
        "obstacle_ratio",
    )
    result: dict[str, MetricSummary] = {}
    for metric_name in names:
        result[f"selected.{metric_name}"] = summarize_metric(
            metric_name,
            {
                case.case_id: cast(MetricObservation, getattr(case, metric_name))
                for case in cases["selected"]
            },
            report_seed=request.config.report_seed,
            bootstrap_resamples=request.config.bootstrap_resamples,
        )
    baseline_metric_names: tuple[MetricName, MetricName] = (
        "velocity_rel_l2",
        "cd_head_pct",
    )
    for predictor_name in ("mean", "nearest"):
        for metric_name in baseline_metric_names:
            result[f"{predictor_name}.{metric_name}"] = summarize_metric(
                metric_name,
                {
                    case.case_id: cast(MetricObservation, getattr(case, metric_name))
                    for case in cases[predictor_name]
                },
                report_seed=request.config.report_seed,
                bootstrap_resamples=request.config.bootstrap_resamples,
            )
    return result


def _valid_summary_value(summary: MetricSummary, name: str) -> float:
    value = getattr(summary, name)
    if summary.status != "valid" or not isinstance(value, (int, float)):
        raise ArtifactIntegrityError(
            f"REMOTE_VALIDATE_SUMMARY_INVALID: {summary.name}.{name} is invalid"
        )
    return float(value)


def _plot_data(
    request: RemoteValidationRequest,
    *,
    cases: tuple[_SelectedCase, ...],
    summaries: Mapping[str, MetricSummary],
) -> ValidationPlotData:
    middle = median(item.point.velocity_rel_l2 for item in cases)
    representative = min(
        cases,
        key=lambda item: (abs(item.point.velocity_rel_l2 - middle), item.point.case_id),
    )
    worst = min(cases, key=lambda item: (-item.point.velocity_rel_l2, item.point.case_id))

    def field_data(
        item: _SelectedCase,
        selection: Literal["representative", "worst"],
    ) -> FieldComparisonData:
        solver = np.ascontiguousarray(item.solver_speed[::4, ::2], dtype=np.float64)
        surrogate = np.ascontiguousarray(item.surrogate_speed[::4, ::2], dtype=np.float64)
        return FieldComparisonData(
            selection=selection,
            case_id=item.point.case_id,
            solver=tuple(tuple(float(value) for value in row) for row in solver),
            surrogate=tuple(tuple(float(value) for value in row) for row in surrogate),
        )

    return ValidationPlotData(
        representative_fields=field_data(representative, "representative"),
        worst_fields=field_data(worst, "worst"),
        cases=tuple(item.point for item in cases),
        baselines=(
            BaselinePlotSeries(
                kind="selected_fno",
                artifact_id=request.selected_model_id,
                median_velocity_rel_l2=_valid_summary_value(
                    summaries["selected.velocity_rel_l2"], "median"
                ),
                median_cd_pct=_valid_summary_value(summaries["selected.cd_head_pct"], "median"),
            ),
            BaselinePlotSeries(
                kind="mean_field",
                artifact_id=request.config.baseline_ids[0],
                median_velocity_rel_l2=_valid_summary_value(
                    summaries["mean.velocity_rel_l2"], "median"
                ),
                median_cd_pct=_valid_summary_value(summaries["mean.cd_head_pct"], "median"),
            ),
            BaselinePlotSeries(
                kind="nearest_design",
                artifact_id=request.config.baseline_ids[1],
                median_velocity_rel_l2=_valid_summary_value(
                    summaries["nearest.velocity_rel_l2"], "median"
                ),
                median_cd_pct=_valid_summary_value(summaries["nearest.cd_head_pct"], "median"),
            ),
        ),
    )


def _probe_batch(
    probe: ProbeGeometry,
    *,
    reynolds: float | int,
    rotation: Any,
    torch: Any,
    device: str,
) -> PredictionBatch:
    dtype = torch.float32
    x = torch.arange(OUTPUT_GRID_NX, device=device, dtype=dtype)[None, :] - 0.30 * (
        OUTPUT_GRID_NX - 1
    )
    y = torch.arange(OUTPUT_GRID_NY, device=device, dtype=dtype)[:, None] - 0.50 * (
        OUTPUT_GRID_NY - 1
    )
    angle = rotation * (math.pi / 180.0)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    major = 0.5 * _REFERENCE_DIAMETER_LU * probe.scale
    minor = major * probe.aspect_ratio
    x_prime = cosine * x + sine * y
    y_prime = -sine * x + cosine * y
    sdf = (torch.sqrt((x_prime / major) ** 2 + (y_prime / minor) ** 2) - 1.0) * min(major, minor)
    sdf = sdf.to(dtype=dtype)
    reynolds_value = torch.as_tensor(float(reynolds), device=device, dtype=dtype)
    reynolds_normalized = 2.0 * (reynolds_value - 40.0) / 260.0 - 1.0
    inputs = torch.stack(
        (
            torch.clamp(sdf / _REFERENCE_DIAMETER_LU, -1.0, 1.0),
            torch.ones_like(sdf) * reynolds_normalized,
        ),
        dim=0,
    )[None, ...].contiguous()
    fluid = (sdf > 0.0)[None, None, ...].contiguous()
    design = torch.stack(
        (
            torch.as_tensor(4.0 * (probe.aspect_ratio - 0.5) - 1.0, device=device, dtype=dtype),
            2.0 * rotation / 30.0 - 1.0,
            torch.as_tensor(4.0 * (probe.scale - 0.75) - 1.0, device=device, dtype=dtype),
            reynolds_normalized,
        )
    )[None, ...].contiguous()
    return PredictionBatch(inputs=inputs, fluid_mask=fluid, design_params=design)


def _ood_evaluation(
    request: RemoteValidationRequest,
    *,
    rows: Sequence[Any],
    predictors: tuple[FnoPredictor, FnoPredictor, FnoPredictor],
    statistics: Any,
    torch: Any,
    meter: _GpuMeter,
) -> Any:
    probes = select_probe_geometries(rows, selection="ood")
    identities = tuple(
        ProbeModelIdentity(model_id=reference.artifact_id, model_sha256=reference.sha256)
        for reference in request.models
    )
    training_variance = (
        statistics.outputs.u_mean.standard_deviation**2,
        statistics.outputs.v_mean.standard_deviation**2,
        statistics.outputs.rho_delta.standard_deviation**2,
    )
    results = []
    for probe in probes:
        for reynolds in ALL_PROBE_REYNOLDS:
            rotation = torch.as_tensor(probe.rotation_deg, device="cuda:0", dtype=torch.float32)
            batch = _probe_batch(
                probe,
                reynolds=reynolds,
                rotation=rotation,
                torch=torch,
                device="cuda:0",
            )
            predictions = []
            for identity, predictor in zip(identities, predictors, strict=True):
                output = _predict(
                    predictor,
                    batch,
                    torch=torch,
                    precision=request.precision,
                    meter=meter,
                )
                normalized = np.ascontiguousarray(
                    output.fields_normalized.detach().cpu().numpy(), dtype=np.float32
                )
                physical = denormalize_fields(normalized, statistics)[0]
                predictions.append(
                    EnsembleFieldPrediction(
                        model=identity,
                        fields=np.ascontiguousarray(physical, dtype=np.float32),
                        fluid_mask=np.ascontiguousarray(
                            cast(Any, batch.fluid_mask)[0, 0].detach().cpu().numpy(),
                            dtype=np.bool_,
                        ),
                    )
                )
            results.append(
                evaluate_ensemble_variance(
                    probe,
                    reynolds=reynolds,
                    predictions=predictions,
                    training_output_variance=training_variance,
                )
            )
    return summarize_ood_evaluation(results)


@dataclass(frozen=True, slots=True)
class _RotationAdapter:
    predictor: FnoPredictor
    identity: ProbeModelIdentity
    torch: Any
    precision: Literal["bf16", "fp16"]
    meter: _GpuMeter

    @property
    def model(self) -> ProbeModelIdentity:
        return self.identity

    def _direct(self, probe: ProbeGeometry, rotation: Any) -> Any:
        batch = _probe_batch(
            probe,
            reynolds=probe.source_reynolds,
            rotation=rotation,
            torch=self.torch,
            device="cuda:0",
        )

        def run() -> Any:
            dtype = self.torch.bfloat16 if self.precision == "bf16" else self.torch.float16
            with (
                self.torch.inference_mode(),
                self.torch.autocast(device_type="cuda", dtype=dtype, enabled=True),
            ):
                return cast(Any, self.predictor.forward(batch).cd_head)[0]

        return self.meter.measure(run)

    def cd_and_rotation_gradient(self, probe: ProbeGeometry) -> AutogradCdResult:
        rotation = self.torch.tensor(
            probe.rotation_deg,
            device="cuda:0",
            dtype=self.torch.float32,
            requires_grad=True,
        )

        def run() -> tuple[Any, Any]:
            batch = _probe_batch(
                probe,
                reynolds=probe.source_reynolds,
                rotation=rotation,
                torch=self.torch,
                device="cuda:0",
            )
            dtype = self.torch.bfloat16 if self.precision == "bf16" else self.torch.float16
            with (
                self.torch.enable_grad(),
                self.torch.autocast(device_type="cuda", dtype=dtype, enabled=True),
            ):
                cd = cast(Any, self.predictor.forward(batch).cd_head)[0]
                gradient = self.torch.autograd.grad(
                    cd,
                    rotation,
                    retain_graph=False,
                    create_graph=False,
                )[0]
            return cd, gradient

        cd, gradient = cast(tuple[Any, Any], self.meter.measure(run))
        return AutogradCdResult(
            cd_head=float(cd.detach().cpu().item()),
            rotation_gradient_cd_per_degree=float(gradient.detach().cpu().item()),
        )

    def cd_at_rotation(self, probe: ProbeGeometry, rotation_deg: float) -> float:
        rotation = self.torch.tensor(rotation_deg, device="cuda:0", dtype=self.torch.float32)
        cd = self._direct(probe, rotation)
        return float(cd.detach().cpu().item())


def _gate_evidence(
    summaries: Mapping[str, MetricSummary],
    cases: Mapping[str, tuple[CaseMetrics, ...]],
    ood: Any,
    sensitivity: Any,
) -> tuple[GateEvidence, ...]:
    selected_velocity = _valid_summary_value(summaries["selected.velocity_rel_l2"], "median")
    selected_cd = _valid_summary_value(summaries["selected.cd_head_pct"], "median")
    obstacle_p95 = _valid_summary_value(summaries["selected.obstacle_ratio"], "p95")
    mean_velocity = _valid_summary_value(summaries["mean.velocity_rel_l2"], "median")
    nearest_velocity = _valid_summary_value(summaries["nearest.velocity_rel_l2"], "median")
    mean_cd = _valid_summary_value(summaries["mean.cd_head_pct"], "median")
    nearest_cd = _valid_summary_value(summaries["nearest.cd_head_pct"], "median")
    return (
        GateEvidence(name="field_error", value=selected_velocity, evidence=("test:200",)),
        GateEvidence(name="cd_head_error", value=selected_cd, evidence=("test:200",)),
        head_field_consistency_gate_evidence(
            {case.case_id: case.head_field_gap_pct for case in cases["selected"]}
        ),
        divergence_gate_evidence(
            summaries["selected.prediction_div_mean_abs"],
            summaries["selected.solver_div_mean_abs"],
        ),
        GateEvidence(name="obstacle_compliance", value=obstacle_p95, evidence=("test:200",)),
        GateEvidence(
            name="mean_baseline_field",
            value=selected_velocity,
            comparison_threshold=mean_velocity,
            evidence=("same-test-membership:200",),
        ),
        GateEvidence(
            name="nearest_baseline_field",
            value=selected_velocity,
            comparison_threshold=nearest_velocity,
            evidence=("same-test-membership:200",),
        ),
        GateEvidence(
            name="mean_baseline_cd",
            value=selected_cd,
            comparison_threshold=mean_cd,
            evidence=("same-test-membership:200",),
        ),
        GateEvidence(
            name="nearest_baseline_cd",
            value=selected_cd,
            comparison_threshold=nearest_cd,
            evidence=("same-test-membership:200",),
        ),
        ood_gate_evidence(ood),
        sensitivity_gate_evidence(sensitivity),
        GateEvidence(
            name="evidence_integrity",
            value=True,
            evidence=("all-parent-digests-and-fixed-memberships-verified",),
        ),
    )


def _publish_report(root: Path, report: ValidationReport) -> ArtifactRef:
    directory = root / "validation" / report.report_id
    if directory.is_symlink():
        raise ArtifactIntegrityError(
            "REMOTE_VALIDATE_STORE_UNSAFE: report target must not be a symbolic link"
        )
    report_path = directory / "report.json"
    artifacts = render_validation_artifacts(report, plot_directory="report.plots")
    if (report_path.exists() or report_path.is_symlink()) and (
        load_validation_report(report_path) != report
    ):
        raise ArtifactIntegrityError(
            "REMOTE_VALIDATE_STORE_IDENTITY: immutable report ID already differs"
        )
    write_validation_artifacts(report_path, artifacts)
    if load_validation_report(report_path) != report:
        raise ArtifactIntegrityError("REMOTE_VALIDATE_STORE_IDENTITY: published report differs")
    stale = check_validation_artifacts(report_path, artifacts)
    if stale:
        raise ArtifactIntegrityError(
            "REMOTE_VALIDATE_STORE_IDENTITY: published report companions differ"
        )
    expected_files = {
        report_path,
        directory / "report.md",
        directory / "report.plots.json",
        *(directory / "report.plots" / name for name in artifacts.plots),
    }
    observed_files = {path for path in directory.rglob("*") if path.is_file()}
    if observed_files != expected_files or any(path.is_symlink() for path in directory.rglob("*")):
        raise ArtifactIntegrityError("REMOTE_VALIDATE_STORE_LAYOUT: report artifact members differ")
    size = sum(path.stat().st_size for path in expected_files)
    return ArtifactRef(
        artifact_type="validation_report",
        artifact_id=report.report_id,
        sha256=report.report_sha256,
        size_bytes=size,
        uri=f"validation/{report.report_id}",
    )


def execute_validation(
    request: RemoteValidationRequest,
    *,
    root: Path,
    build: RuntimeBuildManifest,
    reload_volume: Callable[[], None],
    commit_volume: Callable[[], None],
) -> ValidationReceipt:
    """Evaluate exact parents once and publish a report even when gates are red."""

    if not isinstance(request, RemoteValidationRequest):
        raise TypeError("request must be a RemoteValidationRequest")
    assert_request_matches_build(request, build)
    started_at = utc_now()
    reload_volume()
    dataset = open_manifest_dataset(root, request.dataset)
    statistics = fit_preprocessing_statistics(dataset.iter_samples("train"))
    model_store = LocalModelBundleStore(root)
    bundles = tuple(model_store.open(reference) for reference in request.models)
    if any(
        bundle.metadata.dataset_sha256 != request.dataset.sha256
        or bundle.metadata.code_revision != request.model_source_revision
        or bundle.metadata.lock_digest != request.lock_sha256
        or bundle.preprocessing != statistics
        for bundle in bundles
    ):
        raise ArtifactIntegrityError(
            "REMOTE_VALIDATE_MODEL_PARENT_MISMATCH: bundle lineage differs"
        )
    experiment_ids = {bundle.metadata.experiment_id for bundle in bundles}
    model_source_revisions = {bundle.metadata.code_revision for bundle in bundles}
    if (
        len(experiment_ids) != 1
        or model_source_revisions != {request.model_source_revision}
        or len({bundle.metadata.seed for bundle in bundles}) != 3
    ):
        raise ArtifactIntegrityError(
            "REMOTE_VALIDATE_MODEL_PARENT_MISMATCH: ensemble training identities differ"
        )
    baselines = fit_baselines(dataset, statistics)
    observed_baseline_sha256s = tuple(item.metadata.baseline_sha256 for item in baselines)
    if observed_baseline_sha256s != request.baseline_sha256s:
        raise ArtifactIntegrityError(
            "REMOTE_VALIDATE_BASELINE_MISMATCH: fitted baseline identities differ"
        )
    solver_sha256 = canonical_sha256(list(dataset.parent_run_sha256))
    if solver_sha256 != request.solver_sha256:
        raise ArtifactIntegrityError(
            "REMOTE_VALIDATE_SOLVER_MISMATCH: dataset solver lineage differs"
        )

    torch = importlib.import_module("torch")
    device_name = str(torch.cuda.get_device_name(0))
    if request.requested_device_class.casefold() not in device_name.casefold().replace(" ", ""):
        raise ArtifactIntegrityError(
            "REMOTE_VALIDATE_DEVICE_MISMATCH: allocated GPU differs from the request"
        )
    torch.cuda.reset_peak_memory_stats(0)
    meter = _GpuMeter(torch)
    predictors = cast(
        tuple[FnoPredictor, FnoPredictor, FnoPredictor],
        tuple(instantiate_bundle_predictor(bundle, device="cuda:0") for bundle in bundles),
    )
    for predictor in predictors:
        predictor.eval()
    selected_index = request.config.ensemble_model_ids.index(request.selected_model_id)
    cases, selected_cases = _evaluate_test_set(
        request,
        dataset=dataset,
        statistics=statistics,
        selected=predictors[selected_index],
        baselines=baselines,
        torch=torch,
        meter=meter,
    )
    summaries = _summaries(request, cases)
    ood = _ood_evaluation(
        request,
        rows=dataset.split_rows("test"),
        predictors=predictors,
        statistics=statistics,
        torch=torch,
        meter=meter,
    )
    sensitivity_probes = select_probe_geometries(
        dataset.split_rows("test"), selection="sensitivity"
    )
    selected_bundle = bundles[selected_index]
    sensitivity = evaluate_sensitivity_probes(
        sensitivity_probes,
        _RotationAdapter(
            predictor=predictors[selected_index],
            identity=ProbeModelIdentity(
                model_id=selected_bundle.metadata.model_id,
                model_sha256=selected_bundle.metadata.model_sha256,
            ),
            torch=torch,
            precision=request.precision,
            meter=meter,
        ),
    )
    gates = evaluate_required_gates(_gate_evidence(summaries, cases, ood, sensitivity))
    completed_at = utc_now()
    report = ValidationReport.create(
        dataset_id=request.dataset.artifact_id,
        selected_model_id=request.selected_model_id,
        ensemble_model_ids=request.config.ensemble_model_ids,
        baseline_ids=request.config.baseline_ids,
        metrics=summaries,
        gates=gates,
        overall_status=overall_gate_status(gates),
        provenance=Provenance(
            source_revision=request.source_revision,
            source_dirty=False,
            python_version=build.python_version,
            lock_sha256=request.lock_sha256,
            packages=build.packages,
            os=platform.system().casefold(),
            architecture=platform.machine(),
            device_class=request.requested_device_class,
            dtype_policy=f"{request.precision}-inference-fields-fp32-metrics-fp64",
            config_sha256=request.config.config_digest,
            parent_sha256=request.expected_parent_sha256,
            seeds=cast(tuple[int, int, int], tuple(bundle.metadata.seed for bundle in bundles)),
            deterministic=True,
            started_at=started_at,
            completed_at=completed_at,
            gpu_seconds=meter.seconds,
        ),
        ood=ood,
        sensitivity=sensitivity,
        plot_data=_plot_data(request, cases=selected_cases, summaries=summaries),
    )
    assert_report_matches_request(report, request)
    reference = _publish_report(root, report)
    commit_volume()
    reload_volume()
    accounting_completed_at = utc_now()
    allocated = int(torch.cuda.max_memory_allocated(0))
    reserved = int(torch.cuda.max_memory_reserved(0))
    accounting = ExecutionAccounting(
        started_at=started_at,
        completed_at=accounting_completed_at,
        wall_seconds=(accounting_completed_at - started_at).total_seconds(),
        gpu_seconds=meter.seconds,
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
        device_class=request.requested_device_class,
        device_name=device_name,
        precision=request.precision,
        source_revision=request.source_revision,
        lock_sha256=request.lock_sha256,
    )
    return ValidationReceipt.create(
        report=reference,
        overall_status=report.overall_status,
        accounting=accounting,
        parent_sha256=request.expected_parent_sha256,
    )


__all__ = ["execute_validation"]
