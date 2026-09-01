from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import infra.solve_execution as solve_execution
from infra.remote_execution import (
    MAX_REMOTE_INPUT_BYTES,
    RemoteSolveRequest,
    RemoteSweepRequest,
    SolveSummary,
    SweepSummary,
    encode_remote_model,
    load_remote_request,
    parse_remote_model,
    publish_remote_request,
    smoke_design,
)
from infra.runtime_manifest import RuntimeBuildManifest
from infra.solve_execution import execute_solve_request, run_solver_case
from infra.sweep_execution import orchestrate_smoke_sweep
from soufflerie.config import SweepConfig, load_config
from soufflerie.errors import ArtifactIntegrityError, ConfigurationError, RemoteExecutionError
from soufflerie.geometry import ellipse_sdf, obstacle_mask
from soufflerie.schemas import ArtifactRef, CaseConfig, FlowFields, SolverResult

ROOT = Path(__file__).parents[2]


def _build() -> RuntimeBuildManifest:
    return RuntimeBuildManifest.create(
        lock_sha256="b" * 64,
        source_revision="a" * 40,
        source_dirty=False,
        packages={"numpy": "2.2.6", "soufflerie": "0.1.0"},
    )


def _sweep_request(*, force_failure_once: bool = True) -> RemoteSweepRequest:
    config = load_config(ROOT / "configs/sweeps/mvp-v1.yaml", SweepConfig)
    build = _build()
    return RemoteSweepRequest.create(
        config=config,
        requested_device_class="L40S",
        source_revision=build.source_revision,
        lock_sha256=build.lock_sha256,
        force_failure_once=force_failure_once,
    )


def _result_for(case: CaseConfig, template: SolverResult) -> SolverResult:
    sdf = ellipse_sdf(case.shape, case.grid)
    fields = FlowFields(
        u=template.fields.u,
        v=template.fields.v,
        rho=template.fields.rho,
        sdf=sdf,
        obstacle_mask=obstacle_mask(sdf),
    )
    provenance = template.provenance.model_copy(
        update={
            "source_revision": "a" * 40,
            "lock_sha256": "b" * 64,
            "config_sha256": case.sha256,
            "seeds": (case.seed,),
        }
    )
    return SolverResult(
        case_id=case.case_id,
        fields=fields,
        cd=template.cd,
        cl_mean=template.cl_mean,
        strouhal=template.strouhal,
        force_steps=template.force_steps,
        cd_history=template.cd_history,
        cl_history=template.cl_history,
        diagnostics=template.diagnostics,
        provenance=provenance,
    )


def test_remote_models_require_bounded_exact_canonical_json() -> None:
    request = _sweep_request()
    encoded = encode_remote_model(request)
    assert parse_remote_model(encoded, RemoteSweepRequest) == request
    assert len(encoded) < MAX_REMOTE_INPUT_BYTES

    with pytest.raises(ConfigurationError, match="canonical JSON encoding"):
        parse_remote_model(encoded + b"\n", RemoteSweepRequest)
    with pytest.raises(ConfigurationError, match=r"1\.\.16384"):
        parse_remote_model(b"x" * (MAX_REMOTE_INPUT_BYTES + 1), RemoteSweepRequest)
    with pytest.raises(ConfigurationError, match="does not match"):
        parse_remote_model(b'{"schema_version":1,"schema_version":1}', RemoteSweepRequest)

    tampered = request.model_dump(mode="python")
    tampered["source_revision"] = "c" * 40
    with pytest.raises(ValidationError, match="request_digest"):
        RemoteSweepRequest.model_validate(tampered)


def test_attempt_and_forced_failure_do_not_change_logical_solve_identity() -> None:
    sweep = _sweep_request()
    point = smoke_design(sweep)[0]
    first = RemoteSolveRequest.create(
        operation_kind="smoke-sweep",
        sweep_digest=sweep.request_digest,
        design_id=point.design_id,
        split=point.split,
        case=point.case,
        requested_device_class=sweep.requested_device_class,
        source_revision=sweep.source_revision,
        lock_sha256=sweep.lock_sha256,
        attempt_id="s1-case",
        force_retryable_failure=True,
    )
    second = RemoteSolveRequest.create(
        operation_kind="smoke-sweep",
        sweep_digest=sweep.request_digest,
        design_id=point.design_id,
        split=point.split,
        case=point.case,
        requested_device_class=sweep.requested_device_class,
        source_revision=sweep.source_revision,
        lock_sha256=sweep.lock_sha256,
        attempt_id="s2-case",
        force_retryable_failure=False,
    )
    assert first.request_digest == second.request_digest
    assert encode_remote_model(first) != encode_remote_model(second)


def test_smoke_design_is_stratified_distinct_and_does_not_touch_global_rng() -> None:
    request = _sweep_request()
    before = np.random.get_state()
    points = smoke_design(request)
    after = np.random.get_state()

    assert len(points) == 8
    assert len({point.design_id for point in points}) == 8
    assert len({point.case.case_id for point in points}) == 8
    assert [point.split for point in points].count("train") == 5
    assert [point.split for point in points].count("validation") == 2
    assert [point.split for point in points].count("test") == 1
    assert request.request_digest != request.config.config_digest
    for left, right in zip(before, after, strict=True):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right

    ranges_and_values = (
        (request.config.aspect_ratio, [point.case.shape.aspect_ratio for point in points]),
        (request.config.rotation_deg, [point.case.shape.rotation_deg for point in points]),
        (request.config.scale, [point.case.shape.scale for point in points]),
        (request.config.reynolds, [point.case.reynolds for point in points]),
    )
    for bounds, values in ranges_and_values:
        strata = {
            int((value - bounds.minimum) / (bounds.maximum - bounds.minimum) * 8)
            for value in values
        }
        assert strata == set(range(8))


def test_remote_request_publication_is_content_addressed_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    content = encode_remote_model(_sweep_request())

    def reject_hard_link(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("Modal Volumes do not support hard links")

    monkeypatch.setattr("infra.remote_execution.os.link", reject_hard_link)
    first = publish_remote_request(root, content)
    second = publish_remote_request(root, content)
    assert first == second
    assert load_remote_request(root, first) == _sweep_request()

    (root / first.uri).write_bytes(content + b"\n")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        load_remote_request(root, first)


def test_forced_failure_resumes_only_missing_case_and_preserves_success_digests(
    tmp_path: Path,
    solver_result: SolverResult,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    sweep_request = _sweep_request(force_failure_once=True)
    request_ref = publish_remote_request(root, encode_remote_model(sweep_request))
    build = _build()
    submitted_rounds: list[tuple[str, ...]] = []
    sync_calls: list[str] = []
    first_round_digests: dict[str, str] = {}

    def submit(payloads: tuple[bytes, ...], owners: tuple[str, ...]) -> None:
        case_ids: list[str] = []
        for payload, owner in zip(payloads, owners, strict=True):
            request = parse_remote_model(payload, RemoteSolveRequest)
            case_id = request.case.case_id
            case_ids.append(case_id)
            try:
                reference = execute_solve_request(
                    payload,
                    owner,
                    root=root,
                    build=build,
                    reload_volume=lambda: sync_calls.append("worker.reload"),
                    commit_volume=lambda: sync_calls.append("worker.commit"),
                    solve_case=lambda solve_request, _build: _result_for(
                        solve_request.case,
                        solver_result,
                    ),
                )
            except RemoteExecutionError as error:
                assert str(error) == "forced smoke preemption before numerical execution"
                continue
            if not submitted_rounds:
                first_round_digests[case_id] = reference.sha256
        submitted_rounds.append(tuple(case_ids))

    first = orchestrate_smoke_sweep(
        request_ref,
        root=root,
        build=build,
        reload_volume=lambda: sync_calls.append("reload"),
        commit_volume=lambda: sync_calls.append("commit"),
        submit=submit,
    )
    assert first.final_state == "succeeded"
    assert first.succeeded_count == 8
    assert first.attempt_count == 9
    assert first.retry_count == 1
    assert [len(round_ids) for round_ids in submitted_rounds] == [8, 1]
    assert submitted_rounds[1] == (smoke_design(sweep_request)[0].case.case_id,)
    assert first.resumed_case_ids == submitted_rounds[1]
    final_digests = {
        reference.uri.split("/")[1]: reference.sha256 for reference in first.run_references
    }
    assert final_digests | first_round_digests == final_digests
    assert all(final_digests[case_id] == digest for case_id, digest in first_round_digests.items())
    assert "commit" in sync_calls and "reload" in sync_calls

    submitted_rounds.clear()
    second = orchestrate_smoke_sweep(
        request_ref,
        root=root,
        build=build,
        reload_volume=lambda: None,
        commit_volume=lambda: None,
        submit=submit,
    )
    assert submitted_rounds == []
    assert second.attempt_count == 0
    assert second.skipped_case_ids == tuple(sorted(final_digests))
    assert {
        reference.uri.split("/")[1]: reference.sha256 for reference in second.run_references
    } == final_digests
    assert SweepSummary.model_validate_json(second.model_dump_json()) == second


def test_build_or_device_changes_create_a_distinct_smoke_identity() -> None:
    first = _sweep_request()
    second = RemoteSweepRequest.create(
        config=first.config,
        requested_device_class="A10G",
        source_revision=first.source_revision,
        lock_sha256=first.lock_sha256,
    )
    third = RemoteSweepRequest.create(
        config=first.config,
        requested_device_class="L40S",
        source_revision="c" * 40,
        lock_sha256=first.lock_sha256,
    )
    assert len({first.request_digest, second.request_digest, third.request_digest}) == 3


def test_cuda_solver_assembly_binds_fields_histories_and_runtime_provenance(
    monkeypatch: pytest.MonkeyPatch,
    solver_result: SolverResult,
) -> None:
    sweep = _sweep_request(force_failure_once=False)
    point = smoke_design(sweep)[0]
    request = RemoteSolveRequest.create(
        operation_kind="smoke-sweep",
        sweep_digest=sweep.request_digest,
        design_id=point.design_id,
        split=point.split,
        case=point.case,
        requested_device_class=sweep.requested_device_class,
        source_revision=sweep.source_revision,
        lock_sha256=sweep.lock_sha256,
        attempt_id="s1-assembly",
    )

    class FakeStepper:
        def __init__(self, device: str, *, initial_seed: int) -> None:
            assert device == "cuda:0"
            assert initial_seed == point.case.seed

        def force_history(self) -> object:
            return SimpleNamespace(
                steps=solver_result.force_steps,
                cd=solver_result.cd_history,
                cl=solver_result.cl_history,
                count=solver_result.force_steps.size,
            )

    monkeypatch.setattr(
        "warp.get_device",
        lambda _device: SimpleNamespace(is_cuda=True, name="NVIDIA L40S"),
    )
    monkeypatch.setattr(solve_execution, "WarpObstacleStepper", FakeStepper)
    monkeypatch.setattr(
        solve_execution,
        "run_lifecycle",
        lambda _derived, _mask, *, stepper: SimpleNamespace(
            mean_fields=SimpleNamespace(
                u=solver_result.fields.u,
                v=solver_result.fields.v,
                rho=solver_result.fields.rho,
            ),
            diagnostics=solver_result.diagnostics,
        ),
    )
    monkeypatch.setattr(
        solve_execution,
        "mean_force_coefficients",
        lambda _history: SimpleNamespace(cd=1.25, cl=-0.02),
    )
    monkeypatch.setattr(
        solve_execution,
        "estimate_strouhal",
        lambda _history, _derived: SimpleNamespace(strouhal=0.18),
    )

    result = run_solver_case(request, _build())
    assert result.case_id == point.case.case_id
    assert result.fields.shape == (640, 512)
    assert result.cd == 1.25
    assert result.cl_mean == -0.02
    assert result.strouhal == 0.18
    assert result.provenance.source_revision == request.source_revision
    assert result.provenance.config_sha256 == point.case.sha256
    assert result.provenance.device_class == "L40S"
    assert result.provenance.dtype_policy == "fp32-lbm-fp64-reduction"

    reference = ArtifactRef(
        artifact_type="run",
        artifact_id="d" * 20,
        sha256="d" * 64,
        size_bytes=1,
        uri=f"runs/{point.case.case_id}/{'d' * 64}",
    )
    receipt = SolveSummary.create(
        artifact=reference,
        case_id=point.case.case_id,
        source_revision=request.source_revision,
        device_class="L40S",
        wall_seconds=0.5,
        gpu_seconds=0.5,
        final_state="succeeded",
    )
    assert SolveSummary.model_validate_json(receipt.model_dump_json()) == receipt
