from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from soufflerie.artifacts import (
    ArtifactEnvelope,
    LineageNode,
    LineagePolicy,
    ParentLink,
    ParentTypeRule,
    SourceState,
    artifact_content_sha256,
    capture_provenance,
    validate_release_provenance,
    verify_consumer_identities,
    verify_lineage,
)
from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import Provenance, sha256_bytes

NOW = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


def _digest(label: str) -> str:
    return sha256_bytes(label.encode())


def _provenance(
    *,
    parents: dict[str, str] | None = None,
    dirty: bool = False,
    deterministic: bool = True,
    started_at: datetime = NOW,
) -> Provenance:
    return Provenance(
        source_revision="a" * 40,
        source_dirty=dirty,
        python_version="3.11.14",
        lock_sha256="b" * 64,
        packages={"numpy": "2.2.6", "soufflerie": "0.1.0"},
        os="linux",
        architecture="x86_64",
        device_class="cpu",
        dtype_policy="solver-fp32-metrics-fp64",
        config_sha256="c" * 64,
        parent_sha256=parents or {},
        seeds=(7,),
        deterministic=deterministic,
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=2),
        gpu_seconds=0.0,
    )


def _node(
    artifact_type: str,
    label: str,
    *parents: tuple[str, str, str],
) -> LineageNode:
    digest = _digest(label)
    return LineageNode(
        artifact_type=artifact_type,
        artifact_id=digest[:20],
        sha256=digest,
        parents=tuple(
            ParentLink(role=role, artifact_type=parent_type, sha256=parent_digest)
            for role, parent_type, parent_digest in parents
        ),
    )


def _valid_graph() -> tuple[LineageNode, LineageNode, LineageNode, LineageNode]:
    run = _node("run", "run")
    dataset = _node("dataset", "dataset", ("run", "run", run.sha256))
    model = _node("model", "model", ("dataset", "dataset", dataset.sha256))
    report = _node(
        "report",
        "report",
        ("dataset", "dataset", dataset.sha256),
        ("model", "model", model.sha256),
    )
    return run, dataset, model, report


def test_capture_and_release_validation_bind_reviewed_runtime_state(tmp_path: Path) -> None:
    lock = b"version = 1\n"
    (tmp_path / "uv.lock").write_bytes(lock)
    provenance = capture_provenance(
        source_root=tmp_path,
        config={"reynolds": 100.0, "seed": 7},
        parent_sha256={"dataset": _digest("dataset")},
        package_names=("numpy", "soufflerie"),
        package_versions={"numpy": "2.2.6", "soufflerie": "0.1.0"},
        source_state=SourceState(source_revision="a" * 40, source_dirty=False),
        device_class="cpu",
        dtype_policy="solver-fp32-metrics-fp64",
        seeds=(7,),
        deterministic=True,
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=2),
        gpu_seconds=0.0,
    )
    assert provenance.lock_sha256 == sha256_bytes(lock)
    assert provenance.config_sha256 == _digest('{"reynolds":100.0,"seed":7}')
    validate_release_provenance(
        provenance,
        expected_source_revision="a" * 40,
        expected_lock_sha256=sha256_bytes(lock),
        expected_config_sha256=_digest('{"reynolds":100.0,"seed":7}'),
        expected_packages={"numpy": "2.2.6", "soufflerie": "0.1.0"},
        required_parent_sha256={"dataset": _digest("dataset")},
    )

    for changed, message in (
        (provenance.model_copy(update={"source_dirty": True}), "clean source"),
        (provenance.model_copy(update={"deterministic": False}), "deterministic"),
        (
            provenance.model_copy(update={"parent_sha256": {}}),
            "parent evidence",
        ),
    ):
        with pytest.raises(ArtifactIntegrityError, match=message):
            validate_release_provenance(
                changed,
                expected_source_revision="a" * 40,
                expected_lock_sha256=sha256_bytes(lock),
                expected_config_sha256=_digest('{"reynolds":100.0,"seed":7}'),
                expected_packages={"numpy": "2.2.6", "soufflerie": "0.1.0"},
                required_parent_sha256={"dataset": _digest("dataset")},
            )
    with pytest.raises(ArtifactIntegrityError, match="packages"):
        validate_release_provenance(
            provenance,
            expected_source_revision="a" * 40,
            expected_lock_sha256=sha256_bytes(lock),
            expected_config_sha256=_digest('{"reynolds":100.0,"seed":7}'),
            expected_packages={"numpy": "9.9.9"},
            required_parent_sha256={"dataset": _digest("dataset")},
        )
    for field, value, message in (
        ("expected_source_revision", "d" * 40, "source revision"),
        ("expected_lock_sha256", "d" * 64, "lock digest"),
        ("expected_config_sha256", "d" * 64, "config digest"),
    ):
        expectations = {
            "expected_source_revision": "a" * 40,
            "expected_lock_sha256": sha256_bytes(lock),
            "expected_config_sha256": _digest('{"reynolds":100.0,"seed":7}'),
        }
        expectations[field] = value
        with pytest.raises(ArtifactIntegrityError, match=message):
            validate_release_provenance(
                provenance,
                expected_source_revision=expectations["expected_source_revision"],
                expected_lock_sha256=expectations["expected_lock_sha256"],
                expected_config_sha256=expectations["expected_config_sha256"],
                expected_packages={"numpy": "2.2.6", "soufflerie": "0.1.0"},
                required_parent_sha256={"dataset": _digest("dataset")},
            )
    with pytest.raises(ValidationError):
        Provenance.model_validate({**provenance.model_dump(), "source_revision": ""})


def test_capture_rejects_implicit_or_incomplete_package_evidence(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    def capture(
        package_names: tuple[str, ...],
        package_versions: dict[str, str] | None = None,
    ) -> Provenance:
        return capture_provenance(
            source_root=tmp_path,
            config={"seed": 7},
            parent_sha256={},
            package_names=package_names,
            package_versions=package_versions,
            device_class="cpu",
            dtype_policy="fp32",
            seeds=(7,),
            deterministic=True,
            started_at=NOW,
            completed_at=NOW,
            gpu_seconds=0.0,
            source_state=SourceState(source_revision="a" * 40, source_dirty=False),
        )

    with pytest.raises(ArtifactIntegrityError, match="allowlist"):
        capture(())
    with pytest.raises(ArtifactIntegrityError, match="exactly match"):
        capture(("numpy", "soufflerie"), {"numpy": "2.2.6"})


@given(
    second_offset=st.integers(min_value=0, max_value=86_400),
    uri=st.sampled_from(("reports/a.json", "relocated/a.json", "archive/2026/a.json")),
)
def test_content_identity_excludes_location_and_wall_clock_time(
    second_offset: int,
    uri: str,
) -> None:
    provenance = _provenance(started_at=NOW + timedelta(seconds=second_offset))
    members = {"report.json": _digest("member")}
    digest = artifact_content_sha256(
        artifact_type="report",
        logical_metadata={"gate": "green", "score": 0.25},
        member_sha256=members,
        provenance=provenance,
    )
    envelope = ArtifactEnvelope(
        artifact_type="report",
        artifact_id=digest[:20],
        sha256=digest,
        logical_metadata={"gate": "green", "score": 0.25},
        member_sha256=members,
        provenance=provenance,
        created_at=NOW + timedelta(days=second_offset),
        uri=uri,
    )
    assert envelope.sha256 == artifact_content_sha256(
        artifact_type="report",
        logical_metadata={"gate": "green", "score": 0.25},
        member_sha256=members,
        provenance=_provenance(started_at=NOW),
    )


def test_content_identity_rejects_mutable_metadata_and_detects_tampering() -> None:
    provenance = _provenance()
    with pytest.raises(ArtifactIntegrityError, match="mutable field"):
        artifact_content_sha256(
            artifact_type="report",
            logical_metadata={"output": {"path": "renamed/report.json"}},
            member_sha256={"report.json": _digest("member")},
            provenance=provenance,
        )
    digest = artifact_content_sha256(
        artifact_type="report",
        logical_metadata={"gate": "green"},
        member_sha256={"report.json": _digest("member")},
        provenance=provenance,
    )
    with pytest.raises(ValidationError, match="does not match"):
        ArtifactEnvelope(
            artifact_type="report",
            artifact_id=digest[:20],
            sha256=digest,
            logical_metadata={"gate": "red"},
            member_sha256={"report.json": _digest("member")},
            provenance=provenance,
            created_at=NOW,
            uri="reports/report.json",
        )


def test_lineage_and_consumer_identity_accept_one_exact_dag() -> None:
    graph = _valid_graph()
    ordered = verify_lineage(reversed(graph))
    assert [node.artifact_type for node in ordered] == ["run", "dataset", "model", "report"]
    run, dataset, model, report = graph
    accepted = verify_consumer_identities(
        (report, run, model, dataset),
        dataset_id=dataset.artifact_id,
        model_id=model.artifact_id,
        report_id=report.artifact_id,
    )
    assert accepted.dataset == dataset
    assert accepted.model == model
    assert accepted.report == report


def test_lineage_rejects_missing_wrong_and_disallowed_parents() -> None:
    run, dataset, model, report = _valid_graph()
    with pytest.raises(ArtifactIntegrityError, match="is missing"):
        verify_lineage((dataset, model, report))

    wrong_type = report.model_copy(
        update={
            "parents": (
                ParentLink(role="dataset", artifact_type="run", sha256=dataset.sha256),
                ParentLink(role="model", artifact_type="model", sha256=model.sha256),
            )
        }
    )
    with pytest.raises(ArtifactIntegrityError, match="type mismatch"):
        verify_lineage((run, dataset, model, wrong_type))

    no_model = report.model_copy(
        update={
            "parents": (
                ParentLink(role="dataset", artifact_type="dataset", sha256=dataset.sha256),
            )
        }
    )
    with pytest.raises(ArtifactIntegrityError, match="required parent types: model"):
        verify_lineage((run, dataset, model, no_model))

    forbidden = model.model_copy(
        update={
            "parents": (
                ParentLink(role="run", artifact_type="run", sha256=run.sha256),
            )
        }
    )
    with pytest.raises(ArtifactIntegrityError, match="cannot depend"):
        verify_lineage((run, dataset, forbidden))


def test_lineage_rejects_cycles_and_swapped_consumer_identities() -> None:
    first = _node("run", "first")
    second = _node("run", "second")
    first = first.model_copy(
        update={
            "parents": (
                ParentLink(role="previous", artifact_type="run", sha256=second.sha256),
            )
        }
    )
    second = second.model_copy(
        update={
            "parents": (
                ParentLink(role="previous", artifact_type="run", sha256=first.sha256),
            )
        }
    )
    cyclic_policy = LineagePolicy(
        rules={"run": ParentTypeRule(required=("run",), allowed=("run",))}
    )
    with pytest.raises(ArtifactIntegrityError, match="cycle"):
        verify_lineage((first, second), policy=cyclic_policy)

    run, dataset, model, report = _valid_graph()
    with pytest.raises(ArtifactIntegrityError, match="consumer dataset identity has type"):
        verify_consumer_identities(
            (run, dataset, model, report),
            dataset_id=model.artifact_id,
            model_id=dataset.artifact_id,
            report_id=report.artifact_id,
        )


def test_envelope_builds_typed_parent_links_only_from_exact_roles() -> None:
    parent_digest = _digest("dataset")
    provenance = _provenance(parents={"dataset": parent_digest})
    members = {"model.safetensors": _digest("weights")}
    digest = artifact_content_sha256(
        artifact_type="model",
        logical_metadata={"architecture": "fno"},
        member_sha256=members,
        provenance=provenance,
    )
    envelope = ArtifactEnvelope(
        artifact_type="model",
        artifact_id=digest[:20],
        sha256=digest,
        logical_metadata={"architecture": "fno"},
        member_sha256=members,
        provenance=provenance,
        created_at=NOW,
        uri="models/model.json",
    )
    node = envelope.lineage_node(parent_types={"dataset": "dataset"})
    assert node.parents[0].sha256 == parent_digest
    with pytest.raises(ArtifactIntegrityError, match="exactly match"):
        envelope.lineage_node(parent_types={})
