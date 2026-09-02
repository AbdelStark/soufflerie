from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soufflerie.cli as cli
from soufflerie.cli import (
    CommandResult,
    DatasetValidationResult,
    ModelInspectionResult,
    VersionResult,
    app,
    cli_schema_documents,
)


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def solve(self, config: Path, output: Path, *, device: str) -> CommandResult:
        self.calls.append(("solve", config, output, device))
        return CommandResult(message=f"wrote solver result to {output}")

    def validate_dataset(self, manifest: Path) -> DatasetValidationResult:
        self.calls.append(("dataset", manifest))
        return DatasetValidationResult(manifest=str(manifest), dataset_id="d" * 20, case_count=1000)

    def inspect_model(self, bundle: Path) -> ModelInspectionResult:
        self.calls.append(("model", bundle))
        return ModelInspectionResult(
            bundle=str(bundle),
            model_id="model-abc",
            dataset_id="dataset-abc",
            architecture="fno-v1",
        )

    def validate(self, config: Path, output_dir: Path, *, device: str) -> CommandResult:
        self.calls.append(("validate", config, output_dir, device))
        return CommandResult(message=f"validation green at {output_dir}")

    def demo(self, bundle: Path, *, host: str, port: int) -> CommandResult:
        self.calls.append(("demo", bundle, host, port))
        return CommandResult(message=f"demo listening on http://{host}:{port}")


RUNNER = CliRunner()


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--help"], ("solve", "dataset", "model", "validate", "demo", "version")),
        (["solve", "--help"], ("--config", "--output", "--device")),
        (["dataset", "validate", "--help"], ("--manifest", "--json")),
        (["model", "inspect", "--help"], ("--bundle", "--json")),
        (["validate", "--help"], ("--config", "--output-dir", "--device")),
        (["demo", "--help"], ("--bundle", "--host", "--port")),
        (["version", "--help"], ("--json",)),
    ],
)
def test_every_documented_command_and_flag_appears_in_help(
    arguments: list[str], expected: tuple[str, ...]
) -> None:
    result = RUNNER.invoke(app, arguments)
    assert result.exit_code == 0
    assert result.stderr == ""
    for value in expected:
        assert value in result.stdout


def test_version_human_and_json_outputs_are_stable() -> None:
    human = RUNNER.invoke(app, ["version"])
    assert human.exit_code == 0
    assert human.stdout.startswith("soufflerie 0.1.0 (Python 3.11.")
    assert human.stderr == ""

    machine = RUNNER.invoke(app, ["version", "--json"])
    assert machine.exit_code == 0
    payload = json.loads(machine.stdout)
    result = VersionResult.model_validate(payload)
    assert result.package == "soufflerie"
    assert result.version == "0.1.0"
    assert machine.stderr == ""


def test_machine_output_schemas_are_versioned_and_closed() -> None:
    documents = cli_schema_documents()
    assert documents.keys() == {
        "cli-error",
        "cli-version",
        "dataset-validation-result",
        "model-inspection-result",
    }
    for name, document in documents.items():
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        identifier = document["$id"]
        assert isinstance(identifier, str)
        assert identifier.endswith(f"/schemas/v1/{name}.json")
        assert document["additionalProperties"] is False


def test_dataset_and_model_json_are_schema_valid_and_stdout_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(cli, "_backend_factory", lambda: backend)
    manifest = tmp_path / "manifest.json"
    bundle = tmp_path / "bundle"

    dataset = RUNNER.invoke(app, ["dataset", "validate", "--manifest", str(manifest), "--json"])
    assert dataset.exit_code == 0
    dataset_result = DatasetValidationResult.model_validate(json.loads(dataset.stdout))
    assert dataset_result.dataset_id == "d" * 20
    assert dataset_result.case_count == 1000
    assert dataset.stderr == ""

    model = RUNNER.invoke(app, ["model", "inspect", "--bundle", str(bundle), "--json"])
    assert model.exit_code == 0
    model_result = ModelInspectionResult.model_validate(json.loads(model.stdout))
    assert model_result.model_id == "model-abc"
    assert model_result.architecture == "fno-v1"
    assert model.stderr == ""


def test_default_backend_validates_the_checked_parquet_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_backend_factory", cli.DefaultCliBackend)
    result = RUNNER.invoke(
        app,
        [
            "dataset",
            "validate",
            "--manifest",
            "tests/fixtures/dataset/manifest.parquet",
            "--json",
        ],
    )

    assert result.exit_code == 0
    validated = DatasetValidationResult.model_validate(json.loads(result.stdout))
    assert validated.dataset_id == "83d400f135848978b152"
    assert validated.case_count == 1_000
    assert result.stderr == ""


def test_thin_commands_forward_typed_arguments_and_emit_human_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(cli, "_backend_factory", lambda: backend)
    config = tmp_path / "config.yaml"
    output = tmp_path / "result"

    solve = RUNNER.invoke(
        app,
        [
            "solve",
            "--config",
            str(config),
            "--output",
            str(output),
            "--device",
            "CUDA:2",
        ],
    )
    validation = RUNNER.invoke(
        app,
        [
            "validate",
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
    )
    demo = RUNNER.invoke(
        app,
        ["demo", "--bundle", str(output), "--host", "localhost", "--port", "9000"],
    )

    assert solve.exit_code == validation.exit_code == demo.exit_code == 0
    assert solve.stdout == f"wrote solver result to {output}\n"
    assert validation.stdout == f"validation green at {output}\n"
    assert demo.stdout == "demo listening on http://localhost:9000\n"
    assert solve.stderr == validation.stderr == demo.stderr == ""
    assert backend.calls == [
        ("solve", config, output, "cuda:2"),
        ("validate", config, output, "cpu"),
        ("demo", output, "localhost", 9000),
    ]


def test_help_does_not_construct_backend_or_import_optional_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_backend() -> FakeBackend:
        raise AssertionError("help must not construct a domain backend")

    def forbidden_import(name: str, package: str | None = None) -> object:
        raise AssertionError(f"help must not import optional runtime {name!r} ({package!r})")

    monkeypatch.setattr(cli, "_backend_factory", forbidden_backend)
    monkeypatch.setattr(importlib, "import_module", forbidden_import)
    for arguments in (["--help"], ["solve", "--help"], ["demo", "--help"]):
        assert RUNNER.invoke(app, arguments).exit_code == 0
