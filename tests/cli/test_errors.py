from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soufflerie.cli as cli
from soufflerie.cli import CliError, CommandResult, DatasetValidationResult, ModelInspectionResult
from soufflerie.errors import (
    ArtifactIntegrityError,
    CapacityError,
    ConfigurationError,
    DependencyUnavailableError,
    DeviceUnavailableError,
    DomainError,
    InternalInvariantError,
    NonConvergenceError,
    NumericalStabilityError,
    RemoteExecutionError,
    SchemaVersionError,
    ValidationGateError,
)


class FailingBackend:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def solve(self, config: Path, output: Path, *, device: str) -> CommandResult:
        del config, output, device
        raise self.error

    def validate_dataset(self, manifest: Path) -> DatasetValidationResult:
        del manifest
        raise self.error

    def inspect_model(self, bundle: Path) -> ModelInspectionResult:
        del bundle
        raise self.error

    def validate(self, config: Path, output_dir: Path, *, device: str) -> CommandResult:
        del config, output_dir, device
        raise self.error

    def demo(self, bundle: Path, *, host: str, port: int) -> CommandResult:
        del bundle, host, port
        raise self.error


RUNNER = CliRunner()
SOLVE_ARGUMENTS = ["solve", "--config", "case.yaml", "--output", "result"]


@pytest.mark.parametrize(
    ("error", "exit_code", "public_code"),
    [
        (ConfigurationError("invalid config"), 2, "CONFIG_INVALID"),
        (DomainError("outside domain"), 3, "CASE_OUT_OF_DOMAIN"),
        (NumericalStabilityError("unstable"), 3, "SOLVER_UNSTABLE"),
        (NonConvergenceError("not converged"), 3, "SOLVER_NOT_CONVERGED"),
        (ArtifactIntegrityError("digest mismatch"), 4, "ARTIFACT_INTEGRITY"),
        (SchemaVersionError(2), 4, "SCHEMA_UNSUPPORTED"),
        (DependencyUnavailableError("missing extra"), 5, "DEPENDENCY_UNAVAILABLE"),
        (DeviceUnavailableError("missing device"), 5, "DEVICE_UNAVAILABLE"),
        (ValidationGateError("required gate red"), 6, "VALIDATION_RED"),
        (RemoteExecutionError("remote failed"), 7, "REMOTE_EXECUTION"),
        (CapacityError("busy"), 7, "CAPACITY_EXHAUSTED"),
        (InternalInvariantError("internal detail"), 70, "INTERNAL_INVARIANT"),
    ],
)
def test_typed_failures_map_to_stable_exit_codes_and_structured_stderr(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    exit_code: int,
    public_code: str,
) -> None:
    monkeypatch.setattr(cli, "_backend_factory", lambda: FailingBackend(error))
    result = RUNNER.invoke(cli.app, SOLVE_ARGUMENTS)

    assert result.exit_code == exit_code
    assert result.stdout == ""
    record = CliError.model_validate(json.loads(result.stderr))
    assert record.error.code == public_code
    assert record.error.retryable is getattr(error, "retryable", False)


def test_unknown_errors_hide_internal_message_and_exit_70(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "sentinel-internal-secret"
    monkeypatch.setattr(cli, "_backend_factory", lambda: FailingBackend(RuntimeError(sentinel)))

    result = RUNNER.invoke(cli.app, SOLVE_ARGUMENTS)
    assert result.exit_code == 70
    assert result.stdout == ""
    assert sentinel not in result.stderr
    record = CliError.model_validate(json.loads(result.stderr))
    assert record.error.code == "INTERNAL_ERROR"
    assert record.error.message == "unexpected internal error"


def test_invalid_device_is_configuration_error_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_backend() -> FailingBackend:
        raise AssertionError("invalid device must fail before backend construction")

    monkeypatch.setattr(cli, "_backend_factory", forbidden_backend)
    result = RUNNER.invoke(cli.app, [*SOLVE_ARGUMENTS, "--device", "metal"])
    assert result.exit_code == 2
    record = CliError.model_validate(json.loads(result.stderr))
    assert record.error.code == "CONFIG_INVALID"


@pytest.mark.parametrize(
    ("arguments", "extra"),
    [
        (SOLVE_ARGUMENTS, "solver"),
        (["validate", "--config", "validation.yaml", "--output-dir", "report"], "ml"),
        (["demo", "--bundle", "bundle"], "serve"),
    ],
)
def test_missing_optional_extra_is_typed_and_does_not_traceback(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], extra: str
) -> None:
    def missing_import(name: str, package: str | None = None) -> object:
        del package
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(cli, "_backend_factory", cli.DefaultCliBackend)
    monkeypatch.setattr(importlib, "import_module", missing_import)

    result = RUNNER.invoke(cli.app, arguments)
    assert result.exit_code == 5
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    record = CliError.model_validate(json.loads(result.stderr))
    assert record.error.code == "DEPENDENCY_UNAVAILABLE"
    assert f"soufflerie[{extra}]" in record.error.message


@pytest.mark.parametrize(
    "arguments",
    [
        ["dataset", "validate", "--manifest", "manifest.json"],
        ["model", "inspect", "--bundle", "bundle"],
    ],
)
def test_unavailable_domain_backend_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(cli, "_backend_factory", cli.DefaultCliBackend)
    result = RUNNER.invoke(cli.app, arguments)
    assert result.exit_code == 5
    assert result.stdout == ""
    record = CliError.model_validate(json.loads(result.stderr))
    assert record.error.code == "DEPENDENCY_UNAVAILABLE"
    assert "backend is not available in this build" in record.error.message


def test_click_usage_errors_remain_on_stderr_with_exit_2() -> None:
    result = RUNNER.invoke(cli.app, ["solve"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Missing option '--config'" in result.stderr
