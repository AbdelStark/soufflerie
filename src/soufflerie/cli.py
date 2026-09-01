"""Installed, dependency-isolated command-line adapter."""

from __future__ import annotations

import importlib
import json
import re
import sys
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, ParamSpec, Protocol, cast

import typer
from pydantic import BaseModel, Field

from soufflerie import __version__
from soufflerie.errors import (
    ArtifactIntegrityError,
    CapacityError,
    ConfigurationError,
    DependencyUnavailableError,
    DeviceUnavailableError,
    DomainError,
    NonConvergenceError,
    NumericalStabilityError,
    RemoteExecutionError,
    SchemaVersionError,
    SoufflerieError,
    ValidationGateError,
)
from soufflerie.observability import Redactor, safe_exception_fields
from soufflerie.schemas import StrictFrozenModel, VersionedModel

P = ParamSpec("P")

_DEVICE_PATTERN = re.compile(r"^(?:cpu|cuda(?::(?:0|[1-9][0-9]*))?)$")


class CommandResult(StrictFrozenModel):
    """Human result returned by non-JSON command backends."""

    message: Annotated[str, Field(min_length=1, max_length=4096)]


class VersionResult(VersionedModel):
    """Machine-readable installed distribution identity."""

    package: Literal["soufflerie"] = "soufflerie"
    version: Annotated[str, Field(min_length=1, max_length=64)]
    python: Annotated[str, Field(min_length=1, max_length=64)]


class DatasetValidationResult(VersionedModel):
    """Machine-readable successful dataset validation summary."""

    valid: Literal[True] = True
    manifest: Annotated[str, Field(min_length=1, max_length=4096)]
    dataset_id: Annotated[str, Field(min_length=1, max_length=128)]
    case_count: int = Field(ge=0)


class ModelInspectionResult(VersionedModel):
    """Machine-readable successful model bundle inspection summary."""

    valid: Literal[True] = True
    bundle: Annotated[str, Field(min_length=1, max_length=4096)]
    model_id: Annotated[str, Field(min_length=1, max_length=128)]
    dataset_id: Annotated[str, Field(min_length=1, max_length=128)]
    architecture: Annotated[str, Field(min_length=1, max_length=128)]


class CliErrorDetail(StrictFrozenModel):
    """Stable automation-safe CLI error body."""

    code: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$", max_length=64)]
    message: Annotated[str, Field(min_length=1, max_length=4096)]
    retryable: bool


class CliError(VersionedModel):
    """Structured stderr contract for command execution failures."""

    error: CliErrorDetail


CLI_SCHEMA_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "cli-error": CliError,
        "cli-version": VersionResult,
        "dataset-validation-result": DatasetValidationResult,
        "model-inspection-result": ModelInspectionResult,
    }
)


def cli_schema_documents() -> dict[str, dict[str, object]]:
    """Generate schema-v1 JSON documents for machine-readable CLI output."""

    result: dict[str, dict[str, object]] = {}
    for name, model in CLI_SCHEMA_MODELS.items():
        document = cast(dict[str, object], model.model_json_schema())
        document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        document["$id"] = f"https://github.com/AbdelStark/soufflerie/schemas/v1/{name}.json"
        result[name] = document
    return result


def rendered_cli_schema_documents() -> dict[str, str]:
    """Render CLI schemas in the checked-in canonical format."""

    return {
        f"{name}.json": json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        for name, document in cli_schema_documents().items()
    }


class CliBackend(Protocol):
    """Domain-operation boundary kept injectable for CPU contract tests."""

    def solve(self, config: Path, output: Path, *, device: str) -> CommandResult: ...

    def validate_dataset(self, manifest: Path) -> DatasetValidationResult: ...

    def inspect_model(self, bundle: Path) -> ModelInspectionResult: ...

    def validate(self, config: Path, output_dir: Path, *, device: str) -> CommandResult: ...

    def demo(self, bundle: Path, *, host: str, port: int) -> CommandResult: ...


def _require_extra(import_name: str, *, extra: str, operation: str) -> None:
    try:
        importlib.import_module(import_name)
    except ImportError as error:
        raise DependencyUnavailableError(
            f"{operation} requires the '{extra}' extra; install soufflerie[{extra}]"
        ) from error


def _backend_not_implemented(operation: str) -> DependencyUnavailableError:
    return DependencyUnavailableError(
        f"{operation} backend is not available in this build; "
        "install a release containing its domain implementation"
    )


class DefaultCliBackend:
    """Lazy optional-runtime guards for domain backends assigned to later issues."""

    def solve(self, config: Path, output: Path, *, device: str) -> CommandResult:
        del config, output, device
        _require_extra("warp", extra="solver", operation="solve")
        raise _backend_not_implemented("solve")

    def validate_dataset(self, manifest: Path) -> DatasetValidationResult:
        del manifest
        raise _backend_not_implemented("dataset validation")

    def inspect_model(self, bundle: Path) -> ModelInspectionResult:
        del bundle
        raise _backend_not_implemented("model inspection")

    def validate(self, config: Path, output_dir: Path, *, device: str) -> CommandResult:
        del config, output_dir, device
        _require_extra("torch", extra="ml", operation="validation")
        raise _backend_not_implemented("validation")

    def demo(self, bundle: Path, *, host: str, port: int) -> CommandResult:
        del bundle, host, port
        _require_extra("gradio", extra="serve", operation="demo")
        _require_extra("matplotlib", extra="viz", operation="demo")
        raise _backend_not_implemented("demo")


_backend_factory: Callable[[], CliBackend] = DefaultCliBackend


def _exit_code(error: Exception) -> int:
    if isinstance(error, ConfigurationError):
        return 2
    if isinstance(error, (DomainError, NumericalStabilityError, NonConvergenceError)):
        return 3
    if isinstance(error, (ArtifactIntegrityError, SchemaVersionError)):
        return 4
    if isinstance(error, (DependencyUnavailableError, DeviceUnavailableError)):
        return 5
    if isinstance(error, ValidationGateError):
        return 6
    if isinstance(error, (RemoteExecutionError, CapacityError)):
        return 7
    return 70


def _error_record(error: Exception) -> CliError:
    if isinstance(error, SoufflerieError):
        safe = safe_exception_fields(error, redactor=Redactor())
        code = str(safe["error_code"])
        message = str(safe["exception_message"]) or "operation failed"
        retryable = bool(safe["retryable"])
    else:
        code = "INTERNAL_ERROR"
        message = "unexpected internal error"
        retryable = False
    return CliError(error=CliErrorDetail(code=code, message=message, retryable=retryable))


def _render_json(model: VersionedModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _command_boundary(function: Callable[P, None]) -> Callable[P, None]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            function(*args, **kwargs)
        except typer.Exit:
            raise
        except Exception as error:
            typer.echo(_render_json(_error_record(error)), err=True)
            raise typer.Exit(code=_exit_code(error)) from None

    return wrapped


def _validated_device(device: str) -> str:
    normalized = device.strip().lower()
    if _DEVICE_PATTERN.fullmatch(normalized) is None:
        raise ConfigurationError("device must be 'cpu', 'cuda', or 'cuda:<nonnegative index>'")
    return normalized


def _validated_host(host: str) -> str:
    normalized = host.strip()
    if (
        not normalized
        or len(normalized) > 253
        or any(character.isspace() for character in normalized)
    ):
        raise ConfigurationError("host must be a non-empty hostname or address without whitespace")
    return normalized


app = typer.Typer(
    add_completion=False,
    help="Reproducible numerical and learned wind-tunnel workflows.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
dataset_app = typer.Typer(
    add_completion=False,
    help="Dataset artifact commands.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
model_app = typer.Typer(
    add_completion=False,
    help="Model bundle commands.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")


@app.command("solve")
@_command_boundary
def solve_command(
    config: Annotated[Path, typer.Option("--config", help="Case YAML configuration path.")],
    output: Annotated[Path, typer.Option("--output", help="Destination result artifact path.")],
    device: Annotated[
        str, typer.Option("--device", help="Execution device: cpu or cuda[:index].")
    ] = "cpu",
) -> None:
    """Run one validated solver case."""

    normalized_device = _validated_device(device)
    result = _backend_factory().solve(config, output, device=normalized_device)
    typer.echo(result.message)


@dataset_app.command("validate")
@_command_boundary
def dataset_validate_command(
    manifest: Annotated[Path, typer.Option("--manifest", help="Dataset manifest path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit schema-v1 JSON.")] = False,
) -> None:
    """Validate a dataset manifest and all referenced members."""

    result = _backend_factory().validate_dataset(manifest)
    if json_output:
        typer.echo(_render_json(result))
    else:
        typer.echo(
            f"dataset valid: {result.dataset_id} ({result.case_count} cases) at {result.manifest}"
        )


@model_app.command("inspect")
@_command_boundary
def model_inspect_command(
    bundle: Annotated[Path, typer.Option("--bundle", help="Model bundle path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit schema-v1 JSON.")] = False,
) -> None:
    """Inspect and verify a safe model bundle."""

    result = _backend_factory().inspect_model(bundle)
    if json_output:
        typer.echo(_render_json(result))
    else:
        typer.echo(
            f"model valid: {result.model_id} ({result.architecture}, dataset {result.dataset_id})"
        )


@app.command("validate")
@_command_boundary
def validate_command(
    config: Annotated[Path, typer.Option("--config", help="Validation YAML configuration path.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", help="Destination report directory.")],
    device: Annotated[
        str, typer.Option("--device", help="Execution device: cpu or cuda[:index].")
    ] = "cpu",
) -> None:
    """Run the configured validation gates."""

    normalized_device = _validated_device(device)
    result = _backend_factory().validate(config, output_dir, device=normalized_device)
    typer.echo(result.message)


@app.command("demo")
@_command_boundary
def demo_command(
    bundle: Annotated[Path, typer.Option("--bundle", help="Model bundle path.")],
    host: Annotated[str, typer.Option("--host", help="Bind hostname or address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Bind TCP port.")] = 7860,
) -> None:
    """Launch the local interactive model demo."""

    normalized_host = _validated_host(host)
    result = _backend_factory().demo(bundle, host=normalized_host, port=port)
    typer.echo(result.message)


@app.command("version")
@_command_boundary
def version_command(
    json_output: Annotated[bool, typer.Option("--json", help="Emit schema-v1 JSON.")] = False,
) -> None:
    """Show the installed Soufflerie and Python versions."""

    result = VersionResult(
        version=__version__,
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    if json_output:
        typer.echo(_render_json(result))
    else:
        typer.echo(f"soufflerie {result.version} (Python {result.python})")


__all__ = [
    "CLI_SCHEMA_MODELS",
    "CliBackend",
    "CliError",
    "CliErrorDetail",
    "CommandResult",
    "DatasetValidationResult",
    "DefaultCliBackend",
    "ModelInspectionResult",
    "VersionResult",
    "app",
    "cli_schema_documents",
    "rendered_cli_schema_documents",
]
