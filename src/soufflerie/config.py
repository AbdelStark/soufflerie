"""Strict, environment-independent experiment configuration contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, TypeVar

import yaml  # type: ignore[import-untyped]  # PyYAML does not publish typing metadata.
from pydantic import BaseModel, Field, ValidationError, model_validator

from soufflerie.errors import ConfigurationError, SchemaVersionError
from soufflerie.schemas import (
    ContentId,
    GridSpec,
    StrictFrozenModel,
    VersionedModel,
    canonical_sha256,
)

MAX_CONFIG_BYTES = 1_048_576
UINT64_MAX = 2**64 - 1
ENV_REFERENCE = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")
ConfigT = TypeVar("ConfigT", bound=BaseModel)
FinitePositive = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
FiniteNonnegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Seed = Annotated[int, Field(ge=0, le=UINT64_MAX)]


class Range(StrictFrozenModel):
    """A finite increasing closed parameter range."""

    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _minimum_precedes_maximum(self) -> Self:
        if self.minimum >= self.maximum:
            raise ValueError("minimum must be less than maximum")
        return self


class RunSchedule(StrictFrozenModel):
    """Numerical run and diagnostic sampling schedule."""

    steps: int = Field(ge=1)
    warmup_steps: int = Field(ge=0)
    sample_interval: int = Field(ge=1)
    inlet_velocity_lu: float = Field(gt=0.0, le=0.1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _schedule_is_coherent(self) -> Self:
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be less than steps")
        if self.sample_interval > self.steps - self.warmup_steps:
            raise ValueError("sample_interval must fit within the post-warmup window")
        return self


class CanonicalConfig(VersionedModel):
    """A schema-v1 configuration with a location-independent digest."""

    @property
    def config_digest(self) -> str:
        return canonical_sha256(self)


class SweepConfig(CanonicalConfig):
    """Frozen canonical 1,000-case design and solver policy."""

    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")]
    seed: Seed
    samples: Literal[1000]
    shape_family: Literal["ellipse"]
    aspect_ratio: Range
    rotation_deg: Range
    scale: Range
    reynolds: Range
    grid: GridSpec
    run: RunSchedule
    split_counts: tuple[Literal[600], Literal[200], Literal[200]]

    @model_validator(mode="after")
    def _canonical_domain_is_frozen(self) -> Self:
        expected = {
            "aspect_ratio": (0.5, 1.0),
            "rotation_deg": (0.0, 30.0),
            "scale": (0.75, 1.25),
            "reynolds": (40.0, 300.0),
        }
        for name, bounds in expected.items():
            actual = getattr(self, name)
            if (actual.minimum, actual.maximum) != bounds:
                raise ValueError(f"{name} range must be the canonical {bounds}")
        return self


class TrainingConfig(CanonicalConfig):
    """Deterministic three-seed FNO training configuration."""

    dataset_id: ContentId
    seeds: tuple[Seed, Seed, Seed]
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=8, ge=1, le=64)
    learning_rate: FinitePositive = 1e-3
    min_learning_rate: FinitePositive = 1e-5
    weight_decay: FiniteNonnegative = 1e-4
    gradient_clip_norm: FinitePositive = 1.0
    field_weights: tuple[FinitePositive, FinitePositive, FinitePositive] = (1.0, 1.0, 0.25)
    cd_weight: FinitePositive = 0.5
    obstacle_weight: FinitePositive = 0.25
    precision: Literal["bf16", "fp16"] = "bf16"
    num_workers: int = Field(default=4, ge=0, le=64)

    @model_validator(mode="after")
    def _training_policy_is_coherent(self) -> Self:
        if len(set(self.seeds)) != 3:
            raise ValueError("training seeds must be three distinct values")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        return self


class ValidationConfig(CanonicalConfig):
    """Immutable inputs and sampling policy for release validation."""

    dataset_id: ContentId
    ensemble_model_ids: tuple[ContentId, ContentId, ContentId]
    baseline_ids: tuple[ContentId, ContentId]
    report_seed: Seed
    bootstrap_resamples: int = Field(default=2_000, ge=100, le=100_000)
    worst_case_count: Literal[20] = 20
    ood_geometry_count: Literal[10] = 10
    sensitivity_case_count: Literal[10] = 10

    @model_validator(mode="after")
    def _model_identities_are_distinct(self) -> Self:
        identities = (*self.ensemble_model_ids, *self.baseline_ids)
        if len(set(identities)) != len(identities):
            raise ValueError("ensemble and baseline model IDs must be distinct")
        return self


class ServiceConfig(CanonicalConfig):
    """Bounded public service policy without credentials or remote settings."""

    model_id: ContentId
    dataset_id: ContentId
    report_id: ContentId
    host: Literal["127.0.0.1", "0.0.0.0"] = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65_535)
    predict_concurrency: int = Field(default=1, ge=1, le=1)
    predict_queue_capacity: int = Field(default=8, ge=0, le=8)
    solve_enabled: bool = False
    solve_concurrency: int = Field(default=0, ge=0, le=2)
    solve_queue_capacity: int = Field(default=0, ge=0, le=8)
    solve_timeout_seconds: int = Field(default=180, ge=1, le=180)
    predictions_per_minute_client: int = Field(default=60, ge=1, le=60)
    solves_per_hour_client: int = Field(default=2, ge=0, le=2)
    solves_per_day_global: int = Field(default=20, ge=0, le=20)
    solve_gpu_seconds_per_day: float = Field(default=3_600.0, ge=0.0, le=3_600.0)

    @model_validator(mode="after")
    def _disabled_solve_has_no_live_capacity(self) -> Self:
        if not self.solve_enabled and (
            self.solve_concurrency != 0 or self.solve_queue_capacity != 0
        ):
            raise ValueError(
                "solve_concurrency and solve_queue_capacity must be zero "
                "when solve_enabled is false"
            )
        if self.solve_enabled and self.solve_concurrency == 0:
            raise ValueError("solve_concurrency must be positive when solve_enabled is true")
        return self


class _StrictLoader(yaml.SafeLoader):  # type: ignore[misc]  # Runtime subclass is required.
    """Safe YAML loader that forbids anchors, aliases, and duplicate keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise ConfigurationError("YAML aliases are forbidden")
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise ConfigurationError("YAML anchors are forbidden")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, object]:
        if not isinstance(node, yaml.MappingNode):
            raise ConfigurationError("configuration mappings must use YAML objects")
        mapping: dict[str, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigurationError("configuration keys must be strings")
            if key in mapping:
                raise ConfigurationError(f"duplicate configuration key: {key}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


# PyYAML follows YAML 1.1 and otherwise treats yes/no/on/off as booleans. The
# public configuration language accepts only explicit true/false boolean tokens.
_StrictLoader.yaml_implicit_resolvers = {
    key: [(tag, expression) for tag, expression in resolvers if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$"),
    ["t", "f"],
)


def _validate_yaml_values(value: object, *, path: str = "config") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError(f"{path} must not contain NaN or infinity")
        return
    if isinstance(value, str):
        if ENV_REFERENCE.search(value):
            raise ConfigurationError(f"{path} must not contain environment interpolation")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(f"{path} keys must be strings")
            _validate_yaml_values(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_yaml_values(item, path=f"{path}[{index}]")
        return
    raise ConfigurationError(f"{path} contains unsupported YAML type {type(value).__name__}")


def _validation_message(error: ValidationError) -> str:
    messages: list[str] = []
    for issue in error.errors(include_url=False)[:8]:
        location = ".".join(str(part) for part in issue["loc"]) or "config"
        messages.append(f"{location}: {issue['msg']}")
    suffix = "" if len(error.errors()) <= 8 else "; additional errors omitted"
    return "invalid configuration: " + "; ".join(messages) + suffix


def _freeze_yaml_sequences(value: object) -> object:
    """Map YAML sequences to immutable tuples without scalar coercion."""

    if isinstance(value, Mapping):
        return {key: _freeze_yaml_sequences(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_yaml_sequences(item) for item in value)
    return value


def parse_config(data: str | bytes, model: type[ConfigT]) -> ConfigT:
    """Parse one bounded YAML document into a strict frozen model."""

    encoded = data.encode("utf-8") if isinstance(data, str) else data
    if len(encoded) > MAX_CONFIG_BYTES:
        raise ConfigurationError(f"configuration exceeds {MAX_CONFIG_BYTES} bytes")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError("configuration must be valid UTF-8") from exc
    if "\x00" in text:
        raise ConfigurationError("configuration must not contain NUL bytes")
    try:
        raw = yaml.load(text, Loader=_StrictLoader)
    except ConfigurationError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigurationError("configuration is not valid single-document safe YAML") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError("configuration root must be a YAML object")
    _validate_yaml_values(raw)
    try:
        return model.model_validate(_freeze_yaml_sequences(raw))
    except SchemaVersionError:
        raise
    except ValidationError as exc:
        raise ConfigurationError(_validation_message(exc)) from exc


def load_config(path: Path, model: type[ConfigT]) -> ConfigT:
    """Load a bounded config file; paths never participate in its digest."""

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ConfigurationError(f"unable to read configuration file {path.name!r}") from exc
    return parse_config(data, model)


def config_digest(config: BaseModel) -> str:
    return canonical_sha256(config)


CONFIG_SCHEMA_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "service-config": ServiceConfig,
        "sweep-config": SweepConfig,
        "training-config": TrainingConfig,
        "validation-config": ValidationConfig,
    }
)


def config_schema_documents() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, model in CONFIG_SCHEMA_MODELS.items():
        document = model.model_json_schema()
        document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        document["$id"] = f"https://github.com/AbdelStark/soufflerie/schemas/v1/{name}.json"
        result[name] = document
    return result


def rendered_config_schema_documents() -> dict[str, str]:
    return {
        f"{name}.json": json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        for name, document in config_schema_documents().items()
    }


__all__ = [
    "CONFIG_SCHEMA_MODELS",
    "MAX_CONFIG_BYTES",
    "CanonicalConfig",
    "Range",
    "RunSchedule",
    "ServiceConfig",
    "SweepConfig",
    "TrainingConfig",
    "ValidationConfig",
    "config_digest",
    "config_schema_documents",
    "load_config",
    "parse_config",
    "rendered_config_schema_documents",
]
