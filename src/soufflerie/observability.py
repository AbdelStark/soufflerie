"""Structured, vendor-neutral observability and redaction contracts.

The module deliberately has no dependency on global logging configuration or a
hosted monitoring client. Domain code receives a sink or an operation recorder,
and the default sinks discard records. Every serializing sink applies the same
redaction policy at its final trust boundary.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Literal, Protocol, TypeAlias, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

from soufflerie.errors import SoufflerieError
from soufflerie.schemas import JsonScalar, JsonValue, StrictFrozenModel, VersionedModel

Component: TypeAlias = Literal["solver", "datagen", "training", "validation", "service", "infra"]
EventLevel: TypeAlias = Literal["debug", "info", "warning", "error"]
EventName: TypeAlias = Literal[
    "case_started",
    "case_completed",
    "case_failed",
    "artifact_published",
    "sweep_progress",
    "epoch_completed",
    "checkpoint_selected",
    "gate_evaluated",
    "prediction_completed",
    "solve_admitted",
    "solve_terminal",
    "redaction_applied",
    "operation_started",
    "operation_completed",
    "operation_failed",
]
MetricName: TypeAlias = Literal[
    "solver_run_seconds",
    "solver_lups",
    "solver_mass_drift_ratio",
    "sweep_cases_total",
    "remote_gpu_seconds_total",
    "training_epoch_seconds",
    "training_loss",
    "validation_metric",
    "prediction_duration_ms",
    "solve_queue_depth",
    "http_requests_total",
]
MetricScope: TypeAlias = Literal["aggregate", "offline"]
TimerSegment: TypeAlias = Literal["compute", "queue", "io"]

CorrelationId = Annotated[
    str,
    StringConstraints(
        min_length=36,
        max_length=36,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    ),
]
Identity = Annotated[str, StringConstraints(min_length=1, max_length=128, strip_whitespace=True)]
FieldKey = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]
LabelKey = FieldKey
LabelValue = Annotated[str, StringConstraints(min_length=1, max_length=128)]
MetricUnit = Annotated[str, StringConstraints(min_length=1, max_length=64)]
EventFieldValue: TypeAlias = JsonScalar | Annotated[list[JsonScalar], Field(max_length=64)]

MAX_EVENT_FIELDS = 64
MAX_FIELD_LIST_ITEMS = 64
MAX_STRING_LENGTH = 4096
MAX_REDACTION_DEPTH = 12
MAX_REDACTION_ITEMS = 1024
REDACTED = "[REDACTED]"
EXTERNAL_PATH = "[EXTERNAL_PATH]"

_FORBIDDEN_EVENT_KEYS = frozenset(
    {
        "account_id",
        "client_address",
        "client_hash",
        "client_identifier",
        "client_ip",
        "client_key",
        "environment",
        "environment_variables",
        "headers",
        "ip_address",
        "raw_request",
        "rate_limit_key",
        "request_body",
        "source_address",
        "workspace_id",
    }
)
_FORBIDDEN_EVENT_KEY_PATTERN = (
    r"(?:^|_)(?:" + "|".join(sorted(re.escape(key) for key in _FORBIDDEN_EVENT_KEYS)) + r")(?:_|$)"
)
_BASE_SENSITIVE_KEYS = frozenset(
    {"token", "secret", "password", "authorization", "cookie", "api_key", "apikey"}
)
_PATH_KEY_MARKERS = frozenset({"path", "file", "filename", "directory", "artifact_root"})
_URL_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_URL_IN_TEXT = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")
_ABSOLUTE_PATH_IN_TEXT = re.compile(
    r"(?<![A-Za-z0-9:/])(?:/[A-Za-z0-9._~-]+){2,}|(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/](?:[^\\/\s]+[\\/])+[^\\/\s]+)"
)
_FIELD_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _validate_scalar(value: JsonScalar) -> None:
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError("event field values must be JSON scalars or bounded scalar lists")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("event and metric values must be finite")
    if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
        raise ValueError(f"strings must not exceed {MAX_STRING_LENGTH} characters")


def _validate_event_fields(
    value: dict[FieldKey, EventFieldValue],
) -> dict[FieldKey, EventFieldValue]:
    if len(value) > MAX_EVENT_FIELDS:
        raise ValueError(f"events must not exceed {MAX_EVENT_FIELDS} fields")
    invalid = sorted(
        key for key in value if len(key) > 64 or _FIELD_KEY_PATTERN.fullmatch(key) is None
    )
    if invalid:
        raise ValueError(f"invalid event field names: {', '.join(invalid)}")
    forbidden = sorted(key for key in value if _is_forbidden_event_key(key))
    if forbidden:
        raise ValueError(f"forbidden event fields: {', '.join(forbidden)}")
    for item in value.values():
        if isinstance(item, list):
            if len(item) > MAX_FIELD_LIST_ITEMS:
                raise ValueError(f"event field lists must not exceed {MAX_FIELD_LIST_ITEMS} items")
            for scalar in item:
                _validate_scalar(scalar)
        else:
            _validate_scalar(item)
    return value


def _canonical_uuid7(value: str) -> str:
    identifier = UUID(value)
    if identifier.version != 7:
        raise ValueError("correlation_id must be UUIDv7")
    return str(identifier)


class ArtifactIdentities(StrictFrozenModel):
    """Optional artifact identities propagated through one operation."""

    case_id: Identity | None = None
    dataset_id: Identity | None = None
    model_id: Identity | None = None
    job_id: Identity | None = None


EMPTY_ARTIFACT_IDENTITIES = ArtifactIdentities()


class Event(VersionedModel):
    """One bounded schema-v1 application event."""

    timestamp: datetime
    level: EventLevel
    event: EventName
    correlation_id: CorrelationId
    component: Component
    case_id: Identity | None = None
    dataset_id: Identity | None = None
    model_id: Identity | None = None
    job_id: Identity | None = None
    fields: dict[FieldKey, EventFieldValue] = Field(
        default_factory=dict, max_length=MAX_EVENT_FIELDS
    )

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="timestamp")

    @field_validator("correlation_id")
    @classmethod
    def _correlation_is_uuid7(cls, value: str) -> str:
        return _canonical_uuid7(value)

    @field_validator("fields")
    @classmethod
    def _fields_are_bounded_and_safe(
        cls, value: dict[str, EventFieldValue]
    ) -> dict[str, EventFieldValue]:
        return _validate_event_fields(value)


@dataclass(frozen=True, slots=True)
class EventDefinition:
    """Stable registry metadata for an allowed event name."""

    default_level: EventLevel
    default_component: Component
    success_event: EventName | None = None
    failure_event: EventName | None = None


EVENT_REGISTRY: Mapping[EventName, EventDefinition] = MappingProxyType(
    {
        "case_started": EventDefinition("info", "solver", "case_completed", "case_failed"),
        "case_completed": EventDefinition("info", "solver"),
        "case_failed": EventDefinition("error", "solver"),
        "artifact_published": EventDefinition("info", "infra"),
        "sweep_progress": EventDefinition("info", "datagen"),
        "epoch_completed": EventDefinition("info", "training"),
        "checkpoint_selected": EventDefinition("info", "training"),
        "gate_evaluated": EventDefinition("info", "validation"),
        "prediction_completed": EventDefinition("info", "service"),
        "solve_admitted": EventDefinition("info", "service", "solve_terminal", "solve_terminal"),
        "solve_terminal": EventDefinition("info", "service"),
        "redaction_applied": EventDefinition("debug", "infra"),
        "operation_started": EventDefinition(
            "info", "infra", "operation_completed", "operation_failed"
        ),
        "operation_completed": EventDefinition("info", "infra"),
        "operation_failed": EventDefinition("error", "infra"),
    }
)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Stable name, unit, and exact low-cardinality label contract."""

    unit: str | None
    labels: tuple[LabelKey, ...]
    offline_only: bool = False


METRIC_REGISTRY: Mapping[MetricName, MetricDefinition] = MappingProxyType(
    {
        "solver_run_seconds": MetricDefinition("seconds", ("device_class", "outcome")),
        "solver_lups": MetricDefinition("lattice_updates/second", ("device_class", "grid")),
        "solver_mass_drift_ratio": MetricDefinition("ratio", ("case_id",), offline_only=True),
        "sweep_cases_total": MetricDefinition("count", ("state",)),
        "remote_gpu_seconds_total": MetricDefinition(
            "seconds", ("milestone", "device_class", "operation")
        ),
        "training_epoch_seconds": MetricDefinition("seconds", ("seed",)),
        "training_loss": MetricDefinition("ratio", ("seed", "split", "term")),
        "validation_metric": MetricDefinition(None, ("metric", "gate_status")),
        "prediction_duration_ms": MetricDefinition(
            "milliseconds", ("device_class", "validation_status")
        ),
        "solve_queue_depth": MetricDefinition("count", ("state",)),
        "http_requests_total": MetricDefinition("count", ("route", "method", "status_class")),
    }
)


class MetricRecord(VersionedModel):
    """One schema-v1 metric sample validated against ``METRIC_REGISTRY``."""

    timestamp: datetime
    metric: MetricName
    value: float = Field(allow_inf_nan=False)
    unit: MetricUnit
    labels: dict[LabelKey, LabelValue]
    scope: MetricScope = "aggregate"
    correlation_id: CorrelationId | None = None

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="timestamp")

    @field_validator("correlation_id")
    @classmethod
    def _optional_correlation_is_uuid7(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _canonical_uuid7(value)

    @model_validator(mode="after")
    def _matches_registry(self) -> MetricRecord:
        definition = METRIC_REGISTRY[self.metric]
        if set(self.labels) != set(definition.labels):
            expected = ", ".join(definition.labels)
            raise ValueError(f"{self.metric} requires exactly these labels: {expected}")
        if definition.unit is not None and self.unit != definition.unit:
            raise ValueError(f"{self.metric} uses unit {definition.unit!r}")
        if definition.offline_only and self.scope != "offline":
            raise ValueError(f"{self.metric} is an offline report metric")
        if not definition.offline_only and self.scope != "aggregate":
            raise ValueError(f"{self.metric} is an aggregate metric")
        return self


class TimingSummary(VersionedModel):
    """Non-overlapping operation timing totals from one monotonic clock."""

    wall_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    compute_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    queue_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    io_seconds: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _segments_fit_wall_time(self) -> TimingSummary:
        measured = self.compute_seconds + self.queue_seconds + self.io_seconds
        tolerance = max(1e-12, self.wall_seconds * 1e-12)
        if measured > self.wall_seconds + tolerance:
            raise ValueError("timing segments must not exceed wall time")
        return self


def new_correlation_id(*, timestamp: datetime | None = None) -> str:
    """Return a UUIDv7 correlation ID without relying on Python 3.12 APIs."""

    instant = datetime.now(UTC) if timestamp is None else _aware_utc(timestamp, label="timestamp")
    milliseconds = int(instant.timestamp() * 1000)
    if milliseconds < 0 or milliseconds >= 2**48:
        raise ValueError("timestamp is outside the UUIDv7 48-bit millisecond range")
    random_bits = int.from_bytes(secrets.token_bytes(10), "big")
    rand_a = (random_bits >> 68) & 0xFFF
    rand_b = random_bits & ((1 << 62) - 1)
    value = (milliseconds << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return str(UUID(int=value))


def _normalized_key(value: str) -> tuple[str, str]:
    lowered = value.casefold()
    separated = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return separated, separated.replace("_", "")


def _is_sensitive_key(key: str, extensions: frozenset[str]) -> bool:
    separated, compact = _normalized_key(key)
    for marker in _BASE_SENSITIVE_KEYS | _FORBIDDEN_EVENT_KEYS | extensions:
        marker_separated, marker_compact = _normalized_key(marker)
        if marker_separated in separated or marker_compact in compact:
            return True
    return False


def _is_path_key(key: str) -> bool:
    separated, _ = _normalized_key(key)
    tokens = set(separated.split("_"))
    return bool(tokens & _PATH_KEY_MARKERS)


def _is_forbidden_event_key(key: str) -> bool:
    separated, _ = _normalized_key(key)
    padded = f"_{separated}_"
    return any(f"_{_normalized_key(marker)[0]}_" in padded for marker in _FORBIDDEN_EVENT_KEYS)


def _looks_absolute_path(value: str) -> bool:
    return value.startswith("/") or bool(_WINDOWS_ABSOLUTE_PATTERN.match(value))


@dataclass(frozen=True, slots=True)
class Redactor:
    """Immutable recursive redaction policy applied immediately before capture."""

    sensitive_key_extensions: tuple[str, ...] = ()
    secret_values: tuple[str, ...] = ()
    artifact_root: Path | None = None
    max_depth: int = MAX_REDACTION_DEPTH
    max_items: int = MAX_REDACTION_ITEMS
    _extensions: frozenset[str] = field(init=False, repr=False)
    _root: Path | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be nonnegative")
        if self.max_items < 1:
            raise ValueError("max_items must be positive")
        if any(not item for item in self.sensitive_key_extensions):
            raise ValueError("sensitive key extensions must be non-empty")
        if any(not item for item in self.secret_values):
            raise ValueError("configured secret values must be non-empty")
        object.__setattr__(
            self,
            "_extensions",
            frozenset(item.casefold() for item in self.sensitive_key_extensions),
        )
        root = None if self.artifact_root is None else self.artifact_root.resolve(strict=False)
        object.__setattr__(self, "_root", root)

    def redact(self, value: JsonValue, *, key: str | None = None) -> JsonValue:
        """Return a JSON-native copy with sensitive keys and values removed."""

        budget = [self.max_items]
        return self._redact_value(value, key=key, depth=0, budget=budget)

    def _redact_value(
        self,
        value: JsonValue,
        *,
        key: str | None,
        depth: int,
        budget: list[int],
    ) -> JsonValue:
        if depth > self.max_depth:
            raise ValueError(f"redaction input exceeds maximum depth {self.max_depth}")
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError(f"redaction input exceeds maximum size {self.max_items}")
        if key is not None and _is_sensitive_key(key, self._extensions):
            return REDACTED
        if isinstance(value, dict):
            return {
                item_key: self._redact_value(
                    item_value,
                    key=item_key,
                    depth=depth + 1,
                    budget=budget,
                )
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [
                self._redact_value(item, key=key, depth=depth + 1, budget=budget) for item in value
            ]
        if isinstance(value, str):
            return self._redact_string(value, key=key)
        return value

    def _redact_string(self, value: str, *, key: str | None) -> str:
        if _URL_PATTERN.match(value):
            result = self._sanitize_url(value)
        elif (key is not None and _is_path_key(key)) or _looks_absolute_path(value):
            result = self._sanitize_path(value)
        else:
            result = _URL_IN_TEXT.sub(lambda match: self._sanitize_url(match.group()), value)
            result = _ABSOLUTE_PATH_IN_TEXT.sub(EXTERNAL_PATH, result)
        for secret_value in sorted(self.secret_values, key=len, reverse=True):
            result = result.replace(secret_value, REDACTED)
        return result

    def _sanitize_url(self, value: str) -> str:
        try:
            parts = urlsplit(value)
            hostname = parts.hostname or ""
            if not hostname:
                return f"{parts.scheme}://{EXTERNAL_PATH}"
            if ":" in hostname and not hostname.startswith("["):
                hostname = f"[{hostname}]"
            port = "" if parts.port is None else f":{parts.port}"
            return urlunsplit((parts.scheme, f"{hostname}{port}", parts.path, "", ""))
        except ValueError:
            return REDACTED

    def _sanitize_path(self, value: str) -> str:
        if _WINDOWS_ABSOLUTE_PATTERN.match(value):
            # Windows artifact roots cannot be soundly resolved on a POSIX host.
            return EXTERNAL_PATH
        if _looks_absolute_path(value):
            if self._root is None:
                return EXTERNAL_PATH
            candidate = Path(value).resolve(strict=False)
            try:
                return candidate.relative_to(self._root).as_posix()
            except ValueError:
                return EXTERNAL_PATH
        path = PurePosixPath(value.replace("\\", "/"))
        if ".." in path.parts:
            return EXTERNAL_PATH
        normalized = path.as_posix()
        return normalized.removeprefix("./")


def redact(
    value: JsonValue,
    *,
    key: str | None = None,
    sensitive_key_extensions: tuple[str, ...] = (),
    secret_values: tuple[str, ...] = (),
    artifact_root: Path | None = None,
) -> JsonValue:
    """Apply a one-shot redaction policy to a JSON-native value."""

    return Redactor(
        sensitive_key_extensions=sensitive_key_extensions,
        secret_values=secret_values,
        artifact_root=artifact_root,
    ).redact(value, key=key)


def safe_exception_fields(
    error: Exception, *, redactor: Redactor | None = None
) -> dict[str, JsonScalar]:
    """Return an allowlisted exception summary; stack frames are never captured."""

    policy = redactor or Redactor()
    code = error.code if isinstance(error, SoufflerieError) else "UNEXPECTED_ERROR"
    retryable = error.retryable if isinstance(error, SoufflerieError) else False
    message = cast(str, policy.redact(str(error)))
    return {
        "exception_type": type(error).__name__,
        "error_code": code,
        "retryable": retryable,
        "exception_message": message[:MAX_STRING_LENGTH],
    }


def _safe_event(event: Event, redactor: Redactor) -> Event:
    payload = event.model_dump(mode="python")
    for identity in ("case_id", "dataset_id", "model_id", "job_id"):
        value = payload[identity]
        if isinstance(value, str):
            payload[identity] = cast(str, redactor.redact(value, key=identity))
    safe_fields = {
        key: cast(EventFieldValue, redactor.redact(cast(JsonValue, value), key=key))
        for key, value in event.fields.items()
    }
    payload["fields"] = safe_fields
    return Event.model_validate(payload)


def _safe_metric(metric: MetricRecord, redactor: Redactor) -> MetricRecord:
    safe_labels = {
        key: cast(str, redactor.redact(value, key=key)) for key, value in metric.labels.items()
    }
    safe_unit = cast(str, redactor.redact(metric.unit, key="unit"))
    return MetricRecord.model_validate(
        {**metric.model_dump(mode="python"), "unit": safe_unit, "labels": safe_labels}
    )


def event_json(event: Event, *, redactor: Redactor | None = None) -> str:
    """Serialize one redacted event as canonical one-line JSON."""

    safe = _safe_event(event, redactor or Redactor())
    return json.dumps(
        safe.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def metric_json(metric: MetricRecord, *, redactor: Redactor | None = None) -> str:
    """Serialize one redacted metric record as canonical one-line JSON."""

    safe = _safe_metric(metric, redactor or Redactor())
    return json.dumps(
        safe.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


class EventSink(Protocol):
    """Minimal dependency-injection boundary for event capture."""

    def emit(self, event: Event) -> None: ...


class MetricSink(Protocol):
    """Minimal dependency-injection boundary for metric capture."""

    def emit(self, metric: MetricRecord) -> None: ...


@dataclass(frozen=True, slots=True)
class NoOpEventSink:
    """Event sink used by domain functions unless instrumentation is supplied."""

    def emit(self, event: Event) -> None:
        del event


@dataclass(frozen=True, slots=True)
class NoOpMetricSink:
    """Metric sink used by domain functions unless instrumentation is supplied."""

    def emit(self, metric: MetricRecord) -> None:
        del metric


NOOP_EVENT_SINK = NoOpEventSink()
NOOP_METRIC_SINK = NoOpMetricSink()


@dataclass(slots=True)
class InMemoryEventSink:
    """Thread-safe bounded event capture for tests and local operation summaries."""

    redactor: Redactor = field(default_factory=Redactor)
    capacity: int = 1000
    _events: list[Event] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")

    def emit(self, event: Event) -> None:
        safe = _safe_event(event, self.redactor)
        with self._lock:
            if len(self._events) >= self.capacity:
                raise BufferError("event sink capacity exceeded")
            self._events.append(safe)

    @property
    def events(self) -> tuple[Event, ...]:
        with self._lock:
            return tuple(self._events)


@dataclass(slots=True)
class InMemoryMetricSink:
    """Thread-safe bounded metric capture for tests and offline summaries."""

    redactor: Redactor = field(default_factory=Redactor)
    capacity: int = 1000
    _metrics: list[MetricRecord] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")

    def emit(self, metric: MetricRecord) -> None:
        safe = _safe_metric(metric, self.redactor)
        with self._lock:
            if len(self._metrics) >= self.capacity:
                raise BufferError("metric sink capacity exceeded")
            self._metrics.append(safe)

    @property
    def metrics(self) -> tuple[MetricRecord, ...]:
        with self._lock:
            return tuple(self._metrics)


@dataclass(slots=True)
class JsonlEventSink:
    """Append redacted events to a UTF-8 JSONL file and flush every record."""

    path: Path
    redactor: Redactor = field(default_factory=Redactor)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def emit(self, event: Event) -> None:
        line = event_json(event, redactor=self.redactor)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{line}\n")
            handle.flush()


@dataclass(slots=True)
class JsonlMetricSink:
    """Append redacted metric records to a UTF-8 JSONL file."""

    path: Path
    redactor: Redactor = field(default_factory=Redactor)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def emit(self, metric: MetricRecord) -> None:
        line = metric_json(metric, redactor=self.redactor)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{line}\n")
            handle.flush()


@dataclass(slots=True)
class OperationTimer:
    """Monotonic wall timer with non-overlapping queue, I/O, and compute spans."""

    clock: Callable[[], float] = time.perf_counter
    synchronize: Callable[[], None] | None = None
    _started: float = field(init=False, repr=False)
    _finished: float | None = field(default=None, init=False, repr=False)
    _active: TimerSegment | None = field(default=None, init=False, repr=False)
    _totals: dict[TimerSegment, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._started = self.clock()
        self._totals = {"compute": 0.0, "queue": 0.0, "io": 0.0}

    @contextmanager
    def segment(self, segment: TimerSegment) -> Iterator[None]:
        """Measure one span; compute spans synchronize before and after GPU work."""

        if self._finished is not None:
            raise RuntimeError("timer is already finished")
        if self._active is not None:
            raise RuntimeError(f"timing segments cannot overlap ({self._active} is active)")
        if segment not in self._totals:
            raise ValueError(f"unknown timing segment {segment!r}")
        if segment == "compute" and self.synchronize is not None:
            self.synchronize()
        started = self.clock()
        self._active = segment
        try:
            yield
        finally:
            try:
                if segment == "compute" and self.synchronize is not None:
                    self.synchronize()
                completed = self.clock()
                if completed < started:
                    raise RuntimeError("monotonic clock moved backwards")
                self._totals[segment] += completed - started
            finally:
                self._active = None

    def finish(self) -> TimingSummary:
        """Freeze and return coherent wall and segment totals."""

        if self._active is not None:
            raise RuntimeError(f"cannot finish while {self._active} segment is active")
        if self._finished is None:
            self._finished = self.clock()
        if self._finished < self._started:
            raise RuntimeError("monotonic clock moved backwards")
        return TimingSummary(
            wall_seconds=self._finished - self._started,
            compute_seconds=self._totals["compute"],
            queue_seconds=self._totals["queue"],
            io_seconds=self._totals["io"],
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class OperationRecorder:
    """Correlation, identities, event capture, metrics, and timing for one operation."""

    correlation_id: str
    component: Component
    identities: ArtifactIdentities = field(default_factory=ArtifactIdentities)
    event_sink: EventSink = NOOP_EVENT_SINK
    metric_sink: MetricSink = NOOP_METRIC_SINK
    redactor: Redactor = field(default_factory=Redactor)
    timer: OperationTimer = field(default_factory=OperationTimer)
    utc_clock: Callable[[], datetime] = field(default=_utc_now, repr=False)
    _terminal_fields: dict[FieldKey, EventFieldValue] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.correlation_id = _canonical_uuid7(self.correlation_id)
        if self.component not in {
            "solver",
            "datagen",
            "training",
            "validation",
            "service",
            "infra",
        }:
            raise ValueError(f"unknown component {self.component!r}")

    def emit(
        self,
        event: EventName,
        *,
        fields: Mapping[str, EventFieldValue] | None = None,
        level: EventLevel | None = None,
    ) -> Event:
        """Build and emit one event using this operation's correlation context."""

        definition = EVENT_REGISTRY[event]
        record = Event(
            timestamp=self.utc_clock(),
            level=definition.default_level if level is None else level,
            event=event,
            correlation_id=self.correlation_id,
            component=self.component,
            **self.identities.model_dump(),
            fields={} if fields is None else dict(fields),
        )
        safe = _safe_event(record, self.redactor)
        self.event_sink.emit(safe)
        return safe

    def metric(
        self,
        metric: MetricName,
        value: float,
        *,
        unit: str,
        labels: Mapping[str, str],
        scope: MetricScope = "aggregate",
    ) -> MetricRecord:
        """Build and emit one registry-validated metric sample."""

        record = MetricRecord(
            timestamp=self.utc_clock(),
            metric=metric,
            value=value,
            unit=unit,
            labels=dict(labels),
            scope=scope,
            correlation_id=self.correlation_id,
        )
        safe = _safe_metric(record, self.redactor)
        self.metric_sink.emit(safe)
        return safe

    def annotate(self, **fields: EventFieldValue) -> None:
        """Attach bounded fields to the automatic terminal event."""

        candidate = {**self._terminal_fields, **fields}
        self._terminal_fields = _validate_event_fields(candidate)

    def terminal_fields(self) -> dict[FieldKey, EventFieldValue]:
        timing = self.timer.finish()
        values: dict[FieldKey, EventFieldValue] = {
            **self._terminal_fields,
            "wall_seconds": timing.wall_seconds,
            "compute_seconds": timing.compute_seconds,
            "queue_seconds": timing.queue_seconds,
            "io_seconds": timing.io_seconds,
        }
        return values


@contextmanager
def operation_context(
    event: EventName,
    *,
    correlation_id: str | None = None,
    identities: ArtifactIdentities = EMPTY_ARTIFACT_IDENTITIES,
    component: Component | None = None,
    event_sink: EventSink = NOOP_EVENT_SINK,
    metric_sink: MetricSink = NOOP_METRIC_SINK,
    redactor: Redactor | None = None,
    timer: OperationTimer | None = None,
    utc_clock: Callable[[], datetime] = _utc_now,
) -> Iterator[OperationRecorder]:
    """Emit a registered start and automatic terminal event around an operation.

    Failures are summarized using the public error taxonomy and re-raised. The
    context never configures global logging, and its no-op defaults make it safe
    for every component to call unconditionally.
    """

    definition = EVENT_REGISTRY[event]
    recorder = OperationRecorder(
        correlation_id=correlation_id or new_correlation_id(),
        component=component or definition.default_component,
        identities=identities,
        event_sink=event_sink,
        metric_sink=metric_sink,
        redactor=redactor or Redactor(),
        timer=timer or OperationTimer(),
        utc_clock=utc_clock,
    )
    recorder.emit(event)
    try:
        yield recorder
    except Exception as error:
        terminal = definition.failure_event or "operation_failed"
        fields = {
            **recorder.terminal_fields(),
            **safe_exception_fields(error, redactor=recorder.redactor),
            "outcome": "failed",
        }
        recorder.emit(terminal, fields=fields, level="error")
        raise
    else:
        terminal = definition.success_event or "operation_completed"
        fields = {**recorder.terminal_fields(), "outcome": "succeeded"}
        recorder.emit(terminal, fields=fields)


OBSERVABILITY_SCHEMA_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "event": Event,
        "metric": MetricRecord,
    }
)


def _apply_registry_schema_invariants(name: str, document: dict[str, object]) -> None:
    properties = cast(dict[str, dict[str, object]], document["properties"])
    if name == "event":
        fields = properties["fields"]
        fields["additionalProperties"] = False
        property_names = cast(dict[str, object], fields["propertyNames"])
        property_names["not"] = {"pattern": _FORBIDDEN_EVENT_KEY_PATTERN}
        return

    labels = properties["labels"]
    labels["additionalProperties"] = False
    conditions: list[dict[str, object]] = []
    for metric, definition in METRIC_REGISTRY.items():
        label_properties = {
            label: {"maxLength": 128, "minLength": 1, "type": "string"}
            for label in definition.labels
        }
        metric_labels: dict[str, object] = {
            "additionalProperties": False,
            "properties": label_properties,
            "required": list(definition.labels),
            "type": "object",
        }
        then_properties: dict[str, object] = {
            "labels": metric_labels,
            "scope": {"const": "offline" if definition.offline_only else "aggregate"},
        }
        if definition.unit is not None:
            then_properties["unit"] = {"const": definition.unit}
        then: dict[str, object] = {"properties": then_properties}
        if definition.offline_only:
            then["required"] = ["scope"]
        conditions.append(
            {
                "if": {
                    "properties": {"metric": {"const": metric}},
                    "required": ["metric"],
                },
                "then": then,
            }
        )
    document["allOf"] = conditions


def observability_schema_documents() -> dict[str, dict[str, object]]:
    """Generate deterministic JSON Schema draft 2020-12 documents."""

    result: dict[str, dict[str, object]] = {}
    for name, model in OBSERVABILITY_SCHEMA_MODELS.items():
        document = cast(dict[str, object], model.model_json_schema())
        document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        document["$id"] = f"https://github.com/AbdelStark/soufflerie/schemas/v1/{name}.json"
        _apply_registry_schema_invariants(name, document)
        result[name] = document
    return result


def rendered_observability_schema_documents() -> dict[str, str]:
    """Render observability schemas in the checked-in canonical format."""

    return {
        f"{name}.json": json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        for name, document in observability_schema_documents().items()
    }


__all__ = [
    "EMPTY_ARTIFACT_IDENTITIES",
    "EVENT_REGISTRY",
    "EXTERNAL_PATH",
    "METRIC_REGISTRY",
    "NOOP_EVENT_SINK",
    "NOOP_METRIC_SINK",
    "OBSERVABILITY_SCHEMA_MODELS",
    "REDACTED",
    "ArtifactIdentities",
    "Component",
    "Event",
    "EventDefinition",
    "EventFieldValue",
    "EventLevel",
    "EventName",
    "EventSink",
    "InMemoryEventSink",
    "InMemoryMetricSink",
    "JsonlEventSink",
    "JsonlMetricSink",
    "MetricDefinition",
    "MetricName",
    "MetricRecord",
    "MetricScope",
    "MetricSink",
    "NoOpEventSink",
    "NoOpMetricSink",
    "OperationRecorder",
    "OperationTimer",
    "Redactor",
    "TimerSegment",
    "TimingSummary",
    "event_json",
    "metric_json",
    "new_correlation_id",
    "observability_schema_documents",
    "operation_context",
    "redact",
    "rendered_observability_schema_documents",
    "safe_exception_fields",
]
