"""Bounded privacy-preserving admission for the public service boundary."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import math
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Literal, TypeAlias

from soufflerie.config import ServiceConfig
from soufflerie.errors import (
    BudgetExhaustedError,
    ConfigurationError,
    InternalInvariantError,
    RateLimitError,
    SolveDisabledError,
)
from soufflerie.service.contracts import ReadinessProbe

MAX_CLIENT_STATES = 4_096
CLIENT_HMAC_KEY_BYTES = 32
MAX_FORWARDED_ADDRESSES = 16
MAX_FORWARDED_HEADER_CHARACTERS = 1_024

WallClock = Callable[[], datetime]
MonotonicClock = Callable[[], float]
SecretFactory = Callable[[int], bytes]
IpNetwork: TypeAlias = ipaddress.IPv4Network | ipaddress.IPv6Network
ReadinessReason: TypeAlias = Literal[
    "artifact_invalid",
    "runtime_unavailable",
    "solve_disabled",
    "solve_budget_exhausted",
    "validation_red",
    "ready",
]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_networks(values: Sequence[str]) -> tuple[IpNetwork, ...]:
    if len(values) > MAX_FORWARDED_ADDRESSES:
        raise ConfigurationError("trusted proxy list exceeds the configured bound")
    networks: list[IpNetwork] = []
    for value in values:
        try:
            network = ipaddress.ip_network(value, strict=False)
        except ValueError as error:
            raise ConfigurationError(
                "trusted proxy entries must be IP addresses or CIDRs"
            ) from error
        if network.prefixlen == 0:
            raise ConfigurationError("trusted proxy networks must not cover every address")
        networks.append(network)
    return tuple(networks)


def _parse_override_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ConfigurationError("SOUFFLERIE_SOLVE_ENABLED must be exactly true or false")


def _parse_gpu_override(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ConfigurationError(
            "SOUFFLERIE_SOLVE_GPU_SECONDS_PER_DAY must be a finite nonnegative number"
        ) from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ConfigurationError(
            "SOUFFLERIE_SOLVE_GPU_SECONDS_PER_DAY must be a finite nonnegative number"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class AdmissionSettings:
    """Operational settings whose secret value never enters public configuration."""

    client_hmac_key: bytes = field(repr=False)
    trusted_proxy_networks: tuple[str, ...] = ()
    solve_enabled_override: bool | None = None
    solve_gpu_seconds_per_day_override: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.client_hmac_key, bytes):
            raise ConfigurationError("client HMAC key must be bytes")
        if len(self.client_hmac_key) < CLIENT_HMAC_KEY_BYTES:
            raise ConfigurationError(
                f"client HMAC key must contain at least {CLIENT_HMAC_KEY_BYTES} bytes"
            )
        if not isinstance(self.trusted_proxy_networks, tuple):
            raise ConfigurationError("trusted proxy networks must be an immutable tuple")
        _parse_networks(self.trusted_proxy_networks)
        if self.solve_enabled_override is not None and not isinstance(
            self.solve_enabled_override, bool
        ):
            raise ConfigurationError("solve enabled override must be a boolean")
        override = self.solve_gpu_seconds_per_day_override
        if override is not None and (
            isinstance(override, bool) or not math.isfinite(override) or override < 0.0
        ):
            raise ConfigurationError("solve GPU-second override must be finite and nonnegative")


def load_admission_settings(
    environ: Mapping[str, str],
    *,
    secret_factory: SecretFactory = secrets.token_bytes,
) -> AdmissionSettings:
    """Read only the named operational controls from an explicit environment mapping."""

    encoded_key = environ.get("SOUFFLERIE_CLIENT_HMAC_KEY")
    if encoded_key is None:
        key = secret_factory(CLIENT_HMAC_KEY_BYTES)
    else:
        if len(encoded_key) != CLIENT_HMAC_KEY_BYTES * 2:
            raise ConfigurationError(
                "SOUFFLERIE_CLIENT_HMAC_KEY must be exactly 64 hexadecimal characters"
            )
        try:
            key = bytes.fromhex(encoded_key)
        except ValueError as error:
            raise ConfigurationError(
                "SOUFFLERIE_CLIENT_HMAC_KEY must be exactly 64 hexadecimal characters"
            ) from error
        if encoded_key != encoded_key.lower():
            raise ConfigurationError("SOUFFLERIE_CLIENT_HMAC_KEY must use lowercase hexadecimal")

    proxy_value = environ.get("SOUFFLERIE_TRUSTED_PROXIES", "")
    proxy_parts = tuple(item.strip() for item in proxy_value.split(",") if item.strip())
    networks = _parse_networks(proxy_parts)
    canonical_networks = tuple(str(network) for network in networks)

    enabled_value = environ.get("SOUFFLERIE_SOLVE_ENABLED")
    enabled = None if enabled_value is None else _parse_override_bool(enabled_value)
    gpu_value = environ.get("SOUFFLERIE_SOLVE_GPU_SECONDS_PER_DAY")
    gpu_override = None if gpu_value is None else _parse_gpu_override(gpu_value)
    return AdmissionSettings(
        client_hmac_key=key,
        trusted_proxy_networks=canonical_networks,
        solve_enabled_override=enabled,
        solve_gpu_seconds_per_day_override=gpu_override,
    )


@dataclass(frozen=True, slots=True, repr=False)
class _ClientKey:
    digest: bytes


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    """Aggregate-only operational state safe to inspect without client identifiers."""

    prediction_client_states: int
    solve_client_states: int
    solves_admitted_today: int
    gpu_seconds_reserved_today: float
    solve_available: bool


@dataclass(frozen=True, slots=True)
class ServiceReadiness:
    """Internal prediction/solve decision with the fixed failure precedence."""

    prediction_ready: bool
    solve_ready: bool
    reason: ReadinessReason


def evaluate_service_readiness(
    config: ServiceConfig,
    probe: ReadinessProbe,
    *,
    solve_enabled: bool,
    solve_budget_available: bool,
) -> ServiceReadiness:
    """Evaluate readiness without allowing validation red to hide availability."""

    identities_match = (
        probe.model_id == config.model_id
        and probe.model_dataset_id == config.dataset_id
        and probe.report_id == config.report_id
        and probe.report_model_id == config.model_id
        and probe.report_dataset_id == config.dataset_id
    )
    if not identities_match or not (
        probe.model_integrity_verified and probe.report_integrity_verified
    ):
        return ServiceReadiness(False, False, "artifact_invalid")
    if not probe.device_available or not probe.warmup_complete:
        return ServiceReadiness(False, False, "runtime_unavailable")
    if not config.solve_enabled or not solve_enabled:
        return ServiceReadiness(True, False, "solve_disabled")
    if not solve_budget_available:
        return ServiceReadiness(True, False, "solve_budget_exhausted")
    if probe.validation_status == "red":
        return ServiceReadiness(True, True, "validation_red")
    return ServiceReadiness(True, True, "ready")


class AdmissionController:
    """Thread-safe process-local rate, proxy, solve-count, and GPU-budget guard."""

    def __init__(
        self,
        *,
        config: ServiceConfig,
        settings: AdmissionSettings,
        now: WallClock = _utc_now,
        monotonic: MonotonicClock = time.monotonic,
        max_client_states: int = MAX_CLIENT_STATES,
    ) -> None:
        if max_client_states < 1 or max_client_states > MAX_CLIENT_STATES:
            raise ConfigurationError(f"max_client_states must be in [1, {MAX_CLIENT_STATES}]")
        gpu_override = settings.solve_gpu_seconds_per_day_override
        if gpu_override is not None and gpu_override > config.solve_gpu_seconds_per_day:
            raise ConfigurationError(
                "solve GPU-second environment override cannot exceed service configuration"
            )
        self._config = config
        self._hmac_key = settings.client_hmac_key
        self._trusted_proxies = _parse_networks(settings.trusted_proxy_networks)
        self._solve_enabled = config.solve_enabled and settings.solve_enabled_override is not False
        self._gpu_seconds_limit = (
            config.solve_gpu_seconds_per_day if gpu_override is None else gpu_override
        )
        self._now = now
        self._monotonic = monotonic
        self._max_client_states = max_client_states
        self._prediction_buckets: dict[_ClientKey, _Bucket] = {}
        self._solve_buckets: dict[_ClientKey, _Bucket] = {}
        self._budget_day: date | None = None
        self._solves_today = 0
        self._gpu_seconds_reserved_today = 0.0
        self._last_monotonic: float | None = None
        self._lock = threading.Lock()

    @property
    def solve_enabled(self) -> bool:
        return self._solve_enabled

    def admit_prediction(
        self,
        *,
        peer_address: str | None,
        forwarded_for: Sequence[str] = (),
    ) -> None:
        key = self._client_key(peer_address=peer_address, forwarded_for=forwarded_for)
        with self._lock:
            instant = self._monotonic_now()
            self._consume(
                self._prediction_buckets,
                key,
                capacity=self._config.predictions_per_minute_client,
                window_seconds=60.0,
                instant=instant,
            )

    def admit_solve(
        self,
        *,
        peer_address: str | None,
        forwarded_for: Sequence[str] = (),
    ) -> None:
        key = self._client_key(peer_address=peer_address, forwarded_for=forwarded_for)
        with self._lock:
            if not self._solve_enabled:
                raise SolveDisabledError("reference solve admission is disabled")
            wall = self._aware_now()
            self._reset_daily_budget(wall.date())
            retry_after = self._seconds_until_next_day(wall)
            if self._solves_today >= self._config.solves_per_day_global:
                raise BudgetExhaustedError(
                    "daily solve count is exhausted", retry_after_seconds=retry_after
                )
            reservation = float(self._config.solve_timeout_seconds)
            if self._gpu_seconds_reserved_today + reservation > self._gpu_seconds_limit:
                raise BudgetExhaustedError(
                    "daily solve GPU budget is exhausted", retry_after_seconds=retry_after
                )
            instant = self._monotonic_now()
            self._consume(
                self._solve_buckets,
                key,
                capacity=self._config.solves_per_hour_client,
                window_seconds=3_600.0,
                instant=instant,
            )
            self._solves_today += 1
            self._gpu_seconds_reserved_today += reservation

    def snapshot(self) -> AdmissionSnapshot:
        with self._lock:
            wall = self._aware_now()
            self._reset_daily_budget(wall.date())
            instant = self._monotonic_now()
            self._prune(self._prediction_buckets, instant=instant)
            self._prune(self._solve_buckets, instant=instant)
            return AdmissionSnapshot(
                prediction_client_states=len(self._prediction_buckets),
                solve_client_states=len(self._solve_buckets),
                solves_admitted_today=self._solves_today,
                gpu_seconds_reserved_today=self._gpu_seconds_reserved_today,
                solve_available=self._solve_available_locked(),
            )

    def readiness(self, probe: ReadinessProbe) -> ServiceReadiness:
        with self._lock:
            self._reset_daily_budget(self._aware_now().date())
            return evaluate_service_readiness(
                self._config,
                probe,
                solve_enabled=self._solve_enabled,
                solve_budget_available=self._solve_available_locked(),
            )

    def _consume(
        self,
        buckets: dict[_ClientKey, _Bucket],
        key: _ClientKey,
        *,
        capacity: int,
        window_seconds: float,
        instant: float,
    ) -> None:
        self._prune(buckets, instant=instant)
        if capacity <= 0:
            raise RateLimitError(
                "request rate is disabled", retry_after_seconds=math.ceil(window_seconds)
            )
        bucket = buckets.get(key)
        if bucket is None:
            if len(buckets) >= self._max_client_states:
                retry_after = max(
                    1,
                    math.ceil(min(item.expires_at for item in buckets.values()) - instant),
                )
                raise RateLimitError(
                    "rate-limit state capacity is exhausted",
                    retry_after_seconds=retry_after,
                )
            bucket = _Bucket(
                tokens=float(capacity),
                updated_at=instant,
                expires_at=instant + window_seconds,
            )
            buckets[key] = bucket
        else:
            elapsed = instant - bucket.updated_at
            bucket.tokens = min(
                float(capacity), bucket.tokens + elapsed * float(capacity) / window_seconds
            )
            bucket.updated_at = instant
            bucket.expires_at = instant + window_seconds
        if bucket.tokens < 1.0:
            seconds = (1.0 - bucket.tokens) * window_seconds / float(capacity)
            raise RateLimitError(
                "request rate limit is exhausted",
                retry_after_seconds=max(1, math.ceil(seconds)),
            )
        bucket.tokens -= 1.0

    def _client_key(
        self,
        *,
        peer_address: str | None,
        forwarded_for: Sequence[str],
    ) -> _ClientKey:
        address = self._effective_address(peer_address=peer_address, forwarded_for=forwarded_for)
        return _ClientKey(hmac.new(self._hmac_key, address.encode(), hashlib.sha256).digest())

    def _effective_address(
        self,
        *,
        peer_address: str | None,
        forwarded_for: Sequence[str],
    ) -> str:
        peer = self._normalized_address(peer_address)
        if not self._is_trusted(peer) or len(forwarded_for) != 1:
            return peer
        header = forwarded_for[0]
        if len(header) > MAX_FORWARDED_HEADER_CHARACTERS:
            return peer
        parts = tuple(item.strip() for item in header.split(","))
        if not parts or len(parts) > MAX_FORWARDED_ADDRESSES or any(not item for item in parts):
            return peer
        try:
            chain = tuple(self._normalized_address(item, allow_unknown=False) for item in parts)
        except ValueError:
            return peer
        complete_chain = (*chain, peer)
        index = len(complete_chain) - 1
        while index >= 0 and self._is_trusted(complete_chain[index]):
            index -= 1
        return complete_chain[max(0, index)]

    @staticmethod
    def _normalized_address(value: str | None, *, allow_unknown: bool = True) -> str:
        if value is None:
            if allow_unknown:
                return "unknown"
            raise ValueError("missing forwarded address")
        try:
            parsed = ipaddress.ip_address(value)
        except ValueError:
            if allow_unknown:
                return "unknown"
            raise
        if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
            return str(parsed.ipv4_mapped)
        return parsed.compressed

    def _is_trusted(self, address: str) -> bool:
        if address == "unknown":
            return False
        parsed = ipaddress.ip_address(address)
        return any(
            parsed.version == network.version and parsed in network
            for network in self._trusted_proxies
        )

    def _monotonic_now(self) -> float:
        instant = self._monotonic()
        if not math.isfinite(instant):
            raise InternalInvariantError("admission monotonic clock must be finite")
        if self._last_monotonic is not None and instant < self._last_monotonic:
            raise InternalInvariantError("admission monotonic clock regressed")
        self._last_monotonic = instant
        return instant

    def _aware_now(self) -> datetime:
        instant = self._now()
        if not isinstance(instant, datetime) or instant.utcoffset() is None:
            raise InternalInvariantError("admission wall clock must be timezone-aware")
        return instant.astimezone(UTC)

    @staticmethod
    def _prune(buckets: dict[_ClientKey, _Bucket], *, instant: float) -> None:
        expired = [key for key, bucket in buckets.items() if instant >= bucket.expires_at]
        for key in expired:
            del buckets[key]

    def _reset_daily_budget(self, today: date) -> None:
        if self._budget_day == today:
            return
        if self._budget_day is not None and today < self._budget_day:
            raise InternalInvariantError("admission wall clock regressed across a UTC day")
        self._budget_day = today
        self._solves_today = 0
        self._gpu_seconds_reserved_today = 0.0

    def _solve_available_locked(self) -> bool:
        return (
            self._solve_enabled
            and self._solves_today < self._config.solves_per_day_global
            and self._gpu_seconds_reserved_today + self._config.solve_timeout_seconds
            <= self._gpu_seconds_limit
        )

    @staticmethod
    def _seconds_until_next_day(now: datetime) -> int:
        tomorrow = now.date() + timedelta(days=1)
        boundary = datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC)
        return max(1, math.ceil((boundary - now).total_seconds()))


__all__ = [
    "CLIENT_HMAC_KEY_BYTES",
    "MAX_CLIENT_STATES",
    "MAX_FORWARDED_ADDRESSES",
    "MAX_FORWARDED_HEADER_CHARACTERS",
    "AdmissionController",
    "AdmissionSettings",
    "AdmissionSnapshot",
    "ServiceReadiness",
    "evaluate_service_readiness",
    "load_admission_settings",
]
