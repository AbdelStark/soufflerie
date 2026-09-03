from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from soufflerie.config import ServiceConfig
from soufflerie.errors import BudgetExhaustedError, ConfigurationError, RateLimitError
from soufflerie.service.admission import (
    MAX_CLIENT_STATES,
    AdmissionController,
    AdmissionSettings,
    load_admission_settings,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
SECRET = b"s" * 32


class FakeClock:
    def __init__(self) -> None:
        self.wall = NOW
        self.elapsed = 1_000.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.elapsed

    def advance(self, seconds: float) -> None:
        self.wall += timedelta(seconds=seconds)
        self.elapsed += seconds


def _config(
    *,
    predictions_per_minute: int = 60,
    solves_per_hour: int = 2,
    solves_per_day: int = 20,
    gpu_seconds_per_day: float = 3_600.0,
) -> ServiceConfig:
    return ServiceConfig(
        model_id="1" * 20,
        dataset_id="2" * 20,
        report_id="3" * 20,
        solve_enabled=True,
        solve_concurrency=2,
        solve_queue_capacity=8,
        solve_timeout_seconds=180,
        predictions_per_minute_client=predictions_per_minute,
        solves_per_hour_client=solves_per_hour,
        solves_per_day_global=solves_per_day,
        solve_gpu_seconds_per_day=gpu_seconds_per_day,
    )


def _controller(
    config: ServiceConfig,
    clock: FakeClock,
    *,
    settings: AdmissionSettings | None = None,
    max_client_states: int = MAX_CLIENT_STATES,
) -> AdmissionController:
    return AdmissionController(
        config=config,
        settings=settings or AdmissionSettings(client_hmac_key=SECRET),
        now=clock.now,
        monotonic=clock.monotonic,
        max_client_states=max_client_states,
    )


def test_public_service_limits_cannot_be_raised_above_v01_policy() -> None:
    config = _config()
    assert (
        config.predictions_per_minute_client,
        config.solves_per_hour_client,
        config.solves_per_day_global,
        config.solve_concurrency,
        config.solve_queue_capacity,
        config.solve_timeout_seconds,
        config.solve_gpu_seconds_per_day,
    ) == (60, 2, 20, 2, 8, 180, 3_600.0)

    for changes in (
        {"predictions_per_minute_client": 61},
        {"solves_per_hour_client": 3},
        {"solves_per_day_global": 21},
        {"solve_concurrency": 3},
        {"solve_queue_capacity": 9},
        {"solve_timeout_seconds": 181},
        {"solve_gpu_seconds_per_day": 3_600.001},
    ):
        with pytest.raises(ValidationError):
            ServiceConfig.model_validate({**config.model_dump(), **changes})


def test_prediction_bucket_refills_exactly_and_inactive_state_expires() -> None:
    clock = FakeClock()
    controller = _controller(_config(predictions_per_minute=2), clock)

    controller.admit_prediction(peer_address="192.0.2.10")
    controller.admit_prediction(peer_address="192.0.2.10")
    with pytest.raises(RateLimitError) as captured:
        controller.admit_prediction(peer_address="192.0.2.10")
    assert captured.value.retry_after_seconds == 30

    clock.advance(29.999)
    with pytest.raises(RateLimitError) as captured:
        controller.admit_prediction(peer_address="192.0.2.10")
    assert captured.value.retry_after_seconds == 1
    clock.advance(0.001)
    controller.admit_prediction(peer_address="192.0.2.10")

    clock.advance(60.001)
    controller.admit_prediction(peer_address="192.0.2.11")
    assert controller.snapshot().prediction_client_states == 1


def test_solve_client_global_and_gpu_reservation_limits_hold_exactly() -> None:
    clock = FakeClock()
    controller = _controller(_config(), clock)

    controller.admit_solve(peer_address="192.0.2.1")
    controller.admit_solve(peer_address="192.0.2.1")
    with pytest.raises(RateLimitError) as captured:
        controller.admit_solve(peer_address="192.0.2.1")
    assert captured.value.retry_after_seconds == 1_800

    for index in range(2, 20):
        controller.admit_solve(peer_address=f"192.0.2.{index}")
    snapshot = controller.snapshot()
    assert snapshot.solves_admitted_today == 20
    assert snapshot.gpu_seconds_reserved_today == 3_600.0
    assert snapshot.solve_available is False
    with pytest.raises(BudgetExhaustedError) as budget_captured:
        controller.admit_solve(peer_address="198.51.100.1")
    assert budget_captured.value.retry_after_seconds == 43_200

    clock.advance(43_200)
    controller.admit_solve(peer_address="198.51.100.1")
    assert controller.snapshot().solves_admitted_today == 1


def test_lower_gpu_ceiling_closes_admission_without_consuming_client_rate() -> None:
    clock = FakeClock()
    controller = _controller(_config(gpu_seconds_per_day=180.0), clock)
    controller.admit_solve(peer_address="192.0.2.1")

    for address in ("192.0.2.2", "192.0.2.2"):
        with pytest.raises(BudgetExhaustedError):
            controller.admit_solve(peer_address=address)
    assert controller.snapshot().solve_client_states == 1


def test_forwarded_chain_is_used_only_for_an_allowlisted_immediate_proxy() -> None:
    clock = FakeClock()
    config = _config(predictions_per_minute=1)
    untrusted = _controller(config, clock)
    untrusted.admit_prediction(peer_address="203.0.113.8", forwarded_for=("192.0.2.1",))
    with pytest.raises(RateLimitError):
        untrusted.admit_prediction(peer_address="203.0.113.8", forwarded_for=("192.0.2.2",))

    trusted = _controller(
        config,
        clock,
        settings=AdmissionSettings(
            client_hmac_key=SECRET,
            trusted_proxy_networks=("10.0.0.0/8", "2001:db8:ffff::/48"),
        ),
    )
    trusted.admit_prediction(peer_address="10.0.0.5", forwarded_for=("192.0.2.1",))
    trusted.admit_prediction(peer_address="10.0.0.5", forwarded_for=("192.0.2.2",))
    with pytest.raises(RateLimitError):
        trusted.admit_prediction(
            peer_address="10.0.0.5",
            forwarded_for=("192.0.2.1, 10.0.0.4",),
        )


def test_client_state_is_bounded_and_public_snapshot_contains_no_identifier() -> None:
    clock = FakeClock()
    controller = _controller(_config(predictions_per_minute=1), clock, max_client_states=2)
    controller.admit_prediction(peer_address="192.0.2.1")
    controller.admit_prediction(peer_address="192.0.2.2")
    with pytest.raises(RateLimitError, match="state capacity"):
        controller.admit_prediction(peer_address="192.0.2.3")

    snapshot = asdict(controller.snapshot())
    serialized = repr(snapshot)
    assert set(snapshot) == {
        "prediction_client_states",
        "solve_client_states",
        "solves_admitted_today",
        "gpu_seconds_reserved_today",
        "solve_available",
    }
    assert "192.0.2" not in serialized
    assert SECRET.hex() not in serialized


def test_environment_settings_are_strict_and_can_only_lower_gpu_budget() -> None:
    settings = load_admission_settings(
        {
            "SOUFFLERIE_CLIENT_HMAC_KEY": "ab" * 32,
            "SOUFFLERIE_TRUSTED_PROXIES": "10.0.0.0/8,2001:db8::/32",
            "SOUFFLERIE_SOLVE_ENABLED": "false",
            "SOUFFLERIE_SOLVE_GPU_SECONDS_PER_DAY": "90",
        }
    )
    assert settings.client_hmac_key == bytes.fromhex("ab" * 32)
    assert settings.trusted_proxy_networks == ("10.0.0.0/8", "2001:db8::/32")
    assert settings.solve_enabled_override is False
    assert settings.solve_gpu_seconds_per_day_override == 90.0
    assert ("ab" * 32) not in repr(settings)

    ephemeral = load_admission_settings({}, secret_factory=lambda size: b"e" * size)
    assert ephemeral.client_hmac_key == b"e" * 32

    for environment in (
        {"SOUFFLERIE_CLIENT_HMAC_KEY": "short"},
        {"SOUFFLERIE_SOLVE_ENABLED": "FALSE"},
        {"SOUFFLERIE_SOLVE_GPU_SECONDS_PER_DAY": "nan"},
        {"SOUFFLERIE_TRUSTED_PROXIES": "*"},
        {"SOUFFLERIE_TRUSTED_PROXIES": "0.0.0.0/0"},
    ):
        with pytest.raises(ConfigurationError):
            load_admission_settings(environment)

    with pytest.raises(ConfigurationError, match="cannot exceed"):
        _controller(
            _config(gpu_seconds_per_day=90.0),
            FakeClock(),
            settings=AdmissionSettings(
                client_hmac_key=SECRET,
                solve_gpu_seconds_per_day_override=91.0,
            ),
        )
