from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import soufflerie.datagen.design as design_module
from soufflerie.config import SweepConfig, load_config
from soufflerie.datagen.design import (
    CANDIDATE_COUNT,
    DesignSummary,
    case_config_for_point,
    generate_design_summary,
    render_design_summary,
    sample_design,
)
from soufflerie.schemas import CaseConfig

CONFIG_PATH = Path("configs/sweeps/mvp-v1.yaml")
REPORT_PATH = Path("reports/data/design-mvp-v1.json")


def _config() -> SweepConfig:
    return load_config(CONFIG_PATH, SweepConfig)


def _disable_dense_preflight(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
    lattice: list[str] = []
    geometry: list[str] = []

    def record_lattice(case: CaseConfig, *, sample_interval: int) -> None:
        assert sample_interval == 10
        lattice.append(case.case_id)

    def record_geometry(*args: object, **kwargs: object) -> None:
        case_shape = args[0]
        grid = args[1]
        geometry.append(f"{case_shape!r}:{grid!r}")

    monkeypatch.setattr(design_module, "derive_lattice", record_lattice)
    monkeypatch.setattr(design_module, "validate_geometry", record_geometry)
    return lattice, geometry


def _legacy_random_state() -> tuple[object, ...]:
    state = np.random.get_state()
    return (state[0], state[1].copy(), state[2], state[3], state[4])  # type: ignore[index]


def _assert_legacy_random_state_equal(
    left: tuple[object, ...],
    right: tuple[object, ...],
) -> None:
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_canonical_design_is_reproducible_stratified_and_preflighted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lattice, geometry = _disable_dense_preflight(monkeypatch)
    before = _legacy_random_state()

    first = sample_design(_config())
    middle = _legacy_random_state()
    second = sample_design(_config())
    after = _legacy_random_state()

    assert first == second
    assert len(first) == 1_000
    assert len(lattice) == 2_000
    assert len(geometry) == 2_000
    assert len(set(point.design_id for point in first)) == 1_000
    assert Counter(point.split for point in first) == Counter(
        {"train": 600, "validation": 200, "test": 200}
    )
    _assert_legacy_random_state_equal(before, middle)
    _assert_legacy_random_state_equal(before, after)

    config = _config()
    physical = (
        ("aspect_ratio", config.aspect_ratio.minimum, config.aspect_ratio.maximum),
        ("rotation_deg", config.rotation_deg.minimum, config.rotation_deg.maximum),
        ("scale", config.scale.minimum, config.scale.maximum),
        ("reynolds", config.reynolds.minimum, config.reynolds.maximum),
    )
    for name, minimum, maximum in physical:
        values = np.asarray([getattr(point, name) for point in first], dtype=np.float64)
        normalized = (values - minimum) / (maximum - minimum)
        strata = np.floor(normalized * 1_000).astype(np.int64)
        np.testing.assert_array_equal(np.sort(strata), np.arange(1_000, dtype=np.int64))
        assert np.all(normalized >= 0.0)
        assert np.all(normalized < 1.0)


def test_candidate_selection_is_maximin_with_stable_index_tie_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dense_preflight(monkeypatch)
    summary = generate_design_summary(_config())

    assert summary.candidate_count == CANDIDATE_COUNT
    assert summary.selected_candidate_index == 10
    assert summary.selected_minimum_distance == pytest.approx(0.03666928448214767)
    assert summary.candidate_minimum_distances[3] == pytest.approx(0.035259819967303506)
    assert summary.candidate_child_seeds[0] == 11661666064119480592
    assert summary.candidate_child_seeds[-1] == 7469557046758967716
    assert design_module._select_maximin_index((1.0,) * CANDIDATE_COUNT) == 0


def test_summary_freezes_canonical_digests_statistics_and_full_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lattice, geometry = _disable_dense_preflight(monkeypatch)
    summary = generate_design_summary(_config())

    assert (
        summary.config_sha256 == "04ffaf7dcb027482d82c921fa617a429872a06e5d0a92545e3cfaf724a011333"
    )
    assert (
        summary.design_sha256 == "352a060bdb7ef2ff3e9432d7eff4333d6d3c9bd9aca33c6f29f6f56a307250c1"
    )
    assert (
        summary.split_sha256 == "406a35bdef46cd10e77efc7fc6b301ffbb089783c2454dc86c8308a093e9d027"
    )
    assert (
        summary.candidate_seeds_sha256
        == "3a742d5398c7134219807743428650bd1a4a6b5750c10f6839a893a2e0ae8b01"
    )
    assert summary.lattice_preflight_passed == 1_000
    assert summary.geometry_preflight_passed == 1_000
    assert summary.all_preflights_passed
    assert len(lattice) == len(geometry) == 1_000
    assert summary.dimension_statistics.aspect_ratio.minimum >= 0.5
    assert summary.dimension_statistics.aspect_ratio.maximum < 1.0
    assert summary.nearest_neighbor_distance.minimum == pytest.approx(
        summary.selected_minimum_distance
    )
    np.testing.assert_allclose(np.diag(summary.pairwise_correlation), 1.0, atol=1e-14)


def test_summary_rejects_tampered_selection_and_seed_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dense_preflight(monkeypatch)
    payload = generate_design_summary(_config()).model_dump(mode="python")

    wrong_selection = {**payload, "selected_candidate_index": 3}
    with pytest.raises(ValidationError, match="selected minimum distance"):
        DesignSummary.model_validate(wrong_selection)

    seeds = list(payload["candidate_child_seeds"])
    seeds[0] += 1
    with pytest.raises(ValidationError, match="candidate seed digest"):
        DesignSummary.model_validate({**payload, "candidate_child_seeds": tuple(seeds)})


def test_checked_in_summary_is_strict_and_canonically_rendered() -> None:
    content = REPORT_PATH.read_text(encoding="utf-8")
    summary = DesignSummary.model_validate_json(content)

    assert render_design_summary(summary) == content
    assert summary.config_sha256 == _config().config_digest
    assert (
        summary.design_sha256 == "352a060bdb7ef2ff3e9432d7eff4333d6d3c9bd9aca33c6f29f6f56a307250c1"
    )
    assert (
        summary.split_sha256 == "406a35bdef46cd10e77efc7fc6b301ffbb089783c2454dc86c8308a093e9d027"
    )
    assert summary.all_preflights_passed


def test_case_config_binds_numerical_controls_without_changing_design_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dense_preflight(monkeypatch)
    config = _config()
    point = sample_design(config)[0]
    case = case_config_for_point(point, config)

    assert case.shape == point.shape
    assert case.reynolds == point.reynolds
    assert case.grid == config.grid
    assert case.steps == config.run.steps
    assert case.warmup_steps == config.run.warmup_steps
    assert case.seed == config.seed
    assert case.case_id != point.design_id
