from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

import soufflerie.datagen.design as design_module
from soufflerie.config import SweepConfig, load_config
from soufflerie.datagen.design import (
    SPLIT_SALT,
    DesignPoint,
    UnsplitDesignPoint,
    assign_splits,
    sample_design,
)
from soufflerie.schemas import CaseConfig, canonical_json_bytes

CONFIG_PATH = Path("configs/sweeps/mvp-v1.yaml")


def _points(monkeypatch: pytest.MonkeyPatch) -> tuple[DesignPoint, ...]:
    def accept_lattice(case: CaseConfig, *, sample_interval: int) -> None:
        assert sample_interval == 10

    monkeypatch.setattr(design_module, "derive_lattice", accept_lattice)
    monkeypatch.setattr(design_module, "validate_geometry", lambda *args, **kwargs: None)
    return sample_design(load_config(CONFIG_PATH, SweepConfig))


def _unsplit(points: tuple[DesignPoint, ...]) -> tuple[UnsplitDesignPoint, ...]:
    return tuple(
        UnsplitDesignPoint(
            index=point.index,
            aspect_ratio=point.aspect_ratio,
            rotation_deg=point.rotation_deg,
            scale=point.scale,
            reynolds=point.reynolds,
            design_id=point.design_id,
        )
        for point in points
    )


def test_split_membership_is_exact_disjoint_and_order_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _points(monkeypatch)
    unsplit = _unsplit(points)
    forward = assign_splits(unsplit, 20260901)
    reverse = assign_splits(tuple(reversed(unsplit)), 20260901)

    assert Counter(point.split for point in forward) == Counter(
        {"train": 600, "validation": 200, "test": 200}
    )
    assert {point.design_id: point.split for point in forward} == {
        point.design_id: point.split for point in reverse
    }
    memberships = {
        split: {point.design_id for point in forward if point.split == split}
        for split in ("train", "validation", "test")
    }
    assert memberships["train"].isdisjoint(memberships["validation"])
    assert memberships["train"].isdisjoint(memberships["test"])
    assert memberships["validation"].isdisjoint(memberships["test"])


def test_split_rank_uses_full_hash_seed_bytes_and_canonical_design_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _points(monkeypatch)
    unsplit = _unsplit(points)
    assigned = assign_splits(unsplit, 20260901)
    seed_bytes = (20260901).to_bytes(8, byteorder="big", signed=False)
    ranked = sorted(
        (
            hashlib.sha256(
                SPLIT_SALT + seed_bytes + canonical_json_bytes(point.canonical_design())
            ).hexdigest(),
            point.design_id,
        )
        for point in unsplit
    )
    membership = {point.design_id: point.split for point in assigned}

    assert all(membership[design_id] == "train" for _, design_id in ranked[:600])
    assert all(membership[design_id] == "validation" for _, design_id in ranked[600:800])
    assert all(membership[design_id] == "test" for _, design_id in ranked[800:])


def test_split_assignment_rejects_incomplete_duplicate_or_already_split_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _points(monkeypatch)
    unsplit = _unsplit(points)

    with pytest.raises(ValueError, match="exactly 1000"):
        assign_splits(unsplit[:-1], 20260901)
    with pytest.raises(ValueError, match="indices"):
        assign_splits((*unsplit[:-1], unsplit[-2]), 20260901)
    with pytest.raises(TypeError, match="UnsplitDesignPoint"):
        assign_splits(points, 20260901)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        assign_splits(unsplit, -1)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        assign_splits(unsplit, True)
