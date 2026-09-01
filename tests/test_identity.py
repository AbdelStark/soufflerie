from __future__ import annotations

import json

import numpy as np
import pytest

from soufflerie.errors import ArtifactIntegrityError
from soufflerie.schemas import (
    ArtifactRef,
    CaseConfig,
    ShapeParams,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
    verify_sha256,
)


def _case() -> CaseConfig:
    return CaseConfig(
        shape=ShapeParams(aspect_ratio=0.5, rotation_deg=10.0, scale=1.0),
        reynolds=100.0,
        nx=512,
        ny=256,
        steps=20_000,
        warmup_steps=10_000,
        inlet_velocity_lu=0.05,
        seed=42,
    )


def test_canonical_json_is_sorted_compact_utf8_and_normalizes_negative_zero() -> None:
    first = {"z": -0.0, "a": "soufflerie", "nested": {"b": 2, "a": 1}}
    second = {"nested": {"a": 1, "b": 2}, "a": "soufflerie", "z": 0.0}
    expected = '{"a":"soufflerie","nested":{"a":1,"b":2},"z":0.0}'
    assert canonical_json(first) == expected
    assert canonical_json(second) == expected
    assert canonical_json_bytes(first) == expected.encode("utf-8")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        canonical_json({"value": value})


def test_case_identity_is_stable_across_json_key_order() -> None:
    case = _case()
    shuffled = json.dumps(
        {
            "seed": case.seed,
            "inlet_velocity_lu": case.inlet_velocity_lu,
            "warmup_steps": case.warmup_steps,
            "steps": case.steps,
            "ny": case.ny,
            "nx": case.nx,
            "reynolds": case.reynolds,
            "shape": {
                "scale": case.shape.scale,
                "rotation_deg": case.shape.rotation_deg,
                "aspect_ratio": case.shape.aspect_ratio,
            },
            "schema_version": case.schema_version,
        }
    )
    reparsed = CaseConfig.model_validate_json(shuffled)
    assert canonical_sha256(reparsed) == case.sha256
    assert len(case.sha256) == 64
    assert case.case_id == case.sha256[:20]


def test_artifact_identity_is_independent_of_storage_location() -> None:
    content = b"immutable artifact bytes"
    local = ArtifactRef.from_bytes(
        artifact_type="run",
        uri="runs/local/metadata.json",
        content=content,
    )
    remote = ArtifactRef.from_bytes(
        artifact_type="run",
        uri="runs/remote/metadata.json",
        content=content,
    )
    assert local.uri != remote.uri
    assert local.sha256 == remote.sha256 == sha256_bytes(content)
    assert local.artifact_id == remote.artifact_id

    with pytest.raises(ValueError, match="artifact-root-relative"):
        ArtifactRef.from_bytes(
            artifact_type="run", uri="https://secret.example/run", content=content
        )


def test_full_sha256_verification_is_authoritative() -> None:
    content = b"artifact"
    digest = sha256_bytes(content)
    verify_sha256(content, digest)

    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        verify_sha256(content + b"-tampered", digest)
    with pytest.raises(ArtifactIntegrityError, match="lowercase SHA-256"):
        verify_sha256(content, digest[:20])


def test_arrays_are_not_implicitly_canonicalized() -> None:
    with pytest.raises(TypeError, match="array bytes and descriptors"):
        canonical_json(np.zeros((2, 2), dtype=np.float32))
