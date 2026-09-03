from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from soufflerie.demo import PanelMetadata, RenderSpec, render_comparison, render_fields
from soufflerie.errors import ArtifactIntegrityError, DependencyUnavailableError
from soufflerie.schemas import FlowFields, sha256_bytes

GOLDEN_STANDALONE_CONTRACT_SHA256 = (
    "1f57ea4ea54b87ca8d5d5071c5426f4cd518a262c8f70482acf1507bcf6e5f02"
)


def _fields(*, scale: float = 1.0, constant: bool = False) -> FlowFields:
    y, x = np.mgrid[:24, :36]
    x_centered = x - 17.5
    y_centered = y - 11.5
    sdf = np.ascontiguousarray(
        np.sqrt((x_centered / 5.0) ** 2 + (y_centered / 7.0) ** 2) - 1.0,
        dtype=np.float32,
    )
    u: npt.NDArray[np.float32]
    v: npt.NDArray[np.float32]
    rho: npt.NDArray[np.float32]
    if constant:
        u = np.zeros((24, 36), dtype=np.float32)
        v = np.zeros((24, 36), dtype=np.float32)
        rho = np.ones((24, 36), dtype=np.float32)
    else:
        u = np.ascontiguousarray(scale * (0.03 + 0.0005 * x + 0.0002 * y), dtype=np.float32)
        v = np.ascontiguousarray(scale * (-0.01 + 0.0001 * x - 0.00015 * y), dtype=np.float32)
        rho = np.ascontiguousarray(1.0 + scale * (0.0002 * x - 0.0001 * y), dtype=np.float32)
    mask = np.ascontiguousarray(sdf <= 0.0, dtype=np.bool_)
    return FlowFields(u=u, v=v, rho=rho, sdf=sdf, obstacle_mask=mask)


def _raw_bytes(fields: FlowFields) -> tuple[bytes, ...]:
    return tuple(
        value.tobytes(order="C")
        for value in (fields.u, fields.v, fields.rho, fields.sdf, fields.obstacle_mask)
    )


def test_standalone_render_matches_golden_contract_and_is_byte_deterministic() -> None:
    fields = _fields()
    before = _raw_bytes(fields)
    spec = RenderSpec(provenance="Solver reference · case 0123456789abcdefabcd")

    first = render_fields(fields, spec)
    second = render_fields(fields, spec)

    assert _raw_bytes(fields) == before
    assert first.data == second.data
    assert first.sha256 == second.sha256 == sha256_bytes(first.data)
    assert first.contract_sha256 == second.contract_sha256
    assert first.contract_sha256 == GOLDEN_STANDALONE_CONTRACT_SHA256
    assert (first.width_px, first.height_px) == (1_200, 400)
    assert [panel.title for panel in first.panels] == [
        "Velocity magnitude",
        "Pressure proxy",
        "Vorticity",
    ]
    assert [panel.units for panel in first.panels] == [
        "lattice velocity",
        "(rho - 1) / 3",
        "inverse lattice unit",
    ]
    assert [panel.colormap for panel in first.panels] == [
        "viridis",
        "coolwarm",
        "RdBu_r",
    ]
    assert [(panel.minimum, panel.maximum) for panel in first.panels] == pytest.approx(
        [
            (0.0, 0.0524180589451903),
            (-0.0023333231608072915, 0.0023333231608072915),
            (-5.000119330361485e-05, 5.000119330361485e-05),
        ],
        abs=1e-15,
    )
    with Image.open(io.BytesIO(first.data)) as image:
        assert image.size == (1_200, 400)
        assert image.format == "PNG"
        assert image.info["Description"] == first.alt_text
        assert image.info["Title"] == spec.annotation
        assert image.info["Software"] == "Soufflerie 0.1.0"


def test_comparison_uses_reference_scales_and_preserves_every_raw_array() -> None:
    reference = _fields(scale=1.0)
    prediction = _fields(scale=1.8)
    reference_before = _raw_bytes(reference)
    prediction_before = _raw_bytes(prediction)

    artifact = render_comparison(
        prediction,
        reference,
        RenderSpec(height_px=900, provenance="Compared with solver reference"),
    )

    assert _raw_bytes(reference) == reference_before
    assert _raw_bytes(prediction) == prediction_before
    assert (artifact.width_px, artifact.height_px) == (1_200, 900)
    assert len(artifact.panels) == 9
    for index in range(3):
        reference_panel = artifact.panels[index]
        prediction_panel = artifact.panels[index + 3]
        assert reference_panel.role == "reference"
        assert prediction_panel.role == "prediction"
        assert prediction_panel.variable == reference_panel.variable
        assert prediction_panel.colormap == reference_panel.colormap
        assert prediction_panel.minimum == reference_panel.minimum
        assert prediction_panel.maximum == reference_panel.maximum

    velocity_error, pressure_error, vorticity_error = artifact.panels[6:]
    assert velocity_error.role == pressure_error.role == vorticity_error.role == "error"
    assert velocity_error.minimum == 0.0
    assert velocity_error.maximum > 0.0
    assert pressure_error.minimum == -pressure_error.maximum
    assert vorticity_error.minimum == -vorticity_error.maximum
    assert pressure_error.units == "signed proxy error"
    assert vorticity_error.units == "signed inverse lattice unit error"
    assert "share scales" in artifact.alt_text


def test_zero_and_constant_fields_receive_finite_non_degenerate_scales() -> None:
    fields = _fields(constant=True)

    artifact = render_fields(fields, RenderSpec())

    assert artifact.panels[0].minimum == 0.0
    assert artifact.panels[0].maximum == float(np.finfo(np.float32).eps)
    for panel in artifact.panels[1:]:
        assert panel.minimum == -float(np.finfo(np.float32).eps)
        assert panel.maximum == float(np.finfo(np.float32).eps)


def test_renderer_revalidates_mutated_nonfinite_input() -> None:
    fields = _fields()
    fields.u[0, 0] = np.nan

    with pytest.raises(ArtifactIntegrityError, match="must remain finite"):
        render_fields(fields, RenderSpec())


def test_comparison_rejects_incompatible_geometry() -> None:
    reference = _fields()
    prediction = _fields()
    prediction.obstacle_mask[11, 17] = False
    prediction.sdf[11, 17] = np.float32(1.0)

    with pytest.raises(ArtifactIntegrityError, match="obstacle masks differ"):
        render_comparison(prediction, reference, RenderSpec())


def test_render_spec_rejects_unbounded_or_ambiguous_style() -> None:
    invalid_specs: tuple[Callable[[], RenderSpec], ...] = (
        lambda: RenderSpec(schema_version=cast(Any, True)),
        lambda: RenderSpec(width_px=599),
        lambda: RenderSpec(width_px=2_400, height_px=1_601),
        lambda: RenderSpec(dpi=71),
        lambda: RenderSpec(colormap_velocity=cast(Any, "plasma")),
        lambda: RenderSpec(spacing_lu=float("nan")),
        lambda: RenderSpec(annotation="\n\t"),
        lambda: RenderSpec(provenance="x" * 241),
    )

    for factory in invalid_specs:
        with pytest.raises(ValueError):
            factory()


def test_artifact_and_panel_records_reject_tampered_metadata() -> None:
    artifact = render_fields(_fields(), RenderSpec())

    with pytest.raises(ArtifactIntegrityError, match="PNG digest"):
        replace(artifact, sha256="0" * 64)
    with pytest.raises(ArtifactIntegrityError, match="contract digest"):
        replace(artifact, contract_sha256="0" * 64)
    with pytest.raises(ArtifactIntegrityError, match="three or nine panels"):
        replace(artifact, panels=())
    with pytest.raises(ArtifactIntegrityError, match="role is unsupported"):
        PanelMetadata(
            role=cast(Any, "solver"),
            variable="vorticity",
            title="Vorticity",
            units="inverse lattice unit",
            colormap="RdBu_r",
            minimum=-1.0,
            maximum=1.0,
        )


def test_concurrent_renders_remain_byte_identical() -> None:
    fields = _fields()
    spec = RenderSpec()

    def render(_: int) -> bytes:
        return render_fields(fields, spec).data

    with ThreadPoolExecutor(max_workers=3) as executor:
        outputs = tuple(executor.map(render, range(3)))

    assert outputs[0] == outputs[1] == outputs[2]


def test_rendering_dependency_is_lazy_and_has_actionable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def unavailable(name: str, package: str | None = None) -> Any:
        if name == "matplotlib":
            raise ImportError("not installed")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", unavailable)

    with pytest.raises(DependencyUnavailableError, match=r"soufflerie\[viz\]"):
        render_fields(_fields(), RenderSpec())


def test_demo_contract_import_does_not_load_visualization_frameworks() -> None:
    code = """
import json
import sys

before = set(sys.modules)
import soufflerie.demo
loaded = sorted({name.split('.')[0] for name in set(sys.modules) - before})
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert set(json.loads(completed.stdout)).isdisjoint({"gradio", "imageio", "matplotlib", "PIL"})
